"""
GStools subpackage providing the Direct Sampling MPS simulation class.

.. currentmodule:: gstools.mps

The following classes and functions are provided

.. autosummary::
   DirectSampling
"""

import queue
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from gstools import config
from gstools.field.base import Field
from gstools.mps.distance import compute_node_weights
from gstools.mps.model import MPSModel
from gstools.mps.model import _validate_boundary as _mv_validate_boundary
from gstools.mps.model import _validate_max_radius as _mv_validate_max_radius
from gstools.mps.model import _validate_n_neighbors as _mv_validate_n_neighbors
from gstools.mps.model import (
    _validate_scan_fraction as _mv_validate_scan_fraction,
)
from gstools.mps.model import _validate_threshold as _mv_validate_threshold
from gstools.mps.training_image import TrainingImage
from gstools.normalizer.tools import apply_mean_norm_trend
from gstools.random.rng import RNG
from gstools.tools.geometric import (
    matrix_isometrize,
    no_of_angles,
    set_angles,
    set_anis,
)

__all__ = ["DirectSampling"]


# Sentinel variable name used when wrapping a univariate TI for the MV engine.
_MV_VAR = "_v"

# DS-mode scan block size.  Large enough that per-call NumPy overhead is
# negligible (essentially full vectorization speed), small enough that the
# greedy DS scan does not overcompute far past the first accepted match.
# This is a call-overhead-amortization constant, not a cache-tuned one.
_SCAN_BLOCK = 4096


def _resolve_nonstationary_map(param, sim_shape, n_stationary):
    """Return ``None`` or an array broadcastable to ``sim_shape``.

    Scalars and 0-d arrays are broadcast to the grid. A 1-D vector is a
    *stationary* multi-component value (e.g. a 3-element Tait–Bryan angle triple
    for a 3-D grid) and is accepted only when its length equals
    ``n_stationary`` — the number of components a stationary value has for this
    parameter and grid dimension. A *per-node* map must have leading dimensions
    equal to ``sim_shape`` (i.e. a full-shape array, not a flattened vector).

    Parameters
    ----------
    param : scalar or array-like or None
        User-supplied rotation/anis value.
    sim_shape : tuple
        Simulation grid shape.
    n_stationary : int
        Component count of a stationary value for this parameter/dimension
        (``no_of_angles(dim)`` for rotation, ``dim - 1`` for anis).

    Raises
    ------
    ValueError
        If a 1-D vector matches neither a stationary value nor the grid — the
        common cause is a per-node map passed *flattened* instead of shaped
        like the grid, which would otherwise silently apply only element [0].
    """
    if param is None:
        return None
    arr = np.asarray(param, dtype=np.float64)
    if arr.ndim == 0 or arr.size == 1:
        return np.full(sim_shape, float(arr.flat[0]))
    if arr.ndim == 1:
        # Only a genuine stationary multi-component value is accepted as 1-D.
        # A per-node map is multi-dimensional (leading dims == grid); a 1-D
        # array of any other length is almost certainly a flattened per-node
        # map and would silently degrade to its first element — reject it.
        if arr.size == n_stationary:
            return arr
        raise ValueError(
            f"1-D non-stationary value of length {arr.size} matches neither a "
            f"stationary value ({n_stationary} component(s) for a "
            f"{len(sim_shape)}-D grid) nor a per-node map. For per-node values "
            f"pass an array shaped like the grid {tuple(sim_shape)!r}, not a "
            f"flattened vector."
        )
    if arr.shape[: len(sim_shape)] == tuple(sim_shape):
        return arr
    raise ValueError(
        f"Non-stationary map shape {arr.shape!r} is incompatible with "
        f"simulation grid shape {tuple(sim_shape)!r}. Pass a scalar or "
        f"1-D vector for a stationary value, or an array whose leading "
        f"dimensions match the grid."
    )


def _precompute_offsets(shape, max_offset=None):
    """Neighbour offsets from the origin, sorted by Euclidean distance.

    Among equidistant offsets the order is a **canonical lexicographic**
    tie-break (by squared distance, then by each coordinate). This is
    platform- and NumPy-version-independent — unlike the default ``argsort``
    (unstable quicksort), whose tie order is implementation-defined — so that
    ``n_neighbors`` cutting mid-shell yields the same neighbour subset, and the
    simulation is reproducible across machines.

    Parameters
    ----------
    shape : tuple
        Simulation grid shape.
    max_offset : int, optional
        Maximum offset in any dimension.
        Default: ``max(shape)``.

    Returns
    -------
    numpy.ndarray, shape (N, dim)
    """
    dim = len(shape)
    if max_offset is None:
        max_offset = max(shape)
    estimated_elements = (2 * max_offset + 1) ** dim
    if estimated_elements > 5_000_000:
        raise ValueError(
            f"_precompute_offsets would allocate ~{estimated_elements * dim * 8 // 1_000_000} MB "
            f"(max_offset={max_offset}, dim={dim}). Set max_radius to bound the "
            "neighbourhood search radius."
        )
    rng_vals = np.arange(-max_offset, max_offset + 1)
    grid = np.array(np.meshgrid(*[rng_vals] * dim, indexing="ij"))
    offsets = grid.reshape(dim, -1).T
    offsets = offsets[np.any(offsets != 0, axis=1)]
    dist_sq = np.sum(offsets**2, axis=1)
    # lexsort: last key is primary. Primary = squared distance; ties broken by
    # coordinate 0, then 1, ... for a deterministic, portable ordering.
    keys = [offsets[:, d] for d in range(dim - 1, -1, -1)] + [dist_sq]
    idx = np.lexsort(keys)
    return offsets[idx]


def _select_neighbors(
    x_i,
    offset_arr,
    sim_shape_arr,
    sim_shape,
    path_pos_map,
    curr_idx,
    informed,
    max_radius,
    n_neighbors,
):
    """Closest valid neighbours of ``x_i``, with their path indices.

    A candidate is valid when it is in bounds, has path index ``< curr_idx``
    (already-simulated in path order) or ``-1`` (conditioning data), and — if
    ``informed`` is given — is marked informed.  ``offset_arr`` is
    distance-sorted, so slicing the first ``n_neighbors`` survivors yields the
    closest ones.

    Passing ``informed=None`` treats every in-bounds lower-index/conditioning
    cell as available; this is correct when building the dependency DAG, where
    all earlier-path nodes are informed by definition.

    Returns
    -------
    coords : numpy.ndarray, shape (m, dim)
        Neighbour coordinates, ``m <= n_neighbors``.
    path_idx : numpy.ndarray, shape (m,)
        Path index of each neighbour (``-1`` for conditioning data).
    """
    # ``offset_arr`` is distance-sorted, so the closest ``n_neighbors`` valid
    # candidates are the first ``n_neighbors`` survivors and we can stop the
    # moment we have them.  Iterating with an early break avoids masking the
    # whole offset array for every node (O(N**2) on large grids without a
    # ``max_radius`` cap).  Tradeoff: when fewer than ``n_neighbors`` valid
    # candidates exist (sparse early-path nodes), the scan still walks the full
    # ``offset_arr`` in Python; this affects only the first few nodes and is
    # bounded by the ``max_radius`` ball when one is set.
    dim = offset_arr.shape[1]
    r_sq = max_radius * max_radius if max_radius is not None else None
    found_coords = []
    found_vidx = []
    for off in offset_arr:
        # Distance-sorted: the first offset beyond the radius ends the scan.
        if r_sq is not None and float(off @ off) > r_sq:
            break
        cand = x_i + off
        if np.any(cand < 0) or np.any(cand >= sim_shape_arr):
            continue
        vi = path_pos_map[int(np.ravel_multi_index(tuple(cand), sim_shape))]
        if not (vi < curr_idx or vi == -1):
            continue
        if informed is not None and not informed[tuple(cand)]:
            continue
        found_coords.append(cand)
        found_vidx.append(vi)
        if len(found_coords) >= n_neighbors:
            break
    if found_coords:
        return (
            np.array(found_coords, dtype=offset_arr.dtype),
            np.array(found_vidx, dtype=path_pos_map.dtype),
        )
    return (
        np.empty((0, dim), dtype=offset_arr.dtype),
        np.empty(0, dtype=path_pos_map.dtype),
    )


def _build_dag_base(
    path, sim_shape, offset_arr, vmap_dict, n_k_dict, max_radius=None
):
    """Unified dependency-DAG builder for both univariate and multivariate DS.

    Parameters
    ----------
    vmap_dict : dict of {key: numpy.ndarray}
        One position map per key.  For univariate DS pass ``{"": path_pos_map}``.
    n_k_dict : dict of {key: int}
        Maximum neighbours per key, matching ``vmap_dict``.

    Returns
    -------
    indegree : dict of {key: numpy.ndarray of int32, shape (N,)}
    out_edges : list of list of (int, key)
        ``out_edges[j]`` → ``(i, key)`` pairs to process when node ``j`` completes.
    """
    N = len(path)
    sim_shape_arr = np.array(sim_shape)

    indegree = {k: np.zeros(N, dtype=np.int32) for k in vmap_dict}
    out_edges = [[] for _ in range(N)]
    for i in range(N):
        x_i = path[i]
        for key, vmap in vmap_dict.items():
            _, vidx = _select_neighbors(
                x_i,
                offset_arr,
                sim_shape_arr,
                sim_shape,
                vmap,
                i,
                None,
                max_radius,
                n_k_dict[key],
            )
            for j in vidx[vidx >= 0]:
                indegree[key][i] += 1
                out_edges[int(j)].append((i, key))
    return indegree, out_edges


def _transform_lags(lags, M, *arrays):
    """Apply geometric transform M and deduplicate collapsed lags.

    Parameters
    ----------
    lags : numpy.ndarray, shape (k, dim)
        Integer SG lag offsets (float64).
    M : numpy.ndarray, shape (dim, dim)
        Isometrization matrix from :func:`matrix_isometrize`.
    *arrays : numpy.ndarray
        Parallel arrays of length k to slice by the same keep_idx.

    Returns
    -------
    lags_ti : numpy.ndarray, shape (k', dim)
        Transformed lags (rounded to nearest integer, deduped).
    *sliced : numpy.ndarray
        Input arrays sliced to keep_idx.
    """
    lags_ti = np.rint(lags @ M.T)
    if len(lags_ti):
        _, keep_idx = np.unique(lags_ti, axis=0, return_index=True)
        keep_idx = np.sort(keep_idx)
        if len(keep_idx) < len(lags_ti):
            # Constant message (no per-call count) so the stdlib default
            # "once per location" filter collapses the routine anisotropic case
            # to a single warning per session instead of one per node.
            warnings.warn(
                "Anisotropy/rotation transform collapsed neighbour lag(s) onto "
                "duplicate TI positions; the duplicates are excluded from the "
                "data event. Reduce the anisotropy ratio or rotation to retain "
                "them.",
                RuntimeWarning,
                stacklevel=2,
            )
            lags_ti = lags_ti[keep_idx]
            arrays = tuple(a[keep_idx] for a in arrays)
    return (lags_ti,) + arrays


def _scan_window(
    lo, win_shape, start, max_scan, threshold, dist_fn, u_fallback, ti_shape
):
    """Chunked vectorized TI window scan (pure-Python path).

    Parameters
    ----------
    lo : numpy.ndarray
        Lower-left anchor of the search window in TI coordinates.
    win_shape : tuple of int
        Shape of the search window.
    start : int
        Starting position (flat index into window) for the random scan order.
    max_scan : int
        Maximum number of candidates to evaluate.
    threshold : float
        Distance threshold for early exit (DS mode). ``<= 0`` → DSBC (no early exit).
    dist_fn : callable
        ``dist_fn(y_blk)`` → 1-D distance array for a block of candidate
        anchor coordinates ``y_blk`` of shape ``(b, dim)``.
    u_fallback : numpy.ndarray, shape (dim,)
        Unused (retained for signature stability); the no-candidate fallback is
        now handled by the caller, which draws a *defined* TI cell.
    ti_shape : numpy.ndarray or tuple
        Unused (retained for signature stability).

    Returns
    -------
    y : numpy.ndarray, shape (dim,) or None
        Coordinate of the best matching TI anchor, or ``None`` when no candidate
        in the window has a finite distance (every candidate excluded — e.g. an
        all-undefined window on a masked TI). The caller treats ``None`` as the
        empty-neighbourhood case and draws a defined TI cell.
    """
    win_size = int(np.prod(win_shape))
    positions = (start + np.arange(max_scan)) % win_size
    y_all = lo + np.column_stack(np.unravel_index(positions, win_shape))

    # DSBC (threshold <= 0) has no threshold-based early exit, but an exact
    # match d == 0 is the best attainable candidate, so accept it immediately
    # (retained for categorical, where d == 0 is common — Juda2022 §2). Both
    # modes therefore scan in blocks of ``_SCAN_BLOCK`` and track the running
    # best, so a full-window DSBC scan never materialises one giant distance
    # array. Block argmin + strict ``<`` keep the first-in-scan-order winner,
    # matching a single ``argmin`` over the whole window.
    accept = threshold if threshold > 0 else 0.0

    # DS mode (threshold > 0): strict acceptance d < t (Mariethoz2010 ¶23).
    # DSBC mode (threshold <= 0, accept == 0): accept the exact match d == 0,
    # i.e. d <= 0, since distances are non-negative (Juda2022 §2).
    dsbc = threshold <= 0
    best_d, best_y = np.inf, None
    for b0 in range(0, max_scan, _SCAN_BLOCK):
        y_blk = y_all[b0 : b0 + _SCAN_BLOCK]
        d_blk = dist_fn(y_blk)
        under = d_blk <= accept if dsbc else d_blk < accept
        if np.any(under):
            return y_blk[int(np.argmax(under))]
        k = int(np.argmin(d_blk))
        if d_blk[k] < best_d:
            best_d = float(d_blk[k])
            best_y = y_blk[k]
    # ``best_y is None`` means every candidate had a non-finite (excluded)
    # distance — only reachable on a masked TI where the whole window is
    # undefined. Signal the caller to fall back to a defined TI draw (M10
    # para [15] empty-neighbourhood handling). For a fully-defined TI the loop
    # always sets best_y, so this returns a real anchor unchanged.
    return best_y


def _reduce_to_fit(lags_ti, ti_shape, *arrays):
    """Globally reduce an over-large (transformed) data event to fit the TI.

    Implements Mariethoz2010 para [43]: *"rotation or affinity may result in
    large data events that do not fit in the TI. In such cases, the data event
    nodes located outside of the TI are ignored until it becomes possible to
    scan the TI with this new, reduced data event."*

    The most-outside node — the one with the largest ``|lag|`` in the
    most-violating dimension — is dropped first and the test repeated, until a
    non-empty TI scan window exists for the reduced event. Nodes are dropped by
    being *outside the TI* (geometry), **regardless of SG neighbour rank**,
    which is the explicit distinction from :func:`_window_bounds` partial mode
    (furthest-first). The reduction is **global** — computed once per simulation
    node from the transformed lags and the TI shape alone, independent of which
    TI cell is being scanned. The collocated ``h=0`` row is never dropped.

    A window over a lag set exists iff, including the anchor at the origin, the
    per-dimension extent ``max(0, max h_d) - min(0, min h_d)`` is at most
    ``ti_shape_d - 1`` (a single lag with ``|h_d| > ti_d - 1`` therefore does
    not fit and is reduced away).

    Parameters
    ----------
    lags_ti : numpy.ndarray, shape (k, dim)
        Transformed (TI-frame) integer lag vectors.
    ti_shape : array-like, shape (dim,)
        Training image shape.
    *arrays : numpy.ndarray
        Parallel arrays of length ``k`` sliced identically to ``lags_ti``.

    Returns
    -------
    tuple
        ``(reduced_lags_ti, *reduced_arrays)``.
    """
    ti1 = np.asarray(ti_shape) - 1
    idx = np.arange(len(lags_ti))
    while len(idx):
        sub = lags_ti[idx]
        nzmask = np.any(sub != 0, axis=1)
        if not nzmask.any():
            break
        nz = sub[nzmask]
        # Include the anchor (origin) so a lone far lag is correctly infeasible.
        extent = np.maximum(0, nz.max(axis=0)) - np.minimum(0, nz.min(axis=0))
        over = extent - ti1
        if np.all(over <= 0):
            break
        d = int(np.argmax(over))  # most-violating dimension
        cand = np.flatnonzero(nzmask)
        drop = cand[int(np.argmax(np.abs(sub[cand, d])))]
        idx = np.delete(idx, drop)
    return (lags_ti[idx],) + tuple(a[idx] for a in arrays)


def _window_bounds(lags_ti, ti_shape, boundary):
    """Compute the TI search window for a set of lag vectors.

    h=0 rows (collocated constraints) do not constrain the window — they map
    anchor ``y`` to ``y`` itself so every window position is valid for them.

    Parameters
    ----------
    lags_ti : numpy.ndarray, shape (k, dim)
        Lag vectors in TI frame (integer-valued float64).  May include h=0 rows.
    ti_shape : numpy.ndarray, shape (dim,)
        TI shape as an integer array.
    boundary : str
        ``"strict"`` or ``"partial"``.

    Returns
    -------
    win_lo : numpy.ndarray, shape (dim,), dtype int
    win_hi : numpy.ndarray, shape (dim,), dtype int
    valid_count : int
        Number of lags to use (``lags_ti[:valid_count]``).  ``-1`` signals that
        no subset of the data event fits inside the TI; the caller should draw a
        random TI value instead of scanning.
    """
    ti_shape_arr = np.asarray(ti_shape, dtype=int)
    dim = len(ti_shape_arr)
    nz = np.flatnonzero(np.any(lags_ti != 0, axis=1))
    if not len(nz):
        return np.zeros(dim, dtype=int), ti_shape_arr - 1, len(lags_ti)

    lags_nz = lags_ti[nz]
    if boundary == "strict":
        win_lo = np.maximum(0, np.ceil(-lags_nz.min(axis=0))).astype(int)
        win_hi = np.minimum(
            ti_shape_arr - 1,
            np.floor(ti_shape_arr - 1 - lags_nz.max(axis=0)),
        ).astype(int)
        if np.all(win_lo <= win_hi):
            return win_lo, win_hi, len(lags_ti)
        # Strict window infeasible (TI too small for the full data event).
        # Fall through to partial truncation, but do not do so silently.
        warnings.warn(
            "boundary='strict' could not fit the full data event inside the "
            "TI; falling back to partial mode (dropping the furthest "
            "neighbour(s)). Ensure the TI is at least as large as the data "
            "event extent to enforce strict mode.",
            RuntimeWarning,
            stacklevel=2,
        )

    keep = len(lags_ti)
    while keep > 0:
        lv_k = lags_ti[:keep]
        nzk = lv_k[np.any(lv_k != 0, axis=1)]
        if not len(nzk):
            return np.zeros(dim, dtype=int), ti_shape_arr - 1, keep
        sw_lo = np.maximum(0, np.ceil(-nzk.min(axis=0))).astype(int)
        sw_hi = np.minimum(
            ti_shape_arr - 1,
            np.floor(ti_shape_arr - 1 - nzk.max(axis=0)),
        ).astype(int)
        if np.all(sw_lo <= sw_hi):
            return sw_lo, sw_hi, keep
        keep -= 1
    return None, None, -1


def _univar_as_mv_ti(ti):
    """Wrap a univariate TrainingImage as a single-variable multivariate TI.

    The resulting TI is bit-identical to a hand-crafted single-variable MV TI
    using the same data and distance parameters.  Used by ``ds_simulate`` and
    ``DirectSampling.__call__`` to route univariate work through ``ds_simulate``.

    Re-wraps the *same* data the user already constructed, so any NaN/masked-TI
    warning was already raised at the user's construction; suppress the
    duplicate here.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return TrainingImage(
            {_MV_VAR: ti.data},
            categorical={_MV_VAR: ti.categorical},
            distance={_MV_VAR: ti.distance_type},
            distance_power=ti.distance_power,
        )


def _make_progress(progress, total, desc):
    """Build ``(update, close)`` callbacks for an optional progress display.

    Parameters
    ----------
    progress : bool or callable or None
        ``None``/``False`` disables it. ``True`` shows a :mod:`tqdm` bar when
        ``tqdm`` is importable, otherwise prints the percentage at 5 % steps on
        a single overwritten line. A callable is invoked as
        ``progress(n_done, total)`` once per completed simulation node.
    total : int
        Number of nodes that will be simulated (``len(path)``).
    desc : str
        Short label for the bar (e.g. ``"DS"``).

    Returns
    -------
    update : callable
        Call (no arguments) once per completed node.
    close : callable
        Call once when the simulation finishes.
    """
    if not progress or total <= 0:
        return (lambda: None), (lambda: None)
    if callable(progress):
        state = {"done": 0}

        def update():
            state["done"] += 1
            progress(state["done"], total)

        return update, (lambda: None)
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    if tqdm is not None:
        bar = tqdm(total=total, desc=desc, unit="node")
        return (lambda: bar.update(1)), bar.close
    # Dependency-free fallback: overwrite a single line at 5 % steps.
    state = {"done": 0, "last": -1}

    def update():
        state["done"] += 1
        pct = 5 * (state["done"] * 20 // total)
        if pct != state["last"]:
            state["last"] = pct
            print(
                f"\r{desc}: {pct:3d}% ({state['done']}/{total})",
                end="",
                flush=True,
            )

    def close():
        print()

    return update, close


def _run_path(
    path,
    u_start,
    u_fallback,
    simulate_fn,
    write_fn,
    update_fn,
    executor,
    offset_arr,
    vmap,
    n_k,
    sim_shape,
    max_radius,
):
    """Dispatch the node simulation path: serial or parallel DAG.

    simulate_fn(i, x_i, u_start_i, u_fallback_i) → result dict
    write_fn(node_tuple, result) → None
    update_fn() → None

    Extracted so a future syn-processing path strategy (which re-opens
    simulated nodes) can replace this function without touching the scan
    kernel.  Note: syn-processing is incompatible with DAG parallelism;
    replace the whole _run_path, not just the serial branch.
    """
    variables = list(vmap)

    if executor is not None:
        indegree, out_edges = _build_dag_base(
            path, sim_shape, offset_arr, vmap, n_k, max_radius
        )
        remaining = {v: indegree[v].copy() for v in variables}
        submitted = np.zeros(len(path), dtype=bool)
        done_q = queue.Queue()
        counts = {"submitted": 0, "done": 0}

        def _ready(i):
            return all(remaining[v][i] == 0 for v in variables)

        def _run(i):
            return i, simulate_fn(i, path[i], u_start[i], u_fallback[i])

        def _submit(i):
            if submitted[i]:
                return
            submitted[i] = True
            executor.submit(_run, i).add_done_callback(done_q.put)
            counts["submitted"] += 1

        for i in range(len(path)):
            if _ready(i):
                _submit(i)

        while counts["done"] < counts["submitted"]:
            i, result = done_q.get().result()
            counts["done"] += 1
            write_fn(tuple(int(c) for c in path[i]), result)
            update_fn()
            for j, v in out_edges[i]:
                remaining[v][j] -= 1
                if _ready(j):
                    _submit(j)
    else:
        for i in range(len(path)):
            result = simulate_fn(i, path[i], u_start[i], u_fallback[i])
            write_fn(tuple(int(c) for c in path[i]), result)
            update_fn()


def ds_simulate(
    training_image,
    sim_shape,
    n_neighbors,
    threshold,
    scan_fraction,
    rng_path,
    rng_nodes,
    conditions=None,
    cond_weight=1.0,
    boundary="strict",
    max_radius=None,
    num_threads=None,
    rotation_map=None,
    anis_map=None,
    progress=None,
):
    """Node-wise multivariate Direct Sampling (Mariethoz2010 §3, Eq. 8).

    Co-simulates every variable of a dict-valued :class:`TrainingImage` on one
    structured grid, treating all variables equally. The path visits every
    *node* with at least one unknown variable; at each node a single joint scan
    finds the TI cell ``y`` minimising the weighted distance ``Σ_k w_k d_k`` over
    all variables, and that one cell's whole vector ``{TI[v][y]}`` is copied to
    the node's *uninformed* variables. Because every variable at a node is drawn
    from the same TI cell, the joint (cross-variable) relationship is reproduced
    exactly. Variables already known at the node (only ever via conditioning
    data) act as collocated ``h = 0`` constraints, weighted by ``cond_weight``.

    Parameters
    ----------
    training_image : TrainingImage
        Multivariate training image (``training_image.multivariate`` is True).
    sim_shape : tuple
        Simulation grid shape.
    n_neighbors : int or dict
        Maximum neighbours per variable. An ``int`` broadcasts to all variables;
        a dict gives one value per variable name.
    threshold : float
        Distance threshold (Juda2022 §2). ``0.0`` -> DSBC mode.
    scan_fraction : float
        Fraction of the TI to scan per node, capped at the valid search window.
    rng_path : numpy.random.RandomState
        RandomState controlling the simulation path (node visit order).
    rng_nodes : numpy.random.RandomState
        RandomState controlling TI scan entry points and fallback cells.
    conditions : dict, optional
        ``{node_index: {variable: value}}`` conditioning data.
    cond_weight : float, optional
        Weight delta for conditioning nodes (Mariethoz2010 §3 ¶26).
    boundary : str, optional
        ``"strict"`` (default) or ``"partial"`` search-window strategy.
    max_radius : float, optional
        Euclidean cap on neighbour selection.
    num_threads : int or None, optional
        Threads for the node-wise DAG. ``None`` -> ``config.NUM_THREADS``.
    rotation_map : numpy.ndarray or None, optional
        Per-node rotation angles, shape matching the simulation grid. ``None``
        → no rotation (stationary). Use ``DirectSampling.set_nonstationary``
        to produce this array from user-facing scalar or array inputs.
    anis_map : numpy.ndarray or None, optional
        Per-node anisotropy ratios, shape matching the simulation grid.
        ``None`` → isotropic (stationary). All values must be positive.
    progress : bool or callable or None, optional
        Show simulation progress. ``True`` displays a :mod:`tqdm` bar (or a
        plain percentage line if ``tqdm`` is not installed); a callable is
        invoked as ``progress(n_done, n_total)`` once per completed node.
        ``None``/``False`` (default) disables it.

    Returns
    -------
    dict
        ``{variable: numpy.ndarray}`` — one simulated field per variable.
    """
    variables = training_image.variables
    weights = (
        training_image.weights
    )  # hoisted once (avoid per-block dict copy)
    ti_shape = np.array(training_image.shape)
    sim_shape_arr = np.array(sim_shape)
    dim = len(sim_shape)

    n_k = (
        {v: int(n_neighbors[v]) for v in variables}
        if isinstance(n_neighbors, dict)
        else {v: int(n_neighbors) for v in variables}
    )

    sg = {v: np.full(sim_shape, np.nan) for v in variables}
    informed = {v: np.zeros(sim_shape, dtype=bool) for v in variables}
    is_cond = {v: np.zeros(sim_shape, dtype=bool) for v in variables}
    if conditions:
        for idx, vd in conditions.items():
            for v, val in vd.items():
                sg[v][idx] = val
                is_cond[v][idx] = True
                informed[v][idx] = True

    n_threads = (
        num_threads if num_threads is not None else (config.NUM_THREADS or 1)
    )
    executor = (
        ThreadPoolExecutor(max_workers=n_threads) if n_threads > 1 else None
    )
    max_off_int = int(np.ceil(max_radius)) if max_radius is not None else None
    offset_arr = _precompute_offsets(sim_shape, max_off_int)

    # Node path: every node with >= 1 uninformed variable.  Fully-conditioned
    # nodes need no simulation and are excluded (they stay -1 / always available).
    unknown = np.zeros(sim_shape, dtype=bool)
    for v in variables:
        unknown |= np.isnan(sg[v])
    path = np.argwhere(unknown)
    path = path[rng_path.permutation(len(path))]
    n_nodes = len(path)
    u_start = rng_nodes.uniform(size=n_nodes)
    u_fallback = rng_nodes.uniform(size=(n_nodes, dim))

    sg_size = int(np.prod(sim_shape))
    path_flat = (
        np.ravel_multi_index(path.T, sim_shape)
        if len(path)
        else np.empty(0, dtype=np.intp)
    )
    # Per-variable position maps over the node path.  A path node carries its
    # node-path index; a cell conditioned in variable v is set to -1 in vmap[v]
    # (always available for v, like univariate conditioning) — essential for a
    # partially-conditioned node, whose known value must not be gated by path
    # order.  Cells absent from the path are already -1.
    vmap = {}
    for v in variables:
        m = np.full(sg_size, -1, dtype=np.intp)
        m[path_flat] = np.arange(len(path_flat))
        m[np.flatnonzero(is_cond[v].reshape(-1))] = -1
        vmap[v] = m

    # Hoist TI variable arrays once — avoids repeated method-call overhead
    # inside the per-node scan loop; all inner closures read from this dict.
    ti_vars = {v: training_image.variable(v) for v in variables}

    # Masked-TI support: NaN cells are undefined. When present, distances
    # exclude them per-position and fallback draws are restricted to cells that
    # are defined in *every* variable (so a joint draw never yields NaN). The
    # ``ti_has_nan`` gate keeps the fully-defined path byte-identical.
    ti_has_nan = any(
        np.issubdtype(a.dtype, np.floating) and np.isnan(a).any()
        for a in ti_vars.values()
    )
    if ti_has_nan:
        finite_all = np.ones(ti_shape, dtype=bool)
        for a in ti_vars.values():
            if np.issubdtype(a.dtype, np.floating):
                finite_all &= ~np.isnan(a)
        finite_flat = np.flatnonzero(finite_all.reshape(-1))
        if finite_flat.size == 0:
            raise ValueError(
                "TrainingImage has no cell defined in all variables (every "
                "cell is NaN in at least one variable); cannot simulate."
            )
    else:
        finite_flat = None

    def _rand_fallback(targets, u_fb_i):
        # Single random TI cell supplies the whole node-vector (preserves the
        # joint relationship); never an independent draw per variable. On a
        # masked TI the draw is restricted to cells defined in every variable.
        if ti_has_nan:
            flat = finite_flat[int(u_fb_i[0] * len(finite_flat))]
            cell = tuple(
                int(c) for c in np.unravel_index(flat, tuple(ti_shape))
            )
        else:
            cell = tuple(int(u_fb_i[d] * s) for d, s in enumerate(ti_shape))
        return {v: float(ti_vars[v][cell]) for v in targets}

    def _joint_scan(
        lo,
        win_shape,
        int_lags,
        de_v,
        cm_v,
        ln_v,
        u_start_i,
        u_fallback_i,
        scan_targets,
    ):
        # Mirrors _scan_ti: full vectorized argmin for DSBC (threshold <= 0),
        # chunked first-under-threshold for DS (threshold > 0).  Returns the
        # single best TI cell coordinate ``y``; the caller copies TI[v][y] to all
        # uninformed variables.  The window is the intersection of per-variable
        # boxes, so every ``y + lag`` is in bounds and the gather needs no
        # clipping; the h=0 lag maps to ``y`` itself.
        win_size = int(np.prod(win_shape))
        # Scan fraction is of the TI (Mariethoz2010 ¶24, Juda2022 §2), capped at
        # the valid search window so we never wrap around and re-scan anchors.
        ti_size = int(np.prod(ti_shape))
        max_scan = max(1, min(win_size, int(scan_fraction * ti_size)))
        start = int(u_start_i * win_size)

        active_vars = [v for v in variables if v in int_lags]
        active_w_total = sum(weights[v] for v in active_vars)
        # Compute node weights once; reused across the _dist_block scan.
        precomp_w = {
            v: compute_node_weights(
                len(de_v[v]),
                ln_v[v],
                training_image.distance_power,
                cm_v[v],
                cond_weight,
            )
            for v in active_vars
        }

        def _dist_block(y_blk):
            d = np.zeros(len(y_blk))
            for v in active_vars:
                il = int_lags[v]
                coords = y_blk[:, None, :] + il[None, :, :]
                all_de_ti = ti_vars[v][tuple(coords.transpose(2, 0, 1))]
                d += weights[v] * training_image.vec_distance_var(
                    v,
                    de_v[v],
                    all_de_ti,
                    cm_v[v],
                    cond_weight,
                    ln_v[v],
                    weights=precomp_w[v],
                    has_nan=ti_has_nan,
                )
            # Renormalize so the joint distance stays in [0, 1] even when
            # some variables have no data event and are excluded from int_lags.
            # Without this, the threshold fires on a compressed scale and
            # accepts matches that should be rejected.
            if 0.0 < active_w_total < 1.0:
                d /= active_w_total
            if ti_has_nan:
                # Never select an anchor whose pasted value would be undefined:
                # the matched cell must be defined in every target variable.
                center_ok = np.ones(len(y_blk), dtype=bool)
                ys = tuple(y_blk.T)
                for v in scan_targets:
                    cv = ti_vars[v][ys]
                    if np.issubdtype(cv.dtype, np.floating):
                        center_ok &= np.isfinite(cv)
                d = np.where(center_ok, d, np.inf)
            return d

        return _scan_window(
            lo,
            win_shape,
            start,
            max_scan,
            threshold,
            _dist_block,
            u_fallback_i,
            ti_shape,
        )

    def _simulate_node_mv(curr_idx, x_i, u_start_i, u_fallback_i):
        x_i_t = tuple(int(c) for c in x_i)
        targets = [v for v in variables if np.isnan(sg[v][x_i_t])]

        lags_v, de_v, cm_v, ln_v = {}, {}, {}, {}
        for var in variables:
            coords, _ = _select_neighbors(
                x_i,
                offset_arr,
                sim_shape_arr,
                sim_shape,
                vmap[var],
                curr_idx,
                informed[var],
                max_radius,
                n_k[var],
            )
            if len(coords):
                lv = (coords - x_i).astype(np.float64)
                dv = sg[var][tuple(coords.T)]
                cv = is_cond[var][tuple(coords.T)]
                lnv = np.linalg.norm(lv, axis=1)
            else:
                lv = np.empty((0, dim), dtype=np.float64)
                dv = np.empty(0)
                cv = np.empty(0, dtype=bool)
                lnv = np.empty(0)
            # collocated h=0 — a variable known at this very node (only possible
            # via conditioning, as the node's unknown variables are what we fill).
            # Never added for a target variable (uninformed here by definition).
            if informed[var][x_i_t]:
                lv = np.concatenate([np.zeros((1, dim)), lv], axis=0)
                dv = np.concatenate([[sg[var][x_i_t]], dv])
                cv = np.concatenate([[is_cond[var][x_i_t]], cv])
                lnv = np.concatenate([[0.0], lnv])
            lags_v[var], de_v[var], cm_v[var], ln_v[var] = lv, dv, cv, lnv

        # Geometric transform: map all per-variable SG lags into TI frame.
        # de_v / cm_v / ln_v stay on original SG geometry for SG-side ops.
        # dict(lags_v) creates a shallow-copy dict so partial-mode truncation
        # of lags_ti_v does not alias back into lags_v.
        if rotation_map is not None or anis_map is not None:
            angles_i = set_angles(
                dim,
                (
                    rotation_map
                    if rotation_map.ndim == 1
                    else rotation_map[tuple(x_i)]
                )
                if rotation_map is not None
                else 0.0,
            )
            anis_i = set_anis(
                dim,
                (anis_map if anis_map.ndim == 1 else anis_map[tuple(x_i)])
                if anis_map is not None
                else 1.0,
            )
            M = matrix_isometrize(dim, angles_i, anis_i)
            lags_ti_v = {}
            for v in variables:
                lv_ti, lags_v[v], de_v[v], cm_v[v], ln_v[v] = _transform_lags(
                    lags_v[v], M, lags_v[v], de_v[v], cm_v[v], ln_v[v]
                )
                # M10 para [43]: globally reduce an over-large transformed event
                # to fit the TI, dropping the most-outside node first (by TI
                # extent, regardless of SG rank). One reduction per simulation
                # node, independent of the TI scan position.
                lv_ti, lags_v[v], de_v[v], cm_v[v], ln_v[v] = _reduce_to_fit(
                    lv_ti, ti_shape, lags_v[v], de_v[v], cm_v[v], ln_v[v]
                )
                lags_ti_v[v] = lv_ti
        else:
            lags_ti_v = dict(lags_v)

        if all(len(lags_v[v]) == 0 for v in variables):
            return _rand_fallback(targets, u_fallback_i)

        # Per-variable search window computed via _window_bounds, then intersected.
        # h=0 lags are excluded inside _window_bounds — they map to y itself and
        # never constrain the window.
        win_lo = np.zeros(dim, dtype=int)
        win_hi = ti_shape - 1
        for var in variables:
            lv_ti = lags_ti_v[var]
            lo, hi, valid_count = _window_bounds(lv_ti, ti_shape, boundary)
            if valid_count == -1:
                return _rand_fallback(targets, u_fallback_i)
            if valid_count < len(lv_ti):
                # Truncate TI-frame and original arrays to the same keep count.
                # lags_ti_v uses a separate dict so this does not affect lags_v.
                lags_ti_v[var] = lv_ti[:valid_count]
                lags_v[var] = lags_v[var][:valid_count]
                de_v[var] = de_v[var][:valid_count]
                cm_v[var] = cm_v[var][:valid_count]
                ln_v[var] = ln_v[var][:valid_count]
            win_lo = np.maximum(win_lo, lo)
            win_hi = np.minimum(win_hi, hi)

        if np.any(win_lo > win_hi):
            return _rand_fallback(targets, u_fallback_i)

        # Integer lags per variable (already exact integers as float64, incl.
        # the 0.0 h=0 row); reused for the scan and the mean-shift gather.
        int_lags = {
            v: lags_ti_v[v].astype(int) for v in variables if len(lags_ti_v[v])
        }
        y = _joint_scan(
            win_lo,
            tuple(win_hi - win_lo + 1),
            int_lags,
            de_v,
            cm_v,
            ln_v,
            u_start_i,
            u_fallback_i,
            targets,
        )
        if y is None:
            # No candidate in the window was defined (masked TI): treat as the
            # empty-neighbourhood case and draw a defined TI cell. Avoids the
            # ``y + lag`` gather below, which an unconstrained cell could send
            # out of bounds.
            return _rand_fallback(targets, u_fallback_i)
        y_t = tuple(int(c) for c in y)
        # Copy the single matched cell's vector to every uninformed variable.
        result = {}
        for v in targets:
            ti_val = float(ti_vars[v][y_t])
            il = int_lags.get(v)
            de_ti_v = (
                ti_vars[v][tuple((y + il).T)]
                if il is not None
                else np.empty(0)
            )
            result[v] = training_image.adjust_value(
                ti_val, de_v[v], de_ti_v, var=v
            )
        return result

    def _write_result(node, result):
        for v, val in result.items():
            if np.isnan(val):
                raise ValueError(
                    f"Simulation produced NaN for {(v, node)}. Check TI data."
                )
            sg[v][node] = val
            informed[v][node] = True

    update_progress, close_progress = _make_progress(progress, len(path), "DS")

    try:
        _run_path(
            path,
            u_start,
            u_fallback,
            lambda i, x_i, u_st, u_fb: _simulate_node_mv(i, x_i, u_st, u_fb),
            _write_result,
            update_progress,
            executor,
            offset_arr,
            vmap,
            n_k,
            sim_shape,
            max_radius,
        )
    finally:
        close_progress()
        if executor is not None:
            executor.shutdown(wait=True)

    return sg


class DirectSampling(Field):
    """Multiple Point Statistics simulation using Direct Sampling.

    Subclasses :class:`gstools.field.base.Field`. Takes a :class:`TrainingImage`
    (analogous to :class:`CovModel`) and produces fields on structured grids.

    Parameters
    ----------
    ti : TrainingImage
        The training image (the MPS model).
    n_neighbors : int or dict of {str: int}, optional
        Maximum neighbors in the data event. A dict gives one value per variable
        name (multivariate TIs only); an int broadcasts to all variables.
        Default: 32.
    scan_fraction : float, optional
        Fraction of the TI to scan per node (capped at the valid search
        window). Default: 1.
    threshold : float, optional
        Distance threshold. 0.0 -> DSBC mode. Default: 0.0.
    cond_weight : float, optional
        Weight for conditioning nodes in distance. Default: 1.0.
    boundary : str, optional
        Search-window strategy: ``"strict"`` (default) or ``"partial"``.
    max_radius : float, optional
        Exclude SG neighbours beyond this Euclidean distance from the
        data event. Default: ``None`` (no limit).
        The minimum effective value is 1.0 (the grid-cell Euclidean
        distance to the nearest neighbour). Values in ``(0, 1)`` accept
        no neighbours at all, making every node fall back to a random TI
        sample.
    seed : int or nan, optional
        Master RNG seed. Default: nan.
    """

    default_field_names = ["field"]

    @staticmethod
    def _validate_variation_n_neighbors(n_neighbors, ti):
        """Reject variation distance with a single-neighbour data event.

        The variation distance (Mariethoz2010 Eq. 9) compares each event to its
        own local mean. With ``n_neighbors == 1`` the local mean equals the sole
        value, so every deviation is zero and every TI candidate matches — the
        scan degenerates to a random draw. Fail fast instead.
        """
        dist_type = ti.distance_type
        if isinstance(dist_type, dict):
            dist_map = dist_type
            n_map = (
                n_neighbors
                if isinstance(n_neighbors, dict)
                else {v: n_neighbors for v in dist_map}
            )
        else:
            dist_map = {None: dist_type}
            n_map = {None: n_neighbors}
        for var, dt in dist_map.items():
            if isinstance(dt, str) and dt.lower().startswith("variation"):
                if int(n_map[var]) < 2:
                    where = "" if var is None else f" for variable {var!r}"
                    raise ValueError(
                        f"distance='variation' requires n_neighbors >= 2{where}: "
                        "a single-point data event has no local mean deviation."
                    )

    def __init__(
        self,
        model_or_ti,
        seed=np.nan,
        **back_compat_kwargs,
    ):
        if isinstance(model_or_ti, MPSModel):
            if back_compat_kwargs:
                raise TypeError(
                    f"DirectSampling: keyword arguments are not accepted when "
                    f"passing an MPSModel; got: {list(back_compat_kwargs)}"
                )
            model = model_or_ti
        else:
            algo_kw = {
                k: back_compat_kwargs.pop(k)
                for k in (
                    "n_neighbors",
                    "scan_fraction",
                    "threshold",
                    "cond_weight",
                    "boundary",
                    "max_radius",
                )
                if k in back_compat_kwargs
            }
            num_threads = back_compat_kwargs.pop("num_threads", None)
            if back_compat_kwargs:
                raise TypeError(
                    f"DirectSampling: unexpected keyword arguments: "
                    f"{list(back_compat_kwargs)}"
                )
            model = MPSModel(model_or_ti, **algo_kw)

        self._model = model
        super().__init__(model=None, dim=model.ti.ndim, value_type="scalar")
        self._ti = model.ti
        self._n_neighbors = model.n_neighbors
        DirectSampling._validate_variation_n_neighbors(
            self._n_neighbors, self._ti
        )
        self._scan_fraction = model.scan_fraction
        self._threshold = model.threshold
        self._cond_weight = model.cond_weight
        self._boundary = model.boundary
        self._max_radius = model.max_radius
        self._num_threads = num_threads
        self._cond_pos = None
        self._cond_val = None
        self._mv_mean = {}
        self._mv_normalizer = {}
        self._mv_trend = {}
        self._rotation = None
        self._anis = None
        self.rng = RNG(None if np.isnan(seed) else int(seed))
        if model.ti.multivariate:
            for v in model.ti.variables:
                if not v.isidentifier() or v in dir(self):
                    raise ValueError(
                        f"DirectSampling: variable name {v!r} cannot be used as "
                        f"a field name; use a valid Python identifier that does "
                        f"not collide with an existing attribute."
                    )

    def __call__(
        self,
        pos=None,
        seed=np.nan,
        path_seed=np.nan,
        node_seed=np.nan,
        mesh_type="structured",
        post_process=True,
        store=True,
        progress=None,
        num_threads=None,
    ):
        """Generate the spatial random field via Direct Sampling.

        The field is saved as ``self.field`` and is also returned.

        Parameters
        ----------
        pos : :class:`list`, optional
            The position tuple, containing main direction and transversal
            directions. Only structured grids are supported.
        seed : :class:`int`, optional
            Seed for the RNG. If ``np.nan``, the current seed is kept.
            Default: ``np.nan``
        path_seed : :class:`int` or :any:`numpy.nan`, optional
            Seed controlling the order in which simulation grid nodes are
            visited. If ``np.nan`` (default), derived from the master RNG
            together with ``node_seed``. Fix this while varying ``node_seed``
            to study the effect of TI search randomness under a constant
            visit order, or vice versa.
            Default: ``np.nan``
        node_seed : :class:`int` or :any:`numpy.nan`, optional
            Seed controlling the TI scan entry point and fallback cell for
            every node. If ``np.nan`` (default), derived from the master RNG.
            Default: ``np.nan``
        mesh_type : :class:`str`, optional
            Grid type. Must be ``"structured"``.
            Default: ``"structured"``
        post_process : :class:`bool`, optional
            Whether to apply post-processing transformations (mean,
            normalizer, trend) to the field. Default: :any:`True`
        store : :class:`bool` or :class:`str`, optional
            Whether to store the field (``True``), not store it (``False``),
            or store it under a custom name (string).
            Default: :any:`True`
        progress : :class:`bool` or callable or None, optional
            Show simulation progress. ``True`` displays a :mod:`tqdm` bar (or a
            plain percentage line if ``tqdm`` is not installed); a callable is
            invoked as ``progress(n_done, n_total)`` once per completed node.
            ``None``/``False`` (default) disables it.

        Returns
        -------
        field : :class:`numpy.ndarray` or :class:`dict`
            For a univariate training image, the simulated field (also saved as
            ``self.field``). For a multivariate training image, a
            ``{variable: numpy.ndarray}`` dict with all variables on equal
            footing — each is also stored as a named field accessible via
            ``self[variable]`` / :attr:`all_fields`.
        """
        if mesh_type != "structured":
            raise ValueError(
                "DirectSampling: only structured grids are supported."
            )
        if self._ti.multivariate and isinstance(store, str):
            # A multivariate run produces one field per variable; there is no
            # single privileged field to store under a custom name. Reject it
            # explicitly rather than silently dropping the name (the fields are
            # always stored under their variable names).
            raise ValueError(
                "DirectSampling: a custom store name is not supported for "
                "multivariate training images; each variable is stored under "
                "its own name. Use store=True/False instead."
            )
        name, save = self.get_store_config(store)
        pos, shape = self.pre_pos(pos, mesh_type)
        # Stationary component counts disambiguate a 1-D value from a flattened
        # per-node map: rotation has no_of_angles(dim) angles, anis has dim-1.
        rotation_map = _resolve_nonstationary_map(
            self._rotation, shape, no_of_angles(len(shape))
        )
        anis_map = _resolve_nonstationary_map(
            self._anis, shape, max(len(shape) - 1, 1)
        )
        conditions = self._conditions_to_grid(self.pos)
        if not np.isnan(seed):
            self.rng.seed = int(seed)
        rng_path = (
            self.rng.random
            if np.isnan(path_seed)
            else RNG(int(path_seed)).random
        )
        rng_nodes = (
            self.rng.random
            if np.isnan(node_seed)
            else RNG(int(node_seed)).random
        )
        # Call-time num_threads overrides the instance default.
        n_threads = (
            num_threads if num_threads is not None else self._num_threads
        )
        if self._ti.multivariate:
            result = ds_simulate(
                training_image=self._ti,
                sim_shape=shape,
                n_neighbors=self._n_neighbors,
                threshold=self._threshold,
                scan_fraction=self._scan_fraction,
                rng_path=rng_path,
                rng_nodes=rng_nodes,
                conditions=conditions,
                cond_weight=self._cond_weight,
                boundary=self._boundary,
                max_radius=self._max_radius,
                num_threads=n_threads,
                rotation_map=rotation_map,
                anis_map=anis_map,
                progress=progress,
            )
            # Equal treatment: every variable is a first-class named field
            # (no privileged primary). Returned as a dict keyed by variable name.
            # Per-variable transforms (mean/normalizer/trend) are applied here
            # using the dicts set via set_mv_transforms(); variables absent from
            # those dicts fall back to the inherited self.mean/normalizer/trend.
            # post_field is then called with process=False to handle storage only.
            out = {}
            for v in self._ti.variables:
                fld = result[v]
                if post_process:
                    mv_mean = self._mv_mean.get(v, self.mean)
                    mv_norm = self._mv_normalizer.get(v, self.normalizer)
                    mv_trend = self._mv_trend.get(v, self.trend)
                    fld = apply_mean_norm_trend(
                        pos=self.pos,
                        field=fld,
                        mesh_type=self.mesh_type,
                        value_type=self.value_type,
                        mean=mv_mean,
                        normalizer=mv_norm,
                        trend=mv_trend,
                        check_shape=False,
                        stacked=False,
                    )
                out[v] = self.post_field(fld, name=v, process=False, save=save)
            return out
        mv_ti = _univar_as_mv_ti(self._ti)
        mv_cond = (
            {idx: {_MV_VAR: val} for idx, val in conditions.items()}
            if conditions
            else None
        )
        mv_result = ds_simulate(
            training_image=mv_ti,
            sim_shape=shape,
            n_neighbors=self._n_neighbors,
            threshold=self._threshold,
            scan_fraction=self._scan_fraction,
            rng_path=rng_path,
            rng_nodes=rng_nodes,
            conditions=mv_cond,
            cond_weight=self._cond_weight,
            boundary=self._boundary,
            max_radius=self._max_radius,
            num_threads=n_threads,
            rotation_map=rotation_map,
            anis_map=anis_map,
            progress=progress,
        )
        return self.post_field(mv_result[_MV_VAR], name, post_process, save)

    def _conditions_to_grid(self, axes):
        """Snap conditioning points to nearest grid nodes (Mariethoz2010 §3 ¶12).

        Univariate returns ``{idx: value}``; multivariate returns
        ``{idx: {variable: value}}`` with non-finite (NaN) entries skipped. When
        two points snap to the same node, the conflict is resolved
        **per variable**: for each variable the closest colliding point with a
        finite value wins. A farther point therefore still contributes its
        finite values for any variable the closer point left ``NaN``, so valid
        conditioning data is never silently discarded.

        Parameters
        ----------
        axes : tuple of numpy.ndarray
            Raw grid axis arrays (``self.pos``).

        Returns
        -------
        dict
        """
        if self._cond_pos is None:
            return {}
        if self._ti.multivariate and isinstance(self._cond_val, dict):
            if not self._cond_val:
                raise ValueError("cond_val must not be empty")
            # idx -> {variable: (value, dist_sq)}; the closest finite value per
            # variable wins, so a farther point fills variables the closer one
            # left NaN (per-variable collision resolution).
            candidates = {}
            n_cond = len(next(iter(self._cond_val.values())))
            for k in range(n_cond):
                idx, dist_sq = self._snap_to_nearest(axes, k)
                var_best = candidates.setdefault(idx, {})
                for v in self._cond_val:
                    val = self._cond_val[v][k]
                    if not np.isfinite(val):
                        continue
                    if v not in var_best or dist_sq < var_best[v][1]:
                        var_best[v] = (float(val), dist_sq)
            return {
                idx: {v: val for v, (val, _) in var_best.items()}
                for idx, var_best in candidates.items()
            }
        # univariate (unchanged)
        candidates = {}  # idx -> (val, dist_sq)
        for k in range(self._cond_val.shape[0]):
            idx, dist_sq = self._snap_to_nearest(axes, k)
            if idx not in candidates or dist_sq < candidates[idx][1]:
                candidates[idx] = (self._cond_val[k], dist_sq)
        return {idx: val for idx, (val, _) in candidates.items()}

    def _snap_to_nearest(self, axes, k):
        """Return (grid_index_tuple, dist_sq) for the k-th conditioning point."""
        idx = tuple(
            int(np.argmin(np.abs(axes[d] - self._cond_pos[d][k])))
            for d in range(self.dim)
        )
        dist_sq = sum(
            (axes[d][idx[d]] - self._cond_pos[d][k]) ** 2
            for d in range(self.dim)
        )
        return idx, dist_sq

    def set_condition(self, cond_pos, cond_val, cond_weight=None):
        """Set the conditioning data for the simulation.

        Parameters
        ----------
        cond_pos : :class:`list`
            The position tuple of the conditioning data ``(x, [y, z])``.
        cond_val : :class:`numpy.ndarray` or :class:`dict`
            Univariate: values at the conditioning positions. Multivariate: a
            ``{variable: numpy.ndarray}`` mapping (use ``numpy.nan`` for a
            variable that is not conditioned at a given point).
        cond_weight : :class:`float`, optional
            Conditioning weight delta. If given, overrides the ``cond_weight`` set
            at construction. Default: :any:`None` (keep existing weight)
        """
        if cond_weight is not None:
            self._cond_weight = float(cond_weight)
        if self._ti.multivariate and isinstance(cond_val, dict):
            if not cond_val:
                raise ValueError("cond_val must not be empty")
            unknown = set(cond_val) - set(self._ti.variables)
            if unknown:
                raise ValueError(
                    f"cond_val contains unknown variable(s): {sorted(unknown)}. "
                    f"TI variables are: {sorted(self._ti.variables)}"
                )
            cond_pos_arr = np.asarray(cond_pos, dtype=np.double).reshape(
                self.dim, -1
            )
            n_cond = len(next(iter(cond_val.values())))
            for v, arr in cond_val.items():
                if len(arr) != n_cond:
                    raise ValueError(
                        "DirectSampling: all cond_val arrays must have the same "
                        f"length; got {n_cond} for {next(iter(cond_val))!r} but "
                        f"{len(arr)} for {v!r}."
                    )
            if cond_pos_arr.shape[1] != n_cond:
                raise ValueError(
                    "DirectSampling: cond_pos and cond_val length mismatch."
                )
            self._cond_pos = cond_pos_arr
            self._cond_val = {
                v: np.asarray(a, dtype=np.double) for v, a in cond_val.items()
            }
        elif self._ti.multivariate:
            raise ValueError(
                "DirectSampling: cond_val must be a dict {variable: array} "
                "for multivariate TrainingImages."
            )
        else:
            from gstools.krige.tools import set_condition as _gs_set_condition

            self._cond_pos, self._cond_val = _gs_set_condition(
                cond_pos, cond_val, self.dim
            )

    def set_mv_transforms(self, mean=None, normalizer=None, trend=None):
        """Set per-variable post-processing transforms for multivariate simulations.

        Only meaningful when the training image is multivariate. Variables not
        listed here fall back to the instance-level ``self.mean``,
        ``self.normalizer``, and ``self.trend``.

        Parameters
        ----------
        mean : dict of {str: scalar or callable}, optional
            Per-variable mean.
        normalizer : dict of {str: Normalizer}, optional
            Per-variable normalizer.
        trend : dict of {str: scalar or callable}, optional
            Per-variable trend (applied after denormalization).
        """
        from gstools.field.base import _set_mean_trend
        from gstools.normalizer.tools import _check_normalizer

        if mean is not None:
            self._mv_mean = {
                v: _set_mean_trend(m, self.dim) for v, m in mean.items()
            }
        if normalizer is not None:
            self._mv_normalizer = {
                v: _check_normalizer(n) for v, n in normalizer.items()
            }
        if trend is not None:
            self._mv_trend = {
                v: _set_mean_trend(t, self.dim) for v, t in trend.items()
            }

    def set_nonstationary(self, rotation=None, anis=None):
        """Set per-node geometric transform for non-stationary simulation.

        Transforms lag vectors from the simulation-grid frame into the training
        image's own frame before each TI scan (Mariethoz2010 §6.2). Enables
        spatially varying orientation and anisotropy without modifying the TI.

        Parameters
        ----------
        rotation : float or numpy.ndarray, optional
            Rotation angle(s) in radians. Scalar → stationary (same angle at
            every node). Array whose leading dimensions match the simulation
            grid shape → per-node angles. Convention matches
            :class:`gstools.CovModel` ``angles`` (2-D: one angle; 3-D:
            Tait–Bryan yaw/pitch/roll). ``None`` → no rotation applied.
        anis : float or numpy.ndarray, optional
            Anisotropy ratio(s). Scalar → stationary. Array → per-node. Values
            less than 1 compress the TI search in transversal directions.
            Convention matches :class:`gstools.CovModel` ``anis``.
            ``None`` → isotropic.
        """
        if rotation is not None:
            self._rotation = np.asarray(rotation, dtype=np.float64)
        if anis is not None:
            anis_arr = np.asarray(anis, dtype=np.float64)
            if np.any(anis_arr <= 0):
                raise ValueError(
                    f"DirectSampling: anis must be positive everywhere, "
                    f"got minimum value {float(anis_arr.min())!r}"
                )
            self._anis = anis_arr

    @property
    def ti(self):
        """TrainingImage: The training image model."""
        return self._ti

    @property
    def n_neighbors(self):
        """:class:`int` or :class:`dict`: Maximum neighbours in the data event (per-variable dict for multivariate TIs)."""
        return self._n_neighbors

    @n_neighbors.setter
    def n_neighbors(self, value):
        validated = _mv_validate_n_neighbors(value, self._ti)
        DirectSampling._validate_variation_n_neighbors(validated, self._ti)
        self._n_neighbors = validated

    @property
    def scan_fraction(self):
        """:class:`float`: Fraction of the TI to scan per node (capped at the search window)."""
        return self._scan_fraction

    @scan_fraction.setter
    def scan_fraction(self, value):
        self._scan_fraction = _mv_validate_scan_fraction(value)

    @property
    def threshold(self):
        """:class:`float`: Distance threshold (0.0 → DSBC mode)."""
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        self._threshold = _mv_validate_threshold(value)

    @property
    def cond_weight(self):
        """:class:`float`: Weight for conditioning nodes in distance."""
        return self._cond_weight

    @cond_weight.setter
    def cond_weight(self, value):
        self._cond_weight = float(value)

    @property
    def boundary(self):
        """:class:`str`: Search-window strategy (``"strict"`` or ``"partial"``)."""
        return self._boundary

    @boundary.setter
    def boundary(self, value):
        self._boundary = _mv_validate_boundary(value)

    @property
    def max_radius(self):
        """:class:`float` or :any:`None`: Euclidean cap on SG neighbour selection.

        Values in ``(0, 1)`` disable all neighbours (nearest grid cell is
        at distance 1.0), causing every node to fall back to a random TI
        sample.
        """
        return self._max_radius

    @max_radius.setter
    def max_radius(self, value):
        self._max_radius = _mv_validate_max_radius(value)

    @property
    def num_threads(self):
        """:class:`int` or :any:`None`: Number of threads for outer DAG parallelism."""
        return self._num_threads

    @num_threads.setter
    def num_threads(self, value):
        self._num_threads = None if value is None else int(value)

    def __repr__(self):
        parts = [
            f"DirectSampling(dim={self.dim}, ",
            f"n_neighbors={self.n_neighbors}, ",
            f"scan_fraction={self.scan_fraction}, ",
            f"threshold={self.threshold}, ",
            f"boundary={self.boundary!r}",
        ]
        if self._rotation is not None:
            rot_val = (
                float(self._rotation)
                if self._rotation.ndim == 0
                else self._rotation
            )
            parts.append(f", rotation={rot_val!r}")
        if self._anis is not None:
            anis_val = (
                float(self._anis) if self._anis.ndim == 0 else self._anis
            )
            parts.append(f", anis={anis_val!r}")
        return "".join(parts) + ")"
