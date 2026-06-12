"""
GStools subpackage providing the Direct Sampling MPS simulation class.

.. currentmodule:: gstools.mps

The following classes and functions are provided

.. autosummary::
   DirectSampling
"""

import queue
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from gstools import config
from gstools.field.base import Field
from gstools.normalizer.tools import apply_mean_norm_trend
from gstools.random.rng import RNG
from gstools.tools.geometric import matrix_isometrize, set_angles, set_anis

__all__ = ["DirectSampling"]

_VALID_BOUNDARY = ("strict", "partial")

# DS-mode scan block size.  Large enough that per-call NumPy overhead is
# negligible (essentially full vectorization speed), small enough that the
# greedy DS scan does not overcompute far past the first accepted match.
# This is a call-overhead-amortization constant, not a cache-tuned one.
_SCAN_BLOCK = 4096


def _resolve_nonstationary_map(param, sim_shape):
    """Return ``None`` or an array broadcastable to ``sim_shape``.

    Scalars and 0-d arrays are broadcast. For 3-D grids with per-node angle
    vectors the map may have shape ``(*sim_shape, n)``; indexing ``m[i, j, k]``
    then yields the per-node vector.
    """
    if param is None:
        return None
    arr = np.asarray(param, dtype=np.float64)
    if arr.ndim == 0 or arr.size == 1:
        return np.full(sim_shape, float(arr.flat[0]))
    if arr.shape[: len(sim_shape)] == tuple(sim_shape):
        return arr
    raise ValueError(
        f"Non-stationary map shape {arr.shape!r} is incompatible with "
        f"simulation grid shape {tuple(sim_shape)!r}. Pass a scalar for a "
        f"stationary value or an array whose leading dimensions match the grid."
    )


def _precompute_offsets(shape, max_offset=None):
    """Neighbour offsets from the origin, sorted by Euclidean distance.

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
    rng_vals = np.arange(-max_offset, max_offset + 1)
    grid = np.array(np.meshgrid(*[rng_vals] * dim, indexing="ij"))
    offsets = grid.reshape(dim, -1).T
    offsets = offsets[np.any(offsets != 0, axis=1)]
    idx = np.argsort(np.sum(offsets**2, axis=1))
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


def _build_dag_base(path, sim_shape, offset_arr, vmap_dict, n_k_dict, max_radius=None):
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


def _build_dag(
    path,
    n_neighbors,
    sim_shape,
    offset_arr,
    path_pos_map,
    max_radius=None,
):
    """Build the univariate simulation dependency DAG.

    Thin wrapper around :func:`_build_dag_base` that converts the result back
    to the univariate format (scalar indegree array, list-of-int out_edges).
    """
    indegree_d, out_edges_d = _build_dag_base(
        path,
        sim_shape,
        offset_arr,
        {"": path_pos_map},
        {"": n_neighbors},
        max_radius,
    )
    return indegree_d[""], [[i for i, _ in es] for es in out_edges_d]


def ds_simulate(
    training_image,
    sim_shape,
    n_neighbors,
    threshold,
    scan_fraction,
    rng,
    conditions=None,
    cond_weight=1.0,
    boundary="strict",
    max_radius=None,
    num_threads=None,
    rotation_map=None,
    anis_map=None,
):
    """Direct Sampling univariate simulation (Mariethoz2010, Juda2022).

    Parameters
    ----------
    training_image : TrainingImage
        Training image; provides ``training_image.distance()`` and
        ``training_image.adjust_value()``.
    sim_shape : tuple
        Simulation grid shape.
    n_neighbors : int
        Maximum number of neighbours in the data event (Juda2022 §2).
    threshold : float
        Distance threshold for early acceptance (Juda2022 §2).
        ``0.0`` → DSBC mode.
    scan_fraction : float
        Fraction of the per-node search window to scan (Mariethoz2010 §3 ¶24).
        Evaluates at most ``floor(f · |window|)`` candidates per node.
        ``1.0`` → full window scan.
    rng : numpy.random.RandomState
        Random number generator.
    conditions : dict, optional
        ``{tuple_index: value}`` mapping of conditioning data.
    cond_weight : float, optional
        Weight δ for conditioning nodes (Mariethoz2010 §3 ¶26).
    boundary : str, optional
        Search-window strategy: ``"strict"`` (default) or ``"partial"``.
    max_radius : float, optional
        If set, SG neighbours beyond this Euclidean distance are excluded
        from the data event (Mariethoz2010 §3 ¶19).
    num_threads : int or None, optional
        Number of threads for outer DAG parallelism. ``None`` defaults to
        ``config.NUM_THREADS``.
    rotation_map : numpy.ndarray or None, optional
        Per-node rotation angles, shape matching the simulation grid. ``None``
        → no rotation (stationary). Use ``DirectSampling.set_nonstationary``
        to produce this array from user-facing scalar or array inputs.
    anis_map : numpy.ndarray or None, optional
        Per-node anisotropy ratios, shape matching the simulation grid.
        ``None`` → isotropic (stationary). All values must be positive.

    Returns
    -------
    numpy.ndarray
    """
    ti_data = training_image.data
    ti_shape = np.array(ti_data.shape)
    sim_shape_arr = np.array(sim_shape)
    dim = len(sim_shape)
    sg = np.full(sim_shape, np.nan)
    is_cond = np.zeros(sim_shape, dtype=bool)
    informed = np.zeros(sim_shape, dtype=bool)

    if conditions:
        for idx, val in conditions.items():
            sg[idx] = val
            is_cond[idx] = True
            informed[idx] = True

    n_threads = (
        num_threads if num_threads is not None else (config.NUM_THREADS or 1)
    )
    executor = (
        ThreadPoolExecutor(max_workers=n_threads) if n_threads > 1 else None
    )
    max_off_int = int(np.ceil(max_radius)) if max_radius is not None else None
    offset_arr = _precompute_offsets(sim_shape, max_off_int)

    path = np.argwhere(np.isnan(sg))
    path = path[rng.permutation(len(path))]
    node_seeds = rng.randint(0, 2**32, size=len(path), dtype=np.int64)

    path_flat = np.ravel_multi_index(path.T, sim_shape)
    path_pos_map = np.full(int(np.prod(sim_shape)), -1, dtype=np.intp)
    path_pos_map[path_flat] = np.arange(len(path_flat))

    def _rand_ti(node_rng):
        return ti_data[tuple(node_rng.randint(0, s) for s in ti_shape)]

    def _get_neighbors(x_i, informed_in):
        curr_idx = path_pos_map[
            int(np.ravel_multi_index(tuple(x_i), sim_shape))
        ]
        coords, _ = _select_neighbors(
            x_i,
            offset_arr,
            sim_shape_arr,
            sim_shape,
            path_pos_map,
            curr_idx,
            informed_in,
            max_radius,
            n_neighbors,
        )
        return coords

    def _scan_ti(lo, win_shape, lags, de_sim, cm, ln, node_rng):
        # Precondition: win_size >= 1 (callers guarantee win_lo <= win_hi on all axes).
        # Also captures scan_fraction, ti_data, threshold, cond_weight from outer scope.
        win_size = int(np.prod(win_shape))
        max_scan = max(1, int(scan_fraction * win_size))
        start = int(node_rng.randint(0, win_size))

        # All scan positions in visit order — shape (max_scan,)
        positions = (start + np.arange(max_scan)) % win_size
        # Anchor coordinates for each position — shape (max_scan, dim)
        y_all = lo + np.column_stack(np.unravel_index(positions, win_shape))

        # lags are integer-valued float64; cast once, reuse for all candidates
        int_lags = lags.astype(int)  # (k, dim)

        def _de_ti(y_rows):
            # TI data events for the given anchor rows — shape (len(y_rows), k)
            coords = y_rows[:, None, :] + int_lags[None, :, :]
            return ti_data[tuple(coords.transpose(2, 0, 1))]

        # DSBC (threshold == 0): no early exit is possible — the global minimum
        # over the whole scan is required — so evaluate every candidate in a
        # single vectorized call.  This is the fastest path and stays exact.
        if threshold <= 0:
            all_de_ti = _de_ti(y_all)
            all_dists = training_image.vec_distance(
                de_sim, all_de_ti, cm, cond_weight, ln
            )
            best_k = int(np.argmin(all_dists))
            return ti_data[tuple(y_all[best_k])], all_de_ti[best_k]

        # DS (threshold > 0): chunked vectorized scan with an early-exit
        # checkpoint between blocks.  Each block is a full vectorized distance
        # call (so the per-element cost matches the single-call version); only
        # the threshold test runs per block.  Blocks advance in scan order, so
        # the first under-threshold candidate found is the first one globally —
        # identical to the unchunked argmax(under) result.
        best_d = np.inf
        best_y = None
        best_de = None
        for b0 in range(0, max_scan, _SCAN_BLOCK):
            y_blk = y_all[b0 : b0 + _SCAN_BLOCK]
            de_blk = _de_ti(y_blk)
            d_blk = training_image.vec_distance(
                de_sim, de_blk, cm, cond_weight, ln
            )
            under = d_blk <= threshold
            if np.any(under):
                k = int(np.argmax(under))
                return ti_data[tuple(y_blk[k])], de_blk[k]
            # No acceptable match in this block.  Track the running best with a
            # strict ``<`` test so that, if no candidate ever falls below the
            # threshold, the returned fallback equals the global argmin with the
            # same first-occurrence tie-break as the unchunked version.
            k = int(np.argmin(d_blk))
            if d_blk[k] < best_d:
                best_d = float(d_blk[k])
                best_y = y_blk[k]
                best_de = de_blk[k]
        return ti_data[tuple(best_y)], best_de

    def _simulate_node(x_i, node_rng, sg_in, informed_in):
        nbrs = _get_neighbors(x_i, informed_in)
        if len(nbrs) == 0:
            return _rand_ti(node_rng)

        lags = (nbrs - x_i).astype(np.float64)  # (k, dim)
        data_event_sim = sg_in[tuple(nbrs.T)]  # (k,)
        cond_mask = is_cond[tuple(nbrs.T)]  # (k,)
        lag_norms = np.linalg.norm(lags, axis=1)  # (k,)

        # Geometric transform: express SG lags in the TI's own frame.
        # matrix_isometrize = derotate + isotropify (Mariethoz2010 §6.2).
        # lags_ti feeds TI-side operations only; original lags/lag_norms/
        # data_event_sim remain authoritative for all SG-side operations.
        if rotation_map is not None or anis_map is not None:
            angles_i = set_angles(
                dim,
                rotation_map[tuple(x_i)] if rotation_map is not None else 0.0,
            )
            anis_i = set_anis(
                dim,
                anis_map[tuple(x_i)] if anis_map is not None else 1.0,
            )
            M = matrix_isometrize(dim, angles_i, anis_i)
            lags_ti = np.rint(lags @ M.T)  # lags are integer-valued (grid offsets)
        else:
            lags_ti = lags

        if boundary == "strict":
            # Search window Y(L_i) — Juda2022 Eq. 5, Mariethoz2010 §3 ¶19
            win_lo = np.maximum(0, np.ceil(-lags_ti.min(axis=0))).astype(int)
            win_hi = np.minimum(
                ti_shape - 1, np.floor(ti_shape - 1 - lags_ti.max(axis=0))
            ).astype(int)
            if np.any(win_lo > win_hi):
                return _rand_ti(node_rng)
            best_v, best_de_ti = _scan_ti(
                win_lo,
                tuple(win_hi - win_lo + 1),
                lags_ti,
                data_event_sim,
                cond_mask,
                lag_norms,
                node_rng,
            )
            return training_image.adjust_value(
                best_v, data_event_sim, best_de_ti
            )

        else:  # "partial" — Mariethoz2010 §6.2: global template reduction
            # Lags are distance-sorted (closest first) because offset_arr is.
            # Drop farthest neighbours one at a time until the bounding box of
            # the remaining data event fits inside the TI, per the paper's
            # "ignore until it becomes possible to scan" directive (§6.2).
            # Drop-ranking uses original lag order (SG geometry); window check
            # uses lags_ti (TI frame), so both sides stay self-consistent.
            valid_count = len(lags)
            while valid_count > 0:
                lags_ti_p = lags_ti[:valid_count]
                sw_lo = np.maximum(0, np.ceil(-lags_ti_p.min(axis=0))).astype(int)
                sw_hi = np.minimum(
                    ti_shape - 1, np.floor(ti_shape - 1 - lags_ti_p.max(axis=0))
                ).astype(int)
                if np.all(sw_lo <= sw_hi):
                    break
                valid_count -= 1
            else:
                # No subset of the data event fits inside the TI (the closest
                # neighbour's lag already exceeds the TI in some dimension).
                return _rand_ti(node_rng)
            best_v, best_de_ti = _scan_ti(
                sw_lo,
                tuple(sw_hi - sw_lo + 1),
                lags_ti[:valid_count],
                data_event_sim[:valid_count],
                cond_mask[:valid_count],
                lag_norms[:valid_count],
                node_rng,
            )
            return training_image.adjust_value(
                best_v, data_event_sim[:valid_count], best_de_ti
            )

    try:
        if executor is not None:
            indegree, out_edges = _build_dag(
                path,
                n_neighbors,
                sim_shape,
                offset_arr,
                path_pos_map,
                max_radius,
            )
            # Running ready-queue: a node is dispatched the instant its last
            # dependency completes (no per-wave barrier).  Workers read the
            # live sg / informed arrays; this is safe because (1) a node is
            # only submitted once all its dependencies are written, so the
            # values it reads are final, and (2) all shared-state mutation
            # (sg, informed, in-degree, submission) happens on this main
            # thread — workers only read.  Each numpy access holds the GIL for
            # its duration, so element reads never tear against the writes.
            # The result of every node depends only on its seed and its
            # (final) neighbour values, so the output is independent of
            # completion order and stays identical to the serial run.
            done_q = queue.Queue()
            counts = {"submitted": 0, "done": 0}

            def _run(i):
                return i, _simulate_node(
                    path[i],
                    RNG(int(node_seeds[i])).random,
                    sg,
                    informed,
                )

            def _submit(i):
                executor.submit(_run, i).add_done_callback(done_q.put)
                counts["submitted"] += 1

            for i in range(len(path)):
                if indegree[i] == 0:
                    _submit(i)

            while counts["done"] < counts["submitted"]:
                i, val = done_q.get().result()
                counts["done"] += 1
                x_i_t = tuple(path[i])
                if np.isnan(val):
                    raise ValueError(
                        f"Simulation produced NaN at {path[i]}. Check TI data."
                    )
                sg[x_i_t] = val
                informed[x_i_t] = True
                for j in out_edges[i]:
                    indegree[j] -= 1
                    if indegree[j] == 0:
                        _submit(j)
        else:
            for i, x_i in enumerate(path):
                x_i_t = tuple(x_i)
                val = _simulate_node(
                    x_i,
                    RNG(int(node_seeds[i])).random,
                    sg,
                    informed,
                )
                if np.isnan(val):
                    raise ValueError(
                        f"Simulation produced NaN at {x_i}. Check TI data."
                    )
                sg[x_i_t] = val
                informed[x_i_t] = True
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    return sg


def _build_dag_mv(
    path,
    variables,
    n_k,
    sim_shape,
    offset_arr,
    vmap,
    max_radius=None,
):
    """Node-vertex dependency DAG for node-wise multivariate simulation.

    Thin wrapper around :func:`_build_dag_base`.  Per-variable in-degrees and
    ``(node, variable)`` out-edges are returned directly from the base builder.

    Returns
    -------
    indegree : dict of {str: numpy.ndarray}
        ``indegree[v][i]`` is node ``i``'s dependency count for variable ``v``.
    out_edges : list of list of (int, str)
        ``out_edges[j]`` holds ``(i, v)`` pairs: completing node ``j`` decrements
        ``indegree[v][i]``.
    """
    return _build_dag_base(path, sim_shape, offset_arr, vmap, n_k, max_radius)


def ds_simulate_mv(
    training_image,
    sim_shape,
    n_neighbors,
    threshold,
    scan_fraction,
    rng,
    conditions=None,
    cond_weight=1.0,
    boundary="strict",
    max_radius=None,
    num_threads=None,
    rotation_map=None,
    anis_map=None,
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
        Fraction of the per-node search window to scan.
    rng : numpy.random.RandomState
        Master RNG (path permutation + per-component seeds).
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

    Returns
    -------
    dict
        ``{variable: numpy.ndarray}`` — one simulated field per variable.
    """
    variables = training_image.variables
    weights = training_image.weights  # hoisted once (avoid per-block dict copy)
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
    path = path[rng.permutation(len(path))]
    node_seeds = rng.randint(0, 2**32, size=len(path), dtype=np.int64)

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

    def _rand_fallback(targets, node_rng):
        # Single random TI cell supplies the whole node-vector (preserves the
        # joint relationship); never an independent draw per variable.
        cell = tuple(int(node_rng.randint(0, s)) for s in ti_shape)
        return {v: float(ti_vars[v][cell]) for v in targets}

    def _joint_scan(lo, win_shape, int_lags, de_v, cm_v, ln_v, node_rng):
        # Mirrors _scan_ti: full vectorized argmin for DSBC (threshold <= 0),
        # chunked first-under-threshold for DS (threshold > 0).  Returns the
        # single best TI cell coordinate ``y``; the caller copies TI[v][y] to all
        # uninformed variables.  The window is the intersection of per-variable
        # boxes, so every ``y + lag`` is in bounds and the gather needs no
        # clipping; the h=0 lag maps to ``y`` itself.
        win_size = int(np.prod(win_shape))
        max_scan = max(1, int(scan_fraction * win_size))
        start = int(node_rng.randint(0, win_size))
        positions = (start + np.arange(max_scan)) % win_size
        y_all = lo + np.column_stack(np.unravel_index(positions, win_shape))

        def _dist_block(y_blk):
            d = np.zeros(len(y_blk))
            active_w = 0.0
            for v in variables:
                il = int_lags.get(v)
                if il is None:
                    continue
                coords = y_blk[:, None, :] + il[None, :, :]
                all_de_ti = ti_vars[v][tuple(coords.transpose(2, 0, 1))]
                d += weights[v] * training_image.vec_distance_var(
                    v, de_v[v], all_de_ti, cm_v[v], cond_weight, ln_v[v]
                )
                active_w += weights[v]
            # Renormalize so the joint distance stays in [0, 1] even when
            # some variables have no data event and are excluded from int_lags.
            # Without this, the threshold fires on a compressed scale and
            # accepts matches that should be rejected.
            if 0.0 < active_w < 1.0:
                d /= active_w
            return d

        if threshold <= 0:
            return y_all[int(np.argmin(_dist_block(y_all)))]

        best_d, best_y = np.inf, None
        for b0 in range(0, max_scan, _SCAN_BLOCK):
            y_blk = y_all[b0 : b0 + _SCAN_BLOCK]
            d_blk = _dist_block(y_blk)
            under = d_blk <= threshold
            if np.any(under):
                return y_blk[int(np.argmax(under))]
            k = int(np.argmin(d_blk))
            if d_blk[k] < best_d:
                best_d = float(d_blk[k])
                best_y = y_blk[k]
        return best_y

    def _simulate_node_mv(curr_idx, x_i, node_rng):
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
                rotation_map[tuple(x_i)] if rotation_map is not None else 0.0,
            )
            anis_i = set_anis(
                dim,
                anis_map[tuple(x_i)] if anis_map is not None else 1.0,
            )
            M = matrix_isometrize(dim, angles_i, anis_i)
            # lags are integer-valued (grid offsets), so rint is lossless
            lags_ti_v = {v: np.rint(lags_v[v] @ M.T) for v in variables}
        else:
            lags_ti_v = dict(lags_v)

        if all(len(lags_v[v]) == 0 for v in variables):
            return _rand_fallback(targets, node_rng)

        # Per-variable search window, intersected.  h=0 lags are excluded — they
        # map to y itself and never constrain the window.
        win_lo = np.zeros(dim, dtype=int)
        win_hi = ti_shape - 1
        for var in variables:
            lv_ti = lags_ti_v[var]
            # h=0 rows (collocated, or SG lag that rounded to 0 in TI space)
            # do not constrain the window.
            nz = np.flatnonzero(np.any(lv_ti != 0, axis=1))
            if not len(nz):
                continue
            if boundary == "strict":
                lv_nz = lv_ti[nz]
                lo = np.maximum(0, np.ceil(-lv_nz.min(axis=0))).astype(int)
                hi = np.minimum(
                    ti_shape - 1, np.floor(ti_shape - 1 - lv_nz.max(axis=0))
                ).astype(int)
                if np.any(lo > hi):
                    return _rand_fallback(targets, node_rng)
            else:  # "partial" — drop farthest until TI box fits
                keep = len(lv_ti)
                while keep > 0:
                    lv_k = lv_ti[:keep]
                    nzk = lv_k[np.any(lv_k != 0, axis=1)]
                    if not len(nzk):
                        lo = np.zeros(dim, dtype=int)
                        hi = ti_shape - 1
                        break
                    lo = np.maximum(0, np.ceil(-nzk.min(axis=0))).astype(int)
                    hi = np.minimum(
                        ti_shape - 1, np.floor(ti_shape - 1 - nzk.max(axis=0))
                    ).astype(int)
                    if np.all(lo <= hi):
                        break
                    keep -= 1
                else:
                    return _rand_fallback(targets, node_rng)
                # Truncate TI-frame and original arrays to the same keep count.
                # lags_ti_v uses a separate dict so this does not affect lags_v.
                lags_ti_v[var] = lv_ti[:keep]
                lags_v[var] = lags_v[var][:keep]
                de_v[var] = de_v[var][:keep]
                cm_v[var] = cm_v[var][:keep]
                ln_v[var] = ln_v[var][:keep]
            win_lo = np.maximum(win_lo, lo)
            win_hi = np.minimum(win_hi, hi)

        if np.any(win_lo > win_hi):
            return _rand_fallback(targets, node_rng)

        # Integer lags per variable (already exact integers as float64, incl.
        # the 0.0 h=0 row); reused for the scan and the mean-shift gather.
        int_lags = {
            v: lags_ti_v[v].astype(int)
            for v in variables
            if len(lags_ti_v[v])
        }
        y = _joint_scan(
            win_lo,
            tuple(win_hi - win_lo + 1),
            int_lags,
            de_v,
            cm_v,
            ln_v,
            node_rng,
        )
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
            result[v] = training_image.adjust_value_var(
                v, ti_val, de_v[v], de_ti_v
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

    try:
        if executor is not None:
            indegree, out_edges = _build_dag_mv(
                path, variables, n_k, sim_shape, offset_arr, vmap, max_radius
            )
            # Running ready-queue over nodes: a node is dispatched the instant
            # every variable's neighbourhood has committed (all per-variable
            # in-degrees zero).  Workers only read the live sg / informed dicts;
            # all mutation happens on this main thread.  A node's whole vector
            # depends only on its seed and its (final) neighbour vectors, so the
            # output is identical to the serial run for the same master seed,
            # regardless of completion order.
            remaining = {v: indegree[v].copy() for v in variables}
            submitted = np.zeros(len(path), dtype=bool)
            done_q = queue.Queue()
            counts = {"submitted": 0, "done": 0}

            def _ready(i):
                return all(remaining[v][i] == 0 for v in variables)

            def _run(i):
                return i, _simulate_node_mv(
                    i, path[i], RNG(int(node_seeds[i])).random
                )

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
                _write_result(tuple(int(c) for c in path[i]), result)
                for j, v in out_edges[i]:
                    remaining[v][j] -= 1
                    if _ready(j):
                        _submit(j)
        else:
            for i in range(len(path)):
                result = _simulate_node_mv(
                    i, path[i], RNG(int(node_seeds[i])).random
                )
                _write_result(tuple(int(c) for c in path[i]), result)
    finally:
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
        Fraction of the per-node search window to scan. Default: 1.
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
    def _validate_n_neighbors(value, ti):
        """Validate and normalise *n_neighbors*; return ``int`` or ``dict of int``."""
        if isinstance(value, dict):
            if not ti.multivariate:
                raise ValueError(
                    "DirectSampling: dict n_neighbors is only valid for "
                    "multivariate TrainingImages."
                )
            missing = set(ti.variables) - set(value)
            extra = set(value) - set(ti.variables)
            if missing or extra:
                raise ValueError(
                    f"DirectSampling: n_neighbors dict keys must match TI "
                    f"variables {ti.variables!r}. Missing: {sorted(missing)}, "
                    f"extra: {sorted(extra)}."
                )
            for k, v in value.items():
                if int(v) < 1:
                    raise ValueError(
                        f"DirectSampling: n_neighbors[{k!r}] must be >= 1, "
                        f"got {v!r}"
                    )
            return {k: int(v) for k, v in value.items()}
        else:
            if int(value) < 1:
                raise ValueError(
                    f"DirectSampling: n_neighbors must be >= 1, got {value!r}"
                )
            return int(value)

    def __init__(
        self,
        ti,
        n_neighbors=32,
        scan_fraction=1,
        threshold=0.0,
        cond_weight=1.0,
        boundary="strict",
        max_radius=None,
        num_threads=None,
        seed=np.nan,
    ):
        if boundary not in _VALID_BOUNDARY:
            raise ValueError(
                f"DirectSampling: boundary must be one of {_VALID_BOUNDARY!r}, "
                f"got {boundary!r}"
            )
        DirectSampling._validate_n_neighbors(n_neighbors, ti)  # raises on invalid
        if not (0 < float(scan_fraction) <= 1):
            raise ValueError(
                f"DirectSampling: scan_fraction must be in (0, 1], "
                f"got {scan_fraction!r}"
            )
        if float(threshold) < 0:
            raise ValueError(
                f"DirectSampling: threshold must be >= 0, got {threshold!r}"
            )
        if float(threshold) > 1.0:
            import warnings

            warnings.warn(
                "threshold > 1.0 guarantees the first candidate is always accepted.",
                stacklevel=2,
            )
        if max_radius is not None and float(max_radius) <= 0:
            raise ValueError(
                f"DirectSampling: max_radius must be a positive float, "
                f"got {max_radius!r}"
            )
        super().__init__(model=None, dim=ti.ndim, value_type="scalar")
        self._ti = ti
        self._n_neighbors = DirectSampling._validate_n_neighbors(n_neighbors, ti)
        self._scan_fraction = float(scan_fraction)
        self._threshold = float(threshold)
        self._cond_weight = float(cond_weight)
        self._boundary = boundary
        self._max_radius = (
            float(max_radius) if max_radius is not None else None
        )
        self._num_threads = num_threads
        self._cond_pos = None
        self._cond_val = None
        self._mv_mean = {}
        self._mv_normalizer = {}
        self._mv_trend = {}
        self._rotation = None
        self._anis = None
        self.rng = RNG(None if np.isnan(seed) else int(seed))
        # Mirror post_field's own name-collision guard (base.py) exactly.
        # Runs after super().__init__() so self.field_names is available —
        # avoids the hardcoded v != "field" workaround.
        if ti.multivariate:
            for v in ti.variables:
                if not v.isidentifier() or (
                    v not in self.field_names and v in dir(self)
                ):
                    raise ValueError(
                        f"DirectSampling: variable name {v!r} cannot be used as "
                        f"a field name; use a valid Python identifier that does "
                        f"not collide with an existing attribute."
                    )

    def __call__(
        self,
        pos=None,
        seed=np.nan,
        mesh_type="structured",
        post_process=True,
        store=True,
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
        name, save = self.get_store_config(store)
        pos, shape = self.pre_pos(pos, mesh_type)
        rotation_map = _resolve_nonstationary_map(self._rotation, shape)
        anis_map = _resolve_nonstationary_map(self._anis, shape)
        conditions = self._conditions_to_grid(self.pos)
        if not np.isnan(seed):
            self.rng.seed = int(seed)
        rng = np.random.RandomState(
            int(self.rng.random.randint(0, 2**32, dtype=np.int64))
        )
        if self._ti.multivariate:
            result = ds_simulate_mv(
                training_image=self._ti,
                sim_shape=shape,
                n_neighbors=self._n_neighbors,
                threshold=self._threshold,
                scan_fraction=self._scan_fraction,
                rng=rng,
                conditions=conditions,
                cond_weight=self._cond_weight,
                boundary=self._boundary,
                max_radius=self._max_radius,
                num_threads=self._num_threads,
                rotation_map=rotation_map,
                anis_map=anis_map,
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
        field = ds_simulate(
            training_image=self._ti,
            sim_shape=shape,
            n_neighbors=self._n_neighbors,
            threshold=self._threshold,
            scan_fraction=self._scan_fraction,
            rng=rng,
            conditions=conditions,
            cond_weight=self._cond_weight,
            boundary=self._boundary,
            max_radius=self._max_radius,
            num_threads=self._num_threads,
            rotation_map=rotation_map,
            anis_map=anis_map,
        )
        return self.post_field(field, name, post_process, save)

    def _conditions_to_grid(self, axes):
        """Snap conditioning points to nearest grid nodes (Mariethoz2010 §3 ¶12).

        Univariate returns ``{idx: value}``; multivariate returns
        ``{idx: {variable: value}}`` with non-finite (NaN) entries skipped. When
        two points snap to the same node, the one closer to the node centre wins
        for all of its variables.
        When the closer point carries ``NaN`` for a variable, that variable is
        not conditioned at the node even if a farther colliding point had a
        finite value there.

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
            candidates = {}  # idx -> (val_dict, dist_sq)
            n_cond = len(next(iter(self._cond_val.values())))
            for k in range(n_cond):
                idx, dist_sq = self._snap_to_nearest(axes, k)
                val_dict = {
                    v: float(self._cond_val[v][k])
                    for v in self._cond_val
                    if np.isfinite(self._cond_val[v][k])
                }
                if idx not in candidates or dist_sq < candidates[idx][1]:
                    candidates[idx] = (val_dict, dist_sq)
            return {idx: vd for idx, (vd, _) in candidates.items()}
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
        self._rotation = (
            None if rotation is None else np.asarray(rotation, dtype=np.float64)
        )
        if anis is not None:
            anis_arr = np.asarray(anis, dtype=np.float64)
            if np.any(anis_arr <= 0):
                raise ValueError(
                    f"DirectSampling: anis must be positive everywhere, "
                    f"got minimum value {float(anis_arr.min())!r}"
                )
            self._anis = anis_arr
        else:
            self._anis = None

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
        self._n_neighbors = DirectSampling._validate_n_neighbors(value, self._ti)

    @property
    def scan_fraction(self):
        """:class:`float`: Fraction of the per-node search window to scan."""
        return self._scan_fraction

    @scan_fraction.setter
    def scan_fraction(self, value):
        if not (0 < float(value) <= 1):
            raise ValueError(
                f"DirectSampling: scan_fraction must be in (0, 1], got {value!r}"
            )
        self._scan_fraction = float(value)

    @property
    def threshold(self):
        """:class:`float`: Distance threshold (0.0 → DSBC mode)."""
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        if float(value) < 0:
            raise ValueError(
                f"DirectSampling: threshold must be >= 0, got {value!r}"
            )
        if float(value) > 1.0:
            import warnings

            warnings.warn(
                "threshold > 1.0 guarantees the first candidate is always accepted.",
                stacklevel=2,
            )
        self._threshold = float(value)

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
        if value not in _VALID_BOUNDARY:
            raise ValueError(
                f"DirectSampling: boundary must be one of {_VALID_BOUNDARY!r}, "
                f"got {value!r}"
            )
        self._boundary = value

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
        if value is not None and float(value) <= 0:
            raise ValueError(
                f"DirectSampling: max_radius must be a positive float, "
                f"got {value!r}"
            )
        self._max_radius = float(value) if value is not None else None

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
            rot_val = float(self._rotation) if self._rotation.ndim == 0 else self._rotation
            parts.append(f", rotation={rot_val!r}")
        if self._anis is not None:
            anis_val = float(self._anis) if self._anis.ndim == 0 else self._anis
            parts.append(f", anis={anis_val!r}")
        return "".join(parts) + ")"
