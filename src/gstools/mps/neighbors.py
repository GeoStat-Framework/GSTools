"""Data-event geometry for Direct Sampling.

Neighbour selection, geometric lag transform, and TI search-window bounds.
Pure NumPy + gstools.tools.geometric; no TrainingImage, no simulation state.
"""

import warnings

import numpy as np

from gstools.tools.geometric import matrix_isometrize, set_angles, set_anis


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


def _lag_transform_matrix(dim, rotation_map, anis_map, x_i):
    """Isometrization matrix M for a node's per-node rotation/anisotropy.

    Mirrors the inline build in the engine: a 1-D map is a stationary value
    broadcast to all nodes; a full-shape map is indexed at ``x_i``.

    Parameters
    ----------
    dim : int
        Spatial dimensionality of the simulation grid.
    rotation_map : numpy.ndarray or None
        Rotation angle(s).  ``None`` → no rotation (identity contribution).
        A 1-D array is a stationary multi-component angle vector applied to
        every node; a full-shape array is indexed at ``x_i`` for per-node
        angles.
    anis_map : numpy.ndarray or None
        Anisotropy ratio(s).  ``None`` → isotropic (ratio 1.0 in all
        transversal directions).  Same broadcast rules as ``rotation_map``.
    x_i : numpy.ndarray, shape (dim,)
        Integer grid coordinates of the current simulation node.  Used to
        index into per-node maps; ignored when the maps are stationary (1-D).

    Returns
    -------
    M : numpy.ndarray, shape (dim, dim)
        Isometrization matrix from :func:`gstools.tools.geometric.matrix_isometrize`.
        Apply as ``lags_ti = lags_sg @ M.T`` to transform SG lag vectors into
        the TI frame.
    """
    angles_i = set_angles(
        dim,
        (rotation_map if rotation_map.ndim == 1 else rotation_map[tuple(x_i)])
        if rotation_map is not None
        else 0.0,
    )
    anis_i = set_anis(
        dim,
        (anis_map if anis_map.ndim == 1 else anis_map[tuple(x_i)])
        if anis_map is not None
        else 1.0,
    )
    return matrix_isometrize(dim, angles_i, anis_i)
