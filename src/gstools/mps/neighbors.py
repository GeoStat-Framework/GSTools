"""Data-event geometry for Direct Sampling.

Neighbour selection, geometric lag transform, and TI search-window bounds.
Pure NumPy + gstools.tools.geometric; no TrainingImage, no simulation state.
"""

import warnings

import numpy as np

from gstools.tools.geometric import matrix_detransform, no_of_angles


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
    """Closest valid neighbours of ``x_i``, with their path indices (Mariethoz2010 ¶15, Juda2022 Eq. 1).

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
    # Chunked vectorized scan: process offsets in blocks to amortize Python
    # overhead while preserving distance-sorted order and early-stop.
    # offset_arr is distance-sorted (ascending dist_sq), so the scan processes
    # shells in order.  When max_radius is set, dist_sq is monotone-increasing
    # and we break the moment the first offset in a chunk exceeds the radius.
    # The informed ravel is looked up via flat_safe = np.where(inbounds,flat,0)
    # which is safe because the inbounds mask gates every use of that value.
    _CHUNK = 256
    dim = offset_arr.shape[1]
    r_sq = max_radius * max_radius if max_radius is not None else None
    # C-order strides: stride[d] = prod(sim_shape[d+1:])
    strides = np.array(
        [int(np.prod(sim_shape[d + 1 :])) for d in range(dim)], dtype=np.intp
    )
    # Precompute ravel of informed for vectorized lookup when informed is not None.
    informed_flat = informed.ravel() if informed is not None else None

    found_coords = []
    found_vidx = []
    n_found = 0

    n_off = len(offset_arr)
    b0 = 0
    while b0 < n_off and n_found < n_neighbors:
        chunk = offset_arr[b0 : b0 + _CHUNK]  # shape (C, dim)

        # Early-stop on radius: dist_sq is monotone-increasing; if the first
        # offset in this chunk already exceeds r_sq, all subsequent ones do too.
        if r_sq is not None:
            chunk_dsq = np.einsum("ij,ij->i", chunk, chunk)
            first_over = np.searchsorted(chunk_dsq, r_sq, side="right")
            if first_over == 0:
                break  # entire remaining offset_arr is beyond radius
            chunk = chunk[:first_over]
            chunk_dsq = chunk_dsq[:first_over]

        cands = x_i + chunk  # (C, dim)

        # In-bounds mask
        inbounds = np.all((cands >= 0) & (cands < sim_shape_arr), axis=1)

        # Flat indices — use flat_safe (clamp to 0 for out-of-bounds rows) so
        # array indexing is always valid; only results where inbounds=True are used.
        flat = cands @ strides  # (C,)
        flat_safe = np.where(inbounds, flat, 0)
        vi = path_pos_map[flat_safe]  # (C,)

        # Path-order / conditioning filter
        idx_ok = (vi < curr_idx) | (vi == -1)

        # Informed filter: when informed is None every in-bounds earlier-path
        # cell is available (DAG mode); otherwise check the boolean flat.
        if informed_flat is not None:
            inf_ok = informed_flat[flat_safe]
        else:
            inf_ok = np.ones(len(chunk), dtype=bool)

        valid = inbounds & idx_ok & inf_ok
        hits = np.flatnonzero(valid)  # distance-sorted order preserved

        need = n_neighbors - n_found
        if len(hits) > need:
            hits = hits[:need]

        if len(hits):
            found_coords.append(cands[hits])
            found_vidx.append(vi[hits])
        n_found += len(hits)

        b0 += len(chunk)  # advance by the (possibly truncated) chunk size

    if found_coords:
        return (
            np.concatenate(found_coords).astype(offset_arr.dtype, copy=False),
            np.concatenate(found_vidx).astype(path_pos_map.dtype, copy=False),
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
        Detransform matrix from :func:`_lag_transform_matrix`.
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
                "gstools.mps: anisotropy/rotation transform collapsed neighbour lag(s) onto "
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
    """Compute the TI search window for a set of lag vectors (Mariethoz2010 ¶18, Juda2022 Eq. 5).

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
            "gstools.mps: boundary='strict' could not fit the full data event inside the "
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


def _lag_transform_matrix(dim, rotation_map, scale_map, x_i):
    """Detransform matrix M for a node's per-node rotation/scale.

    Consumes the canonical per-node maps produced by
    :func:`gstools.mps.nonstationary.resolve_spec` — always shaped
    ``grid_shape + (n_comp,)`` (stationary specs are zero-stride broadcast
    views), so indexing is uniform and no ndim heuristics exist (B5).

    Parameters
    ----------
    dim : int
        Spatial dimensionality of the simulation grid.
    rotation_map : numpy.ndarray or None
        Canonical rotation map, shape ``grid_shape + (no_of_angles(dim),)``.
        ``None`` → no rotation.
    scale_map : numpy.ndarray or None
        Canonical scale map, shape ``grid_shape + (dim,)``. ``None`` → no
        dilation. No axis-0 pin: a uniform value is M10's affinity ``r``.
    x_i : numpy.ndarray, shape (dim,)
        Integer grid coordinates of the current simulation node.

    Returns
    -------
    M : numpy.ndarray, shape (dim, dim)
        ``matrix_detransform(dim, θ(x_i), s(x_i))`` — the SG→TI inverse map
        (Mariethoz2010 §6.2: simulated structures are θ-rotated and s×-sized
        relative to the TI, so lags are pulled back by the inverse). Lags are
        row vectors, applied as ``lags_ti = lags_sg @ M.T``.
    """
    idx = tuple(int(c) for c in x_i)
    angles = (
        rotation_map[idx]
        if rotation_map is not None
        else np.zeros(no_of_angles(dim))
    )
    scale = scale_map[idx] if scale_map is not None else np.ones(dim)
    return matrix_detransform(dim, angles, scale)
