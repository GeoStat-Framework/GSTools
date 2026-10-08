"""
Pure Python/NumPy prototype for local kriging of *many fields at once*
(e.g. daily station data on a fixed target grid) with reuse of the
kriging weights.

Idea
----
For a target point ``j`` the kriging weights only depend on positions,
covariance model, drift and *which of its neighbors have data* -- never
on the values. So instead of one call per field (day), the loop order is
swapped:

    for each target j (parallel in Rust):
        neighbors, full covariance matrix, drift      <- once per target
        cache = {}                                    <- per target only
        for each field t:
            mask = which neighbors have data at t     <- cache key
            weights = cache[mask] or solve(subsystem of full matrix)
            field[j, t] = weights . values

The cache lives only while one target is processed, so memory stays at
``O(#distinct masks * k)`` per thread.

``calc_field_krige_local_multi_python`` is the NumPy stand-in for the
planned Rust function ``gstools_core.calc_field_krige_local_multi`` (same
role as ``calc_field_krige_local_python`` in ``local_kriging_numpy.py``).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from local_kriging_numpy import _calc_cond_drift, _calc_target_drift

CACHE_MODES = ("none", "last", "hash")


def local_krige_multi(
    krige,
    pos,
    cond_val,
    local_radius=None,
    mesh_type="unstructured",
    ext_drift=None,
    return_var=True,
    min_neighbors=1,
    cache="hash",
    return_info=False,
):
    """
    Evaluate local kriging for many fields sharing the same conditioning
    positions (e.g. one field per day).

    Parameters
    ----------
    krige : :any:`gstools.krige.Krige`
        Template instance holding model, *all* conditioning positions,
        drift functions, external drift at the conditions, ``unbiased``,
        ``exact``, ``cond_err``, ``mean``, ``trend`` and ``normalizer``.
        It must contain every station, so build it without NaN values,
        e.g. with ``cond_val=np.zeros(n)``. Prefer a ``Local*`` class:
        global classes build the full ``n x n`` kriging matrix on init.
    pos : :class:`list`
        Position tuple of the target points (x, [y, z]).
    cond_val : :class:`numpy.ndarray`
        Shape ``(cond_no, n_fields)``; NaN marks a missing value
        (that station is left out for that field only).
    local_radius : :class:`float`, optional
        Search radius. Default: ``krige.local_radius``.
    mesh_type : :class:`str`, optional
        'structured' / 'unstructured'. Default: "unstructured"
    ext_drift : :class:`numpy.ndarray` or :any:`None`, optional
        External drift at the target positions (fixed for all fields).
    return_var : :class:`bool`, optional
        Whether to also return the kriging error variance. Default: True
    min_neighbors : :class:`int`, optional
        Targets with fewer available neighbors in a field get NaN there.
        Default: 1
    cache : :class:`str`, optional
        "hash" (all masks seen for this target), "last" (only the previous
        field's mask, as EDK does) or "none" (solve every time).
        Only for comparison in this prototype. Default: "hash"
    return_info : :class:`bool`, optional
        Also return ``n_used`` (shape like the field) and the solve stats.

    Returns
    -------
    field : :class:`numpy.ndarray`
        Shape ``(n_fields, *field_shape)``. Raw, i.e. *not* post-processed
        with mean/normalizer/trend (as in :any:`local_krige`).
    krige_var : :class:`numpy.ndarray`, optional
        Same shape, only if ``return_var``.
    info : :class:`dict`, optional
        ``n_used`` and solve statistics, only if ``return_info``.
    """
    if local_radius is None:
        local_radius = getattr(krige, "local_radius", None)
        if local_radius is None:
            raise ValueError("local_krige_multi: no local_radius given")
    model = krige.model
    cond_no = krige.cond_no
    cond_val = np.asarray(cond_val, dtype=np.double)
    if cond_val.ndim == 1:
        cond_val = cond_val[:, np.newaxis]
    if cond_val.shape[0] != cond_no:
        raise ValueError(
            f"local_krige_multi: cond_val has {cond_val.shape[0]} rows but the "
            f"template krige has {cond_no} conditions. Build the template "
            "without NaN values so it holds every station."
        )
    iso_cond = krige._krige_pos
    iso_targ, shape = krige.pre_pos(pos, mesh_type)
    pnt_cnt = iso_targ.shape[1]

    # same preprocessing as Krige._krige_cond, per field (NaN stays NaN)
    trend = _as_column(krige.cond_trend, cond_no)
    mean = _as_column(krige.cond_mean, cond_no)
    val = krige.normalizer.normalize(cond_val - trend) - mean

    cond_err = krige.cond_err
    cond_err_arr = (
        np.full(cond_no, cond_err, dtype=np.double)
        if np.isscalar(cond_err)
        else np.asarray(cond_err, dtype=np.double)
    )
    drift_cond = _calc_cond_drift(krige)
    ext_drift = krige._pre_ext_drift(pnt_cnt, ext_drift)
    drift_target = _calc_target_drift(krige, iso_targ, ext_drift, pnt_cnt)

    # ================================================================
    # RUST BOUNDARY: in production this becomes a single call to
    #   gstools_core.calc_field_krige_local_multi(...)
    # ================================================================
    field, error, n_used, stats = calc_field_krige_local_multi_python(
        cond_pos=iso_cond,
        cond_val=val,
        target_pos=iso_targ,
        model=model,
        cond_err=cond_err_arr,
        drift_cond=drift_cond,
        drift_target=drift_target,
        unbiased=krige.unbiased,
        exact=krige.exact,
        local_radius=local_radius,
        min_neighbors=min_neighbors,
        cache=cache,
    )
    # ================================================================
    # END RUST BOUNDARY
    # ================================================================

    n_fields = cond_val.shape[1]
    out_shape = (n_fields,) + tuple(np.atleast_1d(shape))
    res = [field.T.reshape(out_shape)]
    if return_var:
        res.append(np.maximum(model.sill - error, 0).T.reshape(out_shape))
    if return_info:
        res.append({"n_used": n_used.T.reshape(out_shape), **stats})
    return res[0] if len(res) == 1 else tuple(res)


def _as_column(values, cond_no):
    """Scalar or (cond_no,) trend/mean values as a (cond_no, 1) column."""
    values = np.asarray(values, dtype=np.double)
    return np.broadcast_to(values, (cond_no,)).reshape(-1, 1)


def calc_field_krige_local_multi_python(
    cond_pos,
    cond_val,
    target_pos,
    model,
    cond_err,
    drift_cond,
    drift_target,
    unbiased,
    exact,
    local_radius,
    min_neighbors=1,
    cache="hash",
    num_threads=None,
):
    """
    NumPy stand-in for the planned Rust function
    ``gstools_core.calc_field_krige_local_multi``.

    Differences to ``calc_field_krige_local_python``:

    * ``cond_val`` is ``(n, T)`` with NaN = missing, everything else is
      unchanged (positions, errors and drift are fixed over the fields)
    * outer loop over targets (``par_iter`` in Rust), inner loop over fields
    * covariances of the full neighborhood are evaluated once per target;
      a field's system is a sub-selection of that matrix
    * weights are cached per target, keyed by the neighbor availability
      mask
    * no ``NoNeighbors`` error: targets with fewer than ``min_neighbors``
      available neighbors get NaN for that field, ``n_used`` tells why

    Parameters
    ----------
    cond_pos : (d, n) ndarray -- isometrized conditioning positions
    cond_val : (n, T) ndarray -- centered values, NaN = missing
    target_pos : (d, m) ndarray -- isometrized target positions
    model : :any:`gstools.CovModel` -- stand-in for ``&CovModel``
    cond_err : (n,) ndarray
    drift_cond : (p, n) ndarray -- may have p == 0
    drift_target : (p, m) ndarray
    unbiased : :class:`bool`
    exact : :class:`bool`
    local_radius : :class:`float`
    min_neighbors : :class:`int`
        Minimal number of available neighbors to krige. Default: 1
    cache : :class:`str`
        "hash", "last" or "none" -- prototype only, to count solves.
    num_threads : :class:`int` or :any:`None`
        Unused here; signature parity with the future Rust call.

    Returns
    -------
    field : (m, T) ndarray -- raw, NaN where not kriged
    error : (m, T) ndarray -- raw RHS . w, NaN where not kriged
    n_used : (m, T) int ndarray -- available neighbors per target/field
    stats : dict -- ``n_solves``, ``n_hits_last``, ``n_hits_hash``,
        ``n_skipped``, ``max_masks_per_target``
    """
    if cache not in CACHE_MODES:
        raise ValueError(f"cache must be one of {CACHE_MODES}, got '{cache}'")
    min_neighbors = max(int(min_neighbors), 1)
    cond_val = np.asarray(cond_val, dtype=np.double)
    if cond_val.ndim == 1:
        cond_val = cond_val[:, np.newaxis]
    pnt_cnt = target_pos.shape[1]
    t_cnt = cond_val.shape[1]

    # neighborhoods over *all* conditioning points, once
    tree = cKDTree(cond_pos.T)
    neighbor_lists = tree.query_ball_point(
        target_pos.T, r=local_radius, return_sorted=True
    )
    avail_all = np.isfinite(cond_val)
    cov_fct = model.cov_nugget if exact else model.covariance

    field = np.full((pnt_cnt, t_cnt), np.nan)
    error = np.full((pnt_cnt, t_cnt), np.nan)
    n_used = np.zeros((pnt_cnt, t_cnt), dtype=np.int64)
    stats = dict(
        n_solves=0, n_hits_last=0, n_hits_hash=0, n_skipped=0,
        max_masks_per_target=0,
    )

    for j in range(pnt_cnt):  # <- becomes `par_iter` over target points in Rust
        nbrs = np.asarray(neighbor_lists[j], dtype=np.intp)
        k = nbrs.size
        if k == 0:
            stats["n_skipped"] += t_cnt
            continue

        # --- once per target: full neighborhood system parts ---
        nbr_pos = cond_pos[:, nbrs]
        cov_full = model.covariance(cdist(nbr_pos.T, nbr_pos.T))
        cov_full[np.diag_indices(k)] += cond_err[nbrs]
        rhs_full = cov_fct(
            np.linalg.norm(nbr_pos - target_pos[:, j : j + 1], axis=0)
        )
        drift_nbrs = drift_cond[:, nbrs]
        drift_targ = drift_target[:, j]

        # --- availability masks of all fields at once ---
        nbr_avail = avail_all[nbrs]  # (k, T)
        nbr_val = cond_val[nbrs]  # (k, T)
        n_used[j] = nbr_avail.sum(axis=0)
        keys = np.packbits(nbr_avail, axis=0)  # bitset per field: (ceil(k/8), T)

        weights_cache = {}  # mask bytes -> (sel, weights, error)
        last_key, last_entry = None, None
        for t in range(t_cnt):
            if n_used[j, t] < min_neighbors:
                stats["n_skipped"] += 1
                continue
            key = keys[:, t].tobytes()
            if cache != "none" and key == last_key:
                entry = last_entry
                stats["n_hits_last"] += 1
            elif cache == "hash" and key in weights_cache:
                entry = weights_cache[key]
                stats["n_hits_hash"] += 1
            else:
                sel = np.flatnonzero(nbr_avail[:, t])
                weights, err = _solve_subsystem(
                    cov_full, rhs_full, drift_nbrs, drift_targ, sel, unbiased
                )
                entry = (sel, weights, err)
                stats["n_solves"] += 1
                if cache == "hash":
                    weights_cache[key] = entry
            last_key, last_entry = key, entry

            sel, weights, err = entry
            field[j, t] = nbr_val[sel, t] @ weights[: sel.size]
            error[j, t] = err
        stats["max_masks_per_target"] = max(
            stats["max_masks_per_target"], len(weights_cache)
        )

    return field, error, n_used, stats


def _solve_subsystem(cov_full, rhs_full, drift_nbrs, drift_targ, sel, unbiased):
    """
    Solve the kriging system of the neighbors ``sel`` (indices into the
    full neighborhood) by sub-selecting the precomputed full matrices.

    The single place doing linear algebra: shared factorizations (idea 4)
    or downdating a full inverse (idea 5) would plug in here.

    Returns
    -------
    weights : (k + pad,) ndarray
    error : float -- RHS . weights
    """
    k = sel.size
    drift_no = drift_nbrs.shape[0]
    pad = drift_no + int(unbiased)
    size = k + pad

    lhs = np.zeros((size, size), dtype=np.double)
    lhs[:k, :k] = cov_full[np.ix_(sel, sel)]
    rhs = np.zeros(size, dtype=np.double)
    rhs[:k] = rhs_full[sel]
    if unbiased:
        lhs[k, :k] = 1
        lhs[:k, k] = 1
        rhs[k] = 1
    if drift_no:
        lhs[-drift_no:, :k] = drift_nbrs[:, sel]
        lhs[:k, -drift_no:] = drift_nbrs[:, sel].T
        rhs[-drift_no:] = drift_targ

    try:
        weights = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return weights, rhs @ weights
