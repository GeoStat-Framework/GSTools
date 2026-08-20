"""
Pure Python/NumPy prototype for *local* (moving-neighborhood) Kriging.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


def local_krige(
    krige,
    pos,
    local_radius,
    mesh_type="unstructured",
    ext_drift=None,
    return_var=True,
):
    """
    Evaluate local (moving-neighborhood) kriging at the given positions.

    Parameters
    ----------
    krige : :any:`gstools.krige.Krige`
        An already-conditioned Krige instance (Simple/Ordinary/Universal/
        ExtDrift/Detrended). Its ``model``, ``cond_pos``, ``cond_val``,
        drift functions, external drift, ``unbiased``, ``exact`` and
        ``cond_err`` settings are reused as-is. ``pseudo_inv`` is *not*
        used here (see notes in :any:`calc_field_krige_local`).
    pos : :class:`list`
        Position tuple of the target points (x, [y, z]).
    local_radius : :class:`float`
        Search radius (in the model's isometrized/isotropic distance):
        every conditioning point within this distance of a target
        point enters its local kriging system. This is the *only*
        neighborhood-selection criterion -- there is no separate
        neighbor-count cap.
    mesh_type : :class:`str`, optional
        'structured' / 'unstructured'. Default: "unstructured"
    ext_drift : :class:`numpy.ndarray` or :any:`None`, optional
        External drift values at the target positions (only relevant
        for :any:`gstools.krige.ExtDrift`). Same semantics as
        ``Krige.__call__``'s ``ext_drift`` argument.
    return_var : :class:`bool`, optional
        Whether to also return the kriging error variance. Default: True

    Returns
    -------
    field : :class:`numpy.ndarray`
        The local-kriging field (raw, i.e. *not* post-processed with
        mean/normalizer/trend -- see :any:`Krige.post_field` for that).
    krige_var : :class:`numpy.ndarray`, optional
        The kriging error variance (only if ``return_var`` is True).
    """
    model = krige.model # Kovarianzmodell
    iso_cond = krige._krige_pos  # (dim, cond_no), isometrized cond. pos.
    iso_targ, shape = krige.pre_pos(pos, mesh_type)  # (dim, pnt_cnt), isometrized target pos.
    pnt_cnt = iso_targ.shape[1] 
    cond_no = krige.cond_no 

    # cond values already normalized/detrended/zero-mean (without padding)
    cond_val = krige._krige_cond[:cond_no]

    cond_err = krige.cond_err
    cond_err_arr = (
        np.full(cond_no, cond_err, dtype=np.double)
        if np.isscalar(cond_err)
        else np.asarray(cond_err, dtype=np.double)
    )

    # drift terms as plain arrays, evaluated once -- see docstrings there
    drift_cond = _calc_cond_drift(krige)
    ext_drift = krige._pre_ext_drift(pnt_cnt, ext_drift)
    drift_target = _calc_target_drift(krige, iso_targ, ext_drift, pnt_cnt)

    # ================================================================
    # RUST BOUNDARY: in production this becomes a single call to
    #   gstools_core.krige.calc_field_krige_local(...)
    # ================================================================
    field, error = calc_field_krige_local(
        cond_pos=iso_cond,
        cond_val=cond_val,
        target_pos=iso_targ,
        model=model,  # stand-in for &CovModelCore
        cond_err=cond_err_arr,
        drift_cond=drift_cond,
        drift_target=drift_target,
        unbiased=krige.unbiased,
        exact=krige.exact,
        local_radius=local_radius,
        num_threads=None,
    )
    # ================================================================
    # END RUST BOUNDARY
    # ================================================================

    field = np.reshape(field, shape)
    if return_var:
        krige_var = np.reshape(np.maximum(model.sill - error, 0), shape)
        return field, krige_var
    return field


def _calc_cond_drift(krige):
    """
    Evaluate all drift terms at the conditioning points *once*, as a
    plain array -- mirrors ``Krige.cond_drift`` (there cached on the
    instance; here just computed fresh per :any:`local_krige` call,
    since this prototype has no persistent object to cache on).

    Internal (functional) drift and external drift are unified into a
    single ``(p, cond_no)`` array, ``p = int_drift_no + ext_drift_no``:
    external drift is already array-shaped, so it is simply stacked
    below the evaluated functional drift rows. This way Universal- and
    External-Drift-Kriging need no callback at all on the Rust side --
    the local solver only ever indexes into ``drift_cond`` with the
    neighbor indices.

    Parameters
    ----------
    krige : :any:`gstools.krige.Krige`

    Returns
    -------
    :class:`numpy.ndarray`
        Shape ``(drift_no, cond_no)``.
    """
    cond_pos = np.asarray(krige.cond_pos)  # (dim, cond_no), original frame
    int_functions = krige.drift_functions
    int_no = len(int_functions)
    ext_no = krige.ext_drift_no
    drift_no = int_no + ext_no

    drift_cond = np.empty((drift_no, krige.cond_no), dtype=np.double)
    for i, f in enumerate(int_functions):
        drift_cond[i] = f(*cond_pos)
    if ext_no > 0:
        drift_cond[int_no:] = krige.cond_ext_drift
    return drift_cond


def _calc_target_drift(krige, iso_targ, ext_drift, pnt_cnt):
    """
    Evaluate all drift terms at the target positions *once*, as a
    plain array -- mirrors ``Krige._calc_target_drift``: computed a
    single time for the whole call, instead of re-evaluating drift
    functions for every neighbor inside the (future Rust) solve loop.

    Parameters
    ----------
    krige : :any:`gstools.krige.Krige`
    iso_targ : (dim, pnt_cnt) ndarray
        Isometrized target positions.
    ext_drift : :class:`numpy.ndarray`
        Preprocessed external drift values at the target positions
        (output of ``krige._pre_ext_drift``).
    pnt_cnt : :class:`int`

    Returns
    -------
    :class:`numpy.ndarray`
        Shape ``(drift_no, pnt_cnt)``.
    """
    int_functions = krige.drift_functions
    int_no = len(int_functions)
    ext_no = krige.ext_drift_no
    drift_no = int_no + ext_no

    drift_target = np.empty((drift_no, pnt_cnt), dtype=np.double)
    if int_no > 0:
        # drift functions live in the original (anisotropic) frame
        targ_pos = krige.model.anisometrize(iso_targ)
        for i, f in enumerate(int_functions):
            drift_target[i] = f(*targ_pos)
    if ext_no > 0:
        drift_target[int_no:] = ext_drift
    return drift_target


def calc_field_krige_local(
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
    num_threads=None,
):
    """
    NumPy stand-in for the planned Rust extension function
    ``gstools_core.krige.calc_field_krige_local`` (see module docstring
    for the exact Rust signature this mirrors).

    Everything in this function -- building the KD-tree once, then per
    target point finding neighbors within ``local_radius``, assembling
    and solving the local kriging system, and accumulating field/error
    -- is what moves into ``gstools-core/src/krige.rs``. There the
    ``for j in range(pnt_cnt)`` loop below becomes a ``rayon``
    ``par_iter``.

    Note this function only ever receives plain arrays and scalars, no
    Python callables and no ``Krige``/``CovModel`` Python objects with
    behaviour beyond ``covariance``/``cov_nugget`` -- ``model`` stands
    in for the future ``&CovModelCore`` (a lightweight, serializable
    model spec on the Rust side; ``as_core_model(model)`` would build
    it in the Python prep step).

    Unlike global kriging, this does *not* use ``Krige``'s
    ``pseudo_inv``/``pseudo_inv_type``: local systems are typically
    small and (with nugget > 0) well conditioned, so this solves
    directly (``numpy.linalg.solve``, i.e. the NumPy equivalent of the
    LDL^T decomposition sketched for the Rust core) instead of forming
    a pseudo-inverse. This matches the planned Rust signature, which
    has no ``pseudo_inv`` parameter at all.

    Parameters
    ----------
    cond_pos : (d, n) ndarray -- isometrized conditioning positions
    cond_val : (n,) ndarray -- unpadded, centered conditioning values
    target_pos : (d, m) ndarray -- isometrized target positions
    model : :any:`gstools.CovModel` -- stand-in for ``&CovModelCore``
    cond_err : (n,) ndarray
    drift_cond : (p, n) ndarray -- may have p == 0
    drift_target : (p, m) ndarray
    unbiased : :class:`bool`
    exact : :class:`bool`
    local_radius : :class:`float`
        Search radius (isometrized distance). Every conditioning point
        within this distance of a target point enters its local
        system -- the only neighborhood-selection criterion, no count
        cap.
    num_threads : :class:`int` or :any:`None`
        Unused by this sequential NumPy stand-in; kept for signature
        parity with the future Rust call.

    Returns
    -------
    field : (m,) ndarray -- raw, not yet post-processed
    error : (m,) ndarray -- raw RHS . w, not yet ``sill - error``
    """
    pnt_cnt = target_pos.shape[1]
    drift_no = drift_cond.shape[0]
    pad = drift_no + int(unbiased)

    tree = cKDTree(cond_pos.T)
    # Diese Querry ist halt nur abhängig vom Radius. Es könnte also sein dass kein Nachbar gefunden wird
    # Damit müsste man irgendwie noch umgehen
    neighbor_lists = tree.query_ball_point(target_pos.T, r=local_radius)

    cov_fct = model.cov_nugget if exact else model.covariance

    field = np.empty(pnt_cnt, dtype=np.double)
    error = np.empty(pnt_cnt, dtype=np.double)

    for j in range(pnt_cnt):  # <- becomes `par_iter` over target points in Rust
        nbrs = np.asarray(neighbor_lists[j], dtype=np.intp)
        k = nbrs.size
        if k == 0:
            raise ValueError(
                f"local_krige: no conditioning points within "
                f"local_radius={local_radius} for target point {j}. "
                "Increase local_radius."
            )
        sys_size = k + pad
        nbr_pos = cond_pos[:, nbrs]  # (d, k)

        # --- local LHS (kriging matrix) ---
        lhs = np.zeros((sys_size, sys_size), dtype=np.double)
        lhs[:k, :k] = model.covariance(cdist(nbr_pos.T, nbr_pos.T))
        lhs[np.diag_indices(k)] += cond_err[nbrs]
        if unbiased:
            lhs[k, :k] = 1
            lhs[:k, k] = 1
        if drift_no:
            lhs[-drift_no:, :k] = drift_cond[:, nbrs]
            lhs[:k, -drift_no:] = drift_cond[:, nbrs].T

        # --- local RHS (kriging vector) ---
        rhs = np.empty(sys_size, dtype=np.double)
        dists = np.linalg.norm(nbr_pos - target_pos[:, j : j + 1], axis=0)
        rhs[:k] = cov_fct(dists)
        if unbiased:
            rhs[k] = 1
        if drift_no:
            rhs[-drift_no:] = drift_target[:, j]

        # --- solve & accumulate ---
        try:
            weights = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            weights = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        local_cond = np.concatenate([cond_val[nbrs], np.zeros(pad)])
        field[j] = local_cond @ weights
        error[j] = rhs @ weights

    return field, error
