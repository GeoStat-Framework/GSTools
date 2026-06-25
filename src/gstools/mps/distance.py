"""Pure distance functions for MPS pattern comparison.

No class state — takes arrays and scalars, returns floats.
``TrainingImage.distance()`` uses these internally; other algorithms
can import them directly.
"""

import numpy as np

__all__ = [
    "compute_node_weights",
    "vec_categorical_dist",
    "vec_l1_dist",
    "vec_l2_dist",
    "vec_lp_dist",
    "vec_variation_dist",
]


def compute_node_weights(
    n, lag_norms, distance_power, cond_mask=None, cond_weight=1.0
):
    """Compute normalized spatial-decay weights for a data event.

    Combines spatial decay (Mariethoz2010 Eq. 5) with conditioning data
    multipliers (Mariethoz2010 §3 ¶26).

    Parameters
    ----------
    n : int
        Number of neighbours in the data event.
    lag_norms : array-like or None, shape (n,)
        Euclidean norms ``‖h_i‖`` of each lag vector. ``None`` or
        ``distance_power == 0`` → uniform spatial weights. A zero lag-norm
        (collocated ``h=0`` entry) keeps the unit baseline weight and is not
        amplified by the spatial decay — its weight is scaled by ``cond_weight``.
    distance_power : float
        Exponent δ. ``0.0`` → uniform.
    cond_mask : array-like of bool, optional
        ``True`` where the neighbour is a conditioning datum.
    cond_weight : float, optional
        Bonus weight multiplier for conditioning nodes.

    Returns
    -------
    numpy.ndarray, shape (n,)
        Node weights normalized to sum to 1.
    """
    raw_w = np.ones(n, dtype=np.float64)
    if lag_norms is not None and distance_power != 0.0:
        norms = np.asarray(lag_norms, dtype=np.float64)
        # Only non-zero lags decay with distance.  A *true* zero lag-norm is a
        # collocated/conditioning entry (h=0, e.g. the multivariate same-node
        # constraint); it keeps the unit-cell baseline weight 1.0 rather than the
        # divergent norm**(-power), and its importance is governed by the
        # cond_weight multiplier below (it always carries cond_mask=True).
        nz = norms != 0.0
        raw_w[nz] = norms[nz] ** (-distance_power)

    if cond_mask is not None:
        # ``raw_w`` is a freshly allocated array (np.ones above, modified in
        # place), so no defensive copy is needed before scaling.
        raw_w[np.asarray(cond_mask, dtype=bool)] *= cond_weight

    total = raw_w.sum()
    if total == 0.0:
        # Every neighbour was zeroed out — the canonical case is an all-
        # conditioning data event with cond_weight == 0 (δ_c = 0 → conditioning
        # ignored entirely, Me13 p.323 → unconditional behaviour). Returning
        # zero weights makes the data event non-informative so the node is drawn
        # unconditionally, rather than re-weighting the ignored nodes uniformly.
        return np.zeros(n, dtype=np.float64)
    if not np.isfinite(total):
        # Defensive: non-finite weight sum should be unreachable for grid lags
        # (‖h‖ >= 1) — fall back to uniform rather than emit NaNs.
        return np.full(n, 1.0 / n, dtype=np.float64)
    return raw_w / total


# ---------------------------------------------------------------------------
# Vectorized variants — same maths, operate on all TI candidates at once.
# Each accepts all_de_ti of shape (max_scan, n) and returns (max_scan,).
# np.dot(X, w) with X (max_scan, n) and w (n,) is a standard BLAS matvec.
#
# NaN handling (``has_nan``): when the TI contains undefined (NaN) cells, a
# candidate data event may have NaN at some positions. With ``has_nan=True``
# those positions are excluded per-row and the weights are renormalized over
# the defined positions (a proper convex combination); a candidate whose data
# event is *entirely* undefined gets distance ``+inf`` so it is never selected.
# ``has_nan=False`` (the default) is the original, NaN-free computation, kept
# byte-identical and overhead-free for the common case.
# ---------------------------------------------------------------------------


def _masked_weights(all_de_ti, node_weights):
    """Per-row weights with undefined (NaN) TI positions zeroed.

    Returns
    -------
    valid : numpy.ndarray of bool, shape (max_scan, n)
        ``True`` where the TI value is defined (finite).
    we : numpy.ndarray, shape (max_scan, n)
        ``node_weights`` broadcast, zeroed at undefined positions.
    wsum : numpy.ndarray, shape (max_scan,)
        Row sums of ``we``; ``0`` for a fully-undefined data event.
    """
    valid = ~np.isnan(all_de_ti)
    we = node_weights[np.newaxis, :] * valid
    return valid, we, we.sum(axis=1)


def _renorm(numerator, wsum):
    """``numerator / wsum`` per row, with ``+inf`` where ``wsum == 0``."""
    out = np.full(numerator.shape, np.inf, dtype=np.float64)
    nz = wsum > 0
    out[nz] = numerator[nz] / wsum[nz]
    return out


def vec_categorical_dist(
    data_event_sim, all_de_ti, node_weights, has_nan=False
):
    """Vectorized categorical distance over all TI scan candidates.

    Parameters
    ----------
    data_event_sim : numpy.ndarray, shape (n,)
    all_de_ti : numpy.ndarray, shape (max_scan, n)
    node_weights : numpy.ndarray, shape (n,)
    has_nan : bool, optional
        Enable per-row NaN exclusion. Default ``False``.

    Returns
    -------
    numpy.ndarray, shape (max_scan,)
        Distance in [0, 1] for each candidate.
    """
    if not has_nan:
        return np.dot(
            (data_event_sim != all_de_ti).astype(np.float64), node_weights
        )
    _, we, wsum = _masked_weights(all_de_ti, node_weights)
    # NaN positions compare unequal, but ``we`` is zero there so they do not
    # contribute to the mismatch sum.
    mism = (we * (data_event_sim != all_de_ti)).sum(axis=1)
    return _renorm(mism, wsum)


def vec_l1_dist(data_event_sim, all_de_ti, node_weights, d_max, has_nan=False):
    """Vectorized L1 distance over all TI scan candidates.

    Parameters
    ----------
    data_event_sim : numpy.ndarray, shape (n,)
    all_de_ti : numpy.ndarray, shape (max_scan, n)
    node_weights : numpy.ndarray, shape (n,)
    d_max : float
    has_nan : bool, optional

    Returns
    -------
    numpy.ndarray, shape (max_scan,)
    """
    if not has_nan:
        return np.dot(np.abs(data_event_sim - all_de_ti) / d_max, node_weights)
    valid, we, wsum = _masked_weights(all_de_ti, node_weights)
    ad = np.where(valid, np.abs(data_event_sim - all_de_ti) / d_max, 0.0)
    return _renorm((we * ad).sum(axis=1), wsum)


def vec_l2_dist(data_event_sim, all_de_ti, node_weights, d_max, has_nan=False):
    """Vectorized L2 distance over all TI scan candidates.

    Parameters
    ----------
    data_event_sim : numpy.ndarray, shape (n,)
    all_de_ti : numpy.ndarray, shape (max_scan, n)
    node_weights : numpy.ndarray, shape (n,)
    d_max : float
    has_nan : bool, optional

    Returns
    -------
    numpy.ndarray, shape (max_scan,)
    """
    if not has_nan:
        return np.sqrt(
            np.dot(((data_event_sim - all_de_ti) / d_max) ** 2, node_weights)
        )
    valid, we, wsum = _masked_weights(all_de_ti, node_weights)
    sq = np.where(valid, ((data_event_sim - all_de_ti) / d_max) ** 2, 0.0)
    return np.sqrt(_renorm((we * sq).sum(axis=1), wsum))


def vec_lp_dist(
    data_event_sim, all_de_ti, node_weights, d_max, p, has_nan=False
):
    """Vectorized Lp distance over all TI scan candidates.

    Parameters
    ----------
    data_event_sim : numpy.ndarray, shape (n,)
    all_de_ti : numpy.ndarray, shape (max_scan, n)
    node_weights : numpy.ndarray, shape (n,)
    d_max : float
    p : float
    has_nan : bool, optional

    Returns
    -------
    numpy.ndarray, shape (max_scan,)
    """
    if not has_nan:
        diffs = np.abs(data_event_sim - all_de_ti) / d_max
        return np.dot(diffs**p, node_weights) ** (1.0 / p)
    valid, we, wsum = _masked_weights(all_de_ti, node_weights)
    dp = np.where(
        valid, (np.abs(data_event_sim - all_de_ti) / d_max) ** p, 0.0
    )
    return _renorm((we * dp).sum(axis=1), wsum) ** (1.0 / p)


def vec_variation_dist(
    data_event_sim, all_de_ti, node_weights, d_max, p=2.0, has_nan=False
):
    """Vectorized variation distance over all TI scan candidates.

    Parameters
    ----------
    data_event_sim : numpy.ndarray, shape (n,)
    all_de_ti : numpy.ndarray, shape (max_scan, n)
    node_weights : numpy.ndarray, shape (n,)
    d_max : float
    p : float, optional
        Lp aggregation exponent. Default ``2.0``.
    has_nan : bool, optional

    Returns
    -------
    numpy.ndarray, shape (max_scan,)
        Distance in [0, 1].
    """
    if not has_nan:
        de_sim_c = data_event_sim - data_event_sim.mean()
        all_de_ti_c = all_de_ti - all_de_ti.mean(axis=1, keepdims=True)
        diffs = de_sim_c - all_de_ti_c
        # 2*d_max normalises the common case to [0, 1]; SG values are not
        # bounded by the TI range (conditioning / mean-shifts), so clamp.
        return np.minimum(
            1.0,
            np.dot(np.abs(diffs / (2 * d_max)) ** p, node_weights)
            ** (1.0 / p),
        )
    # De-mean each event over its *defined* positions only, so the comparison
    # uses a common support per candidate row.
    valid = ~np.isnan(all_de_ti)
    cnt = valid.sum(axis=1, keepdims=True)
    safe_cnt = np.where(cnt > 0, cnt, 1)
    m_sim = (
        np.where(valid, data_event_sim[np.newaxis, :], 0.0).sum(
            axis=1, keepdims=True
        )
        / safe_cnt
    )
    m_ti = (
        np.where(valid, all_de_ti, 0.0).sum(axis=1, keepdims=True) / safe_cnt
    )
    diffs = (data_event_sim[np.newaxis, :] - m_sim) - (all_de_ti - m_ti)
    contr = np.where(valid, np.abs(diffs / (2 * d_max)) ** p, 0.0)
    we = node_weights[np.newaxis, :] * valid
    agg = _renorm((we * contr).sum(axis=1), we.sum(axis=1)) ** (1.0 / p)
    # Clamp only finite aggregates so a fully-undefined row keeps its +inf
    # (never selected), matching the other metrics' exclusion contract.
    return np.where(np.isfinite(agg), np.minimum(1.0, agg), np.inf)
