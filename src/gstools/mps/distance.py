"""Pure distance functions for MPS pattern comparison.

No class state — takes arrays and scalars, returns floats.
``TrainingImage.distance()`` uses these internally; other algorithms
can import them directly.
"""

import numpy as np

__all__ = [
    "compute_node_weights",
    "categorical_dist",
    "l1_dist",
    "l2_dist",
    "lp_dist",
    "variation_dist",
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


def categorical_dist(data_event_sim, data_event_ti, node_weights):
    """Weighted categorical distance (Mariethoz2010 Eq. 3).

    Parameters
    ----------
    data_event_sim : numpy.ndarray, shape (n,)
    data_event_ti : numpy.ndarray, shape (n,)
    node_weights : numpy.ndarray, shape (n,)
        Normalized spatial and conditioning weights.

    Returns
    -------
    float
        Distance in [0, 1].
    """
    return float(
        np.dot(
            node_weights,
            (data_event_sim != data_event_ti).astype(np.float64),
        )
    )


def l1_dist(data_event_sim, data_event_ti, node_weights, d_max):
    """Weighted L1 distance / Manhattan (Mariethoz2010 Eq. 6).

    Parameters
    ----------
    data_event_sim : numpy.ndarray, shape (n,)
    data_event_ti : numpy.ndarray, shape (n,)
    node_weights : numpy.ndarray, shape (n,)
        Normalized spatial and conditioning weights.
    d_max : float
        Data range for normalization.

    Returns
    -------
    float
        Distance in [0, 1].
    """
    return float(
        np.dot(node_weights, np.abs(data_event_sim - data_event_ti) / d_max)
    )


def l2_dist(data_event_sim, data_event_ti, node_weights, d_max):
    """Weighted L2 / RMS distance (Mariethoz2010 Eq. 4–5).

    Parameters
    ----------
    data_event_sim : numpy.ndarray, shape (n,)
    data_event_ti : numpy.ndarray, shape (n,)
    node_weights : numpy.ndarray, shape (n,)
        Normalized spatial and conditioning weights.
    d_max : float
        Data range for normalization.

    Returns
    -------
    float
        Distance in [0, 1].
    """
    return float(
        np.sqrt(
            np.dot(
                node_weights,
                ((data_event_sim - data_event_ti) / d_max) ** 2,
            )
        )
    )


def lp_dist(data_event_sim, data_event_ti, node_weights, d_max, p):
    """Weighted Lp (Minkowski) distance.

    Warning: Computationally heavier than l1_dist or l2_dist due to
    the generic C-level pow() evaluation. Use only when p != 1.0 or 2.0.

    Parameters
    ----------
    data_event_sim : numpy.ndarray, shape (n,)
    data_event_ti : numpy.ndarray, shape (n,)
    node_weights : numpy.ndarray, shape (n,)
        Normalized spatial and conditioning weights.
    d_max : float
        Data range for normalization.
    p : float
        The Minkowski exponent (e.g., 1.5, 3.0, 5.0).

    Returns
    -------
    float
        Distance in [0, 1].
    """
    diffs = np.abs(data_event_sim - data_event_ti) / d_max
    return float(np.sum(node_weights * (diffs**p)) ** (1.0 / p))


def variation_dist(data_event_sim, data_event_ti, node_weights, d_max, p=2.0):
    """Weighted variation distance (Mariethoz2010 Eq. 9, de-meaned).

    Parameters
    ----------
    data_event_sim : numpy.ndarray, shape (n,)
    data_event_ti : numpy.ndarray, shape (n,)
    node_weights : numpy.ndarray, shape (n,)
        Normalized spatial and conditioning weights.
    d_max : float
        Data range for normalization.
    p : float, optional
        Lp aggregation exponent. Default ``2.0`` (RMS, Mariethoz2010 Eq. 9).

    Returns
    -------
    float
        Distance in [0, 1].
    """
    diffs = (data_event_sim - data_event_sim.mean()) - (
        data_event_ti - data_event_ti.mean()
    )
    # 2*d_max normalises the common case to [0, 1]; SG values are not bounded
    # by the TI range (conditioning data / accumulated mean-shifts), so clamp.
    return float(
        min(
            1.0,
            np.dot(node_weights, np.abs(diffs / (2 * d_max)) ** p)
            ** (1.0 / p),
        )
    )


# ---------------------------------------------------------------------------
# Vectorized variants — same maths, operate on all TI candidates at once.
# Each accepts all_de_ti of shape (max_scan, n) and returns (max_scan,).
# np.dot(X, w) with X (max_scan, n) and w (n,) is a standard BLAS matvec.
# ---------------------------------------------------------------------------


def vec_categorical_dist(data_event_sim, all_de_ti, node_weights):
    """Vectorized categorical distance over all TI scan candidates.

    Parameters
    ----------
    data_event_sim : numpy.ndarray, shape (n,)
    all_de_ti : numpy.ndarray, shape (max_scan, n)
    node_weights : numpy.ndarray, shape (n,)

    Returns
    -------
    numpy.ndarray, shape (max_scan,)
        Distance in [0, 1] for each candidate.
    """
    return np.dot(
        (data_event_sim != all_de_ti).astype(np.float64), node_weights
    )


def vec_l1_dist(data_event_sim, all_de_ti, node_weights, d_max):
    """Vectorized L1 distance over all TI scan candidates.

    Parameters
    ----------
    data_event_sim : numpy.ndarray, shape (n,)
    all_de_ti : numpy.ndarray, shape (max_scan, n)
    node_weights : numpy.ndarray, shape (n,)
    d_max : float

    Returns
    -------
    numpy.ndarray, shape (max_scan,)
    """
    return np.dot(np.abs(data_event_sim - all_de_ti) / d_max, node_weights)


def vec_l2_dist(data_event_sim, all_de_ti, node_weights, d_max):
    """Vectorized L2 distance over all TI scan candidates.

    Parameters
    ----------
    data_event_sim : numpy.ndarray, shape (n,)
    all_de_ti : numpy.ndarray, shape (max_scan, n)
    node_weights : numpy.ndarray, shape (n,)
    d_max : float

    Returns
    -------
    numpy.ndarray, shape (max_scan,)
    """
    return np.sqrt(
        np.dot(((data_event_sim - all_de_ti) / d_max) ** 2, node_weights)
    )


def vec_lp_dist(data_event_sim, all_de_ti, node_weights, d_max, p):
    """Vectorized Lp distance over all TI scan candidates.

    Parameters
    ----------
    data_event_sim : numpy.ndarray, shape (n,)
    all_de_ti : numpy.ndarray, shape (max_scan, n)
    node_weights : numpy.ndarray, shape (n,)
    d_max : float
    p : float

    Returns
    -------
    numpy.ndarray, shape (max_scan,)
    """
    diffs = np.abs(data_event_sim - all_de_ti) / d_max
    return np.dot(diffs**p, node_weights) ** (1.0 / p)


def vec_variation_dist(data_event_sim, all_de_ti, node_weights, d_max, p=2.0):
    """Vectorized variation distance over all TI scan candidates.

    Parameters
    ----------
    data_event_sim : numpy.ndarray, shape (n,)
    all_de_ti : numpy.ndarray, shape (max_scan, n)
    node_weights : numpy.ndarray, shape (n,)
    d_max : float
    p : float, optional
        Lp aggregation exponent. Default ``2.0``.

    Returns
    -------
    numpy.ndarray, shape (max_scan,)
        Distance in [0, 1].
    """
    de_sim_c = data_event_sim - data_event_sim.mean()
    all_de_ti_c = all_de_ti - all_de_ti.mean(axis=1, keepdims=True)
    diffs = de_sim_c - all_de_ti_c
    # 2*d_max normalises the common case to [0, 1]; SG values are not bounded
    # by the TI range (conditioning data / accumulated mean-shifts), so clamp.
    return np.minimum(
        1.0,
        np.dot(np.abs(diffs / (2 * d_max)) ** p, node_weights) ** (1.0 / p),
    )
