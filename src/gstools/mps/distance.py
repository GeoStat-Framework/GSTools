"""Node-weight computation for MPS pattern comparison.

The vectorized pattern distances themselves (categorical / L1 / L2 / Lp /
variation) are computed by the Rust backend (``gstools_core.mps_dist_block_*``);
this module keeps only the shared spatial-decay weighting used to build a data
event.
"""

import numpy as np

from gstools.mps._validation import _validate_real_data

__all__ = ["compute_node_weights"]


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
    raw = np.ones(n, dtype=np.float64)
    conditioned = (
        np.zeros(n, dtype=bool)
        if cond_mask is None
        else np.asarray(cond_mask, dtype=bool)
    )
    if lag_norms is not None:
        _validate_real_data(lag_norms, False, "lag_norms")
        norms = np.asarray(lag_norms, dtype=np.float64)
    else:
        norms = np.zeros(n)
    nonzero = (norms != 0.0) & (distance_power != 0.0)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        raw[nonzero] = norms[nonzero] ** (-distance_power)
        raw[conditioned] *= cond_weight
        total = raw.sum()
    if np.isfinite(total) and total > 0.0:
        return raw / total

    # Shift before multiplying by the exponent, matching the native fallback.
    # A zero conditioning multiplier excludes a node even after recomputing.
    eligible = ~(conditioned & (cond_weight == 0.0))
    weights = np.zeros(n, dtype=np.float64)
    if not eligible.any():
        return weights
    log_lag = np.zeros(n)
    log_lag[nonzero] = np.log(norms[nonzero])
    log_lag = log_lag[eligible]
    with np.errstate(over="ignore", under="ignore"):
        log_weights = -distance_power * (log_lag - log_lag.min())
        if cond_weight > 0.0:
            log_weights[conditioned[eligible]] += np.log(cond_weight)
        weights[eligible] = np.exp(log_weights - log_weights.max())
    return weights / weights.sum()
