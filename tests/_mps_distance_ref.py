"""NumPy reference distances for MPS tests (test-only oracle).

These pure-NumPy implementations serve as independent oracles for the
production Rust kernels (``gstools_core.mps_dist_block_*``). They remain in the
test suite so expected values do not share the implementation under test.
"""

import numpy as np


def _masked_weights(all_de_ti, node_weights):
    valid = ~np.isnan(all_de_ti)
    we = node_weights[np.newaxis, :] * valid
    return valid, we, we.sum(axis=1)


def _renorm(numerator, wsum):
    out = np.full(numerator.shape, np.inf, dtype=np.float64)
    nz = wsum > 0
    out[nz] = numerator[nz] / wsum[nz]
    return out


def vec_categorical_dist(
    data_event_sim, all_de_ti, node_weights, has_nan=False
):
    if not has_nan:
        return np.dot(
            (data_event_sim != all_de_ti).astype(np.float64), node_weights
        )
    _, we, wsum = _masked_weights(all_de_ti, node_weights)
    mism = (we * (data_event_sim != all_de_ti)).sum(axis=1)
    return _renorm(mism, wsum)


def vec_l1_dist(data_event_sim, all_de_ti, node_weights, d_max, has_nan=False):
    if not has_nan:
        return np.dot(np.abs(data_event_sim - all_de_ti) / d_max, node_weights)
    valid, we, wsum = _masked_weights(all_de_ti, node_weights)
    ad = np.where(valid, np.abs(data_event_sim - all_de_ti) / d_max, 0.0)
    return _renorm((we * ad).sum(axis=1), wsum)


def vec_l2_dist(data_event_sim, all_de_ti, node_weights, d_max, has_nan=False):
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
    if not has_nan:
        de_sim_c = data_event_sim - data_event_sim.mean()
        all_de_ti_c = all_de_ti - all_de_ti.mean(axis=1, keepdims=True)
        diffs = de_sim_c - all_de_ti_c
        return np.minimum(
            1.0,
            np.dot(np.abs(diffs / (2 * d_max)) ** p, node_weights)
            ** (1.0 / p),
        )
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
    return np.where(np.isfinite(agg), np.minimum(1.0, agg), np.inf)
