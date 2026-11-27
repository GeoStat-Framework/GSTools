"""Shared helpers for the mps tests."""

import contextlib
import warnings

import numpy as np

from gstools.mps.distance import compute_node_weights, variable_distance
from gstools.tools.geometric import generate_grid


def flat_pos(grid_shape):
    """Flat per-dimension coordinate arrays for a unit-spaced grid."""
    return generate_grid([np.arange(s, dtype=float) for s in grid_shape])


def _var_dist(
    var,
    de_sim,
    row,
    cond_mask=None,
    cond_weight=1.0,
    lag_norms=None,
    d_max=None,
):
    """``variable_distance`` with node weights built the way the scan does."""
    w = compute_node_weights(
        len(de_sim), lag_norms, var.distance_power, cond_mask, cond_weight
    )
    return variable_distance(
        var, de_sim, row, w, var.d_max if d_max is None else d_max
    )


def uni_dist(ti, de_sim, de_ti, **kw):
    """Univariate data-event distance via the vectorized production path."""
    row = np.asarray(de_ti, dtype=float)[np.newaxis, :]
    return float(_var_dist(ti.variable(), de_sim, row, **kw)[0])


def mv_dist(ti, de_sim, de_ti, **kw):
    """Multivariate joint data-event distance via the vectorized path."""
    total = 0.0
    for v in ti.variables:
        row = np.asarray(de_ti[v.name], dtype=float)[np.newaxis, :]
        total += ti.weights[v.name] * float(
            _var_dist(v, de_sim[v.name], row, **kw)[0]
        )
    return total


@contextlib.contextmanager
def ignoring(*categories):
    """Silence library warnings of the given categories inside the block."""
    with warnings.catch_warnings():
        for category in categories:
            warnings.simplefilter("ignore", category)
        yield


def as_dict(ds, out):
    """Map a multivariate result tuple to ``{variable name: field}``."""
    return dict(zip((v.name for v in ds.ti.variables), out))
