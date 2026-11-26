"""Vectorized conditioning snap vs a per-point reference.

``DirectSampling._conditions_to_grid`` snaps all points with NumPy. These
tests compare it with the straightforward per-point rule it replaced: nearest
node per axis (lower index on ties) and, per variable, the closest finite
point on a shared node (the first one on ties).
"""

import warnings

import numpy as np
import pytest

from gstools.mps import DirectSampling, MPSModel, TrainingImage, Variable


def _reference(axes, pos, val):
    """Per-point snap; returns ``{node: {variable: value}}``."""
    best = {}
    for k in range(pos.shape[1]):
        idx = tuple(
            int(np.argmin(np.abs(ax - pos[d, k]))) for d, ax in enumerate(axes)
        )
        dist_sq = sum(
            (ax[idx[d]] - pos[d, k]) ** 2 for d, ax in enumerate(axes)
        )
        node = best.setdefault(idx, {})
        for v, x in val.items():
            if np.isfinite(x[k]) and (v not in node or dist_sq < node[v][1]):
                node[v] = (float(x[k]), dist_sq)
    return {i: {v: x for v, (x, _) in n.items()} for i, n in best.items() if n}


def _as_dict(conditions):
    if conditions is None:
        return {}
    nodes, values = conditions
    out = {}
    for i, node in enumerate(zip(*(n.tolist() for n in nodes))):
        out[node] = {
            v: float(c[i]) for v, c in values.items() if np.isfinite(c[i])
        }
    return out


def _generator(dim):
    rng = np.random.default_rng(0)
    shape = (6,) * dim
    ti = TrainingImage(
        [
            Variable("a", rng.integers(0, 3, shape)),
            Variable("b", rng.integers(0, 3, shape)),
        ]
    )
    return DirectSampling(MPSModel(ti, scan_fraction=0.5))


@pytest.mark.parametrize("seed", range(40))
def test_matches_per_point_reference(seed):
    rng = np.random.default_rng(seed)
    dim = int(rng.integers(1, 4))
    axes = []
    for _ in range(dim):
        n = int(rng.integers(2, 8))
        kind = rng.integers(0, 3)
        if kind == 0:
            axes.append(np.arange(n) * 0.5)
        elif kind == 1:
            axes.append(np.cumsum(rng.uniform(0.1, 2.0, n)))
        else:
            axes.append(np.arange(n)[::-1] * 0.25)
    n_pts = int(rng.integers(1, 50))
    pos = np.array(
        [
            # node positions, exact midpoints (ties) and outside points
            rng.choice(
                np.concatenate(
                    [ax, (ax[1:] + ax[:-1]) / 2, [ax.min() - 1, ax.max() + 1]]
                ),
                n_pts,
            )
            for ax in axes
        ]
    )
    val = {}
    for v in ("a", "b"):
        x = rng.integers(0, 3, n_pts).astype(float)
        x[rng.random(n_pts) < 0.4] = np.nan
        val[v] = x
    ds = _generator(dim)
    ds.set_condition(list(pos), val)
    with warnings.catch_warnings():  # outside points, collisions
        warnings.simplefilter("ignore")
        got = ds._conditions_to_grid(axes)
    expected = _reference(axes, pos, val)
    assert _as_dict(got) == expected


def test_exhaustive_grid_is_identity():
    # One point per node (an auxiliary field known everywhere): every node
    # maps to itself, without collisions.
    ds = _generator(3)
    axes = [np.arange(5) * 0.05, np.arange(4) * 0.05, np.arange(3) * 0.05]
    grid = np.meshgrid(*axes, indexing="ij")
    b = np.arange(grid[0].size, dtype=float)
    ds.set_condition(
        [g.ravel() for g in grid], {"a": np.full(b.size, np.nan), "b": b}
    )
    nodes, values = ds._conditions_to_grid(axes)
    assert np.array_equal(
        np.ravel_multi_index(nodes, (5, 4, 3)), np.arange(b.size)
    )
    assert np.array_equal(values["b"], b)
    assert np.isnan(values["a"]).all()
