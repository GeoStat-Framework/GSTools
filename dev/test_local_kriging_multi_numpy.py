"""
Validation script for the NumPy multi-field local-kriging prototype.

Checks that kriging many fields at once with cached weights
(local_kriging_multi_numpy) gives the same result as calling the
single-field prototype once per field with only the available conditioning points,
and that the cache actually saves solves.

Run with:
    uv run pytest dev/test_local_kriging_multi_numpy.py -v
"""

import numpy as np
import pytest
from scipy.spatial import cKDTree

import gstools as gs
from local_kriging_multi_numpy import (
    RUST_AVAILABLE,
    calc_field_krige_local_multi,
    calc_field_krige_local_multi_python,
    local_krige_multi,
)
from local_kriging_numpy import calc_field_krige_local_python

FULL_RADIUS = 1e6
RADIUS = 4.0


def make_data(n=40, m=25, t_cnt=30, missing=0.2, seed=1):
    """Conditioning points in [0, 10]^2 with random gaps, target points in the same box."""
    rng = np.random.RandomState(seed)
    cond_pos = rng.uniform(0, 10, (2, n))
    target_pos = rng.uniform(0, 10, (2, m))
    cond_val = rng.uniform(-1, 1, (n, t_cnt))
    cond_val[rng.uniform(size=(n, t_cnt)) < missing] = np.nan
    # smooth external drift, e.g. an elevation
    drift_cond = (cond_pos[0] + 0.5 * cond_pos[1])[np.newaxis]
    drift_target = (target_pos[0] + 0.5 * target_pos[1])[np.newaxis]
    return cond_pos, cond_val, target_pos, drift_cond, drift_target


MODEL = gs.Exponential(dim=2, var=1.5, len_scale=3, nugget=0.1)

# (unbiased, use external drift): simple, ordinary, external drift kriging
SETUPS = [(False, False), (True, False), (True, True)]

needs_rust = pytest.mark.skipif(
    not RUST_AVAILABLE, reason="GSTools-Core not available"
)
BACKENDS = ["python", pytest.param("rust", marks=needs_rust)]


def run_multi(cond_pos, cond_val, target_pos, drift_cond, drift_target,
              unbiased, radius=RADIUS, min_neighbors=1):
    n = cond_pos.shape[1]
    return calc_field_krige_local_multi_python(
        cond_pos, cond_val, target_pos, MODEL,
        cond_err=np.full(n, MODEL.nugget),
        drift_cond=drift_cond, drift_target=drift_target,
        unbiased=unbiased, exact=False, local_radius=radius,
        min_neighbors=min_neighbors,
    )


@pytest.mark.parametrize("unbiased, ext", SETUPS)
def test_matches_single_field_calls(unbiased, ext):
    cond_pos, cond_val, target_pos, d_cond, d_targ = make_data()
    if not ext:
        d_cond, d_targ = d_cond[:0], d_targ[:0]
    field, error, n_used, _ = run_multi(
        cond_pos, cond_val, target_pos, d_cond, d_targ, unbiased
    )
    for t in range(cond_val.shape[1]):
        avail = np.isfinite(cond_val[:, t])
        # independent neighbor count with only this field's conditioning points
        counts = np.array([
            len(lst) for lst in cKDTree(cond_pos[:, avail].T).query_ball_point(
                target_pos.T, r=RADIUS)
        ])
        np.testing.assert_array_equal(n_used[:, t], counts)
        ok = counts > 0
        f_ref, e_ref = calc_field_krige_local_python(
            cond_pos[:, avail], cond_val[avail, t], target_pos[:, ok], MODEL,
            np.full(avail.sum(), MODEL.nugget), d_cond[:, avail], d_targ[:, ok],
            unbiased, False, RADIUS,
        )
        np.testing.assert_allclose(field[ok, t], f_ref, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(error[ok, t], e_ref, rtol=1e-10, atol=1e-12)
        assert np.isnan(field[~ok, t]).all()


def test_one_solve_per_target_without_missing_data():
    cond_pos, cond_val, target_pos, d_cond, d_targ = make_data(missing=0.0)
    m, t_cnt = target_pos.shape[1], cond_val.shape[1]
    stats = run_multi(cond_pos, cond_val, target_pos, d_cond, d_targ, True)[3]
    assert stats["n_solves"] == m
    assert stats["n_hits_last"] == (t_cnt - 1) * m


def test_cache_reuses_recurring_masks():
    """Alternating outages A, B, A, B, ...: the hash map finds them."""
    cond_pos, cond_val, target_pos, d_cond, d_targ = make_data(missing=0.0)
    cond_val[0, ::2] = np.nan
    cond_val[1, 1::2] = np.nan
    m, t_cnt = target_pos.shape[1], cond_val.shape[1]
    args = (cond_pos, cond_val, target_pos, d_cond, d_targ, True, FULL_RADIUS)
    stats = run_multi(*args)[3]
    assert stats["n_solves"] == 2 * m
    assert stats["n_hits_hash"] == (t_cnt - 2) * m


def test_min_neighbors_gives_nan():
    cond_pos, cond_val, target_pos, d_cond, d_targ = make_data(missing=0.0)
    cond_val[2:, 0] = np.nan  # only 2 points with a value in field 0
    field, error, n_used, stats = run_multi(
        cond_pos, cond_val, target_pos, d_cond, d_targ, True,
        radius=FULL_RADIUS, min_neighbors=3,
    )
    assert (n_used[:, 0] == 2).all()
    assert np.isnan(field[:, 0]).all() and np.isnan(error[:, 0]).all()
    assert np.isfinite(field[:, 1:]).all()
    assert stats["n_skipped"] == target_pos.shape[1]


@pytest.mark.parametrize("backend", BACKENDS)
def test_wrapper_ext_drift_matches_global_per_day(backend):
    cond_pos, cond_val, target_pos, d_cond, d_targ = make_data()
    template = gs.krige.LocalExtDrift(
        MODEL, cond_pos, np.zeros(cond_pos.shape[1]),
        local_radius=FULL_RADIUS, ext_drift=d_cond[0],
    )
    field, var = local_krige_multi(
        template, target_pos, cond_val, ext_drift=d_targ[0], backend=backend
    )
    assert field.shape == (cond_val.shape[1], target_pos.shape[1])
    for t in range(cond_val.shape[1]):
        avail = np.isfinite(cond_val[:, t])
        ref = gs.krige.ExtDrift(
            MODEL, cond_pos[:, avail], cond_val[avail, t], ext_drift=d_cond[0, avail]
        )
        f_ref, v_ref = ref(target_pos, ext_drift=d_targ[0], post_process=False, store=False)
        np.testing.assert_allclose(field[t], f_ref, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(var[t], v_ref, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("backend", BACKENDS)
def test_wrapper_simple_with_mean_matches_global_per_day(backend):
    cond_pos, cond_val, target_pos, _, _ = make_data()
    template = gs.krige.LocalSimple(
        MODEL, cond_pos, np.zeros(cond_pos.shape[1]),
        local_radius=FULL_RADIUS, mean=0.3,
    )
    field = local_krige_multi(
        template, target_pos, cond_val, return_var=False, backend=backend
    )
    for t in range(cond_val.shape[1]):
        avail = np.isfinite(cond_val[:, t])
        ref = gs.krige.Simple(MODEL, cond_pos[:, avail], cond_val[avail, t], mean=0.3)
        f_ref = ref(target_pos, return_var=False, post_process=False, store=False)
        np.testing.assert_allclose(field[t], f_ref, rtol=1e-6, atol=1e-8)


def test_wrapper_rejects_template_without_all_conditioning_points():
    cond_pos, cond_val, target_pos, _, _ = make_data()
    first_day = cond_val[:, 0]  # contains NaN -> template drops conditioning points
    template = gs.krige.LocalOrdinary(MODEL, cond_pos, first_day, local_radius=RADIUS)
    with pytest.raises(ValueError, match="holds every conditioning point"):
        local_krige_multi(template, target_pos, cond_val)


@needs_rust
@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("unbiased, ext", SETUPS)
def test_rust_matches_python(unbiased, ext, exact):
    cond_pos, cond_val, target_pos, d_cond, d_targ = make_data()
    if not ext:
        d_cond, d_targ = d_cond[:0], d_targ[:0]
    args = (
        cond_pos, cond_val, target_pos, MODEL, np.full(cond_pos.shape[1], MODEL.nugget),
        d_cond, d_targ, unbiased, exact, RADIUS,
    )
    f_py, e_py, n_py, s_py = calc_field_krige_local_multi(
        *args, min_neighbors=2, backend="python"
    )
    f_rs, e_rs, n_rs, s_rs = calc_field_krige_local_multi(
        *args, min_neighbors=2, backend="rust"
    )
    np.testing.assert_allclose(f_rs, f_py, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(e_rs, e_py, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(n_rs, n_py)
    assert s_rs == s_py


@needs_rust
def test_rust_rejects_bad_input():
    cond_pos, cond_val, target_pos, d_cond, d_targ = make_data()
    n = cond_pos.shape[1]
    args = (cond_pos, cond_val[:-1], target_pos, MODEL, np.full(n, 0.1),
            d_cond, d_targ, True, False, RADIUS)
    with pytest.raises(ValueError, match="cond_val has"):
        calc_field_krige_local_multi(*args, backend="rust")
