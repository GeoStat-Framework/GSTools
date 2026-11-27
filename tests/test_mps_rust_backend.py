"""Rust-only MPS regression tests.

The independent NumPy distance references remain in _mps_distance_ref; these
tests exercise the installed Core exports and the complete DS engine.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from _mps_distance_ref import (
    vec_categorical_dist,
    vec_l1_dist,
    vec_l2_dist,
    vec_lp_dist,
    vec_variation_dist,
)

from gstools.mps import DirectSampling, MPSModel, TrainingImage
from gstools.mps import simulate as _simulate
from gstools.mps.training_image import Variable

gstools_core = pytest.importorskip("gstools_core")


@pytest.fixture(autouse=True)
def _restore_mps_engine_state():
    original_hook = _simulate._MPS_RUST_ENGINE_STATS_HOOK
    yield
    _simulate._MPS_RUST_ENGINE_STATS_HOOK = original_hook


def _positions(shape):
    return [np.arange(float(n)) for n in shape]


def _training_data(shape=(32, 32)):
    coords = np.indices(shape)
    return ((coords[0] // 4 + coords[1] // 5) % 3).astype(float)


@pytest.mark.parametrize(
    ("distance", "reference", "export", "categorical"),
    [
        ("categorical", vec_categorical_dist, "mps_dist_block_cat", True),
        ("l1", vec_l1_dist, "mps_dist_block_l1", False),
        ("l2", vec_l2_dist, "mps_dist_block_l2", False),
        ("lp", vec_lp_dist, "mps_dist_block_lp", False),
        ("variation", vec_variation_dist, "mps_dist_block_variation", False),
    ],
)
@pytest.mark.parametrize("masked", [False, True])
def test_distance_exports_match_independent_reference(
    distance, reference, export, categorical, masked
):
    rng = np.random.default_rng(20260809)
    n_lags, n_candidates = 8, 17
    data_event = (
        rng.integers(0, 3, n_lags).astype(float)
        if categorical
        else rng.normal(size=n_lags)
    )
    candidates = (
        rng.integers(0, 3, (n_candidates, n_lags)).astype(float)
        if categorical
        else rng.normal(size=(n_candidates, n_lags))
    )
    if masked:
        candidates[rng.random(candidates.shape) < 0.2] = np.nan
        candidates[0] = np.nan
        export += "_masked"
    weights = rng.random(n_lags)
    weights /= weights.sum()
    base = np.arange(n_candidates, dtype=np.int64) * n_lags
    lags = np.arange(n_lags, dtype=np.int64)
    d_max = 4.0
    p = 3.0 if distance == "lp" else 2.0
    if categorical:
        expected = reference(data_event, candidates, weights, has_nan=masked)
        args = (data_event, candidates.ravel(), base, lags, weights)
    elif distance == "lp":
        expected = reference(
            data_event, candidates, weights, d_max, p, has_nan=masked
        )
        args = (data_event, candidates.ravel(), base, lags, weights, d_max, p)
    elif distance == "variation":
        expected = reference(
            data_event, candidates, weights, d_max, p=p, has_nan=masked
        )
        args = (data_event, candidates.ravel(), base, lags, weights, d_max, p)
    else:
        expected = reference(
            data_event, candidates, weights, d_max, has_nan=masked
        )
        args = (data_event, candidates.ravel(), base, lags, weights, d_max)
    np.testing.assert_allclose(
        getattr(gstools_core, export)(*args), expected, rtol=1e-12, atol=1e-13
    )


def test_rayon_categorical_export_matches_serial():
    rng = np.random.default_rng(1701)
    n_lags, n_candidates = 8, 4097
    data_event = rng.integers(0, 4, n_lags).astype(float)
    candidates = rng.integers(0, 4, (n_candidates, n_lags)).astype(float)
    base = np.arange(n_candidates, dtype=np.int64) * n_lags
    lags = np.arange(n_lags, dtype=np.int64)
    weights = rng.random(n_lags)
    weights /= weights.sum()
    args = (data_event, candidates.ravel(), base, lags, weights)
    np.testing.assert_array_equal(
        gstools_core.mps_dist_block_cat_rayon(*args),
        gstools_core.mps_dist_block_cat(*args),
    )


@pytest.mark.parametrize(
    ("categorical", "distance"),
    [
        (True, "l1"),
        (False, "l1"),
        (False, "l2"),
        (False, "l3"),
        (False, "variation"),
        (False, "variation1.5"),
    ],
)
def test_complete_engine_is_reproducible_across_threads(categorical, distance):
    data = (
        _training_data()
        if categorical
        else np.sin(np.indices((32, 32))[0] / 5)
    )
    ti = TrainingImage(
        data, categorical=categorical, distance=distance, n_neighbors=8
    )
    model = MPSModel(ti, scan_fraction=0.3, threshold=0.0)

    def run(threads):
        ds = DirectSampling(model, seed=42)
        return ds(_positions((10, 10)), num_threads=threads, store=False)

    first = run(1)
    np.testing.assert_array_equal(first, run(1))
    np.testing.assert_array_equal(first, run(4))
    assert np.isfinite(first).all()


def test_multivariate_masked_conditions_and_threads():
    clean = _training_data()
    masked = clean.copy()
    masked[8:12, 9:14] = np.nan
    with pytest.warns(UserWarning, match="contains NaN"):
        ti = TrainingImage(
            [
                Variable("clean", clean, categorical=True, n_neighbors=8),
                Variable("masked", masked, categorical=True, n_neighbors=8),
            ]
        )
    model = MPSModel(ti, scan_fraction=0.3)

    def run(threads):
        ds = DirectSampling(model, seed=42)
        ds.set_condition(
            [[1.0, 4.0], [2.0, 5.0]],
            {
                "clean": np.array([1.0, np.nan]),
                "masked": np.array([np.nan, 2.0]),
            },
        )
        return ds(_positions((10, 10)), num_threads=threads, store=False)

    one, four = run(1), run(4)
    for variable in ("clean", "masked"):
        np.testing.assert_array_equal(one[variable], four[variable])
        assert np.isfinite(one[variable]).all()
    assert one["clean"][1, 2] == 1.0
    assert one["masked"][4, 5] == 2.0


def test_nonstationary_explicit_path_uses_complete_engine(monkeypatch):
    calls = []
    original = _simulate._mps_simulate_gsc

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(_simulate, "_mps_simulate_gsc", spy)
    ti = TrainingImage(_training_data(), categorical=True, n_neighbors=8)
    ds = DirectSampling(
        MPSModel(ti, scan_fraction=0.3, rotation=np.pi / 6, scale=[1.0, 0.8]),
        seed=42,
    )
    shape = (9, 9)
    path = np.argwhere(np.ones(shape, dtype=bool))[::-1]
    result = ds(_positions(shape), path=path, store=False)
    assert len(calls) == 1
    assert np.isfinite(result).all()


def test_progress_callback_is_not_in_api():
    assert (
        "progress" not in inspect.signature(DirectSampling.__call__).parameters
    )
    assert (
        "progress" not in inspect.signature(_simulate.ds_simulate).parameters
    )


def test_stats_hook_reports_rust_engine(monkeypatch):
    captured = []
    monkeypatch.setattr(
        _simulate, "_MPS_RUST_ENGINE_STATS_HOOK", captured.append
    )
    ti = TrainingImage(_training_data(), categorical=True)
    ds = DirectSampling(MPSModel(ti, scan_fraction=0.3), seed=42)
    result = ds(_positions((24, 24)), num_threads=4, store=False)
    assert np.isfinite(result).all()
    assert captured[-1]["requested_threads"] == 4
    assert 1 <= captured[-1]["used_threads"] <= 4
