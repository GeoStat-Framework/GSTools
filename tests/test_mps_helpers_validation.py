"""Independent helper, representation and compatibility regressions (F3/F5/F6)."""

import importlib
import sys
import types

import numpy as np
import pytest

from gstools.mps import (
    DirectSampling,
    MPSModel,
    TrainingImage,
    Variable,
    _core,
)
from gstools.mps.model import _validate_post_processing
from gstools.mps.training_image import _data_range, _validate_n_neighbors


@pytest.mark.parametrize("name", [None, "named"])
def test_common_support_adjustment(name):
    ti = TrainingImage(
        Variable(
            name,
            [0.0, 10.0],
            categorical=False,
            distance="variation",
            n_neighbors=2,
        )
    )
    assert ti.adjust_value(10.0, [100.0, 200.0], [1.0, np.nan]) == 109.0
    assert ti.adjust_value(10.0, [np.nan], [1.0]) == 10.0
    assert ti.adjust_value(10.0, [], []) == 10.0
    with pytest.raises(ValueError, match="equal-length"):
        ti.adjust_value(10.0, [1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="rescale"):
        ti.adjust_value(1e308, [1e308], [-1e308])


@pytest.mark.parametrize("name", [None, "named"])
def test_asymmetric_penalty_helper_and_supplied_missingness(name):
    ti = TrainingImage(
        Variable(
            name,
            [0, 1],
            n_neighbors=2,
            penalty_matrix=[[0.0, 0.2], [0.7, 0.0]],
        )
    )
    # T[simulation, TI], weighted by surviving support per candidate.
    actual = ti.vec_distance_var(
        None,
        [0.0, 1.0],
        [[1.0, 0.0], [np.nan, 0.0], [np.nan, np.nan]],
        weights=np.array([0.25, 0.75]),
    )
    np.testing.assert_allclose(actual, [0.575, 0.7, np.inf])
    np.testing.assert_allclose(
        ti.vec_distance_var(None, [0.0], [[1.0]]), [0.2]
    )


def test_condition_weight_default_and_override():
    ti = TrainingImage(Variable("x", [0, 1], n_neighbors=2, cond_weight=3.0))
    args = (None, [0.0, 0.0], [[1.0, 0.0]])
    np.testing.assert_allclose(
        ti.vec_distance_var(*args, cond_mask=[True, False]), [0.75]
    )
    np.testing.assert_allclose(
        ti.vec_distance_var(*args, cond_mask=[True, False], cond_weight=1.0),
        [0.5],
    )


@pytest.mark.parametrize("metric", ["l1", "l2", "l3", "variation"])
def test_supplied_nan_block_enables_masking(metric):
    ti = TrainingImage(
        np.array([0.0, 1.0]), categorical=False, distance=metric, n_neighbors=2
    )
    # One shared finite point: variation residual is zero, Lp error is one.
    result = ti.vec_distance_var(None, [1.0, 1.0], [[np.nan, 0.0]])
    np.testing.assert_allclose(result, [0.0 if metric == "variation" else 1.0])


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(weights=[-1.0]),
        dict(weights=[np.inf]),
        dict(weights=[1.0, 1.0]),
        dict(cond_mask=[True, False]),
        dict(lag_norms=[np.nan]),
        dict(cond_weight=-1.0),
    ],
)
def test_helper_malformed_inputs_raise(kwargs):
    ti = TrainingImage(np.array([0, 1]), n_neighbors=1)
    with pytest.raises(ValueError):
        ti.vec_distance_var(None, [0.0], [[1.0]], **kwargs)


@pytest.mark.parametrize(
    "value",
    [
        1,
        np.int64(3),
        np.uint64(2**63),
        2**53 + 1,
        2**80,
        3.0,
        True,
        np.bool_(True),
    ],
)
def test_count_conversion_is_exact(value):
    assert _validate_n_neighbors(value, None) == int(value)
    assert _validate_post_processing(value) == int(value)


@pytest.mark.parametrize(
    "value", [1.9, 0.9, -0.5, -1, np.nan, np.inf, -np.inf, "3"]
)
def test_count_conversion_rejects_loss_or_invalid(value):
    with pytest.raises(ValueError):
        _validate_n_neighbors(value, None)
    with pytest.raises(ValueError):
        _validate_post_processing(value)


def test_count_zero_policy():
    assert _validate_post_processing(False) == 0
    with pytest.raises(ValueError):
        _validate_n_neighbors(0, None)


@pytest.mark.parametrize("categorical", [False, True])
@pytest.mark.parametrize(
    "data",
    [
        np.array([2**60, 2**60 + 3], np.int64),
        np.array([np.iinfo(np.int64).max]),
        np.array([np.iinfo(np.uint64).max], np.uint64),
    ],
)
def test_lossy_integer_data_rejected(data, categorical):
    with pytest.raises(
        ValueError, match="recode" if categorical else "center/rescale"
    ):
        TrainingImage(data, categorical=categorical)


@pytest.mark.parametrize(
    "data",
    [
        np.array([-(2**63), 0], np.int64),
        np.array([0, 2**63], np.uint64),
        np.array([2**60, 2**60 + 256], np.int64),
        np.array([-128, 127], np.int8),
    ],
)
def test_exact_large_integers_supported(data):
    ti = TrainingImage(data, categorical=False)
    assert ti.variable().d_max == float(int(data.max()) - int(data.min()))


def test_integer_range_subtracts_before_conversion():
    assert _data_range(np.array([2**60, 2**60 + 3])) == 3.0


@pytest.mark.parametrize("radius", [np.nan, np.inf, -np.inf, 0, -1])
def test_radius_requires_none_or_finite_positive(radius):
    with pytest.raises(ValueError, match="max_radius"):
        TrainingImage(np.array([0, 1]), max_radius=radius)


@pytest.mark.parametrize(
    "version, supported",
    [
        ("1.4.0", True),
        ("1.4.0+local", True),
        ("1.4.0.post1", True),
        ("1.4.0rc1", False),
        ("1.4.0.dev1", False),
        ("1.3.99", False),
        ("garbage", False),
        ("1.5.0", True),
    ],
)
def test_core_pep440(version, supported):
    assert _core._supported_version(version) is supported


@pytest.mark.parametrize("revision", [6, 7, 9, 100, "8", None, 8.5])
def test_unknown_interface_revision_rejected(revision):
    fake = types.ModuleType("gstools_core")
    for name in _core._REQUIRED_EXPORTS:
        setattr(fake, name, getattr(_core._gstools_core, name))
    fake.__version__ = "1.4.0"
    fake.mps_api_version = lambda: revision
    saved = {
        name: sys.modules.get(name)
        for name in ("gstools_core", "gstools.mps._core")
    }
    try:
        sys.modules["gstools_core"] = fake
        sys.modules.pop("gstools.mps._core", None)
        with pytest.raises(ImportError, match="exactly 8"):
            importlib.import_module("gstools.mps._core")
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@pytest.mark.parametrize("threshold", [0, 0.0, -0.0, None])
def test_threshold_mode_and_zone_namespace(threshold):
    import gstools as gs

    ti = TrainingImage(np.array([0.0, 1.0]))
    assert MPSModel(ti, threshold=threshold).threshold == 0.0
    assert isinstance(gs.mps.Zone(ti, where=np.ones(2, bool)), gs.mps.Zone)


def test_penalty_helper_rejects_overflowing_weight_total():
    ti = TrainingImage(
        Variable(
            None,
            [0, 1],
            n_neighbors=2,
            penalty_matrix=[[0.0, 0.2], [0.7, 0.0]],
        )
    )
    with pytest.raises(ValueError, match="weights"):
        ti.vec_distance_var(
            None, [0.0, 1.0], [[1.0, 0.0]], weights=[1e308, 1e308]
        )


@pytest.mark.parametrize(
    "values",
    [
        np.array([2**60 + 1, 2**60 + 3], dtype=object),
        np.array([1 + 2j, 1 + 3j]),
    ],
)
def test_real_data_required_before_float64_conversion(values):
    with pytest.raises(ValueError, match="real numeric"):
        TrainingImage(values)
    image = TrainingImage(np.array([0, 1]))
    ds = DirectSampling(MPSModel(image))
    with pytest.raises(ValueError, match="real numeric"):
        ds.set_condition([[0, 1]], values)
    with pytest.raises(ValueError, match="real numeric"):
        image.vec_distance_var(None, [0, 1], [values])
    with pytest.raises(ValueError, match="real numeric"):
        image.adjust_value(0, values, [0, 1])


def test_extreme_event_weights_preserve_candidate_ranking():
    image = TrainingImage(np.array([0, 1]))
    actual = image.vec_distance_var(
        None,
        [0, 0, 0],
        [[0, 0, 1], [1, 1, 0]],
        cond_mask=[True, True, False],
        cond_weight=1e308,
    )
    assert actual[0] < actual[1]
    image = TrainingImage(np.array([0, 1]), distance_power=2048)
    actual = image.vec_distance_var(
        None, [0, 0], [[0, 1], [1, 0]], lag_norms=[2, 3]
    )
    np.testing.assert_array_equal(actual, [0, 1])
