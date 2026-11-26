"""Variable settings, penalty matrices, replacement, and TI windows."""

import numpy as np
import pytest

import gstools as gs
from gstools.mps import Variable


# --- Variable validation (finite, >= 0) ---
@pytest.mark.parametrize("bad", [-1.0, np.inf, np.nan])
def test_variable_distance_power_rejects_nonfinite_or_negative(bad):
    with pytest.raises(ValueError, match="distance_power"):
        Variable("a", np.zeros((3, 3)), distance_power=bad)


@pytest.mark.parametrize("bad", [-1.0, np.inf, np.nan])
def test_variable_cond_weight_rejects_nonfinite_or_negative(bad):
    with pytest.raises(ValueError, match="cond_weight"):
        Variable("a", np.zeros((3, 3)), cond_weight=bad)


# --- API placement: MPSModel no longer takes cond_weight ---
def test_mpsmodel_rejects_cond_weight_kwarg():
    ti = gs.mps.TrainingImage(np.zeros((9, 9)))
    with pytest.raises(TypeError):
        gs.mps.MPSModel(ti, cond_weight=2.0)


# --- TI sugar: bare array forwards distance_power/cond_weight to the variable ---
def test_training_image_sugar_forwards_distance_power_and_cond_weight():
    ti = gs.mps.TrainingImage(
        np.zeros((9, 9)),
        categorical=False,
        distance_power=0.7,
        cond_weight=3.0,
    )
    v = ti.variable()
    assert v.distance_power == pytest.approx(0.7)
    assert v.cond_weight == pytest.approx(3.0)
    assert ti.distance_power == pytest.approx(0.7)


# --- TI rejects distance_power/cond_weight when given Variable(s) ---
def test_training_image_rejects_per_run_settings_with_variable_list():
    with pytest.raises(ValueError, match="distance_power/cond_weight"):
        gs.mps.TrainingImage(
            [Variable("a", np.zeros((4, 4))), Variable("b", np.zeros((4, 4)))],
            cond_weight=2.0,
        )


def test_training_image_rejects_per_run_settings_with_single_variable():
    with pytest.raises(ValueError, match="distance_power/cond_weight"):
        gs.mps.TrainingImage(
            Variable("a", np.zeros((4, 4))), distance_power=1.0
        )


# --- DirectSampling.cond_weight reflects per-variable values ---
def test_directsampling_cond_weight_view_scalar_and_dict():
    ti_uni = gs.mps.TrainingImage(np.zeros((9, 9)), cond_weight=2.0)
    ds = gs.mps.DirectSampling(gs.mps.MPSModel(ti_uni))
    assert ds.cond_weight == pytest.approx(2.0)  # scalar when all equal

    ti_mv = gs.mps.TrainingImage(
        [
            Variable("a", np.zeros((9, 9), dtype=int), cond_weight=1.0),
            Variable("b", np.zeros((9, 9), dtype=int), cond_weight=4.0),
        ]
    )
    ds_mv = gs.mps.DirectSampling(gs.mps.MPSModel(ti_mv))
    assert ds_mv.cond_weight == {"a": 1.0, "b": 4.0}  # dict when they differ


# --- Native path: non-uniform per-variable distance_power changes the field ---
def test_non_uniform_distance_power_changes_output():
    rng = np.random.default_rng(0)
    data_a = rng.random((24, 24)) * 10.0
    data_b = rng.random((24, 24)) * 10.0
    pos = [np.arange(10, dtype=float)] * 2

    def run(dp_a):
        ti = gs.mps.TrainingImage(
            [
                Variable(
                    "a",
                    data_a,
                    categorical=False,
                    distance="l2",
                    n_neighbors=16,
                    distance_power=dp_a,
                ),
                Variable(
                    "b",
                    data_b,
                    categorical=False,
                    distance="l2",
                    n_neighbors=16,
                    distance_power=0.0,
                ),
            ]
        )
        return gs.mps.DirectSampling(gs.mps.MPSModel(ti, scan_fraction=0.4))(
            pos, seed=1
        )

    uniform = run(0.0)
    nonuniform = run(4.0)
    assert not np.array_equal(uniform["a"], nonuniform["a"]), (
        "per-variable distance_power had no effect on the field"
    )


# --- Native path: non-uniform per-variable cond_weight changes the field ---
def test_non_uniform_cond_weight_changes_output():
    rng = np.random.default_rng(3)
    data_a = rng.integers(0, 3, (24, 24)).astype(float)
    data_b = rng.integers(0, 3, (24, 24)).astype(float)
    pos = [np.arange(12, dtype=float)] * 2
    cpos = [np.array([5.0, 8.0]), np.array([5.0, 8.0])]

    def run(cw_a):
        ti = gs.mps.TrainingImage(
            [
                Variable(
                    "a",
                    data_a,
                    categorical=True,
                    n_neighbors=12,
                    cond_weight=cw_a,
                ),
                Variable(
                    "b",
                    data_b,
                    categorical=True,
                    n_neighbors=12,
                    cond_weight=1.0,
                ),
            ]
        )
        ds = gs.mps.DirectSampling(gs.mps.MPSModel(ti, scan_fraction=0.5))
        ds.set_condition(
            cpos,
            {"a": np.array([1.0, 2.0]), "b": np.array([0.0, 1.0])},
        )
        return ds(pos, seed=5)

    assert not np.array_equal(run(1.0)["a"], run(8.0)["a"]), (
        "per-variable cond_weight had no effect on the field"
    )


# ----------------------------------------------------------------------
# Penalty-matrix validation and storage
# ----------------------------------------------------------------------
def test_penalty_matrix_accepted_and_frozen():
    v = Variable(
        "c",
        np.array([[0, 1], [1, 0]]),
        penalty_matrix=[[0.0, 0.5], [0.3, 0.0]],
    )
    assert v.penalty_matrix.shape == (2, 2)
    assert not v.penalty_matrix.flags.writeable  # frozen


def test_penalty_matrix_rejects_non_square():
    with pytest.raises(ValueError, match="2D square"):
        Variable("c", np.array([0, 1]), penalty_matrix=[[0.0, 1.0, 1.0]])


def test_penalty_matrix_rejects_nonzero_diagonal():
    with pytest.raises(ValueError, match="diagonal"):
        Variable(
            "c", np.array([0, 1]), penalty_matrix=[[0.1, 1.0], [1.0, 0.0]]
        )


@pytest.mark.parametrize("bad", [0.0, -0.5, 1.5, np.nan])
def test_penalty_matrix_rejects_bad_offdiagonal(bad):
    with pytest.raises(ValueError, match="off-diagonal"):
        Variable(
            "c", np.array([0, 1]), penalty_matrix=[[0.0, bad], [1.0, 0.0]]
        )


def test_penalty_matrix_rejects_on_continuous():
    with pytest.raises(ValueError, match="categorical"):
        Variable(
            "c",
            np.zeros((3, 3)),
            categorical=False,
            penalty_matrix=[[0.0, 1.0], [1.0, 0.0]],
        )


def test_penalty_matrix_rejects_out_of_range_codes():
    # data holds code 5 but the 2x2 matrix only covers codes {0, 1}
    with pytest.raises(ValueError, match="category codes"):
        Variable(
            "c", np.array([0.0, 5.0]), penalty_matrix=[[0.0, 1.0], [1.0, 0.0]]
        )


def test_penalty_matrix_simulation_runs():
    # The Rust engine applies the categorical penalty distance,
    # so a simulation with a penalty_matrix runs (no NotImplementedError).
    ti = gs.mps.TrainingImage(
        Variable(
            "c",
            np.tile([0, 1], (9, 5))[:, :9].astype(float),
            penalty_matrix=[[0.0, 0.5], [0.5, 0.0]],
        )
    )
    field = gs.mps.DirectSampling(gs.mps.MPSModel(ti))([range(3)] * 2, seed=1)
    assert field.shape == (3, 3)
    assert np.all(np.isin(field, [0.0, 1.0]))


# ----------------------------------------------------------------------
# Variable.replace()
# ----------------------------------------------------------------------
def test_replace_shares_buffer_and_overrides_only_config():
    v = Variable("a", np.zeros((4, 4)), n_neighbors=10, cond_weight=2.0)
    v2 = v.replace(n_neighbors=5)
    assert v2.data is v.data  # no copy; shares the frozen buffer
    assert v2.n_neighbors == 5
    assert v2.cond_weight == 2.0  # carried over
    assert v.n_neighbors == 10  # original untouched


def test_replace_rejects_data_and_categorical_and_unknown():
    v = Variable("a", np.zeros((4, 4)))
    with pytest.raises(ValueError, match="data"):
        v.replace(data=np.ones((4, 4)))
    with pytest.raises(ValueError, match="categorical"):
        v.replace(categorical=False)
    with pytest.raises(ValueError, match="unknown field"):
        v.replace(not_a_field=1)


def test_replace_preserves_penalty_matrix():
    v = Variable(
        "c",
        np.array([[0, 1], [1, 0]]),
        penalty_matrix=[[0.0, 1.0], [1.0, 0.0]],
    )
    v2 = v.replace(n_neighbors=4)
    np.testing.assert_array_equal(v2.penalty_matrix, v.penalty_matrix)


# ----------------------------------------------------------------------
# TrainingImage.window()
# ----------------------------------------------------------------------
def test_window_returns_view_with_recomputed_range():
    data = np.arange(100, dtype=float).reshape(10, 10)
    ti = gs.mps.TrainingImage(data, categorical=False)
    sub = ti.window(np.s_[:5, 2:7])
    assert sub.shape == (5, 5)
    # shares memory with the parent buffer (a view, not a copy)
    assert np.shares_memory(sub.variable().data, ti.variable().data)
    # d_max recomputed from the sub-region
    expected = float(data[:5, 2:7].max() - data[:5, 2:7].min())
    assert sub.variable().d_max == pytest.approx(expected)


def test_window_multivariate_preserves_settings():
    a = np.arange(36, dtype=float).reshape(6, 6)
    b = (np.arange(36).reshape(6, 6) % 2).astype(float)
    ti = gs.mps.TrainingImage(
        [
            Variable("a", a, categorical=False, n_neighbors=7),
            Variable("b", b, categorical=True, cond_weight=3.0),
        ]
    )
    sub = ti.window(np.s_[1:4, 1:4])
    assert sub.shape == (3, 3)
    assert sub.variable("a").n_neighbors == 7
    assert sub.variable("b").cond_weight == 3.0


@pytest.mark.parametrize("sl", [np.s_[::2, :], np.s_[:20, :], np.s_[3:3, :]])
def test_window_rejects_bad_slices(sl):
    ti = gs.mps.TrainingImage(np.zeros((10, 10)))
    with pytest.raises(ValueError):
        ti.window(sl)


# ----------------------------------------------------------------------
# Named univariate TrainingImage is usable end-to-end
# ----------------------------------------------------------------------
def test_named_univariate_training_image_runs():
    ti = gs.mps.TrainingImage(Variable("f", np.zeros((9, 9))))
    assert ti.variable().name == "f"
    field = gs.mps.DirectSampling(gs.mps.MPSModel(ti))([range(3)] * 2, seed=2)
    # univariate return is a bare array (not a dict), all zeros from the TI
    assert isinstance(field, np.ndarray)
    np.testing.assert_array_equal(field, 0.0)


def _ti_data():
    rng = np.random.default_rng(0)
    return rng.integers(0, 3, (24, 24)).astype(float)


def _run(penalty, seed=1):
    data = _ti_data()
    var = Variable("c", data, categorical=True, n_neighbors=12)
    if penalty is not None:
        var = var.replace(penalty_matrix=penalty)
    ti = gs.mps.TrainingImage(var)
    return gs.mps.DirectSampling(gs.mps.MPSModel(ti, scan_fraction=0.4))(
        [np.arange(10, dtype=float)] * 2, seed=seed
    )


def test_all_ones_penalty_matches_plain_categorical():
    # Off-diagonal all 1.0 reproduces the plain 0/1 mismatch -> identical field.
    ones = 1.0 - np.eye(3)
    np.testing.assert_array_equal(_run(None), _run(ones))


def test_penalty_changes_output_vs_plain():
    # Small off-diagonal costs make near-miss categories cheaper -> different.
    soft = np.array([[0.0, 0.2, 0.9], [0.2, 0.0, 0.2], [0.9, 0.2, 0.0]])
    assert not np.array_equal(_run(None), _run(soft))


def test_asymmetric_penalty_differs_from_transpose():
    # Orientation T[sim, ti] must matter: P and P.T give different fields.
    p = np.array([[0.0, 0.2, 0.9], [0.8, 0.0, 0.1], [0.3, 0.7, 0.0]])
    f_p = _run(p, seed=5)
    f_pt = _run(p.T.copy(), seed=5)
    assert not np.array_equal(f_p, f_pt)


def test_penalty_with_masked_ti_runs():
    # A NaN (undefined) cell in the TI still simulates with a penalty matrix.
    data = _ti_data()
    data[0, 0] = np.nan
    var = Variable("c", data, categorical=True, n_neighbors=12).replace(
        penalty_matrix=1.0 - np.eye(3)
    )
    ti = gs.mps.TrainingImage(var)
    field = gs.mps.DirectSampling(gs.mps.MPSModel(ti, scan_fraction=0.4))(
        [np.arange(8, dtype=float)] * 2, seed=2
    )
    assert np.all(np.isfinite(field))
    assert np.all(np.isin(field, [0.0, 1.0, 2.0]))


def test_multivariate_penalty_on_one_variable_runs():
    rng = np.random.default_rng(7)
    ti = gs.mps.TrainingImage(
        [
            Variable(
                "facies",
                rng.integers(0, 3, (20, 20)).astype(float),
                categorical=True,
                n_neighbors=8,
                penalty_matrix=np.array(
                    [[0.0, 0.3, 0.6], [0.3, 0.0, 0.3], [0.6, 0.3, 0.0]]
                ),
            ),
            Variable(
                "cont",
                rng.random((20, 20)),
                categorical=False,
                distance="l1",
                n_neighbors=8,
            ),
        ]
    )
    out = gs.mps.DirectSampling(gs.mps.MPSModel(ti, scan_fraction=0.3))(
        [np.arange(8, dtype=float)] * 2, seed=0
    )
    assert np.all(np.isin(out["facies"], [0.0, 1.0, 2.0]))
    assert np.all(np.isfinite(out["cont"]))
