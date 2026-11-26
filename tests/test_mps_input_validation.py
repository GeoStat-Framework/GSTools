"""Input validation for training images, variables, and conditioning data."""

import numpy as np
import pytest

import gstools as gs
from gstools.mps import Variable


# Infinite training-image values are rejected; NaN marks missing data.
def test_inf_training_image_data_rejected_variable():
    with pytest.raises(ValueError, match="inf"):
        Variable("c", np.full((3, 3), np.inf), categorical=False)


def test_inf_training_image_data_rejected_bare_array():
    with pytest.raises(ValueError, match="inf"):
        gs.mps.TrainingImage(np.full((3, 3), np.inf), categorical=False)


def test_negative_inf_training_image_data_rejected():
    data = np.zeros((3, 3))
    data[0, 0] = -np.inf
    with pytest.raises(ValueError, match="inf"):
        Variable("c", data, categorical=False)


def test_nan_training_image_data_still_allowed():
    # NaN is the documented missing-value marker; it must NOT be rejected.
    data = np.zeros((3, 3))
    data[0, 0] = np.nan
    with pytest.warns(UserWarning, match="NaN"):
        var = Variable("c", data, categorical=False)
    assert var.has_nan is True


# --- Variable weights must be finite and non-negative ---
@pytest.mark.parametrize("bad", [-1.0, -0.001, np.inf, np.nan])
def test_variable_weight_rejects_nonfinite_or_negative(bad):
    with pytest.raises(ValueError, match="weight"):
        Variable("a", np.zeros((2, 2)), weight=bad)


def test_negative_multivariate_weight_rejected_via_variable():
    # The Variable constructor rejects the bad weight before TI assembly.
    with pytest.raises(ValueError, match="weight"):
        gs.mps.TrainingImage(
            [
                Variable("a", np.zeros((2, 2)), weight=2.0),
                Variable("b", np.zeros((2, 2)), weight=-1.0),
            ]
        )


def test_zero_weight_variable_is_allowed():
    # Zero is a valid (finite, >= 0) weight; only all-zero totals are rejected.
    var = Variable("a", np.zeros((2, 2)), weight=0.0)
    assert var.weight == 0.0


# --- Lp / variation exponent must be finite and > 0 (rejects 'linf'/'lnan') ---
@pytest.mark.parametrize("dist", ["linf", "lnan", "l0", "l-1"])
def test_lp_exponent_rejects_nonfinite_or_nonpositive(dist):
    with pytest.raises(ValueError):
        Variable("c", np.zeros((3, 3)), categorical=False, distance=dist)


@pytest.mark.parametrize(
    "dist", ["variationinf", "variationnan", "variation0"]
)
def test_variation_exponent_rejects_nonfinite_or_nonpositive(dist):
    with pytest.raises(ValueError):
        Variable("c", np.zeros((3, 3)), categorical=False, distance=dist)


# --- P2: conditioning positions must be finite (set_condition) ---
@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_set_condition_rejects_nonfinite_position(bad):
    ds = gs.mps.DirectSampling(
        gs.mps.MPSModel(gs.mps.TrainingImage(np.zeros((9, 9))))
    )
    with pytest.raises(ValueError, match="finite"):
        ds.set_condition([[bad], [0.0]], [0.0])


# --- P2: out-of-grid conditioning warns when snapped to an edge node ---
def test_out_of_grid_conditioning_warns_when_snapped():
    ds = gs.mps.DirectSampling(
        gs.mps.MPSModel(gs.mps.TrainingImage(np.zeros((9, 9))))
    )
    ds.set_condition([[100.0], [0.0]], [0.0])
    with pytest.warns(UserWarning, match="outside"):
        ds([range(3)] * 2, seed=1)


# --- Conditioning collision on an occupied grid node warns ---
def test_conditioning_collision_warns():
    ds = gs.mps.DirectSampling(
        gs.mps.MPSModel(gs.mps.TrainingImage(np.zeros((9, 9))))
    )
    # Both points snap to grid node (0, 0) on the [0, 1, 2] axes.
    ds.set_condition([[0.0, 0.1], [0.0, 0.0]], [0.0, 0.0])
    with pytest.warns(UserWarning, match="collided"):
        ds([range(3)] * 2, seed=1)
