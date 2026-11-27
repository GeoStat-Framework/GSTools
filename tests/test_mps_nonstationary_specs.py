"""Validation and simulation behavior for MPS rotation and scale specs."""

import numpy as np
import pytest

import gstools as gs
from gstools.mps.model import _validate_rotation, _validate_scale
from gstools.mps.nonstationary import resolve_spec
from gstools.tools.geometric import no_of_angles, set_angles


def _pos(grid_shape):
    """One flat zero coordinate array per dimension (callables ignore values)."""
    n = int(np.prod(grid_shape))
    return tuple(np.zeros(n) for _ in grid_shape)


class TestSpecResolver:
    def test_none_resolves_to_none_stationary(self):
        arr, stat = resolve_spec(None, (5, 5), 1, _pos((5, 5)), "rotation")
        assert arr is None and stat is True

    def test_scalar_rotation_broadcasts_stationary(self):
        g, nc = (5, 5), no_of_angles(2)  # 1
        arr, stat = resolve_spec(45.0, g, nc, _pos(g), "rotation", angles=True)
        assert arr.shape == g + (nc,) and stat is True
        assert np.all(arr == 45.0)

    def test_scalar_rotation_uses_set_angles_convention_3d(self):
        # A scalar rotation must set only the first plane (GSTools convention),
        # not fill all angles (cf. GeoStat-Framework/GSTools#416).
        g, nc = (2, 2, 2), no_of_angles(3)  # 3
        arr, stat = resolve_spec(0.5, g, nc, _pos(g), "rotation", angles=True)
        assert stat is True
        assert np.allclose(arr[0, 0, 0], set_angles(3, 0.5))  # [0.5, 0, 0]

    def test_scalar_scale_dilates_all_axes(self):
        # Contrast: a scalar scale fills every component (all axes dilate).
        g, nc = (2, 2, 2), 3
        arr, _ = resolve_spec(2.0, g, nc, _pos(g), "scale", positive=True)
        assert np.all(arr[0, 0, 0] == 2.0)

    def test_stationary_view_is_readonly_zero_stride(self):
        g, nc = (4, 4), 1
        arr, stat = resolve_spec(2.0, g, nc, _pos(g), "scale", positive=True)
        assert stat is True
        assert not arr.flags.writeable  # read-only broadcast
        assert 0 in arr.strides  # shares its compact backing vector
        with pytest.raises(ValueError):
            arr[0, 0, 0] = 1.0

    def test_component_vector_is_stationary(self):
        g, nc = (5, 5), 2  # scale on 2-D: dim components
        arr, stat = resolve_spec(
            [2.0, 3.0], g, nc, _pos(g), "scale", positive=True
        )
        assert arr.shape == g + (nc,) and stat is True
        assert np.all(arr[..., 0] == 2.0) and np.all(arr[..., 1] == 3.0)

    def test_grid_scalar_map_nonstationary(self):
        g, nc = (5, 5), 1
        m = np.arange(25.0).reshape(g)
        arr, stat = resolve_spec(m, g, nc, _pos(g), "rotation")
        assert arr.shape == g + (nc,) and stat is False
        assert np.array_equal(arr[..., 0], m)

    def test_grid_full_map_nonstationary(self):
        g, nc = (5, 5), 2
        m = (np.arange(50.0) + 1.0).reshape(g + (nc,))
        arr, stat = resolve_spec(m, g, nc, _pos(g), "scale", positive=True)
        assert arr.shape == g + (nc,) and stat is False
        assert np.array_equal(arr, m)

    def test_callable_one_value_per_node(self):
        g, nc, n = (4, 4), 1, 16
        arr, stat = resolve_spec(
            lambda *p: np.full(n, 30.0), g, nc, _pos(g), "rotation"
        )
        assert arr.shape == g + (nc,) and stat is False
        assert np.all(arr == 30.0)

    def test_callable_per_node_components(self):
        g, nc, n = (4, 4), 2, 16
        arr, stat = resolve_spec(
            lambda *p: np.tile([2.0, 3.0], (n, 1)),
            g,
            nc,
            _pos(g),
            "scale",
            positive=True,
        )
        assert arr.shape == g + (nc,) and stat is False

    def test_ambiguous_1d_grid_raises(self):
        with pytest.raises(ValueError, match="ambiguous"):
            resolve_spec(
                np.ones(3), (3,), 3, _pos((3,)), "scale", positive=True
            )

    def test_flattened_map_raises(self):
        with pytest.raises(ValueError, match="matches no accepted form"):
            resolve_spec(np.zeros(25), (5, 5), 1, _pos((5, 5)), "rotation")

    def test_callable_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="callable must return"):
            resolve_spec(
                lambda *p: np.zeros((3, 3)),
                (4, 4),
                1,
                _pos((4, 4)),
                "rotation",
            )

    def test_nonfinite_rotation_raises(self):
        with pytest.raises(ValueError, match="finite"):
            resolve_spec(np.inf, (5, 5), 1, _pos((5, 5)), "rotation")

    def test_scale_nonpositive_raises(self):
        with pytest.raises(ValueError, match="> 0"):
            resolve_spec(0.0, (5, 5), 2, _pos((5, 5)), "scale", positive=True)

    def test_scale_nonfinite_raises(self):
        with pytest.raises(ValueError, match="finite"):
            resolve_spec(
                np.inf, (5, 5), 2, _pos((5, 5)), "scale", positive=True
            )


class TestMPSModelNonstationarySpecs:
    @staticmethod
    def _ti(dim=2):
        return gs.TrainingImage(np.zeros((6,) * dim), categorical=True)

    def test_defaults_are_none(self):
        m = gs.MPSModel(self._ti())
        assert m.rotation is None and m.scale is None

    def test_store_rotation_and_scale(self):
        m = gs.MPSModel(self._ti(), rotation=45.0, scale=2.0)
        assert float(m.rotation) == 45.0 and float(m.scale) == 2.0

    def test_rotation_rejected_in_1d(self):
        # Direct validator check (1-D TIs are unusual); no_of_angles(1) == 0.
        with pytest.raises(ValueError, match="undefined|1-D"):
            _validate_rotation(45.0, 1)

    def test_scale_supported_in_1d(self):
        out = _validate_scale(2.0, 1)
        assert float(out) == 2.0

    def test_scale_nonpositive_raises(self):
        with pytest.raises(ValueError, match="> 0"):
            gs.MPSModel(self._ti(), scale=0.0)

    def test_scale_nonfinite_raises(self):
        with pytest.raises(ValueError, match="finite"):
            gs.MPSModel(self._ti(), scale=np.inf)

    def test_rotation_nonfinite_raises(self):
        with pytest.raises(ValueError, match="finite"):
            gs.MPSModel(self._ti(), rotation=np.nan)

    def test_nan_threshold_rejected(self):
        # cf. GeoStat-Framework/GSTools#416: NaN slipped past </> checks.
        with pytest.raises(ValueError, match="finite"):
            gs.MPSModel(self._ti(), threshold=np.nan)

    def test_callable_stored_unvalidated(self):
        def f(*pos):
            return np.zeros(1)

        m = gs.MPSModel(self._ti(), rotation=f, scale=f)
        assert m.rotation is f and m.scale is f

    def test_setters_revalidate(self):
        m = gs.MPSModel(self._ti())
        m.rotation = 30.0
        m.scale = [1.0, 2.0]
        assert float(m.rotation) == 30.0
        assert np.array_equal(m.scale, [1.0, 2.0])
        with pytest.raises(ValueError, match="> 0"):
            m.scale = -1.0


def _ti():
    rng = np.random.default_rng(0)
    return gs.mps.TrainingImage(rng.integers(0, 3, (24, 24)), n_neighbors=8)


def _run(**model_kw):
    ds = gs.mps.DirectSampling(
        gs.mps.MPSModel(_ti(), scan_fraction=0.3, **model_kw)
    )
    return ds([np.arange(10, dtype=float)] * 2, seed=4)


def test_scalar_vector_callable_rotation_agree():
    theta = np.pi / 5
    f_scalar = _run(rotation=theta)
    # 2-D: no_of_angles == 1, so the stationary component vector is [theta].
    f_vector = _run(rotation=[theta])
    f_callable = _run(rotation=lambda x, y: np.full_like(x, theta))
    np.testing.assert_array_equal(f_scalar, f_vector)
    np.testing.assert_array_equal(f_scalar, f_callable)


def test_rotation_actually_changes_output_vs_plain():
    # A model-level rotation must reach the simulation engine.
    f_plain = _run()
    f_rot = _run(rotation=np.pi / 4)
    assert not np.array_equal(f_plain, f_rot)


def test_callable_scale_changes_output():
    f_plain = _run()
    # anisotropic absolute scale via a callable (compress axis 1)
    f_scale = _run(
        scale=lambda x, y: np.stack(
            [np.ones_like(x), 0.5 * np.ones_like(y)], axis=-1
        )
    )
    assert not np.array_equal(f_plain, f_scale)


def test_identity_rotation_scale_is_bit_identical_to_plain():
    f_plain = _run()
    f_id = _run(rotation=0.0, scale=1.0)
    np.testing.assert_array_equal(f_plain, f_id)


def test_rotation_rejected_on_1d_training_image():
    ti = gs.mps.TrainingImage(np.zeros(12))
    with pytest.raises(ValueError, match="rotation"):
        gs.mps.MPSModel(ti, rotation=0.5)


def test_scale_rejected_nonpositive_at_construction():
    ti = gs.mps.TrainingImage(np.zeros((9, 9)))
    with pytest.raises(ValueError, match="scale"):
        gs.mps.MPSModel(ti, scale=[1.0, 0.0])


def test_3d_scalar_rotation_runs_and_matches_padded_vector():
    rng = np.random.default_rng(1)
    data = rng.integers(0, 2, (12, 12, 12))

    def run(rot):
        ds = gs.mps.DirectSampling(
            gs.mps.MPSModel(
                gs.mps.TrainingImage(data, n_neighbors=4),
                scan_fraction=0.1,
                rotation=rot,
            )
        )
        return ds([np.arange(5, dtype=float)] * 3, seed=0)

    # set_angles convention: a 3-D scalar pads with zeros -> [theta, 0, 0].
    f_scalar = run(0.5)
    f_padded = run([0.5, 0.0, 0.0])
    np.testing.assert_array_equal(f_scalar, f_padded)
