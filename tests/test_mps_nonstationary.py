"""Tests for gstools.mps.nonstationary."""

import unittest
import warnings

import numpy as np
from mps_fixtures import as_dict, flat_pos, ignoring

import gstools as gs
from gstools.mps.direct_sampling import DirectSampling
from gstools.mps.model import MPSModel
from gstools.mps.nonstationary import (
    build_zone_selector,
    resolve_spec,
    resolve_where,
)
from gstools.mps.training_image import (
    TrainingImage,
    Variable,
)


class TestNonstationarity(unittest.TestCase):
    """Geometric non-stationarity for DirectSampling (MPSModel rotation/scale)."""

    def _make_ds(self, ti_shape=(20, 20), **kw):
        rng = np.random.default_rng(0)
        data = rng.integers(0, 3, ti_shape)
        n_neighbors = kw.pop("n_neighbors", 4)
        ti = gs.mps.TrainingImage(data, n_neighbors=n_neighbors)
        defaults = dict(scan_fraction=0.2)
        defaults.update(kw)
        return gs.mps.DirectSampling(MPSModel(ti, **defaults)), ti

    def test_scalar_rotation_valid_values(self):
        rng = np.random.default_rng(0)
        ti = gs.mps.TrainingImage(rng.integers(0, 3, (20, 20)), n_neighbors=4)
        ds = gs.mps.DirectSampling(
            MPSModel(ti, scan_fraction=0.2, rotation=np.pi / 4)
        )
        with ignoring(RuntimeWarning):
            field = ds([np.arange(8, dtype=float)] * 2, seed=0)
        self.assertEqual(field.shape, (8, 8))
        self.assertTrue(np.all(np.isin(field, [0, 1, 2])))

    def test_rotation_changes_output(self):
        ds_plain, ti = self._make_ds(ti_shape=(30, 30), n_neighbors=8)
        pos = [np.arange(10, dtype=float)] * 2
        f_plain = ds_plain(pos, seed=7)

        ds_rot = gs.mps.DirectSampling(
            MPSModel(ti, scan_fraction=0.2, rotation=np.pi / 4)
        )
        with ignoring(RuntimeWarning):
            f_rot = ds_rot(pos, seed=7)
        self.assertFalse(np.array_equal(f_plain, f_rot))

    def test_array_rotation_map_runs(self):
        rng = np.random.default_rng(0)
        ti = gs.mps.TrainingImage(rng.integers(0, 3, (20, 20)), n_neighbors=4)
        angle_map = np.linspace(0, np.pi / 2, 64).reshape(8, 8)
        ds = gs.mps.DirectSampling(
            MPSModel(ti, scan_fraction=0.2, rotation=angle_map)
        )
        with ignoring(RuntimeWarning):
            field = ds([np.arange(8, dtype=float)] * 2, seed=1)
        self.assertEqual(field.shape, (8, 8))
        self.assertTrue(np.all(np.isin(field, [0, 1, 2])))

    def test_anis_changes_output(self):
        ds_plain, ti = self._make_ds(ti_shape=(30, 30), n_neighbors=8)
        pos = [np.arange(10, dtype=float)] * 2
        f_plain = ds_plain(pos, seed=3)

        ds_anis = gs.mps.DirectSampling(
            MPSModel(ti, scan_fraction=0.2, scale=[1.0, 0.5])
        )
        f_anis = ds_anis(pos, seed=3)
        self.assertFalse(np.array_equal(f_plain, f_anis))

    def test_combined_rotation_anis_runs(self):
        rng = np.random.default_rng(0)
        ti = gs.mps.TrainingImage(rng.integers(0, 3, (20, 20)), n_neighbors=4)
        ds = gs.mps.DirectSampling(
            MPSModel(
                ti, scan_fraction=0.2, rotation=np.pi / 6, scale=[1.0, 0.5]
            )
        )
        field = ds([np.arange(8, dtype=float)] * 2, seed=2)
        self.assertEqual(field.shape, (8, 8))
        self.assertTrue(np.all(np.isin(field, [0, 1, 2])))

    def test_conditioning_preserved_with_rotation(self):
        rng = np.random.default_rng(0)
        ti = gs.mps.TrainingImage(rng.integers(0, 3, (20, 20)), n_neighbors=4)
        ds = gs.mps.DirectSampling(
            MPSModel(ti, scan_fraction=0.3, rotation=np.pi / 4)
        )
        ds.set_condition([[4.0], [4.0]], [2])
        with ignoring(RuntimeWarning):
            field = ds([np.arange(8, dtype=float)] * 2, seed=0)
        self.assertEqual(int(field[4, 4]), 2)

    def test_3d_rotation_runs(self):
        rng = np.random.default_rng(0)
        data = rng.integers(0, 2, (12, 12, 12))
        ti = gs.mps.TrainingImage(data, n_neighbors=4)
        ds = gs.mps.DirectSampling(
            MPSModel(ti, scan_fraction=0.1, rotation=np.pi / 4)
        )
        with ignoring(RuntimeWarning):
            field = ds([np.arange(5, dtype=float)] * 3, seed=0)
        self.assertEqual(field.shape, (5, 5, 5))
        self.assertTrue(np.all(np.isin(field, [0, 1])))

    def test_collapsed_window_fallback(self):
        # 90° rotation on a tiny TI can force all windows to collapse.
        # Output must be finite and within TI values — no crash, no NaN.
        rng = np.random.default_rng(0)
        data = rng.integers(0, 2, (4, 4))
        ti = gs.mps.TrainingImage(data)
        ds = gs.mps.DirectSampling(
            MPSModel(ti, scan_fraction=1.0, rotation=np.pi / 2)
        )
        field = ds([np.arange(6, dtype=float)] * 2, seed=0)
        self.assertTrue(np.all(np.isfinite(field)))
        self.assertTrue(np.all(np.isin(field, [0, 1])))

    def test_identity_matches_no_transform(self):
        # θ=0, scale=1 must produce bit-identical output to the plain path.
        ds_plain, ti = self._make_ds(ti_shape=(20, 20), n_neighbors=8)
        pos = [np.arange(8, dtype=float)] * 2
        f_plain = ds_plain(pos, seed=5)

        ds_id = gs.mps.DirectSampling(
            MPSModel(ti, scan_fraction=0.2, rotation=0.0, scale=1.0)
        )
        f_id = ds_id(pos, seed=5)
        np.testing.assert_array_equal(f_plain, f_id)

    def test_multivariate_rotation_valid_output(self):
        # Exercises the multivariate transform path: shape, value subsets,
        # and that rotation changes output vs no rotation (same seed).
        rng = np.random.default_rng(0)
        ti = gs.mps.TrainingImage(
            [
                Variable(
                    "a",
                    rng.integers(0, 3, (20, 20)),
                    categorical=True,
                    n_neighbors=4,
                ),
                Variable(
                    "b",
                    rng.random((20, 20)),
                    categorical=False,
                    distance="l1",
                    n_neighbors=4,
                ),
            ]
        )
        pos = [np.arange(8, dtype=float)] * 2

        ds_plain = gs.mps.DirectSampling(MPSModel(ti, scan_fraction=0.2))
        result_plain = ds_plain(pos, seed=9)

        ds_rot = gs.mps.DirectSampling(
            MPSModel(ti, scan_fraction=0.2, rotation=np.pi / 4)
        )
        with ignoring(RuntimeWarning):
            result_rot = ds_rot(pos, seed=9)
            result_rot = as_dict(ds_rot, result_rot)

        self.assertEqual(result_rot["a"].shape, (8, 8))
        self.assertEqual(result_rot["b"].shape, (8, 8))
        self.assertTrue(np.all(np.isin(result_rot["a"], [0, 1, 2])))
        self.assertTrue(np.all(np.isfinite(result_rot["b"])))
        self.assertFalse(np.array_equal(result_plain[0], result_rot["a"]))

    def test_flattened_rotation_map_raises(self):
        # A per-node map passed flattened (length Nx*Ny) on a 2-D grid must
        # raise, not silently apply only element [0].
        rng = np.random.default_rng(0)
        ti = gs.mps.TrainingImage(rng.integers(0, 3, (20, 20)), n_neighbors=4)
        ds = gs.mps.DirectSampling(
            MPSModel(ti, scan_fraction=0.2, rotation=np.linspace(0, np.pi, 25))
        )
        with self.assertRaisesRegex(ValueError, "shape"):
            ds([np.arange(5, dtype=float)] * 2, seed=0)

    def test_3d_stationary_rotation_triple_runs(self):
        # A genuine stationary multi-component value (3 Tait-Bryan angles for a
        # 3-D grid) is still accepted (length == no_of_angles(3) == 3).
        rng = np.random.default_rng(0)
        ti = gs.mps.TrainingImage(
            rng.integers(0, 2, (12, 12, 12)), n_neighbors=4
        )
        ds = gs.mps.DirectSampling(
            MPSModel(ti, scan_fraction=0.1, rotation=np.array([0.1, 0.2, 0.3]))
        )
        field = ds([np.arange(5, dtype=float)] * 3, seed=0)
        self.assertEqual(field.shape, (5, 5, 5))
        self.assertTrue(np.all(np.isin(field, [0, 1])))

    def test_scalar_vs_materialized_map_bit_identical(self):
        # Scalar spec (hoisted-M stationary path) vs a hand-materialized
        # constant per-node map (per-node path) must agree exactly.
        rng = np.random.default_rng(0)
        ti = gs.mps.TrainingImage(rng.integers(0, 3, (20, 20)), n_neighbors=4)
        pos = [np.arange(8, dtype=float)] * 2
        f_scalar = gs.mps.DirectSampling(
            MPSModel(ti, scan_fraction=0.2, rotation=0.3)
        )(pos, seed=11)
        f_map = gs.mps.DirectSampling(
            MPSModel(ti, scan_fraction=0.2, rotation=np.full((8, 8), 0.3))
        )(pos, seed=11)
        np.testing.assert_array_equal(f_scalar, f_map)

    def test_callable_vs_materialized_bit_identical(self):
        # Callable form vs a hand-materialized array of the same field must agree exactly.
        rng = np.random.default_rng(0)
        ti = gs.mps.TrainingImage(rng.integers(0, 3, (20, 20)), n_neighbors=4)
        pos = [np.arange(8, dtype=float)] * 2

        def rot(x, y):
            return (y / 8.0) * (np.pi / 4.0)

        gx, gy = np.meshgrid(pos[0], pos[1], indexing="ij")
        rot_arr = (gy / 8.0) * (np.pi / 4.0)

        with ignoring(RuntimeWarning):
            f_call = gs.mps.DirectSampling(
                MPSModel(ti, scan_fraction=0.2, rotation=rot)
            )(pos, seed=3)
            f_arr = gs.mps.DirectSampling(
                MPSModel(ti, scan_fraction=0.2, rotation=rot_arr)
            )(pos, seed=3)
        np.testing.assert_array_equal(f_call, f_arr)

    def test_uniform_dilation_runs_and_differs(self):
        # Uniform scale=0.5 (pure dilation) runs, stays in the TI value set,
        # and differs from the identity run.
        rng = np.random.default_rng(0)
        ti = gs.mps.TrainingImage(rng.integers(0, 3, (30, 30)), n_neighbors=8)
        pos = [np.arange(10, dtype=float)] * 2
        f_id = gs.mps.DirectSampling(MPSModel(ti, scan_fraction=0.2))(
            pos, seed=7
        )
        f_half = gs.mps.DirectSampling(
            MPSModel(ti, scan_fraction=0.2, scale=0.5)
        )(pos, seed=7)
        self.assertTrue(np.all(np.isin(f_half, [0, 1, 2])))
        self.assertFalse(np.array_equal(f_id, f_half))

    def test_nonuniform_axis_spacing_warns(self):
        # Non-uniform axis spacing breaks the index-space geometry that lag
        # transforms assume; the warning needs no rotation or scale.
        rng = np.random.default_rng(0)
        ti = gs.mps.TrainingImage(rng.integers(0, 2, (15, 15)), n_neighbors=4)
        ds = gs.mps.DirectSampling(MPSModel(ti, scan_fraction=0.3))
        with self.assertWarns(UserWarning):
            ds([np.array([0.0, 1.0, 3.0, 7.0]), np.arange(4.0)], seed=0)

    def test_unequal_axis_spacing_warns(self):
        rng = np.random.default_rng(0)
        ti = gs.mps.TrainingImage(rng.integers(0, 2, (15, 15)), n_neighbors=4)
        ds = gs.mps.DirectSampling(MPSModel(ti, scan_fraction=0.3, scale=0.5))
        with self.assertWarns(UserWarning):
            ds([np.arange(4.0), 2.0 * np.arange(4.0)], seed=0)

    def test_uniform_spacing_no_warning(self):
        rng = np.random.default_rng(0)
        ti = gs.mps.TrainingImage(rng.integers(0, 2, (15, 15)), n_neighbors=4)
        ds = gs.mps.DirectSampling(
            MPSModel(ti, scan_fraction=0.3, rotation=np.pi / 4)
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # Lag collapse under the 45 degree rotation is a separate warning.
            warnings.simplefilter("ignore", RuntimeWarning)
            ds([np.arange(6.0)] * 2, seed=0)
        self.assertEqual(len(caught), 0)

    def test_set_nonstationary_removed(self):
        rng = np.random.default_rng(0)
        ti = gs.mps.TrainingImage(rng.integers(0, 2, (10, 10)))
        ds = gs.mps.DirectSampling(MPSModel(ti))
        self.assertFalse(hasattr(ds, "set_nonstationary"))


class TestSpecResolver(unittest.TestCase):
    """Exact-shape spec resolution for rotation/scale (stationary vs per-node)."""

    def test_none_passthrough(self):
        arr, stat = resolve_spec(None, (4, 4), 1, flat_pos((4, 4)), "rotation")
        self.assertIsNone(arr)
        self.assertTrue(stat)

    def test_scalar_is_stationary_broadcast_view(self):
        arr, stat = resolve_spec(0.7, (4, 5), 1, flat_pos((4, 5)), "rotation")
        self.assertTrue(stat)
        self.assertEqual(arr.shape, (4, 5, 1))
        self.assertTrue(np.all(arr == 0.7))
        # zero-copy: broadcast view, not materialized
        self.assertEqual(arr.strides[:2], (0, 0))
        self.assertFalse(arr.flags.writeable)

    def test_stationary_component_vector(self):
        # 3 Tait-Bryan angles on a 3-D grid (n_comp = no_of_angles(3) = 3)
        arr, stat = resolve_spec(
            np.array([0.1, 0.2, 0.3]),
            (4, 4, 4),
            3,
            flat_pos((4, 4, 4)),
            "rotation",
        )
        self.assertTrue(stat)
        self.assertEqual(arr.shape, (4, 4, 4, 3))
        np.testing.assert_array_equal(arr[2, 1, 3], [0.1, 0.2, 0.3])

    def test_grid_shape_form_4a(self):
        vals = np.linspace(0, 1, 20).reshape(4, 5)
        arr, stat = resolve_spec(
            vals, (4, 5), 2, flat_pos((4, 5)), "scale", positive=False
        )
        self.assertFalse(stat)
        self.assertEqual(arr.shape, (4, 5, 2))
        self.assertEqual(arr[3, 4, 0], arr[3, 4, 1])  # broadcast across comps

    def test_grid_plus_comp_form_4b(self):
        vals = np.random.default_rng(0).random((4, 5, 2)) + 0.1
        arr, stat = resolve_spec(vals, (4, 5), 2, flat_pos((4, 5)), "scale")
        self.assertFalse(stat)
        np.testing.assert_array_equal(arr, vals)

    def test_collision_n_nodes_equals_n_comp_raises(self):
        # 1-D grid of length n_comp — both readings match; must be a loud
        # ValueError naming both interpretations.
        with self.assertRaisesRegex(
            ValueError, "stationary.*per-node|per-node.*stationary"
        ):
            resolve_spec(
                np.array([0.1, 0.2, 0.3]), (3,), 3, flat_pos((3,)), "rotation"
            )

    def test_per_node_map_on_1d_grid_is_form_4a(self):
        # Valid per-node maps on 1-D grids must be expressible.
        vals = np.linspace(0.5, 2.0, 7)
        arr, stat = resolve_spec(vals, (7,), 1, flat_pos((7,)), "scale")
        self.assertFalse(stat)
        self.assertEqual(arr.shape, (7, 1))
        np.testing.assert_array_equal(arr[:, 0], vals)

    def test_callable_form(self):
        def f(x, y):
            return x * 0.1 + y

        pos = flat_pos((3, 4))
        arr, stat = resolve_spec(f, (3, 4), 1, pos, "rotation")
        self.assertFalse(stat)
        self.assertEqual(arr.shape, (3, 4, 1))
        self.assertAlmostEqual(float(arr[2, 3, 0]), 2 * 0.1 + 3)

    def test_callable_with_component_axis(self):
        def f(x, y):
            return np.stack([np.ones_like(x), 0.5 * np.ones_like(x)], axis=-1)

        arr, stat = resolve_spec(f, (3, 4), 2, flat_pos((3, 4)), "scale")
        self.assertEqual(arr.shape, (3, 4, 2))
        np.testing.assert_array_equal(arr[1, 1], [1.0, 0.5])

    def test_callable_wrong_shape_raises(self):
        with self.assertRaisesRegex(ValueError, "callable"):
            resolve_spec(
                lambda x, y: np.zeros((5,)),
                (3, 4),
                1,
                flat_pos((3, 4)),
                "rotation",
            )

    def test_positive_validation(self):
        with self.assertRaisesRegex(ValueError, "positive|> 0"):
            resolve_spec(
                -1.0, (4, 4), 2, flat_pos((4, 4)), "scale", positive=True
            )
        with self.assertRaisesRegex(ValueError, "positive|> 0"):
            resolve_spec(
                np.zeros((4, 4)),
                (4, 4),
                2,
                flat_pos((4, 4)),
                "scale",
                positive=True,
            )


class TestWhereAndSelector(unittest.TestCase):
    def test_resolve_where_array(self):
        mask = np.zeros((4, 4), dtype=bool)
        mask[2:] = True
        out = resolve_where(mask, (4, 4), flat_pos((4, 4)))
        np.testing.assert_array_equal(out, mask)

    def test_resolve_where_callable(self):
        out = resolve_where(lambda x, y: x >= 2, (4, 4), flat_pos((4, 4)))
        self.assertEqual(out.shape, (4, 4))
        self.assertTrue(out[3, 0])
        self.assertFalse(out[1, 0])

    def test_resolve_where_callable_non_bool_raises(self):
        with self.assertRaisesRegex(ValueError, "bool"):
            resolve_where(lambda x, y: x + y, (4, 4), flat_pos((4, 4)))

    def test_resolve_where_wrong_shape_raises(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            resolve_where(
                np.zeros((3, 3), dtype=bool), (4, 4), flat_pos((4, 4))
            )

    def test_selector_values(self):
        ti = gs.mps.TrainingImage(np.zeros((5, 5)))
        m1 = np.zeros((4, 4), dtype=bool)
        m1[:2] = True
        m2 = np.zeros((4, 4), dtype=bool)
        m2[3:] = True
        sel = build_zone_selector(
            [gs.mps.Zone(ti, where=m1), gs.mps.Zone(ti, where=m2)],
            (4, 4),
            flat_pos((4, 4)),
        )
        self.assertEqual(sel[0, 0], 1)
        self.assertEqual(sel[3, 0], 2)
        self.assertEqual(sel[2, 0], 0)  # uncovered -> primary

    def test_overlapping_zones_raise(self):
        # Overlapping zones must raise, not silently resolve via precedence order.
        ti = gs.mps.TrainingImage(np.zeros((5, 5)))
        m1 = np.zeros((4, 4), dtype=bool)
        m1[:2] = True
        m2 = np.zeros((4, 4), dtype=bool)
        m2[1:] = True  # overlaps row 1
        with self.assertRaisesRegex(ValueError, "overlap"):
            build_zone_selector(
                [gs.mps.Zone(ti, where=m1), gs.mps.Zone(ti, where=m2)],
                (4, 4),
                flat_pos((4, 4)),
            )


class TestRotationSpecs(unittest.TestCase):
    """Scalar and non-finite rotation specifications."""

    def setUp(self):
        rng = np.random.default_rng(416)
        self.codes = rng.integers(0, 2, (12, 12)).astype(float)  # {0, 1}
        self.cont = rng.random((12, 12))
        self.pos = [np.arange(8.0)] * 2

    def test_one_value_rotation_fills_first_angle_only(self):
        # One value per node is the x-y angle with the others zero (as in
        # gs.tools.set_angles); a scale broadcasts to every axis.
        grid, pos = (2, 2, 2), flat_pos((2, 2, 2))
        specs = {
            "scalar": 0.5,
            "grid map": np.full(grid, 0.5),
            "callable": lambda x, y, z: np.full(x.size, 0.5),
        }
        for name, spec in specs.items():
            with self.subTest(name):
                arr, _ = resolve_spec(spec, grid, 3, pos, "rotation", pad=True)
                np.testing.assert_array_equal(arr[0, 0, 0], [0.5, 0.0, 0.0])
        arr, _ = resolve_spec(2.0, grid, 3, pos, "scale", positive=True)
        np.testing.assert_array_equal(arr[0, 0, 0], [2.0, 2.0, 2.0])

    def test_scalar_rotation_keeps_horizontal_layers(self):
        # A scalar rotation turns the x-y plane only, so a TI whose value
        # depends on z alone must give a constant value in every z-slice.
        layers = np.zeros((12, 12, 8))
        layers += np.arange(8) % 3
        model = MPSModel(TrainingImage(layers, n_neighbors=12), rotation=0.5)
        pos = [np.arange(8.0), np.arange(8.0), np.arange(6.0)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # lag collisions
            field = DirectSampling(model, seed=0)(pos, store=False)
        purity = np.mean(
            [
                np.bincount(s.astype(int).ravel()).max() / s.size
                for s in np.moveaxis(field, 2, 0)
            ]
        )
        self.assertEqual(purity, 1.0)

    def test_non_finite_rotation_callable_rejected_at_call(self):
        # A callable is only evaluated on the simulation grid.
        ti = TrainingImage(self.cont, categorical=False)
        model = MPSModel(ti, rotation=lambda x, y: np.full(x.shape, np.nan))
        with self.assertRaises(ValueError):
            DirectSampling(model, seed=1)(self.pos, store=False)


class TestIdentityTransform(unittest.TestCase):
    """An identity rotation/scale gives the same field as no transform."""

    def test_identity_transform_equals_no_transform(self):
        ti = gs.mps.TrainingImage(
            np.array([[0, 1], [1, 0], [0, 1], [0, 1], [0, 1]]), n_neighbors=3
        )
        base = np.argwhere(np.ones((5, 5), bool))
        path = np.vstack([[2, 2], base[np.any(base != [2, 2], axis=1)]])
        pos = (np.arange(5), np.arange(5))

        def run(**kw):
            ds = gs.mps.DirectSampling(gs.mps.MPSModel(ti, **kw))
            ds.set_condition([[1, 1, 4], [1, 3, 2]], [1, 0, 1])
            with ignoring(RuntimeWarning):  # the tiny TI forces the fallback
                return ds(
                    pos, path=path, node_seed=0, num_threads=1, store=False
                )

        ref = run()
        for kw in (
            dict(scale=1.0),
            dict(scale=[1.0, 1.0]),
            dict(rotation=0.0),
        ):
            with self.subTest(**kw):
                np.testing.assert_array_equal(run(**kw), ref)

    def test_zero_rotation_with_real_scale_equals_scale_alone(self):
        ti = gs.mps.TrainingImage(
            np.array([[0, 1], [1, 0], [0, 1], [0, 1], [0, 1]]), n_neighbors=3
        )
        pos = (np.arange(5), np.arange(5))

        def run(**kw):
            ds = gs.mps.DirectSampling(gs.mps.MPSModel(ti, **kw))
            with ignoring(RuntimeWarning):  # the tiny TI forces the fallback
                return ds(
                    pos, node_seed=0, path_seed=0, num_threads=1, store=False
                )

        np.testing.assert_array_equal(
            run(rotation=0.0, scale=0.5), run(scale=0.5)
        )


if __name__ == "__main__":
    unittest.main()
