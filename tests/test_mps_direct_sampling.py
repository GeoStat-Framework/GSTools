"""Tests for gstools.mps.direct_sampling."""

import unittest
import warnings
from unittest import mock

import numpy as np
from mps_fixtures import as_dict, ignoring

import gstools as gs
from gstools import config as gs_config
from gstools.mps.direct_sampling import DirectSampling
from gstools.mps.distance import variable_distance
from gstools.mps.model import MPSModel
from gstools.mps.post_processing import PostProcessing
from gstools.mps.training_image import (
    TrainingImage,
    Variable,
)
from gstools.random.rng import RNG


class TestParallelExecution(unittest.TestCase):
    def test_valid_values(self):
        rng = np.random.default_rng(0)
        data = rng.integers(0, 3, (20, 20))
        ti = TrainingImage(data, n_neighbors=4)
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.2))
        field = ds([np.arange(8, dtype=float)] * 2, seed=0, num_threads=2)
        self.assertEqual(field.shape, (8, 8))
        self.assertTrue(np.all(np.isin(field, [0, 1, 2])))

    def test_reproducible(self):
        # DAG parallelism is deterministic: same seed → same parallel result
        rng = np.random.default_rng(0)
        data = rng.integers(0, 3, (20, 20))
        ti = TrainingImage(data, n_neighbors=4)
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.2))
        pos = [np.arange(8, dtype=float)] * 2
        self.assertTrue(
            np.array_equal(
                ds(pos, seed=7, num_threads=2), ds(pos, seed=7, num_threads=2)
            )
        )

    def test_conditioning_preserved(self):
        rng = np.random.default_rng(0)
        data = rng.integers(0, 3, (20, 20))
        ti = TrainingImage(data, n_neighbors=4)
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.2))
        ds.set_condition([[5.0], [5.0]], [2])
        field = ds([np.arange(10, dtype=float)] * 2, seed=0, num_threads=2)
        self.assertEqual(field[5, 5], 2)

    def test_global_config(self):
        # num_threads=None reads gs_config.NUM_THREADS
        rng = np.random.default_rng(0)
        data = rng.integers(0, 3, (20, 20))
        ti = TrainingImage(data, n_neighbors=4)
        pos = [np.arange(8, dtype=float)] * 2
        old = gs_config.NUM_THREADS
        try:
            gs_config.NUM_THREADS = 2
            field = DirectSampling(MPSModel(ti, scan_fraction=0.2))(
                pos, seed=7
            )
        finally:
            gs_config.NUM_THREADS = old
        self.assertEqual(field.shape, (8, 8))
        self.assertTrue(np.all(np.isin(field, [0, 1, 2])))

    def test_large_batches(self):
        # n_neighbors=2 → sparse DAG → large ready batches
        rng = np.random.default_rng(1)
        data = rng.integers(0, 2, (30, 30))
        ti = TrainingImage(data, n_neighbors=2)
        pos = [np.arange(12, dtype=float)] * 2
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.3))
        field = ds(pos, seed=42, num_threads=4)
        self.assertEqual(field.shape, (12, 12))
        self.assertTrue(np.all(np.isin(field, [0.0, 1.0])))

    def test_stress(self):
        # large grid, sparse DAG, conditioning, varying thread counts
        rng = np.random.default_rng(3)
        data = rng.integers(0, 4, (40, 40))
        ti = TrainingImage(data, n_neighbors=4)
        pos = [np.arange(25, dtype=float)] * 2
        cond_pos = [
            rng.integers(0, 25, 20).astype(float),
            rng.integers(0, 25, 20).astype(float),
        ]
        cond_val = rng.integers(0, 4, 20).astype(float)
        for nt in (2, 4, 8):
            ds = DirectSampling(MPSModel(ti, scan_fraction=0.4))
            with ignoring(UserWarning):  # random positions may collide
                ds.set_condition(cond_pos, cond_val)
                field = ds(pos, seed=11, num_threads=nt)
            self.assertEqual(field.shape, (25, 25))
            self.assertTrue(np.all(np.isin(field, [0, 1, 2, 3])))

    def test_all_nan_ti_rejected(self):
        # No fully-defined TI cell exists, so no value can be pasted.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ti = TrainingImage(
                np.full((5, 5), np.nan),
                categorical=False,
                distance="l2",
                n_neighbors=2,
            )
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.2))
        with self.assertRaises(ValueError):
            ds([np.arange(3, dtype=float)] * 2, seed=42, num_threads=1)


class TestDirectSampling(unittest.TestCase):
    def setUp(self):
        # 1-D categorical TI: alternating 0/1, length 20
        arr1d = np.tile([0, 1], 10).astype(float)
        self.ti1d = TrainingImage(arr1d, categorical=True, n_neighbors=8)

        # 2-D categorical TI: 8×8 checkerboard
        self.ti2d = TrainingImage(
            (np.indices((8, 8)).sum(axis=0) % 2).astype(float),
            categorical=True,
            n_neighbors=8,
        )

        rng = np.random.default_rng(0)
        self.ti2d_rand = TrainingImage(
            rng.integers(0, 2, size=(20, 20)).astype(float),
            categorical=True,
            n_neighbors=8,
        )

        # 1-D continuous TI
        self.ti1d_cont = TrainingImage(
            np.linspace(0.0, 1.0, 20),
            categorical=False,
            distance="l1",
            n_neighbors=8,
        )

        self.x1d = np.arange(10, dtype=float)
        self.x2d = np.arange(6, dtype=float)
        self.y2d = np.arange(6, dtype=float)

    def test_model_is_retained(self):
        # The MPSModel passed in must not be silently discarded.
        model = MPSModel(self.ti1d)
        ds = DirectSampling(model)
        self.assertIs(ds.mps_model, model)
        # Field's covariance-model slot must stay None (DirectSampling is not
        # a CovModel-based field; pre_pos/field_dim rely on this).
        self.assertIsNone(ds.model)

    def test_raise(self):
        ds = DirectSampling(MPSModel(self.ti1d))
        with self.assertRaises(ValueError):
            ds([self.x1d], seed=42, mesh_type="unstructured")

    def test_repr(self):
        ds = DirectSampling(MPSModel(self.ti1d))
        r = repr(ds)
        self.assertIsInstance(r, str)
        self.assertIn("DirectSampling", r)

    def test_properties(self):
        ti = TrainingImage(self.ti1d.variable().replace(cond_weight=2.0))
        ds = DirectSampling(
            MPSModel(
                ti,
                scan_fraction=0.5,
                threshold=0.05,
            )
        )
        self.assertIs(ds.ti, ti)
        self.assertEqual(ds.ti.variable().n_neighbors, 8)
        self.assertAlmostEqual(ds.ti.variable().cond_weight, 2.0)
        self.assertIsNone(ds.ti.variable().max_radius)

    def test_shape_1d(self):
        ds = DirectSampling(MPSModel(self.ti1d, scan_fraction=1.0))
        field = ds([self.x1d], seed=42)
        self.assertEqual(field.shape, (10,))
        self.assertFalse(np.any(np.isnan(field)))

    def test_shape_2d(self):
        ds = DirectSampling(MPSModel(self.ti2d, scan_fraction=1.0))
        field = ds([self.x2d, self.y2d], seed=42)
        self.assertEqual(field.shape, (6, 6))
        self.assertFalse(np.any(np.isnan(field)))
        # All output values must be in the TI value set {0, 1}
        unique_vals = set(np.unique(field))
        self.assertTrue(unique_vals.issubset({0.0, 1.0}))

    def test_seeded_1d_values_pinned(self):
        ds = DirectSampling(MPSModel(self.ti1d, scan_fraction=1.0))
        field = ds([self.x1d], seed=42)
        self.assertAlmostEqual(field[0], 0.0)
        self.assertAlmostEqual(field[5], 1.0)
        self.assertAlmostEqual(field[9], 1.0)

    def test_seeded_2d_values_pinned(self):
        ds = DirectSampling(MPSModel(self.ti2d, scan_fraction=1.0))
        field = ds([self.x2d, self.y2d], seed=42)
        self.assertAlmostEqual(field[0, 0], 1.0)
        self.assertAlmostEqual(field[2, 3], 0.0)
        self.assertAlmostEqual(field[5, 5], 1.0)

    def test_seeded_reproducibility(self):
        ds = DirectSampling(MPSModel(self.ti2d_rand, scan_fraction=0.5))
        pos = [self.x2d, self.y2d]
        fa = ds(pos, seed=99)
        fb = ds(pos, seed=99)
        fc = ds(pos, seed=100)
        # Same seed → identical output
        self.assertTrue(np.allclose(fa, fb))
        # Different seed → different output
        self.assertFalse(np.allclose(fa, fc))
        # Pin two values for seed=99; stable across NumPy versions because DS
        # uses RandomState (MT19937) throughout, matching the rest of GSTools.
        self.assertAlmostEqual(fa[0, 0], 0.0)
        self.assertAlmostEqual(fa[3, 4], 1.0)

    def test_conditioning_honored(self):
        ds = DirectSampling(MPSModel(self.ti1d, scan_fraction=1.0))
        # Three exact grid node positions exercise multi-point handling
        cond_pos = [np.array([2.0, 4.0, 7.0])]
        cond_val = np.array([0.0, 1.0, 1.0])
        ds.set_condition(cond_pos, cond_val)
        field = ds([self.x1d], seed=5)
        self.assertAlmostEqual(field[2], 0.0)
        self.assertAlmostEqual(field[4], 1.0)
        self.assertAlmostEqual(field[7], 1.0)

    def test_tiny_ti_collapse_recovers(self):
        # TI far smaller than lag span → neighbours are dropped, not raise
        ti_tiny = TrainingImage(
            np.random.default_rng(0).random((3, 3)),
            categorical=False,
            distance="l1",
        )
        ds = DirectSampling(
            MPSModel(
                ti_tiny,
                scan_fraction=1.0,
            )
        )
        with self.assertWarnsRegex(RuntimeWarning, "dropping the furthest"):
            field = ds([np.arange(30, dtype=float)] * 2, seed=1)
        self.assertEqual(field.shape, (30, 30))
        self.assertFalse(np.any(np.isnan(field)))

    def test_scan_fraction_window_semantics(self):
        """scan_fraction=0.1 applies to the window, not the TI — no crash, valid output."""
        rng = np.random.default_rng(0)
        ti = TrainingImage(
            rng.integers(0, 2, (20, 20)).astype(float), categorical=True
        )
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.1))
        field = ds([np.arange(6, dtype=float)] * 2, seed=0)
        self.assertEqual(field.shape, (6, 6))
        self.assertFalse(np.any(np.isnan(field)))
        self.assertTrue(set(np.unique(field)).issubset({0.0, 1.0}))

    def test_continuous_ti(self):
        ds = DirectSampling(
            MPSModel(
                self.ti1d_cont,
                scan_fraction=1.0,
                threshold=0.05,
            )
        )
        field = ds([np.arange(8, dtype=float)], seed=42)
        self.assertEqual(field.shape, (8,))
        self.assertFalse(np.any(np.isnan(field)))
        self.assertTrue(np.all(field >= 0.0))
        self.assertTrue(np.all(field <= 1.0))

    def test_empty_search_window_recovery(self):
        # n_neighbors >> TI size collapses search windows: the furthest
        # neighbours are dropped (with a warning) and the run still completes.
        ti_tiny = TrainingImage(np.array([0.0, 1.0, 0.0]), categorical=True)
        ds = DirectSampling(MPSModel(ti_tiny, scan_fraction=1.0))
        with self.assertWarnsRegex(RuntimeWarning, "dropping the furthest"):
            field = ds([np.arange(5, dtype=float)], seed=1)
        self.assertEqual(field.shape, (5,))
        self.assertFalse(np.any(np.isnan(field)))
        self.assertTrue(set(np.unique(field)).issubset({0.0, 1.0}))

    def test_gstools_namespace(self):
        self.assertFalse(hasattr(gs, "TrainingImage"))
        self.assertFalse(hasattr(gs, "Variable"))
        self.assertIs(gs.mps.DirectSampling, DirectSampling)
        self.assertIs(gs.mps.TrainingImage, TrainingImage)
        self.assertIs(gs.mps.Variable, Variable)


class TestMultivariateDirectSampling(unittest.TestCase):
    def test_fills_all_nodes(self):
        # Node-wise path must leave no NaN in any variable.
        rng = np.random.default_rng(42)
        ti = TrainingImage(
            [
                Variable("a", rng.integers(0, 2, (20, 20))),
                Variable("b", rng.integers(0, 2, (20, 20))),
            ]
        )
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.3))
        field = ds([np.arange(6, dtype=float)] * 2, seed=1)
        field = as_dict(ds, field)
        self.assertFalse(np.any(np.isnan(field["a"])))
        self.assertFalse(np.any(np.isnan(field["b"])))

    def test_output_shapes(self):
        rng = np.random.default_rng(0)
        ti = TrainingImage(
            [
                Variable("primary", rng.integers(0, 3, (20, 20))),
                Variable("secondary", rng.integers(0, 2, (20, 20))),
            ]
        )
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.1))
        field = ds([np.arange(10, dtype=float)] * 2, seed=0)
        field = as_dict(ds, field)
        self.assertEqual(field["primary"].shape, (10, 10))
        self.assertIn("secondary", field)
        self.assertEqual(field["secondary"].shape, (10, 10))

    def test_values_valid(self):
        rng = np.random.default_rng(1)
        ti = TrainingImage(
            [
                Variable("a", rng.integers(0, 4, (20, 20))),
                Variable("b", rng.integers(0, 2, (20, 20))),
            ]
        )
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.2))
        field = ds([np.arange(8, dtype=float)] * 2, seed=5)
        field = as_dict(ds, field)
        self.assertTrue(np.all(np.isin(field["a"], [0, 1, 2, 3])))
        self.assertTrue(np.all(np.isin(field["b"], [0, 1])))

    def test_per_variable_n_neighbors(self):
        rng = np.random.default_rng(3)
        ti = TrainingImage(
            [
                Variable(
                    "primary", rng.integers(0, 3, (20, 20)), n_neighbors=8
                ),
                Variable(
                    "secondary", rng.integers(0, 2, (20, 20)), n_neighbors=2
                ),
            ]
        )
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.2))
        field = ds([np.arange(8, dtype=float)] * 2, seed=7)
        field = as_dict(ds, field)
        self.assertEqual(field["primary"].shape, (8, 8))
        self.assertFalse(np.any(np.isnan(field["primary"])))
        self.assertFalse(np.any(np.isnan(field["secondary"])))

    def test_3d_runs(self):
        rng = np.random.default_rng(8)
        ti = TrainingImage(
            [
                Variable("a", rng.integers(0, 2, (10, 10, 10))),
                Variable("b", rng.integers(0, 2, (10, 10, 10))),
            ]
        )
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.2))
        field = ds([np.arange(4, dtype=float)] * 3, seed=0)
        field = as_dict(ds, field)
        self.assertEqual(field["a"].shape, (4, 4, 4))
        self.assertFalse(np.any(np.isnan(field["a"])))
        self.assertFalse(np.any(np.isnan(field["b"])))

    def test_ds_mode_threshold_stops_scan_early(self):
        rng = np.random.default_rng(12)
        ti = TrainingImage(
            [
                Variable("a", rng.integers(0, 3, (20, 20))),
                Variable("b", rng.integers(0, 2, (20, 20))),
            ]
        )
        pos = [np.arange(8, dtype=float)] * 2
        scanned = {}
        for name, threshold in (("ds", 0.3), ("dsbc", None)):
            ds = DirectSampling(
                MPSModel(ti, scan_fraction=0.5, threshold=threshold)
            )
            # Small scan blocks make the early exit visible in the row count.
            with mock.patch("gstools.mps.scan._SCAN_BLOCK", 8), mock.patch(
                "gstools.mps.scan.variable_distance",
                wraps=variable_distance,
            ) as spy:
                field = ds(pos, seed=4)
                field = as_dict(ds, field)
            scanned[name] = sum(c.args[2].shape[0] for c in spy.call_args_list)
            self.assertFalse(np.any(np.isnan(field["a"])))
            self.assertTrue(np.all(np.isin(field["a"], [0, 1, 2])))
            self.assertTrue(np.all(np.isin(field["b"], [0, 1])))
        self.assertGreater(scanned["dsbc"], 0)
        self.assertLess(scanned["ds"], scanned["dsbc"])

    def test_continuous_variation_variable(self):
        # A continuous variable with variation distance exercises the
        # mean-shift adjust_value_var path end to end.
        rng = np.random.default_rng(13)
        ti = TrainingImage(
            [
                Variable(
                    "cont",
                    rng.random((20, 20)) * 10.0,
                    categorical=False,
                    distance="variation",
                ),
                Variable(
                    "cat", rng.integers(0, 2, (20, 20)), categorical=True
                ),
            ]
        )
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.3))
        field = ds([np.arange(8, dtype=float)] * 2, seed=6)
        field = as_dict(ds, field)
        self.assertEqual(field["cont"].shape, (8, 8))
        self.assertTrue(np.all(np.isfinite(field["cont"])))
        self.assertTrue(np.all(np.isin(field["cat"], [0, 1])))

    def test_joint_cell_invariant(self):
        # Core acceptance test for node-wise co-simulation: every node's vector
        # must be copied from a SINGLE TI cell.  Build a TI where ``b`` is an
        # injective function of ``a`` (b = a + 100), so a consistent (a, b) pair
        # exists at exactly one TI cell.  If both variables at every node trace
        # to one cell, ``b == a + 100`` must hold everywhere.
        ids = np.arange(64).reshape(8, 8)
        ti = TrainingImage(
            [
                Variable("a", ids, categorical=True),
                Variable("b", ids + 100, categorical=True),
            ]
        )
        ds = DirectSampling(MPSModel(ti, scan_fraction=1.0, threshold=None))
        field = ds([np.arange(6, dtype=float)] * 2, seed=0)
        field = as_dict(ds, field)
        np.testing.assert_array_equal(field["b"], field["a"] + 100)

    def test_equal_treatment_named_fields(self):
        # No privileged primary: all variables are first-class named fields.
        rng = np.random.default_rng(20)
        ti = TrainingImage(
            [
                Variable("x", rng.integers(0, 2, (15, 15))),
                Variable("y", rng.integers(0, 2, (15, 15))),
            ]
        )
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.3))
        field_x, field_y = ds([np.arange(6, dtype=float)] * 2, seed=0)
        self.assertIn("x", ds.field_names)
        self.assertIn("y", ds.field_names)
        np.testing.assert_array_equal(ds["x"], field_x)
        np.testing.assert_array_equal(ds["y"], field_y)

    def test_invalid_variable_name_raises(self):
        with self.assertRaisesRegex(ValueError, "identifier"):
            Variable("a b", np.zeros((5, 5), dtype=int))

    def test_n_neighbors_getter_scalar(self):
        arr = np.zeros((10, 10), dtype=int)
        ti = TrainingImage(
            [
                Variable("a", arr, n_neighbors=8),
                Variable("b", arr, n_neighbors=8),
            ]
        )
        ds = DirectSampling(MPSModel(ti))
        self.assertEqual([ds.ti.variable(n).n_neighbors for n in "ab"], [8, 8])

    def test_n_neighbors_getter_dict(self):
        arr = np.zeros((10, 10), dtype=int)
        ti = TrainingImage(
            [
                Variable("a", arr, n_neighbors=8),
                Variable("b", arr, n_neighbors=4),
            ]
        )
        ds = DirectSampling(MPSModel(ti))
        self.assertEqual(
            {v.name: v.n_neighbors for v in ds.ti.variables}, {"a": 8, "b": 4}
        )

    def test_max_radius_setter_univariate(self):
        # Configure max_radius at construction; getter reflects the value.
        ti = TrainingImage(
            np.zeros((10, 10), dtype=int), n_neighbors=4, max_radius=5.0
        )
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.2))
        self.assertAlmostEqual(ds.ti.variable().max_radius, 5.0)

    def test_max_radius_getter_dict_when_different(self):
        arr = np.zeros((10, 10), dtype=int)
        ti = TrainingImage(
            [
                Variable("a", arr, max_radius=20.0),
                Variable("b", arr, max_radius=10.0),
            ]
        )
        ds = DirectSampling(MPSModel(ti))
        self.assertEqual(
            {v.name: v.max_radius for v in ds.ti.variables},
            {"a": 20.0, "b": 10.0},
        )

    def test_set_condition_basic(self):
        rng = np.random.default_rng(0)
        ti = TrainingImage(
            [
                Variable("a", rng.integers(0, 2, (20, 20))),
                Variable("b", rng.integers(0, 2, (20, 20))),
            ]
        )
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.3))
        ds.set_condition(
            cond_pos=[[2.0, 5.0], [2.0, 5.0]],
            cond_val={"a": np.array([1, 0]), "b": np.array([0, 1])},
        )
        field = ds([np.arange(8, dtype=float)] * 2, seed=0)
        field = as_dict(ds, field)
        self.assertEqual(field["a"][2, 2], 1)
        self.assertEqual(field["b"][2, 2], 0)
        self.assertEqual(field["a"][5, 5], 0)
        self.assertEqual(field["b"][5, 5], 1)

    def test_set_condition_partial_nan(self):
        # NaN for b at the conditioning point -> b unconstrained there; only a
        # is conditioned, and b is filled by the simulation (not NaN).
        rng = np.random.default_rng(1)
        ti = TrainingImage(
            [
                Variable("a", rng.integers(0, 2, (20, 20))),
                Variable("b", rng.integers(0, 2, (20, 20))),
            ]
        )
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.3))
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # partial NaN drops nothing
            ds.set_condition(
                cond_pos=[[3.0], [3.0]],
                cond_val={"a": np.array([1]), "b": np.array([np.nan])},
            )
        field = ds([np.arange(8, dtype=float)] * 2, seed=2)
        field = as_dict(ds, field)
        self.assertEqual(field["a"][3, 3], 1)
        self.assertFalse(np.isnan(field["b"][3, 3]))

    def test_set_condition_collision(self):
        # Both points snap to node (4, 4); (4.1, 4.1) is closer than (4.4, 4.4)
        # -> the closer point's values win for all variables.
        rng = np.random.default_rng(2)
        ti = TrainingImage(
            [
                Variable("a", rng.integers(0, 2, (20, 20))),
                Variable("b", rng.integers(0, 2, (20, 20))),
            ]
        )
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.3))
        with self.assertWarnsRegex(UserWarning, "collided"):
            ds.set_condition(
                cond_pos=[[4.4, 4.1], [4.4, 4.1]],
                cond_val={"a": np.array([0, 1]), "b": np.array([0, 1])},
            )
            field = ds([np.arange(8, dtype=float)] * 2, seed=3)
            field = as_dict(ds, field)
        self.assertEqual(field["a"][4, 4], 1)
        self.assertEqual(field["b"][4, 4], 1)

    def test_collocated_constraint(self):
        # b is a unique index per TI cell; a is a deterministic function of b.
        # Conditioning b at the single simulated node forces the collocated
        # (h=0) term to select the TI cell whose b matches, so the co-simulated
        # a must equal that cell's a.  DSBC full scan => exact global argmin.
        # 1x1 sim grid: exactly one node is simulated; with b conditioned there,
        # the collocated h=0 term alone determines which TI cell is matched.
        nb = np.arange(36).reshape(6, 6)
        na = (nb * 7) % 5
        ti = TrainingImage([Variable("a", na), Variable("b", nb)])
        ds = DirectSampling(MPSModel(ti, scan_fraction=1.0, threshold=None))
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # partial NaN drops nothing
            ds.set_condition(
                cond_pos=[[0.0], [0.0]],
                cond_val={"a": np.array([np.nan]), "b": np.array([20])},
            )
        field = ds([np.arange(1, dtype=float)] * 2, seed=0)
        field = as_dict(ds, field)
        # b=20 sits at TI cell (3, 2); a there is (20*7) % 5 == 0
        self.assertEqual(field["a"][0, 0], (20 * 7) % 5)

    def test_univariate_set_condition(self):
        # Scalar set_condition path of a univariate TI.
        rng = np.random.default_rng(0)
        ti = TrainingImage(rng.integers(0, 3, (20, 20)))
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.2))
        ds.set_condition([[5.0], [5.0]], [2])
        field = ds([np.arange(10, dtype=float)] * 2, seed=0)
        self.assertEqual(field[5, 5], 2)

    def test_set_condition_length_mismatch(self):
        ti = TrainingImage(
            [
                Variable("a", np.zeros((10, 10), dtype=int)),
                Variable("b", np.zeros((10, 10), dtype=int)),
            ]
        )
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.3))
        # cond_pos has 1 point but cond_val has 2 -> length mismatch
        with self.assertRaisesRegex(ValueError, "mismatch"):
            ds.set_condition(
                cond_pos=[[3.0], [3.0]],
                cond_val={"a": np.array([1, 0]), "b": np.array([1, 0])},
            )
        # per-variable arrays of different lengths
        with self.assertRaisesRegex(ValueError, "same"):
            ds.set_condition(
                cond_pos=[[3.0, 4.0], [3.0, 4.0]],
                cond_val={"a": np.array([1, 0]), "b": np.array([1])},
            )

    def test_parallel_valid_values(self):
        rng = np.random.default_rng(4)
        ti = TrainingImage(
            [
                Variable("a", rng.integers(0, 3, (20, 20))),
                Variable("b", rng.integers(0, 2, (20, 20))),
            ]
        )
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.2))
        field = ds([np.arange(8, dtype=float)] * 2, seed=0, num_threads=2)
        field = as_dict(ds, field)
        self.assertTrue(np.all(np.isin(field["a"], [0, 1, 2])))
        self.assertTrue(np.all(np.isin(field["b"], [0, 1])))

    def test_parallel_matches_serial(self):
        # The node-vertex DAG commits every per-variable neighbour before a node
        # runs, so num_threads > 1 is bit-identical to serial.
        rng = np.random.default_rng(5)
        ti = TrainingImage(
            [
                Variable("a", rng.integers(0, 3, (20, 20))),
                Variable("b", rng.integers(0, 2, (20, 20))),
            ]
        )
        pos = [np.arange(8, dtype=float)] * 2
        ds_s = DirectSampling(MPSModel(ti, scan_fraction=0.2))
        ds_p = DirectSampling(MPSModel(ti, scan_fraction=0.2))
        f_s = ds_s(pos, seed=7, num_threads=1)
        f_p = ds_p(pos, seed=7, num_threads=2)
        np.testing.assert_array_equal(f_s[0], f_p[0])
        np.testing.assert_array_equal(f_s[1], f_p[1])

    def test_parallel_matches_serial_partial_conditioning(self):
        # Highest-risk DAG path: partial (NaN) conditioning + threads must stay
        # bit-identical to serial.
        rng = np.random.default_rng(17)
        ti = TrainingImage(
            [
                Variable("a", rng.integers(0, 3, (20, 20))),
                Variable("b", rng.integers(0, 2, (20, 20))),
            ]
        )
        pos = [np.arange(8, dtype=float)] * 2
        cond_pos = [np.array([1.0, 4.0, 6.0]), np.array([2.0, 5.0, 7.0])]
        cond_val = {
            "a": np.array([1.0, np.nan, 2.0]),  # b unconstrained at point 0
            "b": np.array([np.nan, 1.0, 0.0]),  # a unconstrained at point 1
        }
        ds_s = DirectSampling(MPSModel(ti, scan_fraction=0.3))
        ds_p = DirectSampling(MPSModel(ti, scan_fraction=0.3))
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # partial NaN drops nothing
            ds_s.set_condition(cond_pos, cond_val)
            ds_p.set_condition(cond_pos, cond_val)
        f_s = ds_s(pos, seed=7, num_threads=1)
        f_p = ds_p(pos, seed=7, num_threads=4)
        np.testing.assert_array_equal(f_s[0], f_p[0])
        np.testing.assert_array_equal(f_s[1], f_p[1])

    def test_parallel_matches_serial_mismatched_max_radius(self):
        # The DAG neighbour cache is built with the shared/global radius (a
        # superset of any variable's own, possibly smaller, radius) and must
        # be re-filtered per variable before use in _gather_neighborhood --
        # otherwise num_threads > 1 silently diverges from serial whenever
        # variables declare different max_radius values (including an
        # unbounded None mixed with a finite sibling).
        rng = np.random.default_rng(23)
        ti = TrainingImage(
            [
                Variable(
                    "a",
                    rng.integers(0, 3, (20, 20)),
                    max_radius=None,
                    n_neighbors=12,
                ),
                Variable(
                    "b",
                    rng.integers(0, 2, (20, 20)),
                    max_radius=3.0,
                    n_neighbors=12,
                ),
            ]
        )
        pos = [np.arange(8, dtype=float)] * 2
        ds_s = DirectSampling(MPSModel(ti, scan_fraction=0.3))
        ds_p = DirectSampling(MPSModel(ti, scan_fraction=0.3))
        f_s = ds_s(pos, seed=7, num_threads=1)
        f_p = ds_p(pos, seed=7, num_threads=4)
        np.testing.assert_array_equal(f_s[0], f_p[0])
        np.testing.assert_array_equal(f_s[1], f_p[1])

    def test_parallel_matches_serial_mismatched_n_neighbors(self):
        # The DAG neighbour cache is stored flat, one fixed-stride buffer per
        # variable, with the stride being that variable's own n_neighbors. A
        # stride shared across variables (or taken from the wrong one) would
        # read a neighbour slice offset into the previous node's entry, so
        # pin the case where the two strides genuinely differ.
        rng = np.random.default_rng(29)
        ti = TrainingImage(
            [
                Variable("a", rng.integers(0, 3, (20, 20)), n_neighbors=20),
                Variable("b", rng.integers(0, 2, (20, 20)), n_neighbors=5),
            ]
        )
        pos = [np.arange(8, dtype=float)] * 2
        ds_s = DirectSampling(MPSModel(ti, scan_fraction=0.3))
        ds_p = DirectSampling(MPSModel(ti, scan_fraction=0.3))
        f_s = ds_s(pos, seed=7, num_threads=1)
        f_p = ds_p(pos, seed=7, num_threads=4)
        np.testing.assert_array_equal(f_s[0], f_p[0])
        np.testing.assert_array_equal(f_s[1], f_p[1])

    def test_parallel_conditioning_preserved(self):
        rng = np.random.default_rng(6)
        ti = TrainingImage(
            [
                Variable("a", rng.integers(0, 2, (20, 20))),
                Variable("b", rng.integers(0, 2, (20, 20))),
            ]
        )
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.3))
        ds.set_condition(
            cond_pos=[[3.0], [3.0]],
            cond_val={"a": np.array([1]), "b": np.array([0])},
        )
        field = ds([np.arange(8, dtype=float)] * 2, seed=9, num_threads=2)
        field = as_dict(ds, field)
        self.assertEqual(field["a"][3, 3], 1)
        self.assertEqual(field["b"][3, 3], 0)

    def test_set_condition_validation_errors(self):
        ti = TrainingImage(
            [
                Variable("a", np.zeros((10, 10), dtype=int)),
                Variable("b", np.zeros((10, 10), dtype=int)),
            ]
        )
        cases = {
            "array_instead_of_dict": (np.array([1]), "dict"),
            "empty_dict": ({}, "empty"),
            "unknown_variable": (
                {"a": np.array([1.0]), "c": np.array([1.0])},
                "unknown variable",
            ),
        }
        for label, (cond_val, regex) in cases.items():
            with self.subTest(label=label):
                ds = DirectSampling(MPSModel(ti, scan_fraction=0.3))
                with self.assertRaisesRegex(ValueError, regex):
                    ds.set_condition([[3.0], [3.0]], cond_val)

    def test_set_condition_collision_per_variable_merge(self):
        # Two points snap to node (4,4). The closer point {a:1, b:nan} wins
        # variable a; the farther point {a:0, b:0} still fills variable b, which
        # the closer point left NaN (per-variable collision resolution, C1).
        rng = np.random.default_rng(7)
        ti = TrainingImage(
            [
                Variable("a", rng.integers(0, 2, (20, 20))),
                Variable("b", rng.integers(0, 2, (20, 20))),
            ]
        )
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.3))
        with self.assertWarnsRegex(UserWarning, "collided"):
            ds.set_condition(
                # point 0 at (4.4,4.4) {a:0,b:0}; point 1 at (4.1,4.1) {a:1,b:nan}
                cond_pos=[[4.4, 4.1], [4.4, 4.1]],
                cond_val={"a": np.array([0, 1]), "b": np.array([0, np.nan])},
            )
            field = ds([np.arange(8, dtype=float)] * 2, seed=1)
            field = as_dict(ds, field)
        # closer point wins a; farther point's finite b is not discarded
        self.assertEqual(field["a"][4, 4], 1)
        self.assertEqual(field["b"][4, 4], 0)

    def test_distance_power_multivariate(self):
        # distance_power > 0 exercises the per-variable lag-norm weighting path.
        rng = np.random.default_rng(8)
        ti = TrainingImage(
            [
                Variable(
                    "cont",
                    rng.random((20, 20)) * 10.0,
                    categorical=False,
                    distance="l2",
                    distance_power=1.0,
                ),
                Variable(
                    "cat",
                    rng.integers(0, 2, (20, 20)),
                    categorical=True,
                    distance_power=1.0,
                ),
            ],
        )
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.3))
        field = ds([np.arange(8, dtype=float)] * 2, seed=3)
        field = as_dict(ds, field)
        self.assertEqual(field["cont"].shape, (8, 8))
        self.assertTrue(np.all(np.isfinite(field["cont"])))
        self.assertTrue(np.all(np.isin(field["cat"], [0, 1])))

    def test_single_variable_multivariate(self):
        rng = np.random.default_rng(9)
        ti = TrainingImage([Variable("only", rng.integers(0, 3, (20, 20)))])
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.3))
        field = ds([np.arange(8, dtype=float)] * 2, seed=0)
        self.assertIsInstance(field, np.ndarray)
        self.assertEqual(field.shape, (8, 8))
        self.assertTrue(np.all(np.isin(field, [0, 1, 2])))

    def test_threshold_renormalization_with_empty_variable(self):
        # When variable 'b' has no informed neighbours (first node on the
        # path), the joint distance must still be renormalized
        # to [0,1] so the threshold comparison is meaningful.
        # Build a TI where 'a' and 'b' are injective: b = a + 100 (same as
        # test_joint_cell_invariant). Use threshold > 0 (DS mode, not DSBC).
        # The first simulated node has zero neighbours for both variables, so
        # it falls back to a random TI cell — both variables must be drawn from
        # the same cell (b == a + 100 for that node too).
        ids = np.arange(64).reshape(8, 8)
        ti = TrainingImage(
            [
                Variable("a", ids, categorical=True),
                Variable("b", ids + 100, categorical=True),
            ]
        )
        # scan_fraction=1.0, threshold=0.01: very strict but must still complete
        # without NaN and must reproduce the joint relationship everywhere.
        ds = DirectSampling(MPSModel(ti, scan_fraction=1.0, threshold=0.01))
        field = ds([np.arange(6, dtype=float)] * 2, seed=0)
        field = as_dict(ds, field)
        self.assertFalse(np.any(np.isnan(field["a"])))
        self.assertFalse(np.any(np.isnan(field["b"])))
        np.testing.assert_array_equal(field["b"], field["a"] + 100)

    def test_mv_store_list_names(self):
        # store=[...] follows Krige: every field is reachable by its name.
        rng = np.random.default_rng(0)
        ti = TrainingImage(
            [
                Variable("a", rng.integers(0, 2, (20, 20))),
                Variable("b", rng.integers(0, 2, (20, 20))),
            ]
        )
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.3))
        out = ds([np.arange(6, dtype=float)] * 2, seed=0, store=["x", "y"])
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        self.assertEqual(list(ds.field_names), ["x", "y"])
        np.testing.assert_array_equal(ds["x"], out[0])
        np.testing.assert_array_equal(ds["y"], out[1])

    def test_mv_store_false_not_stored(self):
        # store=False must still work for MV (no fields persisted on the model).
        rng = np.random.default_rng(0)
        ti = TrainingImage(
            [
                Variable("a", rng.integers(0, 2, (20, 20))),
                Variable("b", rng.integers(0, 2, (20, 20))),
            ]
        )
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.3))
        out = ds([np.arange(6, dtype=float)] * 2, seed=0, store=False)
        self.assertEqual(len(out), 2)
        self.assertEqual(list(ds.field_names), [])

    def test_mv_run_does_not_set_self_field(self):
        # Equal-treatment MV: no single self.field; per-variable named fields only.
        rng = np.random.default_rng(0)
        ti = TrainingImage(
            [
                Variable("a", rng.integers(0, 2, (15, 15))),
                Variable("b", rng.integers(0, 2, (15, 15))),
            ]
        )
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.3))
        ds([np.arange(6, dtype=float)] * 2, seed=0)
        self.assertFalse(hasattr(ds, "field"))
        self.assertIn("a", ds.field_names)
        self.assertIn("b", ds.field_names)


class TestSeededOutputPinned(unittest.TestCase):
    """Bit-identical full-array pins of seeded simulations.

    The arrays were captured with fixed seeds and must stay identical
    (np.testing.assert_array_equal).
    """

    # --- Reference TIs (seeded deterministically) ---
    @classmethod
    def setUpClass(cls):
        arr1d = np.tile([0.0, 1.0], 10)
        cls.ti1d = TrainingImage(arr1d, categorical=True)
        cls.ti2d = TrainingImage(
            (np.indices((8, 8)).sum(axis=0) % 2).astype(float),
            categorical=True,
        )
        rng0 = np.random.default_rng(0)
        cls.ti2d_rand = TrainingImage(
            rng0.integers(0, 3, (20, 20)).astype(float), categorical=True
        )
        rng_mv = np.random.default_rng(7)
        cls.ti_mv = TrainingImage(
            [
                Variable("a", rng_mv.integers(0, 3, (20, 20)).astype(float)),
                Variable("b", rng_mv.integers(0, 2, (20, 20)).astype(float)),
            ]
        )
        cls.x1d = np.arange(10, dtype=float)
        cls.x2d = np.arange(6, dtype=float)
        cls.y2d = np.arange(6, dtype=float)

    def test_seeded_field_pinned(self):
        # --- univariate 1D, seed=42 ---
        ds = DirectSampling(MPSModel(self.ti1d, scan_fraction=1.0))
        f = ds([self.x1d], seed=42)
        np.testing.assert_array_equal(
            f,
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            err_msg="snap_uni_1d seed=42",
        )
        # seed=99
        f = ds([self.x1d], seed=99)
        np.testing.assert_array_equal(
            f,
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            err_msg="snap_uni_1d seed=99",
        )

        # --- univariate 2D checkerboard, seed=42 ---
        ds2 = DirectSampling(MPSModel(self.ti2d, scan_fraction=1.0))
        f = ds2([self.x2d, self.y2d], seed=42)
        np.testing.assert_array_equal(
            f,
            [
                [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            ],
            err_msg="snap_uni_2d_checker seed=42",
        )
        # seed=99
        f = ds2([self.x2d, self.y2d], seed=99)
        np.testing.assert_array_equal(
            f,
            [
                [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            ],
            err_msg="snap_uni_2d_checker seed=99",
        )

        # --- univariate 2D random TI, seed=42 ---
        ds3 = DirectSampling(MPSModel(self.ti2d_rand, scan_fraction=0.5))
        f = ds3([self.x2d, self.y2d], seed=42)
        np.testing.assert_array_equal(
            f,
            [
                [1.0, 2.0, 2.0, 1.0, 2.0, 2.0],
                [2.0, 0.0, 1.0, 2.0, 2.0, 1.0],
                [1.0, 2.0, 2.0, 0.0, 0.0, 2.0],
                [2.0, 0.0, 1.0, 2.0, 1.0, 2.0],
                [0.0, 2.0, 2.0, 2.0, 1.0, 2.0],
                [1.0, 1.0, 2.0, 2.0, 1.0, 1.0],
            ],
            err_msg="snap_uni_rand seed=42",
        )

        # --- multivariate 2D, seed=42 ---
        ds_mv = DirectSampling(MPSModel(self.ti_mv, scan_fraction=0.5))
        res = ds_mv([self.x2d, self.y2d], seed=42)
        res = as_dict(ds_mv, res)
        np.testing.assert_array_equal(
            res["a"],
            [
                [1.0, 1.0, 1.0, 2.0, 2.0, 0.0],
                [0.0, 0.0, 1.0, 2.0, 2.0, 1.0],
                [2.0, 2.0, 0.0, 1.0, 0.0, 0.0],
                [2.0, 2.0, 1.0, 1.0, 2.0, 1.0],
                [0.0, 1.0, 2.0, 1.0, 0.0, 1.0],
                [2.0, 2.0, 2.0, 1.0, 1.0, 2.0],
            ],
            err_msg="snap_mv var=a seed=42",
        )
        np.testing.assert_array_equal(
            res["b"],
            [
                [0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            ],
            err_msg="snap_mv var=b seed=42",
        )


class TestPathAndNodeSeeds(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        data = rng.integers(0, 3, (20, 20))
        self.ti = TrainingImage(data.astype(float), n_neighbors=4)
        self.pos = [np.arange(8, dtype=float)] * 2

    def test_fixed_path_seed_different_node_seed(self):
        # Same visit order, different TI search → different output
        ds = DirectSampling(MPSModel(self.ti, scan_fraction=0.3))
        fa = ds(self.pos, path_seed=1, node_seed=10)
        fb = ds(self.pos, path_seed=1, node_seed=99)
        self.assertFalse(np.array_equal(fa, fb))

    def test_fixed_node_seed_different_path_seed(self):
        # Same TI search, different visit order → different output
        ds = DirectSampling(MPSModel(self.ti, scan_fraction=0.3))
        fa = ds(self.pos, path_seed=1, node_seed=10)
        fb = ds(self.pos, path_seed=99, node_seed=10)
        self.assertFalse(np.array_equal(fa, fb))

    def test_both_seeds_fixed_reproducible(self):
        # Explicit seeds on both → fully reproducible across calls
        ds = DirectSampling(MPSModel(self.ti, scan_fraction=0.3))
        fa = ds(self.pos, path_seed=7, node_seed=42)
        fb = ds(self.pos, path_seed=7, node_seed=42)
        self.assertTrue(np.array_equal(fa, fb))


class TestStatisticalValidity(unittest.TestCase):
    """Core correctness contract: relaxed thresholds and large TIs must reproduce TI statistics."""

    def test_categorical_histogram_reproduced(self):
        """Simulated category proportions must stay within 0.08 of the TI proportions."""
        rng = np.random.default_rng(0)
        data = rng.choice(
            [0, 1, 2], size=(100, 100), p=[0.5, 0.3, 0.2]
        ).astype(float)
        ti = TrainingImage(data, categorical=True, n_neighbors=8)
        ti_props = np.array([np.mean(data == v) for v in [0, 1, 2]])

        ds = DirectSampling(
            MPSModel(
                ti,
                scan_fraction=1.0,
                threshold=None,
            )
        )
        pos = [np.arange(50, dtype=float)] * 2
        field = ds(pos, seed=0)

        # (a) every category must appear — collapse to a single value is a bug
        self.assertEqual(
            set(np.unique(field)),
            {0.0, 1.0, 2.0},
            msg="At least one TI category is missing from the simulated field.",
        )

        # (b) per-category proportions within 0.08 of TI proportions
        sim_props = np.array([np.mean(field == v) for v in [0, 1, 2]])
        np.testing.assert_allclose(
            sim_props,
            ti_props,
            atol=0.08,
            err_msg="Simulated category proportions deviate more than 0.08 from TI.",
        )

    def test_continuous_mean_preserved(self):
        """Simulated mean and std must be within 0.25 / 0.30 of the TI values."""
        rng = np.random.default_rng(0)
        data = rng.standard_normal((100, 100))
        ti = TrainingImage(
            data, categorical=False, distance="l2", n_neighbors=8
        )
        ti_mean = float(data.mean())
        ti_std = float(data.std())

        ds = DirectSampling(
            MPSModel(
                ti,
                scan_fraction=1.0,
                threshold=0.05,
            )
        )
        pos = [np.arange(40, dtype=float)] * 2
        field = ds(pos, seed=0)

        self.assertAlmostEqual(
            float(field.mean()),
            ti_mean,
            delta=0.25,
            msg="Simulated field mean deviates by more than 0.25 from TI mean.",
        )
        self.assertAlmostEqual(
            float(field.std()),
            ti_std,
            delta=0.30,
            msg="Simulated field std deviates by more than 0.30 from TI std.",
        )


class TestMVTransformsAndReporting(unittest.TestCase):
    """Behaviour tests for mean/transform handling, post_process, and cond_weight."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.mv_data = {
            "a": rng.integers(0, 3, (20, 20)).astype(float),
            "b": rng.integers(0, 2, (20, 20)).astype(float),
        }
        cls.mv_ti = TrainingImage(
            [
                Variable(
                    "a", cls.mv_data["a"], categorical=False, n_neighbors=4
                ),
                Variable(
                    "b", cls.mv_data["b"], categorical=False, n_neighbors=4
                ),
            ]
        )
        cls.pos = [np.arange(6, dtype=float)] * 2

    def test_constructor_mean_applied_to_continuous_variables(self):
        """DirectSampling(mean=100.0) must shift every continuous variable by 100."""
        ds = DirectSampling(
            MPSModel(self.mv_ti, scan_fraction=0.3), mean=100.0
        )

        field_pp = ds(self.pos, seed=5)
        field_raw = ds(self.pos, seed=5, post_process=False)

        for name, pp, raw in zip(("a", "b"), field_pp, field_raw):
            np.testing.assert_allclose(
                pp,
                raw + 100.0,
                err_msg=f"Post-processed {name!r} should equal raw + 100.",
            )

    def test_transform_named_field_leaves_other_variables(self):
        """ds.transform(method, field='a') must change only variable 'a'."""
        ds = DirectSampling(MPSModel(self.mv_ti, scan_fraction=0.3))
        out = ds(self.pos, seed=5)
        out = as_dict(ds, out)
        a_before, b_before = out["a"].copy(), out["b"].copy()
        ds.transform("normal_to_lognormal", field="a")
        np.testing.assert_allclose(ds["a"], np.exp(a_before))
        np.testing.assert_array_equal(ds["b"], b_before)

    def test_post_process_false_returns_raw(self):
        """post_process=False must return values that are a subset of TI categories."""
        ds = DirectSampling(
            MPSModel(self.mv_ti, scan_fraction=0.3), mean=100.0
        )
        field_raw = ds(self.pos, seed=7, post_process=False)
        field_raw = as_dict(ds, field_raw)

        ti_a_values = set(np.unique(self.mv_data["a"]))
        sim_a_values = set(np.unique(field_raw["a"]))
        self.assertTrue(
            sim_a_values.issubset(ti_a_values),
            msg=f"post_process=False 'a' values {sim_a_values} are not a subset of TI values {ti_a_values}.",
        )

    def test_cond_weight_changes_output(self):
        """Different cond_weight values must yield different fields while honoring the conditioned node."""
        rng = np.random.default_rng(2)
        data = rng.integers(0, 3, (20, 20)).astype(float)
        pos = [np.arange(8, dtype=float)] * 2
        cond_pos = [[4.0], [4.0]]
        cond_val = [2]

        ti1 = TrainingImage(data, n_neighbors=8, cond_weight=1.0)
        ds1 = DirectSampling(MPSModel(ti1, scan_fraction=0.3))
        ds1.set_condition(cond_pos, cond_val)

        ti5 = TrainingImage(ti1.variable().replace(cond_weight=5.0))
        ds5 = DirectSampling(MPSModel(ti5, scan_fraction=0.3))
        ds5.set_condition(cond_pos, cond_val)

        # A single seed can coincide (4 of 30 did), so require a difference
        # for at least one of 10 seeds (cond_weight propagates through the
        # distance); the conditioned node must keep its value in every run.
        differs = False
        for seed in range(10):
            f1, f5 = ds1(pos, seed=seed), ds5(pos, seed=seed)
            differs |= not np.array_equal(f1, f5)
            self.assertEqual(int(f1[4, 4]), 2)
            self.assertEqual(int(f5[4, 4]), 2)
        self.assertTrue(
            differs,
            msg="cond_weight=1.0 and cond_weight=5.0 produced identical output for all seeds; cond_weight has no effect.",
        )


class TestFieldApi(unittest.TestCase):
    """Mean/normalizer/trend and transform go through the inherited Field API."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(3)
        cls.a = rng.random((20, 20))
        cls.f = rng.integers(0, 3, (20, 20)).astype(float)
        cls.ti = TrainingImage(
            [
                Variable("a", cls.a, categorical=False, n_neighbors=4),
                Variable("f", cls.f, categorical=True, n_neighbors=4),
            ]
        )
        cls.pos = [np.arange(6, dtype=float)] * 2

    def _ds(self, **kw):
        return DirectSampling(MPSModel(self.ti, scan_fraction=0.3), **kw)

    def test_mean_applied_to_continuous_only(self):
        ds = self._ds(mean=10.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = ds(self.pos, seed=1)
            out = as_dict(ds, out)
        raw = self._ds()(self.pos, seed=1, post_process=False)
        np.testing.assert_allclose(out["a"], raw[0] + 10.0)
        self.assertTrue(set(np.unique(out["f"])) <= set(np.unique(self.f)))
        np.testing.assert_array_equal(out["f"], raw[1])
        msgs = [str(x.message) for x in w if x.category is UserWarning]
        self.assertEqual(len(msgs), 1)
        self.assertIn("'f'", msgs[0])
        self.assertNotIn("'a'", msgs[0])

    def test_no_warning_without_mean_normalizer_trend(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._ds()(self.pos, seed=1)

    def test_categorical_univariate_mean_skipped_with_warning(self):
        ti = TrainingImage(self.f, categorical=True, n_neighbors=4)
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.3), mean=10.0)
        with self.assertWarnsRegex(UserWarning, "categorical"):
            out = ds(self.pos, seed=1)
        self.assertTrue(set(np.unique(out)) <= set(np.unique(self.f)))

    def test_transform_named_field(self):
        ds = self._ds()
        out = ds(self.pos, seed=1)
        out = as_dict(ds, out)
        res = ds.transform("normal_to_lognormal", field="a")
        np.testing.assert_allclose(res, np.exp(out["a"]))
        np.testing.assert_allclose(ds["a"], np.exp(out["a"]))

    def test_model_dependent_transform_raises_clear_error(self):
        ds = self._ds()
        ds(self.pos, seed=1)
        for method in (
            "binary",
            "discrete",
            "zinnharvey",
            "normal_force_moments",
            "normal_to_uniform",
            "normal_to_arcsin",
            "normal_to_uquad",
        ):
            with self.assertRaisesRegex(ValueError, "covariance model"):
                ds.transform(method, field="a")

    def test_transform_on_categorical_warns(self):
        ds = self._ds()
        ds(self.pos, seed=1)
        with self.assertWarnsRegex(UserWarning, "'f'.*categorical"):
            ds.transform("normal_to_lognormal", field="f")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ds.transform("normal_to_lognormal", field="a")

    def test_transform_on_categorical_univariate_custom_store_warns(self):
        ti = TrainingImage(self.f, categorical=True, n_neighbors=4)
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.3))
        ds(self.pos, seed=1, store="mine")
        with self.assertWarnsRegex(UserWarning, "categorical"):
            ds.transform("normal_to_lognormal", field="mine")

    def test_normalizer_and_trend_match_field_pipeline(self):
        from gstools.normalizer import LogNormal
        from gstools.normalizer.tools import apply_mean_norm_trend

        def trend(x, y):
            return x + 2.0 * y

        ds = self._ds(mean=1.5, normalizer=LogNormal(), trend=trend)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = ds(self.pos, seed=4)
            out = as_dict(ds, out)
        raw = self._ds()(self.pos, seed=4, post_process=False)
        expected = apply_mean_norm_trend(
            pos=ds.pos,
            field=raw[0],
            mesh_type=ds.mesh_type,
            value_type=ds.value_type,
            mean=ds.mean,
            normalizer=ds.normalizer,
            trend=ds.trend,
            check_shape=False,
            stacked=False,
        )
        np.testing.assert_allclose(out["a"], expected)
        np.testing.assert_array_equal(out["f"], raw[1])

    def test_warning_with_only_normalizer_or_only_trend(self):
        from gstools.normalizer import LogNormal

        for kw in ({"normalizer": LogNormal()}, {"trend": 2.0}):
            ds = self._ds(**kw)
            with self.assertWarnsRegex(UserWarning, "'f'"):
                ds(self.pos, seed=1)

    def test_warning_with_single_value_next_to_dict(self):
        ds = self._ds(mean={"a": 5.0}, trend=2.0)
        with self.assertWarnsRegex(UserWarning, "'f'"):
            ds(self.pos, seed=1)

    def _raw(self, seed):
        return self._ds()(self.pos, seed=seed, post_process=False)

    def test_dict_mean_shifts_only_named_variable(self):
        ds = self._ds(mean={"a": 5.0})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = ds(self.pos, seed=2)
        raw = self._raw(2)
        np.testing.assert_allclose(out[0], raw[0] + 5.0)
        np.testing.assert_array_equal(out[1], raw[1])

    def test_dict_mean_leaves_other_continuous_variable_bit_identical(self):
        ti = TrainingImage(
            [
                Variable("a", self.a, categorical=False, n_neighbors=4),
                Variable(
                    "b", self.a[::-1].copy(), categorical=False, n_neighbors=4
                ),
            ]
        )
        raw = DirectSampling(MPSModel(ti, scan_fraction=0.3))(
            self.pos, seed=2, post_process=False
        )
        out = DirectSampling(MPSModel(ti, scan_fraction=0.3), mean={"a": 5.0})(
            self.pos, seed=2
        )
        np.testing.assert_allclose(out[0], raw[0] + 5.0)
        np.testing.assert_array_equal(out[1], raw[1])

    def test_dict_normalizer_matches_apply_mean_norm_trend(self):
        from gstools.normalizer import LogNormal
        from gstools.normalizer.tools import apply_mean_norm_trend

        for norm in (LogNormal(), LogNormal):
            ds = self._ds(normalizer={"a": norm})
            out = ds(self.pos, seed=2)
            raw = self._raw(2)
            expected = apply_mean_norm_trend(
                ds.pos,
                raw[0].copy(),
                normalizer=LogNormal(),
                mesh_type=ds.mesh_type,
                check_shape=False,
            )
            np.testing.assert_array_equal(out[0], expected)
            np.testing.assert_array_equal(out[1], raw[1])

    def test_dict_trend_callable(self):
        def trend(x, y):
            return x + 2.0 * y

        ds = self._ds(trend={"a": trend})
        out = ds(self.pos, seed=2)
        raw = self._raw(2)
        xx, yy = np.meshgrid(*self.pos, indexing="ij")
        np.testing.assert_allclose(out[0], raw[0] + xx + 2.0 * yy)
        np.testing.assert_array_equal(out[1], raw[1])

    def test_dict_unknown_or_categorical_name_raises(self):
        for kw in ("mean", "trend"):
            for bad in ("zzz", "f"):
                with self.assertRaises(ValueError):
                    self._ds(**{kw: {bad: 1.0}})
        from gstools.normalizer import LogNormal

        with self.assertRaises(ValueError):
            self._ds(normalizer={"f": LogNormal()})
        with self.assertRaises(ValueError):
            self._ds(mean={"a": "x"})
        with self.assertRaises(ValueError):
            self._ds(normalizer={"a": 3})

    def test_dict_names_checked_again_at_call_time(self):
        ds = self._ds(mean={"a": 1.0})
        ti2 = TrainingImage(
            [
                Variable("b", self.a, categorical=False, n_neighbors=4),
                Variable("f", self.f, categorical=True, n_neighbors=4),
            ]
        )
        ds.mps_model.ti = ti2
        with self.assertRaises(ValueError):
            ds(self.pos, seed=1)
        ti3 = TrainingImage(
            [
                Variable("a", self.f, categorical=True, n_neighbors=4),
                Variable("b", self.a, categorical=False, n_neighbors=4),
            ]
        )
        ds.mps_model.ti = ti3
        with self.assertRaises(ValueError):
            ds(self.pos, seed=1)

    def test_dict_setter_and_single_value_unchanged(self):
        ds = self._ds()
        ds.mean = {"a": 2.0}
        self.assertEqual(ds.mean, {"a": 2.0})
        with self.assertRaises(ValueError):
            ds.mean = {"zzz": 2.0}
        ds.mean = 3.0
        self.assertEqual(ds.mean, 3.0)
        raw = self._raw(2)
        with self.assertWarnsRegex(UserWarning, "'f'"):
            out = ds(self.pos, seed=2)
        np.testing.assert_allclose(out[0], raw[0] + 3.0)

    def test_transform_process_with_dicts_raises(self):
        ds = self._ds(mean={"a": 1.0})
        ds(self.pos, seed=1)
        with self.assertRaisesRegex(ValueError, "process"):
            ds.transform("normal_to_lognormal", field="a", process=True)
        ds.transform("normal_to_lognormal", field="a")

    def test_store_names(self):
        ds = self._ds()
        ds(self.pos, seed=1)
        self.assertEqual(ds.field_names, ["a", "f"])
        ti = TrainingImage(self.f, categorical=True, n_neighbors=4)
        uni = DirectSampling(MPSModel(ti, scan_fraction=0.3))
        uni(self.pos, seed=1)
        self.assertEqual(uni.field_names, ["field"])
        uni2 = DirectSampling(MPSModel(ti, scan_fraction=0.3))
        uni2(self.pos, seed=1, store="mine")
        self.assertEqual(uni2.field_names, ["mine"])
        stored = {n: ds[n].copy() for n in ds.field_names}
        ds(self.pos, seed=1, store=["x", "y"])
        self.assertEqual(ds.field_names, ["a", "f", "x", "y"])
        for name, field in stored.items():
            np.testing.assert_array_equal(ds[name], field)


class TestReturnTypeContract(unittest.TestCase):
    """Return-type contract: one variable returns ndarray; several return a tuple."""

    @classmethod
    def setUpClass(cls):
        # Small categorical 2-D TI — fast to simulate.
        rng = np.random.default_rng(55)
        cls.ti_uni = TrainingImage(
            rng.integers(0, 3, (15, 15)).astype(float),
            categorical=True,
            n_neighbors=4,
        )
        # Small 2-variable MV TI.
        rng2 = np.random.default_rng(56)
        cls.ti_mv = TrainingImage(
            [
                Variable(
                    "alpha",
                    rng2.integers(0, 2, (15, 15)).astype(float),
                    n_neighbors=4,
                ),
                Variable(
                    "beta",
                    rng2.integers(0, 3, (15, 15)).astype(float),
                    n_neighbors=4,
                ),
            ]
        )
        cls.pos = [np.arange(5, dtype=float)] * 2
        cls.sim_shape = (5, 5)

    # ------------------------------------------------------------------
    # Univariate: must return a bare numpy.ndarray, correct shape/dtype,
    # values a subset of the TI.
    # ------------------------------------------------------------------

    def test_univariate_contract(self):
        ds = DirectSampling(MPSModel(self.ti_uni, scan_fraction=0.5))
        out = ds(self.pos, seed=101)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, self.sim_shape)
        self.assertTrue(np.issubdtype(out.dtype, np.number))
        ti_vals = set(np.unique(self.ti_uni.variable().data))
        self.assertTrue(set(np.unique(out)).issubset(ti_vals))

    # ------------------------------------------------------------------
    # Multivariate: must return a tuple in variable order, each item
    # a correctly-shaped ndarray with values a subset of its own TI variable.
    # ------------------------------------------------------------------

    def test_multivariate_contract(self):
        ds = DirectSampling(MPSModel(self.ti_mv, scan_fraction=0.5))
        out = ds(self.pos, seed=105)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), len(self.ti_mv.variables))
        for v, arr in zip(self.ti_mv.variables, out):
            self.assertIsInstance(arr, np.ndarray)
            self.assertEqual(arr.shape, self.sim_shape)
            ti_vals = set(np.unique(v.data))
            self.assertTrue(set(np.unique(arr)).issubset(ti_vals))

    # ------------------------------------------------------------------
    # Output shape and type for 1-D and n-D grids
    # ------------------------------------------------------------------

    def test_univariate_nd_returns_ndarray_correct_shape(self):
        cases = [
            (
                np.tile([0.0, 1.0], 10),
                1.0,
                [np.arange(7, dtype=float)],
                (7,),
            ),
            (
                np.random.default_rng(57)
                .integers(0, 2, (8, 8, 8))
                .astype(float),
                0.3,
                [np.arange(4, dtype=float)] * 3,
                (4, 4, 4),
            ),
        ]
        for data, scan_fraction, pos, shape in cases:
            with self.subTest(shape=shape):
                ti = TrainingImage(data, categorical=True)
                ds = DirectSampling(MPSModel(ti, scan_fraction=scan_fraction))
                out = ds(pos, seed=110)
                self.assertIsInstance(out, np.ndarray)
                self.assertEqual(out.shape, shape)


class TestSharedModelCondWeight(unittest.TestCase):
    """cond_weight lives on Variable; DirectSampling reads it through the TI."""

    def setUp(self):
        rng = np.random.default_rng(42)
        self.ti_data = rng.integers(0, 3, (20, 20))

    def test_cond_weight_on_model_is_honored(self):
        """Two DS with identical cond_weight=2 (baked into the TI) produce the same output."""
        ti1 = TrainingImage(self.ti_data, n_neighbors=4, cond_weight=2.0)
        ds1 = DirectSampling(MPSModel(ti1, scan_fraction=0.5))
        cpos = [np.array([5.0, 10.0]), np.array([5.0, 10.0])]
        cval = np.array([1.0, 2.0])
        ds1.set_condition(cpos, cval)
        out1 = ds1([np.arange(15, dtype=float)] * 2, seed=7)

        ti2 = TrainingImage(self.ti_data, n_neighbors=4, cond_weight=2.0)
        ds2 = DirectSampling(MPSModel(ti2, scan_fraction=0.5))
        ds2.set_condition(cpos, cval)
        out2 = ds2([np.arange(15, dtype=float)] * 2, seed=7)

        np.testing.assert_array_equal(out1, out2)


class TestPenaltyMatrixSetCondition(unittest.TestCase):
    def test_univariate_bad_value_raises(self):
        # out-of-range (5.0) and non-integer (1.5) codes both fail fast here.
        t = 1.0 - np.eye(3)
        ti = TrainingImage(
            np.array([0, 1, 2, 0, 1, 2], dtype=float), penalty_matrix=t
        )
        for bad_val in (5.0, 1.5):
            ds = DirectSampling(MPSModel(ti))
            with self.subTest(val=bad_val):
                with self.assertRaisesRegex(ValueError, "penalty_matrix"):
                    ds.set_condition([[0.0]], [bad_val])

    def test_univariate_valid_code_accepted(self):
        t = 1.0 - np.eye(3)
        ti = TrainingImage(
            np.array([0, 1, 2, 0, 1, 2], dtype=float), penalty_matrix=t
        )
        ds = DirectSampling(MPSModel(ti))
        ds.set_condition([[0.0]], [2.0])
        np.testing.assert_array_equal(ds._cond_val["field"], [2.0])
        np.testing.assert_array_equal(ds._cond_pos, [[0.0]])

    def test_multivariate_out_of_range_raises(self):
        t = 1.0 - np.eye(2)
        rng = np.random.default_rng(3)
        ti = gs.mps.TrainingImage(
            [
                Variable("a", rng.integers(0, 2, (10, 10)), penalty_matrix=t),
                Variable("b", rng.integers(0, 2, (10, 10))),
            ]
        )
        ds = DirectSampling(MPSModel(ti))
        with self.assertRaisesRegex(ValueError, "penalty_matrix"):
            ds.set_condition([[0.0], [0.0]], {"a": [9.0], "b": [0.0]})

    def test_multivariate_no_penalty_matrix_unaffected(self):
        rng = np.random.default_rng(4)
        ti = gs.mps.TrainingImage(
            [
                Variable("a", rng.integers(0, 2, (10, 10))),
                Variable("b", rng.integers(0, 2, (10, 10))),
            ]
        )
        ds = DirectSampling(MPSModel(ti))
        with self.assertWarnsRegex(UserWarning, "cannot be reproduced"):
            ds.set_condition([[0.0], [0.0]], {"a": [99.0], "b": [0.0]})
        np.testing.assert_array_equal(ds._cond_val["a"], [99.0])
        np.testing.assert_array_equal(ds._cond_val["b"], [0.0])


class TestPenaltyMatrixOverrideConditioning(unittest.TestCase):
    """A post-pass penalty-matrix override is checked against conditioning codes."""

    def setUp(self):
        rng = np.random.default_rng(416)
        self.codes = rng.integers(0, 2, (12, 12)).astype(float)  # {0, 1}
        self.pos = [np.arange(8.0)] * 2

    def _ti_and_post(self):
        # A TI without penalty matrix, and a 2x2-matrix override of it.
        post = PostProcessing(
            1, overrides={"field": {"penalty_matrix": [[0, 1], [1, 0]]}}
        )
        return TrainingImage(self.codes), post

    def test_override_penalty_matrix_checks_conditioning_codes(self):
        # The post-pass indexes the override's matrix with the conditioning
        # codes, so a code beyond it must fail in set_condition.
        ti, post = self._ti_and_post()
        model = MPSModel(ti, post_processing=post)
        with self.assertRaisesRegex(ValueError, "penalty_matrix"):
            DirectSampling(model).set_condition([[0.0], [0.0]], [2.0])

    def test_override_set_after_condition_rejected_at_call(self):
        ti, post = self._ti_and_post()
        model = MPSModel(ti, post_processing=PostProcessing(1))
        ds = DirectSampling(model, seed=1)
        with ignoring(UserWarning):  # code 2 is not in the 0/1 TI
            ds.set_condition([[0.0], [0.0]], [2.0])  # fine: no matrix yet
        model.post_processing = post
        with self.assertRaisesRegex(ValueError, "penalty_matrix"):
            ds(self.pos, store=False)


class TestTIReplacedAfterSetCondition(unittest.TestCase):
    """Replacing model.ti after set_condition fails clearly at call time."""

    def setUp(self):
        self.pos = [np.arange(6.0)] * 2

    @staticmethod
    def _ti(names):
        rng = np.random.default_rng(1)
        return TrainingImage(
            [Variable(n, rng.integers(0, 2, (10, 10))) for n in names]
        )

    def _conditioned(self):
        model = MPSModel(self._ti("ab"))
        ds = DirectSampling(model, seed=1)
        ds.set_condition([[1.0], [1.0]], {"a": [1.0], "b": [0.0]})
        return model, ds

    def test_ti_without_conditioned_variables_raises(self):
        for names in ("cd", "a"):
            with self.subTest(names=names):
                model, ds = self._conditioned()
                model.ti = self._ti(names)
                with self.assertRaisesRegex(ValueError, "set_condition"):
                    ds(self.pos, store=False)

    def test_replacement_colliding_with_attribute_raises(self):
        model, ds = self._conditioned()
        model.ti = self._ti(["rng", "b"])
        ds.set_condition([[1.0], [1.0]], {"b": [0.0]})
        with self.assertRaisesRegex(ValueError, "collides"):
            ds(self.pos, store=False)

    def test_repeated_calls_do_not_trip_collision_guard(self):
        _, ds = self._conditioned()
        ds(self.pos)
        ds(self.pos)


class TestConditioningOutsideGrid(unittest.TestCase):
    """Conditioning points outside the grid snap to the edge node and warn."""

    def setUp(self):
        rng = np.random.default_rng(416)
        self.codes = rng.integers(0, 2, (12, 12)).astype(float)  # {0, 1}
        self.pos = [np.arange(8.0)] * 2

    def _conditioned_field(self, cond_pos, cond_val):
        ti = TrainingImage(self.codes, n_neighbors=6)
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.5), seed=0)
        ds.set_condition(cond_pos, cond_val)
        return ds(self.pos, store=False)

    def test_condition_far_outside_grid_warns(self):
        # The snap to the nearest edge node is not silent.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            field = self._conditioned_field([[50.0], [-20.0]], [5.0])
        outside = [
            w
            for w in caught
            if issubclass(w.category, UserWarning)
            and "outside the simulation grid" in str(w.message)
        ]
        self.assertEqual(len(outside), 1)
        self.assertEqual(field[7, 0], 5.0)

    def test_condition_within_half_cell_of_edge_is_silent(self):
        # Values the categorical TI reproduces, so only the edge snap is tested.
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            field = self._conditioned_field(
                [[-0.4, 7.4], [7.4, -0.4]], [1.0, 0.0]
            )
        self.assertEqual((field[0, 7], field[7, 0]), (1.0, 0.0))


class TestSeedArguments(unittest.TestCase):
    """Seed arguments: None is accepted and full-width path seeds differ."""

    def setUp(self):
        rng = np.random.default_rng(416)
        self.codes = rng.integers(0, 2, (12, 12)).astype(float)  # {0, 1}
        self.pos = [np.arange(8.0)] * 2

    def test_none_seeds_accepted(self):
        # None means a fresh random seed (as in gs.SRF), not a crash.
        model = MPSModel(TrainingImage(self.codes, n_neighbors=6))
        runs = {
            "init seed": lambda: DirectSampling(model, seed=None)(self.pos),
            "call seed": lambda: DirectSampling(model)(self.pos, seed=None),
            "path_seed": lambda: DirectSampling(model, seed=1)(
                self.pos, path_seed=None
            ),
            "node_seed": lambda: DirectSampling(model, seed=1)(
                self.pos, node_seed=None
            ),
        }
        for name, run in runs.items():
            with self.subTest(name):
                field = run()
                self.assertEqual(field.shape, (8, 8))
                self.assertTrue(np.isin(field, self.codes).all())

    def test_full_width_path_seeds_differ(self):
        # 329 and 505 would collide through a 16-bit draw.
        model = MPSModel(
            TrainingImage(self.codes, n_neighbors=6), scan_fraction=0.5
        )
        ds = DirectSampling(model)
        a = ds(self.pos, path_seed=329, node_seed=1, store=False)
        b = ds(self.pos, path_seed=505, node_seed=1, store=False)
        self.assertFalse(np.array_equal(a, b))

    def test_string_path_and_node_seed_rejected(self):
        model = MPSModel(TrainingImage(self.codes, n_neighbors=6))
        for name in ("path_seed", "node_seed"):
            with self.subTest(name):
                with self.assertRaises(TypeError):
                    DirectSampling(model, seed=1)(self.pos, **{name: "abc"})

    def test_unset_seed_continues_master_stream(self):
        # NaN (the default) continues the master stream: successive calls
        # differ, and the first equals a call with the instance seed itself.
        model = MPSModel(
            TrainingImage(self.codes, n_neighbors=6), scan_fraction=0.5
        )
        ds = DirectSampling(model, seed=5)
        first = ds(self.pos, store=False)
        second = ds(self.pos, store=False)
        self.assertFalse(np.array_equal(first, second))
        again = DirectSampling(model, seed=5)(self.pos, store=False)
        np.testing.assert_array_equal(first, again)

    def test_none_seed_draws_fresh_each_call(self):
        model = MPSModel(
            TrainingImage(self.codes, n_neighbors=6), scan_fraction=0.5
        )
        ds = DirectSampling(model)
        a = ds(self.pos, seed=None, store=False)
        b = ds(self.pos, seed=None, store=False)
        self.assertFalse(np.array_equal(a, b))

    def test_seed_sequence_rng_matches_reference_construction(self):
        for seed in (0, 329, 505, 2**63 + 1, 2**128 + 7):
            with self.subTest(seed=seed):
                master = np.random.SeedSequence(seed)
                rng = RNG(seed, seed_sequence=True)
                for _ in range(4):
                    ref = np.random.RandomState(
                        np.random.MT19937(master.spawn(1)[0])
                    )
                    np.testing.assert_array_equal(
                        rng.random.uniform(size=32), ref.uniform(size=32)
                    )
                rng.seed = seed  # re-seeding restarts the child sequence
                np.testing.assert_array_equal(
                    rng.random.uniform(size=32),
                    RNG(seed, seed_sequence=True).random.uniform(size=32),
                )

    def test_seed_prefix_fingerprints_distinct(self):
        # Bounded check that nearby and huge seeds give distinct streams.
        seeds = list(range(256)) + [329, 505, 2**63 + 1, 2**128 + 7]
        prints = [
            RNG(seed, seed_sequence=True).random.bytes(64) for seed in seeds
        ]
        self.assertEqual(len(set(prints)), len(seeds))


class TestZeroWeightVariables(unittest.TestCase):
    """A zero-weight NaN variable does not poison the scan."""

    def setUp(self):
        rng = np.random.default_rng(416)
        self.codes = rng.integers(0, 2, (12, 12)).astype(float)  # {0, 1}
        self.cont = rng.random((12, 12))
        self.pos = [np.arange(8.0)] * 2

    def test_zero_weight_nan_variable_does_not_poison_scan(self):
        # A weight-0 variable is carried along, never matched: its NaN cells
        # must not turn the joint distance into NaN (0 * NaN).
        cont = self.cont.copy()
        cont[::3, ::3] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ti = TrainingImage(
                [
                    Variable("a", self.codes, weight=1.0),
                    Variable("b", cont, categorical=False, weight=0.0),
                ]
            )
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.5), seed=0)
        field = ds(self.pos, store=False)
        field = as_dict(ds, field)
        self.assertTrue(np.isin(field["a"], [0.0, 1.0]).all())


class TestSetConditionValidation(unittest.TestCase):
    """set_condition input checks, warnings and atomicity."""

    def setUp(self):
        rng = np.random.default_rng(416)
        self.codes = rng.integers(0, 2, (12, 12)).astype(float)  # {0, 1}
        self.pos = [np.arange(8.0)] * 2

    def test_set_condition_rejects_non_finite_positions(self):
        # Non-finite positions raise and leave the previous conditioning intact.
        ti = TrainingImage(self.codes, n_neighbors=6)
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.5))
        ds.set_condition([[1.0], [2.0]], [1.0])
        before = ds(self.pos, seed=3, store=False)
        for bad in (np.nan, np.inf):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "finite"):
                    ds.set_condition([[bad], [2.0]], [0.0])
        np.testing.assert_array_equal(
            ds(self.pos, seed=3, store=False), before
        )

    def test_set_condition_float64_remedy_follows_categorical_flag(self):
        big = np.array([2**60 + 1])
        cases = ((True, "categorical labels"), (False, "continuous values"))
        for categorical, remedy in cases:
            ti = TrainingImage(
                Variable("a", self.codes, categorical=categorical)
            )
            ds = DirectSampling(MPSModel(ti))
            with self.subTest(categorical=categorical):
                with self.assertRaisesRegex(ValueError, remedy):
                    ds.set_condition([[1.0], [2.0]], big)

    def test_set_condition_bad_penalty_code_keeps_previous(self):
        # A code outside the penalty matrix raises and keeps the previous condition.
        data = np.tile([0.0, 1.0], (8, 4))
        var = Variable("c", data, penalty_matrix=[[0, 1], [1, 0]])
        ds = DirectSampling(MPSModel(TrainingImage(var), scan_fraction=0.5))
        ds.set_condition([[1.0], [1.0]], [1.0])
        before = ds(self.pos, seed=3, store=False)
        with self.assertRaises(ValueError):
            ds.set_condition([[3.0], [3.0]], [2.0])
        np.testing.assert_array_equal(
            ds(self.pos, seed=3, store=False), before
        )

    def test_failed_multivariate_update_keeps_previous(self):
        ti = TrainingImage(
            [
                Variable("a", self.codes, n_neighbors=2),
                Variable("b", 1.0 - self.codes, n_neighbors=2),
            ]
        )
        ds = DirectSampling(MPSModel(ti), seed=7)
        ds.set_condition([[0.0], [0.0]], {"a": [1.0], "b": [0.0]})
        with self.assertRaises(ValueError):
            ds.set_condition([[1.0], [1.0]], {"a": ["invalid"], "b": [0.0]})
        np.testing.assert_array_equal(ds._cond_pos, [[0.0], [0.0]])
        self.assertEqual(ds._cond_val["a"].tolist(), [1.0])
        self.assertEqual(ds._cond_val["b"].tolist(), [0.0])

    def test_set_condition_rejects_2d_value_array(self):
        ti = TrainingImage(
            [
                Variable("a", self.codes, n_neighbors=2),
                Variable("b", self.codes, n_neighbors=2),
            ]
        )
        ds = DirectSampling(MPSModel(ti), seed=7)
        with self.assertRaisesRegex(ValueError, "1-D"):
            ds.set_condition(
                [[0.0, 1.0], [0.0, 1.0]],
                {"a": [[0.0, 1.0], [1.0, 0.0]], "b": [0.0, 1.0]},
            )
        self.assertIsNone(ds._cond_val)

    def test_set_condition_warns_unreproducible_and_dropped(self):
        ti = TrainingImage(self.codes, n_neighbors=6)
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.5))
        with self.assertWarnsRegex(UserWarning, "cannot be reproduced"):
            ds.set_condition([[1.0], [2.0]], [99.0])
        with self.assertWarnsRegex(UserWarning, "without any finite"):
            ds.set_condition([[1.0, 3.0], [2.0, 4.0]], [np.nan, 1.0])

    def test_all_nan_point_is_dropped_and_counted_once(self):
        # Only the point with NaN in every variable is dropped; partial NaN
        # points are kept and not counted.
        ti = TrainingImage(
            [Variable("a", self.codes), Variable("b", self.codes)]
        )
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.5))
        nan = np.nan
        with self.assertWarnsRegex(
            UserWarning, "1 conditioning point.*dropped"
        ):
            ds.set_condition(
                [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
                {"a": [1.0, nan, nan], "b": [nan, nan, 0.0]},
            )
        self.assertEqual(ds._cond_pos.shape, (2, 2))

    def test_collisions_warn_once_per_call(self):
        # Three points snap to node (4, 4): two collisions, one warning.
        ti = TrainingImage(self.codes, n_neighbors=6)
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.5))
        ds.set_condition([[4.4, 4.1, 4.3], [4.4, 4.1, 4.3]], [0.0, 1.0, 1.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ds(self.pos, seed=3, store=False)
        collided = [
            str(w.message) for w in caught if "collided" in str(w.message)
        ]
        self.assertEqual(len(collided), 1)
        self.assertIn("2 conditioning value(s)", collided[0])


class TestExactCountsAndIntegers(unittest.TestCase):
    def setUp(self):
        ti = TrainingImage(
            np.random.default_rng(0).integers(0, 3, (10, 10)), n_neighbors=4
        )
        self.ds = DirectSampling(MPSModel(ti, scan_fraction=0.2))

    def test_fractional_num_threads_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            self.ds([np.arange(4.0)] * 2, num_threads=2.5)

    def test_zero_num_threads_rejected(self):
        with self.assertRaises(ValueError):
            self.ds([np.arange(4.0)] * 2, num_threads=0)

    def test_lossy_integer_cond_val_rejected_atomically(self):
        val = np.array([2**60 + 3], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "not exactly representable"):
            self.ds.set_condition([[0.0], [0.0]], val)
        self.assertIsNone(self.ds._cond_val)


class TestExactIntegersMultivariate(unittest.TestCase):
    def test_lossy_integer_cond_val_rejected_in_dict(self):
        rng = np.random.default_rng(0)
        ti = TrainingImage(
            [
                Variable("a", rng.integers(0, 3, (10, 10)), n_neighbors=4),
                Variable("b", rng.integers(0, 3, (10, 10)), n_neighbors=4),
            ]
        )
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.2))
        ds.set_condition([[1.0], [1.0]], {"a": [1.0], "b": [2.0]})
        lossy = np.array([2**60 + 3], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "not exactly representable"):
            ds.set_condition([[0.0], [0.0]], {"a": lossy})
        np.testing.assert_array_equal(ds._cond_pos, [[1.0], [1.0]])
        self.assertEqual(ds._cond_val["a"].tolist(), [1.0])
        self.assertEqual(ds._cond_val["b"].tolist(), [2.0])


class TestLargeSeeds(unittest.TestCase):
    def setUp(self):
        ti = TrainingImage(
            np.random.default_rng(0).integers(0, 3, (10, 10)), n_neighbors=4
        )
        self.ds = DirectSampling(MPSModel(ti, scan_fraction=0.2))
        self.pos = [np.arange(4.0)] * 2

    def test_seed_beyond_64_bits(self):
        big = 2**64 + 1
        fa = self.ds(self.pos, seed=big)
        fb = self.ds(self.pos, seed=big)
        self.assertEqual(fa.shape, (4, 4))
        np.testing.assert_array_equal(fa, fb)

    def test_path_seed_beyond_64_bits(self):
        fa = self.ds(self.pos, seed=1, path_seed=2**70)
        fb = self.ds(self.pos, seed=1, path_seed=2**70)
        np.testing.assert_array_equal(fa, fb)

    def test_node_seed_beyond_64_bits(self):
        fa = self.ds(self.pos, seed=1, node_seed=2**70)
        fb = self.ds(self.pos, seed=1, node_seed=2**70)
        np.testing.assert_array_equal(fa, fb)

    def test_non_numeric_seed_rejected(self):
        with self.assertRaises((TypeError, ValueError)):
            self.ds(self.pos, seed="abc")


class TestSeedValidation(unittest.TestCase):
    """Seeds must be exact non-negative integers (or None / NaN)."""

    def setUp(self):
        ti = TrainingImage(
            np.random.default_rng(0).integers(0, 3, (10, 10)), n_neighbors=4
        )
        self.model = MPSModel(ti, scan_fraction=0.2)
        self.ds = DirectSampling(self.model)
        self.pos = [np.arange(4.0)] * 2

    def test_non_integral_float_seed_rejected(self):
        for name in ("seed", "path_seed", "node_seed"):
            with self.subTest(argument=name):
                with self.assertRaisesRegex(ValueError, name):
                    self.ds(self.pos, **{name: 1.5})
        with self.assertRaisesRegex(ValueError, "seed"):
            DirectSampling(self.model, seed=1.5)

    def test_negative_seed_rejected_by_name(self):
        for name in ("seed", "path_seed", "node_seed"):
            with self.subTest(argument=name):
                with self.assertRaisesRegex(ValueError, f"{name}.*>= 0"):
                    self.ds(self.pos, **{name: -1})
        with self.assertRaisesRegex(ValueError, "seed.*>= 0"):
            DirectSampling(self.model, seed=-3)

    def test_integral_float_and_numpy_integer_seeds_accepted(self):
        ref = self.ds(self.pos, seed=3)
        np.testing.assert_array_equal(self.ds(self.pos, seed=3.0), ref)
        np.testing.assert_array_equal(self.ds(self.pos, seed=np.int64(3)), ref)
        np.testing.assert_array_equal(
            self.ds(self.pos, seed=np.float64(3.0)), ref
        )

    def test_non_finite_grid_axes_rejected(self):
        for bad in (np.nan, np.inf):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(
                    ValueError, "grid coordinates must be finite"
                ):
                    self.ds([np.array([0.0, bad, 2.0]), np.arange(3.0)])


class TestOneVariableModelSimulation(unittest.TestCase):
    """Return type follows the variable count; conditioning is form-independent."""

    def setUp(self):
        self.d = (np.indices((12, 12)).sum(axis=0) % 3).astype(float)

    def test_variable_count_decides_return_type(self):
        d = self.d
        cases = [
            (d, np.ndarray),
            (Variable("a", d), np.ndarray),
            ([Variable("a", d)], np.ndarray),
            ((Variable("a", d),), np.ndarray),
            ([Variable("a", d), Variable("b", d)], tuple),
        ]
        for data, kind in cases:
            out = DirectSampling(MPSModel(TrainingImage(data)), seed=1)(
                [range(4)] * 2
            )
            self.assertIsInstance(out, kind)

    def _ds(self, data):
        return DirectSampling(MPSModel(TrainingImage(data)), seed=1)

    def test_conditioning_semantics_match(self):
        pos = [[1.0, 2.0], [1.0, 2.0]]
        for data in (self.d, [Variable("a", self.d)]):
            ds = self._ds(data)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                ds.set_condition(pos, [1.0, np.nan])
            texts = [str(x.message) for x in w]
            self.assertEqual(
                texts,
                [
                    "DirectSampling.set_condition: 1 conditioning point(s) "
                    "without any finite value were dropped."
                ],
            )
            grid = ds._conditions_to_grid((np.arange(4.0), np.arange(4.0)))
            self.assertEqual(len(grid), 1)
            ds = self._ds(data)
            with self.assertRaises(ValueError):
                ds.set_condition(pos, [1.0, np.inf])
            with self.assertRaises(ValueError):
                ds.set_condition(pos, [1.0, -np.inf])

    def test_array_and_dict_condition_identical(self):
        pos = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
        vals = [0.0, 1.0, 2.0]
        ds_a = self._ds([Variable("a", self.d)])
        ds_b = self._ds([Variable("a", self.d)])
        ds_a.set_condition(pos, vals)
        ds_b.set_condition(pos, {"a": vals})
        ra = ds_a([range(6)] * 2)
        rb = ds_b([range(6)] * 2)
        np.testing.assert_array_equal(ra, rb)

    def test_failed_set_condition_keeps_previous(self):
        ds = self._ds(self.d)
        ds.set_condition([[1.0], [1.0]], [1.0])
        pos = ds._cond_pos.copy()
        vals = {k: v.copy() for k, v in ds._cond_val.items()}
        with self.assertRaises(ValueError):
            ds.set_condition([[1.0, 2.0], [1.0, 2.0]], [1.0, np.inf])
        np.testing.assert_array_equal(ds._cond_pos, pos)
        self.assertEqual(list(ds._cond_val), list(vals))
        for name, v in vals.items():
            np.testing.assert_array_equal(ds._cond_val[name], v)

    def test_output_values_are_ti_values(self):
        n = 16
        base = np.indices((n, n)).sum(axis=0) % 7
        for data in (
            base.astype(np.float32) / 3,
            Variable("c", base.astype(np.int64)),
        ):
            if isinstance(data, np.ndarray):
                ti = TrainingImage(data, categorical=False, distance="l2")
                ref = data
            else:
                ti = TrainingImage(data)
                ref = data.data
            ds = DirectSampling(MPSModel(ti, scan_fraction=0.5), seed=3)
            out = ds([range(8)] * 2)
            self.assertTrue(np.isin(out, ref.astype(np.float64)).all())

    def test_single_element_list_store(self):
        ds = DirectSampling(
            MPSModel(TrainingImage([Variable("a", self.d)])), seed=1
        )
        out = ds([range(4)] * 2)
        self.assertIsInstance(out, np.ndarray)
        np.testing.assert_array_equal(ds.field, out)
        ds([range(4)] * 2, store="x")
        self.assertIn("x", ds.field_names)

    def test_two_variables_tuple_in_order(self):
        ds = self._ds([Variable("a", self.d), Variable("b", self.d + 10.0)])
        out = ds([range(4)] * 2)
        self.assertEqual(len(out), 2)
        np.testing.assert_array_equal(out[1], out[0] + 10.0)
        np.testing.assert_array_equal(ds["a"], out[0])
        np.testing.assert_array_equal(ds["b"], out[1])


if __name__ == "__main__":
    unittest.main()
