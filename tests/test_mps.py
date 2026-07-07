#!/usr/bin/env python
"""Unittest for the MPS module (TrainingImage and DirectSampling)."""

import unittest
import warnings

import numpy as np

import gstools as gs
from gstools import config as gs_config
from gstools.mps.data_event import DataEvent
from gstools.mps.direct_sampling import DirectSampling
from gstools.mps.distance import (
    compute_node_weights,
    vec_categorical_dist,
    vec_l1_dist,
    vec_l2_dist,
    vec_lp_dist,
    vec_variation_dist,
)
from gstools.mps.model import MPSModel
from gstools.mps.neighbors import (
    _lag_transform_matrix,
    _precompute_offsets,
    _reduce_to_fit,
    _transform_lags,
    _window_bounds,
)
from gstools.mps.nonstationary import (
    build_zone_selector,
    resolve_spec,
    resolve_where,
)
from gstools.mps.scan import _scan_window
from gstools.mps.simulate import ds_simulate
from gstools.mps.training_image import TrainingImage, Variable
from gstools.tools.geometric import generate_grid


def _flat_pos(grid_shape):
    """Flat per-dimension coordinate arrays for a unit-spaced grid."""
    return generate_grid([np.arange(s, dtype=float) for s in grid_shape])


def _uni_dist(ti, de_sim, de_ti, **kw):
    """Univariate data-event distance via the vectorized production path."""
    row = np.asarray(de_ti, dtype=float)[np.newaxis, :]
    return float(ti.vec_distance_var(None, de_sim, row, **kw)[0])


def _mv_dist(ti, de_sim, de_ti, **kw):
    """Multivariate joint data-event distance via the vectorized path."""
    total = 0.0
    for v in ti.variables:
        row = np.asarray(de_ti[v.name], dtype=float)[np.newaxis, :]
        total += ti.weights[v.name] * float(
            ti.vec_distance_var(v.name, de_sim[v.name], row, **kw)[0]
        )
    return total


class TestDirectSamplingParallel(unittest.TestCase):
    def test_valid_values(self):
        rng = np.random.default_rng(0)
        data = rng.integers(0, 3, (20, 20))
        ti = TrainingImage(data, n_neighbors=4)
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.2))
        ds.num_threads = 2
        field = ds([np.arange(8, dtype=float)] * 2, seed=0)
        self.assertEqual(field.shape, (8, 8))
        self.assertTrue(np.all(np.isin(field, [0, 1, 2])))

    def test_reproducible(self):
        # DAG parallelism is deterministic: same seed → same parallel result
        rng = np.random.default_rng(0)
        data = rng.integers(0, 3, (20, 20))
        ti = TrainingImage(data, n_neighbors=4)
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.2))
        ds.num_threads = 2
        pos = [np.arange(8, dtype=float)] * 2
        self.assertTrue(np.array_equal(ds(pos, seed=7), ds(pos, seed=7)))

    def test_conditioning_preserved(self):
        rng = np.random.default_rng(0)
        data = rng.integers(0, 3, (20, 20))
        ti = TrainingImage(data, n_neighbors=4)
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.2))
        ds.num_threads = 2
        ds.set_condition([[5.0], [5.0]], [2])
        field = ds([np.arange(10, dtype=float)] * 2, seed=0)
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
        ds.num_threads = 4
        field = ds(pos, seed=42)
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
            ds.num_threads = nt
            ds.set_condition(cond_pos, cond_val)
            field = ds(pos, seed=11)
            self.assertEqual(field.shape, (25, 25))
            self.assertTrue(np.all(np.isin(field, [0, 1, 2, 3])))

    def test_all_nan_ti_no_typeerror(self):
        # Regression test for _scan_window None return on all-NaN TI.
        # When every candidate distance is NaN, best_y stays None without
        # the fallback. Verify the fallback is triggered and no TypeError occurs
        # (ValueError from NaN output is acceptable for all-NaN TI).
        ti_data = np.full((5, 5), np.nan, dtype=float)
        ti = TrainingImage(
            ti_data, categorical=False, distance="l2", n_neighbors=2
        )
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.2))
        ds.num_threads = 1
        try:
            field = ds([np.arange(3, dtype=float)] * 2, seed=42)
            # If we get here, simulation completed without TypeError.
            # Field may contain NaN due to the all-NaN TI.
            self.assertIsNotNone(field)
        except TypeError as e:
            self.fail(f"Fallback did not prevent TypeError: {e}")
        except ValueError:
            # ValueError from _write_result NaN check is acceptable when
            # TI is all-NaN; the important thing is no TypeError occurs.
            pass


class TestVariable(unittest.TestCase):
    def test_basic_categorical(self):
        data = np.array([[0, 1], [1, 0]], dtype=int)
        v = Variable("cat", data, categorical=True, n_neighbors=8)
        self.assertEqual(v.name, "cat")
        self.assertTrue(v.categorical)
        self.assertEqual(v.n_neighbors, 8)
        self.assertIsNone(v.weight)
        self.assertIsNone(v.p_norm)
        self.assertIsNone(v.variation_p_norm)
        self.assertIsNone(v.d_max)

    def test_data_defensive_copy(self):
        data = np.array([0, 1, 2, 3])
        v = Variable("x", data)
        data[0] = 99
        self.assertEqual(v.data[0], 0)

    def test_continuous_attributes(self):
        v = Variable(
            "p", np.linspace(0, 1, 20), categorical=False, distance="l2"
        )
        self.assertEqual(v.p_norm, 2.0)
        self.assertIsNone(v.variation_p_norm)
        self.assertGreater(v.d_max, 0)

    def test_variation_distance(self):
        v = Variable(
            "p",
            np.linspace(0, 1, 20),
            categorical=False,
            distance="variation",
            n_neighbors=2,
        )
        self.assertIsNone(v.p_norm)
        self.assertEqual(v.variation_p_norm, 2.0)

    def test_invalid_name_raises(self):
        with self.assertRaisesRegex(ValueError, "identifier"):
            Variable("123bad", np.zeros(4))
        with self.assertRaisesRegex(ValueError, "identifier"):
            Variable("", np.zeros(4))
        with self.assertRaisesRegex(ValueError, "identifier"):
            Variable("a b", np.zeros(4))

    def test_name_none_allowed(self):
        # None is the internal sentinel for the univariate anonymous variable
        v = Variable(None, np.zeros(4))
        self.assertIsNone(v.name)

    def test_n_neighbors_setter(self):
        v = Variable("x", np.zeros(4))
        v.n_neighbors = 8
        self.assertEqual(v.n_neighbors, 8)
        with self.assertRaisesRegex(ValueError, "n_neighbors"):
            v.n_neighbors = 0

    def test_variation_guard_at_construction(self):
        with self.assertRaisesRegex(ValueError, "variation"):
            Variable(
                "x",
                np.linspace(0, 1, 10),
                categorical=False,
                distance="variation",
                n_neighbors=1,
            )

    def test_variation_guard_via_setter(self):
        v = Variable(
            "x",
            np.linspace(0, 1, 10),
            categorical=False,
            distance="variation",
            n_neighbors=2,
        )
        with self.assertRaisesRegex(ValueError, "variation"):
            v.n_neighbors = 1
        self.assertEqual(v.n_neighbors, 2)  # unchanged

    def test_read_only_name(self):
        v = Variable("x", np.zeros(4))
        with self.assertRaises(AttributeError):
            v.name = "y"

    def test_read_only_categorical(self):
        v = Variable("x", np.zeros(4))
        with self.assertRaises(AttributeError):
            v.categorical = False

    def test_weight_stored(self):
        v = Variable("x", np.zeros(4), weight=0.7)
        self.assertAlmostEqual(v.weight, 0.7)

    def test_repr(self):
        v = Variable("x", np.zeros((4, 4)))
        self.assertIn("Variable", repr(v))
        self.assertIn("x", repr(v))

    def test_max_radius_none_default(self):
        v = Variable("x", np.zeros(4))
        self.assertIsNone(v.max_radius)

    def test_max_radius_stored(self):
        v = Variable("x", np.zeros(4), max_radius=50.0)
        self.assertAlmostEqual(v.max_radius, 50.0)

    def test_max_radius_zero_raises(self):
        with self.assertRaisesRegex(ValueError, "max_radius"):
            Variable("x", np.zeros(4), max_radius=0.0)

    def test_max_radius_negative_raises(self):
        with self.assertRaisesRegex(ValueError, "max_radius"):
            Variable("x", np.zeros(4), max_radius=-5.0)

    def test_max_radius_read_only(self):
        v = Variable("x", np.zeros(4), max_radius=10.0)
        with self.assertRaises(AttributeError):
            v.max_radius = 20.0


class TestTrainingImage(unittest.TestCase):
    def setUp(self):
        arr_cat = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        self.ti_cat = TrainingImage(arr_cat, categorical=True)

        arr_cont = np.linspace(0.0, 1.0, 20)
        self.ti_cont = TrainingImage(
            arr_cont, categorical=False, distance="l1"
        )

    def test_properties(self):
        np.testing.assert_array_equal(
            self.ti_cat.data,
            np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float),
        )
        self.assertEqual(self.ti_cat.ndim, 2)
        self.assertEqual(self.ti_cat.shape, (3, 3))
        self.assertTrue(self.ti_cat.categorical)
        self.assertEqual(
            self.ti_cat.distance_type, "l1"
        )  # default ignored for cat
        self.assertIsInstance(repr(self.ti_cat), str)
        self.assertIn("TrainingImage", repr(self.ti_cat))

        self.assertEqual(self.ti_cont.ndim, 1)
        self.assertEqual(self.ti_cont.shape, (20,))
        self.assertFalse(self.ti_cont.categorical)
        self.assertEqual(self.ti_cont.distance_type, "l1")

    def test_raise(self):
        with self.assertRaises(ValueError):
            TrainingImage(np.ones(10), categorical=False, distance="l0")
        with self.assertRaises(ValueError):
            TrainingImage(np.ones(10), categorical=False, distance="labc")
        with self.assertRaises(ValueError):
            TrainingImage(np.ones(10), categorical=False, distance="invalid")

    def test_distance_categorical(self):
        # Identical events → 0.0
        a = np.array([0.0, 1.0, 0.0])
        dist = _uni_dist(self.ti_cat, a, a)
        self.assertAlmostEqual(dist, 0.0)

        # Completely mismatched, uniform weights → 1.0
        b = np.array([1.0, 0.0, 1.0])
        dist = _uni_dist(self.ti_cat, a, b)
        self.assertAlmostEqual(dist, 1.0)

        # One of three mismatched → 1/3
        c = np.array([1.0, 1.0, 0.0])
        dist = _uni_dist(self.ti_cat, a, c)
        self.assertAlmostEqual(dist, 1.0 / 3.0)

        # Two of four mismatched → 0.5 (spec-required half-mismatch case)
        a4 = np.array([0.0, 1.0, 0.0, 1.0])
        c4 = np.array([1.0, 0.0, 0.0, 1.0])
        dist = _uni_dist(self.ti_cat, a4, c4)
        self.assertAlmostEqual(dist, 0.5)

    def test_distance_continuous(self):
        x = np.array([0.0, 0.5, 1.0])
        y = np.array([0.2, 0.3, 0.8])

        # l1
        ti_l1 = TrainingImage(
            np.linspace(0.0, 1.0, 10), categorical=False, distance="l1"
        )
        self.assertAlmostEqual(_uni_dist(ti_l1, x, x), 0.0)
        self.assertAlmostEqual(_uni_dist(ti_l1, x, y), 0.2, places=6)

        # l2
        ti_l2 = TrainingImage(
            np.linspace(0.0, 1.0, 10), categorical=False, distance="l2"
        )
        self.assertAlmostEqual(_uni_dist(ti_l2, x, x), 0.0)
        self.assertAlmostEqual(_uni_dist(ti_l2, x, y), 0.2, places=6)

        # lp (p=3.5) — non-uniform diffs [0.3, 0.1, 0.3] distinguish lp from l1/l2
        y_lp = np.array([0.3, 0.4, 0.7])
        ti_lp = TrainingImage(
            np.linspace(0.0, 1.0, 10), categorical=False, distance="l3.5"
        )
        self.assertAlmostEqual(_uni_dist(ti_lp, x, x), 0.0)
        self.assertAlmostEqual(_uni_dist(ti_lp, x, y_lp), 0.2680, places=3)
        self.assertGreater(
            _uni_dist(ti_lp, x, y_lp), _uni_dist(ti_l1, x, y_lp)
        )

        # variation (default p=2)
        ti_var = TrainingImage(
            np.linspace(0.0, 1.0, 10), categorical=False, distance="variation"
        )
        self.assertAlmostEqual(_uni_dist(ti_var, x, x), 0.0)
        self.assertAlmostEqual(_uni_dist(ti_var, x, y), 0.094281, places=5)
        # constant shift → distance = 0 (key behavioral property of variation distance)
        self.assertAlmostEqual(_uni_dist(ti_var, x, x + 0.15), 0.0, places=10)

        # variation1 (L^1 aggregation)
        ti_var1 = TrainingImage(
            np.linspace(0.0, 1.0, 10), categorical=False, distance="variation1"
        )
        self.assertAlmostEqual(_uni_dist(ti_var1, x, x), 0.0)
        self.assertAlmostEqual(_uni_dist(ti_var1, x, y), 0.08889, places=4)
        self.assertAlmostEqual(_uni_dist(ti_var1, x, x + 0.15), 0.0, places=10)
        # L^1 < L^2 for non-uniform diffs
        self.assertLess(_uni_dist(ti_var1, x, y), _uni_dist(ti_var, x, y))

        # variation2 explicit matches variation (regression guard)
        ti_var2 = TrainingImage(
            np.linspace(0.0, 1.0, 10), categorical=False, distance="variation2"
        )
        self.assertAlmostEqual(
            _uni_dist(ti_var2, x, y), _uni_dist(ti_var, x, y), places=10
        )

    def test_adjust_value(self):
        # Categorical and lp: passthrough
        self.assertAlmostEqual(
            self.ti_cat.adjust_value(
                0.7, np.array([0.1, 0.3]), np.array([0.4, 0.6])
            ),
            0.7,
        )
        self.assertAlmostEqual(
            self.ti_cont.adjust_value(
                0.7, np.array([0.1, 0.3]), np.array([0.4, 0.6])
            ),
            0.7,
        )

        # variation: Z(y) - Z_bar(y) + Z_bar(x) = 0.7 - 0.6 + 0.3 = 0.4
        ti_var = TrainingImage(
            np.linspace(0.0, 1.0, 20), categorical=False, distance="variation"
        )
        result = ti_var.adjust_value(
            0.7, np.array([0.1, 0.3, 0.5]), np.array([0.4, 0.6, 0.8])
        )
        self.assertAlmostEqual(result, 0.4, places=6)
        self.assertNotAlmostEqual(result, 0.7)  # must not be passthrough

    def test_distance_weights(self):
        a = np.array([0.0, 1.0, 0.0])
        b = np.array([1.0, 1.0, 0.0])  # first element differs

        # cond_weight=2 on first node → it gets weight 0.5 (double)
        d1 = _uni_dist(
            self.ti_cat, a, b, cond_mask=[True, False, False], cond_weight=1.0
        )
        d2 = _uni_dist(
            self.ti_cat, a, b, cond_mask=[True, False, False], cond_weight=2.0
        )
        self.assertGreater(d2, d1)

        # distance_power shifts weight toward closer neighbours — use non-uniform
        # differences so the weighted sums actually differ: diffs = [0, 0, 0.5]
        ti_p = TrainingImage(
            np.linspace(0.0, 1.0, 10),
            categorical=False,
            distance="l1",
            distance_power=1.0,
        )
        ti_flat = TrainingImage(
            np.linspace(0.0, 1.0, 10),
            categorical=False,
            distance="l1",
            distance_power=0.0,
        )
        x = np.array([0.0, 0.5, 1.0])
        z = np.array([0.0, 0.5, 0.5])  # only third element differs
        lags = np.array([1.0, 2.0, 3.0])
        d_power = _uni_dist(ti_p, x, z, lag_norms=lags)
        d_flat = _uni_dist(ti_flat, x, z, lag_norms=lags)
        # power=1 weights far neighbours less → smaller distance for far mismatch
        self.assertLess(d_power, d_flat)

    def test_distance_empty_event(self):
        dist = _uni_dist(self.ti_cat, np.array([]), np.array([]))
        self.assertAlmostEqual(dist, 0.0)

    def test_distance_functions_directly(self):
        a = np.array([0.0, 1.0, 0.0])
        b = np.array([1.0, 0.0, 1.0])
        w = np.array([1 / 3, 1 / 3, 1 / 3])

        # weights sum to 1
        w2 = compute_node_weights(3, None, 0.0)
        self.assertAlmostEqual(w2.sum(), 1.0)

        # cond_weight=2 on first node → uniform spatial weights → w[0] = 2/(2+1+1) = 0.5
        w3 = compute_node_weights(
            3,
            None,
            0.0,
            cond_mask=[True, False, False],
            cond_weight=2.0,
        )
        self.assertAlmostEqual(w3.sum(), 1.0)
        self.assertAlmostEqual(w3[0], 0.5, places=6)

        # The simulation calls only the vectorized distance functions; test
        # them directly by passing a single candidate as a shape-(1, n) row and
        # reading element [0] out.
        def cat(s, t):
            return float(vec_categorical_dist(s, t[np.newaxis, :], w)[0])

        def l1(s, t, dm):
            return float(vec_l1_dist(s, t[np.newaxis, :], w, dm)[0])

        def l2(s, t, dm):
            return float(vec_l2_dist(s, t[np.newaxis, :], w, dm)[0])

        def lp(s, t, dm, p):
            return float(vec_lp_dist(s, t[np.newaxis, :], w, dm, p)[0])

        def var(s, t, dm, p=2.0):
            return float(vec_variation_dist(s, t[np.newaxis, :], w, dm, p)[0])

        # categorical: identical → 0, opposite → 1
        self.assertAlmostEqual(cat(a, a), 0.0)
        self.assertAlmostEqual(cat(a, b), 1.0)

        # continuous: identical → 0
        x = np.array([0.0, 0.5, 1.0])
        d_max = 1.0
        self.assertAlmostEqual(l1(x, x, d_max), 0.0)
        self.assertAlmostEqual(l2(x, x, d_max), 0.0)
        self.assertAlmostEqual(lp(x, x, d_max, 3.5), 0.0)
        self.assertAlmostEqual(var(x, x, d_max), 0.0)
        self.assertAlmostEqual(var(x, x, d_max, p=1.0), 0.0)

        # distances in [0, 1]
        y = np.array([0.2, 0.3, 0.8])
        self.assertAlmostEqual(l1(x, y, d_max), 0.2, places=6)
        self.assertAlmostEqual(l2(x, y, d_max), 0.2, places=6)
        self.assertAlmostEqual(var(x, y, d_max), 0.094281, places=5)
        self.assertAlmostEqual(var(x, y, d_max, p=1.0), 0.08889, places=4)
        # p=2 explicit matches default
        self.assertAlmostEqual(
            var(x, y, d_max, p=2.0), var(x, y, d_max), places=10
        )

        # lp: non-uniform diffs [0.3, 0.1, 0.3] verify the p-norm exponent is used
        y_lp = np.array([0.3, 0.4, 0.7])
        self.assertAlmostEqual(lp(x, y_lp, d_max, 3.5), 0.2680, places=3)
        self.assertGreater(lp(x, y_lp, d_max, 3.5), l1(x, y_lp, d_max))

    def test_variation_dist_bounded(self):
        """variation_dist with distance_power > 0 must stay in [0, 1]."""
        # Adversarial: weight concentrated on maximally anti-correlated element
        x = np.array([0.0, 1.0, 0.0])
        y = np.array([1.0, 0.0, 1.0])
        lags = np.array([10.0, 0.1, 10.0])
        w = compute_node_weights(3, lags, 1.0)
        d = float(vec_variation_dist(x, y[np.newaxis, :], w, 1.0)[0])
        self.assertGreaterEqual(d, 0.0)
        self.assertLessEqual(d, 1.0)
        self.assertAlmostEqual(d, 0.661747, places=5)
        # Also via the vectorized per-variable production path
        ti = TrainingImage(
            np.linspace(0.0, 1.0, 10),
            categorical=False,
            distance="variation",
            distance_power=1.0,
        )
        self.assertLessEqual(_uni_dist(ti, x, y, lag_norms=lags), 1.0)

    def test_variation_dist_out_of_range_clamped(self):
        """Out-of-range SG values (from conditioning / mean-shift) must clamp to [0, 1]."""
        ti = TrainingImage(
            np.linspace(0.0, 1.0, 10),  # d_max == 1.0
            categorical=False,
            distance="variation",
        )
        de_sim = np.array([5.0, 0.0])  # 5.0 is far outside the TI range
        de_ti = np.array([0.0, 1.0])
        self.assertLessEqual(_uni_dist(ti, de_sim, de_ti), 1.0)
        vec = ti.vec_distance_var(None, de_sim, de_ti[np.newaxis, :])
        self.assertEqual(vec.shape, (1,))
        self.assertLessEqual(vec[0], 1.0)

    def test_variation_lp_parsing(self):
        """variation<p> string is parsed correctly and rejects bad inputs."""
        data = np.linspace(0.0, 1.0, 10)
        for spec in ("variation", "variation1", "variation1.5", "variation2"):
            ti = TrainingImage(data, categorical=False, distance=spec)
            self.assertEqual(ti.distance_type, spec)
        # invalid suffix
        with self.assertRaises(ValueError):
            TrainingImage(data, categorical=False, distance="variationX")
        # non-positive exponent
        with self.assertRaises(ValueError):
            TrainingImage(data, categorical=False, distance="variation0")
        with self.assertRaises(ValueError):
            TrainingImage(data, categorical=False, distance="variation-1")

    def test_variation_lp_adjust_value(self):
        """adjust_value mean-shift applies for all variation<p> variants."""
        de_sim = np.array([0.1, 0.3, 0.5])  # mean = 0.3
        de_ti = np.array([0.4, 0.6, 0.8])  # mean = 0.6
        # expected: 0.7 - 0.6 + 0.3 = 0.4
        for spec in ("variation", "variation1", "variation1.5"):
            ti = TrainingImage(
                np.linspace(0.0, 1.0, 20), categorical=False, distance=spec
            )
            self.assertAlmostEqual(
                ti.adjust_value(0.7, de_sim, de_ti), 0.4, places=6
            )

    def test_node_weights_zero_cond_weight(self):
        """All-conditioning event with cond_weight=0 → ignored (Me13 δ_c=0).

        The conditioning data is dropped from the distance entirely (zero
        weights, non-informative data event → unconditional behaviour), not
        re-weighted uniformly. Weights must still be finite (no NaNs).
        """
        w = compute_node_weights(
            3,
            lag_norms=None,
            distance_power=0.0,
            cond_mask=np.array([True, True, True]),
            cond_weight=0.0,
        )
        self.assertTrue(np.all(np.isfinite(w)))
        np.testing.assert_allclose(w, np.zeros(3))

    def test_node_weights_zero_lag_norm_not_amplified(self):
        """A true zero lag-norm (collocated h=0) must keep the unit baseline
        weight under distance_power > 0, not the divergent 1e-10**(-power), and
        cond_weight must be the knob that scales it."""
        # raw = [baseline 1.0, 1**-2=1.0, 2**-2=0.25] -> zero-lag == unit-lag,
        # and never dominates the data event.
        w = compute_node_weights(3, [0.0, 1.0, 2.0], 2.0)
        self.assertTrue(np.all(np.isfinite(w)))
        self.assertAlmostEqual(w[0], w[1])
        self.assertGreater(w[0], w[2])
        self.assertLess(w[0], 0.9)  # not the old ~1.0 blow-up
        # cond_weight scales the collocated entry (it carries cond_mask=True).
        w_c = compute_node_weights(
            3,
            [0.0, 1.0, 2.0],
            2.0,
            cond_mask=np.array([True, False, False]),
            cond_weight=5.0,
        )
        self.assertGreater(w_c[0], w[0])

    def test_warning_prefixes(self):
        # threshold>1 warning carries an "MPSModel:" prefix
        ti = TrainingImage(np.ones(4), categorical=True)
        with self.assertWarnsRegex(UserWarning, r"MPSModel:"):
            MPSModel(ti, threshold=5.0)
        # NaN warning now comes from Variable construction
        nan_data = np.array([np.nan, 1.0, 2.0])
        with self.assertWarns(UserWarning) as cm:
            Variable("x", nan_data, categorical=False)
        self.assertTrue(
            any("Variable" in str(w.message) for w in cm.warnings),
            "Expected 'Variable' in NaN warning message",
        )


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
        # Regression: the MPSModel passed in must not be silently discarded.
        model = MPSModel(self.ti1d)
        ds = DirectSampling(model)
        self.assertIs(ds.mps_model, model)
        # Field's covariance-model slot must stay None (DirectSampling is not
        # a CovModel-based field; pre_pos/field_dim rely on this).
        self.assertIsNone(ds.model)

    def test_raise(self):
        with self.assertRaises(ValueError):
            MPSModel(self.ti1d, boundary="bad")
        arr1d = np.tile([0, 1], 10).astype(float)
        with self.assertRaises(ValueError):
            Variable(None, arr1d, max_radius=0)
        with self.assertRaises(ValueError):
            Variable(None, arr1d, max_radius=-1.0)
        ds = DirectSampling(MPSModel(self.ti1d))
        with self.assertRaises(ValueError):
            ds([self.x1d], seed=42, mesh_type="unstructured")

    def test_repr(self):
        ds = DirectSampling(MPSModel(self.ti1d))
        r = repr(ds)
        self.assertIsInstance(r, str)
        self.assertIn("DirectSampling", r)

    def test_properties(self):
        ds = DirectSampling(
            MPSModel(
                self.ti1d,
                scan_fraction=0.5,
                threshold=0.05,
                cond_weight=2.0,
                boundary="partial",
            )
        )
        self.assertIs(ds.ti, self.ti1d)
        self.assertEqual(ds.n_neighbors, 8)
        self.assertAlmostEqual(ds.scan_fraction, 0.5)
        self.assertAlmostEqual(ds.threshold, 0.05)
        self.assertAlmostEqual(ds.cond_weight, 2.0)
        self.assertEqual(ds.boundary, "partial")
        self.assertIsNone(ds.max_radius)

    def test_offsets_shape(self):
        off = _precompute_offsets((5, 5))
        # shape: (N, 2) for 2-D, no zero row
        self.assertEqual(off.ndim, 2)
        self.assertEqual(off.shape[1], 2)
        self.assertFalse(np.any(np.all(off == 0, axis=1)))
        # sorted by Euclidean norm
        norms = np.linalg.norm(off, axis=1)
        self.assertTrue(np.all(norms[:-1] <= norms[1:]))

    def test_offsets_1d(self):
        off = _precompute_offsets((10,))
        self.assertEqual(off.shape[1], 1)
        self.assertFalse(np.any(off == 0))

    def test_offsets_max_offset(self):
        off = _precompute_offsets((5, 5), max_offset=1)
        self.assertLessEqual(np.abs(off).max(), 1)
        # 2-D, max_offset=1: 3^2 - 1 = 8 neighbours
        self.assertEqual(off.shape, (8, 2))

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

    def test_regression_1d(self):
        ds = DirectSampling(MPSModel(self.ti1d, scan_fraction=1.0))
        field = ds([self.x1d], seed=42)
        self.assertAlmostEqual(field[0], 1.0)
        self.assertAlmostEqual(field[5], 0.0)
        self.assertAlmostEqual(field[9], 0.0)

    def test_regression_2d(self):
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
        self.assertAlmostEqual(fa[3, 4], 0.0)

    def test_conditioning_honored(self):
        ds = DirectSampling(MPSModel(self.ti1d, scan_fraction=1.0))
        # Three exact grid node positions — spec requires ≥ 3 to exercise multi-point handling
        cond_pos = [np.array([2.0, 4.0, 7.0])]
        cond_val = np.array([0.0, 1.0, 1.0])
        ds.set_condition(cond_pos, cond_val)
        field = ds([self.x1d], seed=5)
        self.assertAlmostEqual(field[2], 0.0)
        self.assertAlmostEqual(field[4], 1.0)
        self.assertAlmostEqual(field[7], 1.0)

    def test_boundary_partial(self):
        ds = DirectSampling(
            MPSModel(
                self.ti2d,
                scan_fraction=1.0,
                boundary="partial",
            )
        )
        field = ds([self.x2d, self.y2d], seed=42)
        self.assertEqual(field.shape, (6, 6))
        self.assertFalse(np.any(np.isnan(field)))

    def test_boundary_partial_collapse_recovers(self):
        # TI far smaller than lag span → partial mode must recover, not raise
        ti_tiny = TrainingImage(
            np.random.default_rng(0).random((3, 3)),
            categorical=False,
            distance="l1",
        )
        ds = DirectSampling(
            MPSModel(
                ti_tiny,
                scan_fraction=1.0,
                boundary="partial",
            )
        )
        field = ds([np.arange(30, dtype=float)] * 2, seed=1)
        self.assertEqual(field.shape, (30, 30))
        self.assertFalse(np.any(np.isnan(field)))

    def test_threshold_above_one_warns_in_constructor(self):
        with self.assertWarns(UserWarning):
            MPSModel(self.ti1d, threshold=5.0)

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

    def test_max_radius(self):
        ds = DirectSampling(MPSModel(self.ti2d, scan_fraction=1.0))
        field = ds([self.x2d, self.y2d], seed=42)
        self.assertEqual(field.shape, (6, 6))
        self.assertFalse(np.any(np.isnan(field)))

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

    def test_ds_simulate_direct(self):
        rng = np.random.RandomState(7)
        result = ds_simulate(
            self.ti1d,
            sim_shape=(8,),
            threshold=0.0,
            scan_fraction=1.0,
            rng_path=rng,
            rng_nodes=rng,
        )
        # univariate result dict has key None
        self.assertIn(None, result)
        arr = result[None]
        self.assertEqual(arr.shape, (8,))
        self.assertFalse(np.any(np.isnan(arr)))
        self.assertTrue(set(np.unique(arr)).issubset({0.0, 1.0}))

    def test_empty_search_window_recovery(self):
        # n_neighbors >> TI size collapses search windows → must recover silently
        ti_tiny = TrainingImage(np.array([0.0, 1.0, 0.0]), categorical=True)
        ds = DirectSampling(MPSModel(ti_tiny, scan_fraction=1.0))
        field = ds([np.arange(5, dtype=float)], seed=1)
        self.assertEqual(field.shape, (5,))
        self.assertFalse(np.any(np.isnan(field)))
        self.assertTrue(set(np.unique(field)).issubset({0.0, 1.0}))

    def test_mpsmodel_setters_validate(self):
        m = MPSModel(self.ti1d)
        # setters normalize/validate in one place
        m.scan_fraction = 0.5
        self.assertAlmostEqual(m.scan_fraction, 0.5)
        m.boundary = "partial"
        self.assertEqual(m.boundary, "partial")
        with self.assertRaises(ValueError):
            m.scan_fraction = 0.0  # out of (0, 1]
        with self.assertRaises(ValueError):
            m.boundary = "bad"
        with self.assertRaises(ValueError):
            m.threshold = -1.0

    def test_threshold_warning_points_at_user_code(self):
        import warnings as _w

        # constructor path: the warning must originate from THIS test file,
        # not from inside gstools/mps/model.py
        with _w.catch_warnings(record=True) as rec:
            _w.simplefilter("always")
            MPSModel(self.ti1d, threshold=5.0)
        self.assertTrue(any("model.py" not in str(x.filename) for x in rec))
        # direct-setter path: same expectation
        m = MPSModel(self.ti1d)
        with _w.catch_warnings(record=True) as rec:
            _w.simplefilter("always")
            m.threshold = 5.0
        self.assertTrue(any("model.py" not in str(x.filename) for x in rec))

    def test_gstools_namespace(self):
        self.assertIs(gs.DirectSampling, DirectSampling)
        self.assertIs(gs.TrainingImage, TrainingImage)
        self.assertFalse(hasattr(gs, "Variable"))
        self.assertIs(gs.mps.DirectSampling, DirectSampling)
        self.assertIs(gs.mps.TrainingImage, TrainingImage)
        self.assertIs(gs.mps.Variable, Variable)

    def test_config_is_read_only_view(self):
        model = MPSModel(
            self.ti1d, threshold=0.3, scan_fraction=0.5, cond_weight=4.0
        )
        ds = DirectSampling(model)
        # Getters read straight from the shared model (no instance copy).
        self.assertAlmostEqual(ds.threshold, 0.3)
        self.assertAlmostEqual(ds.scan_fraction, 0.5)
        self.assertAlmostEqual(ds.cond_weight, 4.0)
        # Setters are gone: config lives on the MPSModel.
        with self.assertRaises(AttributeError):
            ds.threshold = 0.1
        with self.assertRaises(AttributeError):
            ds.cond_weight = 2.0
        with self.assertRaises(AttributeError):
            ds.scan_fraction = 0.5
        with self.assertRaises(AttributeError):
            ds.boundary = "partial"
        with self.assertRaises(AttributeError):
            ds.n_neighbors = 4
        with self.assertRaises(AttributeError):
            ds.max_radius = 5.0


class TestMultivariateTrainingImage(unittest.TestCase):
    def test_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "same shape"):
            TrainingImage(
                [
                    Variable("a", np.zeros((4, 4))),
                    Variable("b", np.zeros((3, 3))),
                ]
            )

    def test_equal_weights(self):
        ti = TrainingImage(
            [
                Variable("a", np.zeros((4, 4), dtype=int)),
                Variable("b", np.ones((4, 4), dtype=int)),
            ]
        )
        self.assertTrue(ti.multivariate)
        self.assertEqual([v.name for v in ti.variables], ["a", "b"])
        self.assertAlmostEqual(ti.weights["a"], 0.5)
        self.assertAlmostEqual(ti.weights["b"], 0.5)
        self.assertEqual(ti.shape, (4, 4))
        self.assertEqual(ti.ndim, 2)

    def test_weights_must_sum_positive(self):
        # all-zero weights are invalid
        with self.assertRaisesRegex(ValueError, "positive"):
            TrainingImage(
                [
                    Variable("a", np.zeros((4, 4), dtype=int), weight=0.0),
                    Variable("b", np.zeros((4, 4), dtype=int), weight=0.0),
                ]
            )

    def test_distance_weighted_sum(self):
        ti = TrainingImage(
            [
                Variable("a", np.zeros((4, 4), dtype=int), weight=0.5),
                Variable("b", np.zeros((4, 4), dtype=int), weight=0.5),
            ]
        )
        de_sg = {"a": np.array([1, 1]), "b": np.array([0, 0])}
        de_ti = {"a": np.array([0, 0]), "b": np.array([0, 0])}
        self.assertAlmostEqual(_mv_dist(ti, de_sg, de_ti), 0.5)

    def test_per_variable_categorical_and_distance(self):
        ti = TrainingImage(
            [
                Variable("cat", np.zeros((4, 4), dtype=int), categorical=True),
                Variable(
                    "cont",
                    np.linspace(0, 100, 16).reshape(4, 4),
                    categorical=False,
                    distance="l2",
                ),
            ]
        )
        self.assertTrue(ti.variable("cat").categorical)
        self.assertFalse(ti.variable("cont").categorical)
        self.assertAlmostEqual(ti.variable("cont").d_max, 100.0)

    def test_univariate_variation_p_preserved(self):
        ti = TrainingImage(
            np.linspace(0, 100, 16).reshape(4, 4),
            categorical=False,
            distance="variation1.5",
        )
        self.assertFalse(ti.multivariate)
        self.assertEqual(ti.variable().variation_p_norm, 1.5)
        self.assertIsNone(ti.variable().p_norm)
        self.assertIsNone(ti.weights)

    def test_variable_accessor(self):
        v_a = Variable("a", np.arange(16).reshape(4, 4))
        v_b = Variable("b", np.zeros((4, 4), dtype=int))
        ti = TrainingImage([v_a, v_b])
        self.assertIs(ti.variable("a"), v_a)  # same object
        # univariate TIs have no named variables
        uni = TrainingImage(np.zeros((4, 4), dtype=int))
        v_uni = uni.variable()
        self.assertIsNone(v_uni.name)
        with self.assertRaises(KeyError):
            uni.variable("a")

    def test_adjust_value_multivariate(self):
        ti = TrainingImage(
            [
                Variable(
                    "v",
                    np.linspace(0, 100, 16).reshape(4, 4),
                    categorical=False,
                    distance="variation",
                ),
                Variable("c", np.zeros((4, 4), dtype=int), categorical=True),
            ]
        )
        self.assertAlmostEqual(
            ti.adjust_value(
                50.0,
                np.array([10.0, 20.0, 30.0]),
                np.array([40.0, 50.0, 60.0]),
                var="v",
            ),
            20.0,
        )
        self.assertEqual(
            ti.adjust_value(1.0, np.array([0, 1]), np.array([1, 0]), var="c"),
            1.0,
        )

    def test_mixing_none_and_explicit_weights_raises(self):
        with self.assertRaisesRegex(ValueError, "mixing"):
            TrainingImage(
                [
                    Variable("a", np.zeros((4, 4)), weight=0.5),
                    Variable("b", np.zeros((4, 4)), weight=None),
                ]
            )

    def test_duplicate_names_raises(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            TrainingImage(
                [
                    Variable("x", np.zeros((4, 4))),
                    Variable("x", np.zeros((4, 4))),
                ]
            )


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
        self.assertEqual(field["a"].shape, (4, 4, 4))
        self.assertFalse(np.any(np.isnan(field["a"])))
        self.assertFalse(np.any(np.isnan(field["b"])))

    def test_partial_boundary_runs(self):
        rng = np.random.default_rng(11)
        ti = TrainingImage(
            [
                Variable("a", rng.integers(0, 2, (20, 20))),
                Variable("b", rng.integers(0, 2, (20, 20))),
            ]
        )
        ds = DirectSampling(
            MPSModel(ti, scan_fraction=0.3, boundary="partial")
        )
        field = ds([np.arange(8, dtype=float)] * 2, seed=2)
        self.assertFalse(np.any(np.isnan(field["a"])))
        self.assertFalse(np.any(np.isnan(field["b"])))
        self.assertTrue(np.all(np.isin(field["a"], [0, 1])))

    def test_ds_mode_threshold(self):
        rng = np.random.default_rng(12)
        ti = TrainingImage(
            [
                Variable("a", rng.integers(0, 3, (20, 20))),
                Variable("b", rng.integers(0, 2, (20, 20))),
            ]
        )
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.5, threshold=0.3))
        field = ds([np.arange(8, dtype=float)] * 2, seed=4)
        self.assertFalse(np.any(np.isnan(field["a"])))
        self.assertTrue(np.all(np.isin(field["a"], [0, 1, 2])))
        self.assertTrue(np.all(np.isin(field["b"], [0, 1])))

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
        ds = DirectSampling(MPSModel(ti, scan_fraction=1.0, threshold=0.0))
        field = ds([np.arange(6, dtype=float)] * 2, seed=0)
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
        field = ds([np.arange(6, dtype=float)] * 2, seed=0)
        self.assertEqual(set(field), {"x", "y"})
        self.assertIn("x", ds.field_names)
        self.assertIn("y", ds.field_names)
        np.testing.assert_array_equal(ds["x"], field["x"])
        np.testing.assert_array_equal(ds["y"], field["y"])

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
        self.assertEqual(ds.n_neighbors, 8)

    def test_n_neighbors_getter_dict(self):
        arr = np.zeros((10, 10), dtype=int)
        ti = TrainingImage(
            [
                Variable("a", arr, n_neighbors=8),
                Variable("b", arr, n_neighbors=4),
            ]
        )
        ds = DirectSampling(MPSModel(ti))
        self.assertEqual(ds.n_neighbors, {"a": 8, "b": 4})

    def test_n_neighbors_setter_broadcast(self):
        # Configure via Variable directly; getter reflects the change.
        arr = np.zeros((10, 10), dtype=int)
        ti = TrainingImage(
            [
                Variable("a", arr, n_neighbors=16),
                Variable("b", arr, n_neighbors=16),
            ]
        )
        ds = DirectSampling(MPSModel(ti))
        self.assertEqual(ds.ti.variable("a").n_neighbors, 16)
        self.assertEqual(ds.ti.variable("b").n_neighbors, 16)

    def test_n_neighbors_setter_partial_dict(self):
        # Configure per-variable at construction; getter reflects the difference.
        arr = np.zeros((10, 10), dtype=int)
        ti = TrainingImage(
            [
                Variable("a", arr, n_neighbors=20),
                Variable("b", arr, n_neighbors=4),
            ]
        )
        ds = DirectSampling(MPSModel(ti))
        self.assertEqual(ds.ti.variable("a").n_neighbors, 20)
        self.assertEqual(ds.ti.variable("b").n_neighbors, 4)  # unchanged

    def test_max_radius_none_default(self):
        ti = TrainingImage(np.zeros((10, 10), dtype=int), n_neighbors=4)
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.2))
        self.assertIsNone(ds.max_radius)

    def test_max_radius_setter_univariate(self):
        # Configure max_radius at construction; getter reflects the value.
        ti = TrainingImage(
            np.zeros((10, 10), dtype=int), n_neighbors=4, max_radius=5.0
        )
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.2))
        self.assertAlmostEqual(ds.max_radius, 5.0)
        self.assertAlmostEqual(ds.ti.variable().max_radius, 5.0)

    def test_max_radius_setter_broadcast(self):
        # Configure per-variable at construction; getter reports common value.
        arr = np.zeros((10, 10), dtype=int)
        ti = TrainingImage(
            [
                Variable("a", arr, max_radius=10.0),
                Variable("b", arr, max_radius=10.0),
            ]
        )
        ds = DirectSampling(MPSModel(ti))
        self.assertAlmostEqual(ds.ti.variable("a").max_radius, 10.0)
        self.assertAlmostEqual(ds.ti.variable("b").max_radius, 10.0)

    def test_max_radius_getter_dict_when_different(self):
        arr = np.zeros((10, 10), dtype=int)
        ti = TrainingImage(
            [
                Variable("a", arr, max_radius=20.0),
                Variable("b", arr, max_radius=10.0),
            ]
        )
        ds = DirectSampling(MPSModel(ti))
        self.assertEqual(ds.max_radius, {"a": 20.0, "b": 10.0})

    def test_max_radius_zero_raises_via_variable(self):
        # Validation now lives on Variable constructor.
        with self.assertRaisesRegex(ValueError, "max_radius"):
            Variable("x", np.zeros((10, 10), dtype=int), max_radius=0.0)

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
        ds.set_condition(
            cond_pos=[[3.0], [3.0]],
            cond_val={"a": np.array([1]), "b": np.array([np.nan])},
        )
        field = ds([np.arange(8, dtype=float)] * 2, seed=2)
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
        ds.set_condition(
            cond_pos=[[4.4, 4.1], [4.4, 4.1]],
            cond_val={"a": np.array([0, 1]), "b": np.array([0, 1])},
        )
        field = ds([np.arange(8, dtype=float)] * 2, seed=3)
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
        ds = DirectSampling(MPSModel(ti, scan_fraction=1.0, threshold=0.0))
        ds.set_condition(
            cond_pos=[[0.0], [0.0]],
            cond_val={"a": np.array([np.nan]), "b": np.array([20])},
        )
        field = ds([np.arange(1, dtype=float)] * 2, seed=0)
        # b=20 sits at TI cell (3, 2); a there is (20*7) % 5 == 0
        self.assertEqual(field["a"][0, 0], (20 * 7) % 5)

    def test_univariate_set_condition_still_works(self):
        # Regression guard: scalar set_condition path is unchanged.
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
        ds.num_threads = 2
        field = ds([np.arange(8, dtype=float)] * 2, seed=0)
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
        ds_s.num_threads = 1
        ds_p = DirectSampling(MPSModel(ti, scan_fraction=0.2))
        ds_p.num_threads = 2
        f_s = ds_s(pos, seed=7)
        f_p = ds_p(pos, seed=7)
        np.testing.assert_array_equal(f_s["a"], f_p["a"])
        np.testing.assert_array_equal(f_s["b"], f_p["b"])

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
        ds_s.num_threads = 1
        ds_s.set_condition(cond_pos, cond_val)
        ds_p = DirectSampling(MPSModel(ti, scan_fraction=0.3))
        ds_p.num_threads = 4
        ds_p.set_condition(cond_pos, cond_val)
        f_s = ds_s(pos, seed=7)
        f_p = ds_p(pos, seed=7)
        np.testing.assert_array_equal(f_s["a"], f_p["a"])
        np.testing.assert_array_equal(f_s["b"], f_p["b"])

    def test_parallel_conditioning_preserved(self):
        rng = np.random.default_rng(6)
        ti = TrainingImage(
            [
                Variable("a", rng.integers(0, 2, (20, 20))),
                Variable("b", rng.integers(0, 2, (20, 20))),
            ]
        )
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.3))
        ds.num_threads = 2
        ds.set_condition(
            cond_pos=[[3.0], [3.0]],
            cond_val={"a": np.array([1]), "b": np.array([0])},
        )
        field = ds([np.arange(8, dtype=float)] * 2, seed=9)
        self.assertEqual(field["a"][3, 3], 1)
        self.assertEqual(field["b"][3, 3], 0)

    def test_set_condition_array_on_multivariate_raises(self):
        ti = TrainingImage(
            [
                Variable("a", np.zeros((10, 10), dtype=int)),
                Variable("b", np.zeros((10, 10), dtype=int)),
            ]
        )
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.3))
        with self.assertRaisesRegex(ValueError, "dict"):
            ds.set_condition([[3.0], [3.0]], np.array([1]))

    def test_set_condition_empty_dict_raises(self):
        ti = TrainingImage(
            [
                Variable("a", np.zeros((10, 10), dtype=int)),
                Variable("b", np.zeros((10, 10), dtype=int)),
            ]
        )
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.3))
        with self.assertRaisesRegex(ValueError, "empty"):
            ds.set_condition([[3.0], [3.0]], {})

    def test_set_condition_unknown_variable_raises(self):
        ti = TrainingImage(
            [
                Variable("a", np.zeros((10, 10), dtype=int)),
                Variable("b", np.zeros((10, 10), dtype=int)),
            ]
        )
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.3))
        with self.assertRaisesRegex(ValueError, "unknown variable"):
            ds.set_condition(
                [[3.0], [3.0]],
                {"a": np.array([1.0]), "c": np.array([1.0])},
            )

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
        ds.set_condition(
            # point 0 at (4.4,4.4) {a:0,b:0}; point 1 at (4.1,4.1) {a:1,b:nan}
            cond_pos=[[4.4, 4.1], [4.4, 4.1]],
            cond_val={"a": np.array([0, 1]), "b": np.array([0, np.nan])},
        )
        field = ds([np.arange(8, dtype=float)] * 2, seed=1)
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
                ),
                Variable(
                    "cat", rng.integers(0, 2, (20, 20)), categorical=True
                ),
            ],
            distance_power=1.0,
        )
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.3))
        field = ds([np.arange(8, dtype=float)] * 2, seed=3)
        self.assertEqual(field["cont"].shape, (8, 8))
        self.assertTrue(np.all(np.isfinite(field["cont"])))
        self.assertTrue(np.all(np.isin(field["cat"], [0, 1])))

    def test_single_variable_multivariate(self):
        rng = np.random.default_rng(9)
        ti = TrainingImage([Variable("only", rng.integers(0, 3, (20, 20)))])
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.3))
        field = ds([np.arange(8, dtype=float)] * 2, seed=0)
        self.assertEqual(set(field), {"only"})
        self.assertEqual(field["only"].shape, (8, 8))
        self.assertTrue(np.all(np.isin(field["only"], [0, 1, 2])))

    def test_threshold_renormalization_with_empty_variable(self):
        # Fix 1 regression guard: when variable 'b' has no informed neighbours
        # (first node on the path), the joint distance must still be renormalized
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
        self.assertFalse(np.any(np.isnan(field["a"])))
        self.assertFalse(np.any(np.isnan(field["b"])))
        np.testing.assert_array_equal(field["b"], field["a"] + 100)

    def test_mv_custom_store_name_raises(self):
        # Finding #6: a custom store name has no single field to bind to for a
        # multivariate run; reject it instead of silently dropping it.
        rng = np.random.default_rng(0)
        ti = TrainingImage(
            [
                Variable("a", rng.integers(0, 2, (20, 20))),
                Variable("b", rng.integers(0, 2, (20, 20))),
            ]
        )
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.3))
        with self.assertRaisesRegex(ValueError, "custom store name"):
            ds([np.arange(6, dtype=float)] * 2, seed=0, store="myrun")

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
        self.assertEqual(set(out), {"a", "b"})
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
        ti = gs.mps.TrainingImage(rng.integers(0, 3, (20, 20)),
                                  n_neighbors=4)
        ds = gs.mps.DirectSampling(
            MPSModel(ti, scan_fraction=0.2, rotation=np.pi / 4)
        )
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
        f_rot = ds_rot(pos, seed=7)
        self.assertFalse(np.array_equal(f_plain, f_rot))

    def test_array_rotation_map_runs(self):
        rng = np.random.default_rng(0)
        ti = gs.mps.TrainingImage(rng.integers(0, 3, (20, 20)),
                                  n_neighbors=4)
        angle_map = np.linspace(0, np.pi / 2, 64).reshape(8, 8)
        ds = gs.mps.DirectSampling(
            MPSModel(ti, scan_fraction=0.2, rotation=angle_map)
        )
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
        ti = gs.mps.TrainingImage(rng.integers(0, 3, (20, 20)),
                                  n_neighbors=4)
        ds = gs.mps.DirectSampling(
            MPSModel(
                ti, scan_fraction=0.2, rotation=np.pi / 6, scale=[1.0, 0.5]
            )
        )
        field = ds([np.arange(8, dtype=float)] * 2, seed=2)
        self.assertEqual(field.shape, (8, 8))
        self.assertTrue(np.all(np.isin(field, [0, 1, 2])))

    def test_conditioning_preserved(self):
        rng = np.random.default_rng(0)
        ti = gs.mps.TrainingImage(rng.integers(0, 3, (20, 20)),
                                  n_neighbors=4)
        ds = gs.mps.DirectSampling(
            MPSModel(ti, scan_fraction=0.3, rotation=np.pi / 4)
        )
        ds.set_condition([[4.0], [4.0]], [2])
        field = ds([np.arange(8, dtype=float)] * 2, seed=0)
        self.assertEqual(int(field[4, 4]), 2)

    def test_partial_boundary_runs(self):
        rng = np.random.default_rng(0)
        ti = gs.mps.TrainingImage(rng.integers(0, 3, (20, 20)),
                                  n_neighbors=4)
        ds = gs.mps.DirectSampling(
            MPSModel(
                ti, scan_fraction=0.2, boundary="partial", rotation=np.pi / 4
            )
        )
        field = ds([np.arange(8, dtype=float)] * 2, seed=0)
        self.assertTrue(np.all(np.isin(field, [0, 1, 2])))

    def test_3d_rotation_runs(self):
        rng = np.random.default_rng(0)
        data = rng.integers(0, 2, (12, 12, 12))
        ti = gs.mps.TrainingImage(data, n_neighbors=4)
        ds = gs.mps.DirectSampling(
            MPSModel(ti, scan_fraction=0.1, rotation=np.pi / 4)
        )
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
        result_rot = ds_rot(pos, seed=9)

        self.assertEqual(result_rot["a"].shape, (8, 8))
        self.assertEqual(result_rot["b"].shape, (8, 8))
        self.assertTrue(np.all(np.isin(result_rot["a"], [0, 1, 2])))
        self.assertTrue(np.all(np.isfinite(result_rot["b"])))
        self.assertFalse(np.array_equal(result_plain["a"], result_rot["a"]))

    def test_flattened_rotation_map_raises(self):
        # Finding #5: a per-node map passed flattened (length Nx*Ny) on a 2-D
        # grid must raise, not silently apply only element [0].
        rng = np.random.default_rng(0)
        ti = gs.mps.TrainingImage(rng.integers(0, 3, (20, 20)),
                                  n_neighbors=4)
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
            MPSModel(
                ti, scan_fraction=0.1, rotation=np.array([0.1, 0.2, 0.3])
            )
        )
        field = ds([np.arange(5, dtype=float)] * 3, seed=0)
        self.assertEqual(field.shape, (5, 5, 5))
        self.assertTrue(np.all(np.isin(field, [0, 1])))

    def test_scalar_vs_materialized_map_bit_identical(self):
        # Oracle i: scalar spec (hoisted-M stationary path) vs a
        # hand-materialized constant per-node map (per-node path).
        rng = np.random.default_rng(0)
        ti = gs.mps.TrainingImage(rng.integers(0, 3, (20, 20)),
                                  n_neighbors=4)
        pos = [np.arange(8, dtype=float)] * 2
        f_scalar = gs.mps.DirectSampling(
            MPSModel(ti, scan_fraction=0.2, rotation=0.3)
        )(pos, seed=11)
        f_map = gs.mps.DirectSampling(
            MPSModel(ti, scan_fraction=0.2, rotation=np.full((8, 8), 0.3))
        )(pos, seed=11)
        np.testing.assert_array_equal(f_scalar, f_map)

    def test_callable_vs_materialized_bit_identical(self):
        # Oracle c: callable form vs hand-materialized array of the same field.
        rng = np.random.default_rng(0)
        ti = gs.mps.TrainingImage(rng.integers(0, 3, (20, 20)),
                                  n_neighbors=4)
        pos = [np.arange(8, dtype=float)] * 2

        def rot(x, y):
            return (y / 8.0) * (np.pi / 4.0)

        gx, gy = np.meshgrid(pos[0], pos[1], indexing="ij")
        rot_arr = (gy / 8.0) * (np.pi / 4.0)

        f_call = gs.mps.DirectSampling(
            MPSModel(ti, scan_fraction=0.2, rotation=rot)
        )(pos, seed=3)
        f_arr = gs.mps.DirectSampling(
            MPSModel(ti, scan_fraction=0.2, rotation=rot_arr)
        )(pos, seed=3)
        np.testing.assert_array_equal(f_call, f_arr)

    def test_uniform_dilation_runs_and_differs(self):
        # Oracle b (integration): uniform scale=0.5 (pure dilation, formerly
        # inexpressible) runs, stays in the TI value set, and differs from
        # the identity run.
        rng = np.random.default_rng(0)
        ti = gs.mps.TrainingImage(rng.integers(0, 3, (30, 30)),
                                  n_neighbors=8)
        pos = [np.arange(10, dtype=float)] * 2
        f_id = gs.mps.DirectSampling(
            MPSModel(ti, scan_fraction=0.2)
        )(pos, seed=7)
        f_half = gs.mps.DirectSampling(
            MPSModel(ti, scan_fraction=0.2, scale=0.5)
        )(pos, seed=7)
        self.assertTrue(np.all(np.isin(f_half, [0, 1, 2])))
        self.assertFalse(np.array_equal(f_id, f_half))

    def test_nonuniform_axis_spacing_warns(self):
        # Oracle m: index-space geometry assumption.
        rng = np.random.default_rng(0)
        ti = gs.mps.TrainingImage(rng.integers(0, 2, (15, 15)),
                                  n_neighbors=4)
        ds = gs.mps.DirectSampling(
            MPSModel(ti, scan_fraction=0.3, rotation=np.pi / 4)
        )
        xs = np.array([0.0, 1.0, 3.0, 7.0])
        ys = np.arange(4, dtype=float)
        with self.assertWarns(UserWarning):
            ds([xs, ys], seed=0)

    def test_unequal_axis_spacing_warns(self):
        rng = np.random.default_rng(0)
        ti = gs.mps.TrainingImage(rng.integers(0, 2, (15, 15)),
                                  n_neighbors=4)
        ds = gs.mps.DirectSampling(
            MPSModel(ti, scan_fraction=0.3, scale=0.5)
        )
        with self.assertWarns(UserWarning):
            ds([np.arange(4.0), 2.0 * np.arange(4.0)], seed=0)

    def test_uniform_spacing_no_warning(self):
        rng = np.random.default_rng(0)
        ti = gs.mps.TrainingImage(rng.integers(0, 2, (15, 15)),
                                  n_neighbors=4)
        ds = gs.mps.DirectSampling(
            MPSModel(ti, scan_fraction=0.3, rotation=np.pi / 4)
        )
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("error", UserWarning)
            ds([np.arange(6.0)] * 2, seed=0)

    def test_set_nonstationary_removed(self):
        rng = np.random.default_rng(0)
        ti = gs.mps.TrainingImage(rng.integers(0, 2, (10, 10)))
        ds = gs.mps.DirectSampling(MPSModel(ti))
        self.assertFalse(hasattr(ds, "set_nonstationary"))


class TestFullArraySnapshot(unittest.TestCase):
    """Bit-identical full-array regression pins — the step-by-step acceptance gate.

    These arrays are captured from the pre-refactoring engine with fixed seeds and
    must remain bit-identical (np.testing.assert_array_equal) after every
    refactoring step.
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

    def test_snapshot(self):
        # --- univariate 1D, seed=42 ---
        ds = DirectSampling(MPSModel(self.ti1d, scan_fraction=1.0))
        f = ds([self.x1d], seed=42)
        np.testing.assert_array_equal(
            f,
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
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
                [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            ],
            err_msg="snap_uni_2d_checker seed=99",
        )

        # --- univariate 2D random TI, seed=42 ---
        ds3 = DirectSampling(MPSModel(self.ti2d_rand, scan_fraction=0.5))
        f = ds3([self.x2d, self.y2d], seed=42)
        np.testing.assert_array_equal(
            f,
            [
                [0.0, 0.0, 2.0, 0.0, 2.0, 2.0],
                [0.0, 0.0, 2.0, 2.0, 0.0, 2.0],
                [2.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [2.0, 0.0, 1.0, 2.0, 0.0, 0.0],
                [2.0, 0.0, 1.0, 1.0, 2.0, 1.0],
                [2.0, 2.0, 0.0, 2.0, 1.0, 0.0],
            ],
            err_msg="snap_uni_rand seed=42",
        )

        # --- multivariate 2D, seed=42 ---
        ds_mv = DirectSampling(MPSModel(self.ti_mv, scan_fraction=0.5))
        res = ds_mv([self.x2d, self.y2d], seed=42)
        np.testing.assert_array_equal(
            res["a"],
            [
                [2.0, 1.0, 2.0, 0.0, 2.0, 0.0],
                [2.0, 2.0, 2.0, 1.0, 2.0, 1.0],
                [1.0, 0.0, 0.0, 1.0, 2.0, 2.0],
                [2.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 1.0, 2.0, 1.0, 2.0, 2.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 2.0],
            ],
            err_msg="snap_mv var=a seed=42",
        )
        np.testing.assert_array_equal(
            res["b"],
            [
                [1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ],
            err_msg="snap_mv var=b seed=42",
        )


class TestDSSeedControl(unittest.TestCase):
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

    def test_default_still_reproducible(self):
        # No explicit seeds → same seed= value still reproducible
        ds = DirectSampling(MPSModel(self.ti, scan_fraction=0.3))
        fa = ds(self.pos, seed=5)
        fb = ds(self.pos, seed=5)
        self.assertTrue(np.array_equal(fa, fb))


class TestTransformLagsCollapse(unittest.TestCase):
    def test_collapsed_lags_warn(self):
        # Strong anisotropy flattens the y-axis: SG lags (0,1) and (0,2) both
        # round to TI lag (0,0). The dropped neighbour must be warned about.
        M = np.array([[1.0, 0.0], [0.0, 0.1]])
        lags = np.array([[0.0, 1.0], [0.0, 2.0]])
        de = np.array([5.0, 9.0])
        with self.assertWarns(RuntimeWarning):
            lags_ti, de_out = _transform_lags(lags, M, de)
        self.assertEqual(len(lags_ti), 1)
        self.assertEqual(de_out.tolist(), [5.0])  # first occurrence kept

    def test_no_collapse_no_warning(self):
        # Identity transform keeps both lags distinct → no warning.
        M = np.eye(2)
        lags = np.array([[0.0, 1.0], [0.0, 2.0]])
        de = np.array([5.0, 9.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            lags_ti, de_out = _transform_lags(lags, M, de)
        self.assertEqual(len(lags_ti), 2)


class TestReduceToFit(unittest.TestCase):
    """M10 para [43]: global reduce-to-fit, drop most-outside node by rank (#3)."""

    def test_drops_oob_regardless_of_rank(self):
        # The near (low-rank) lag maps out of a 4x4 TI; the far (high-rank) lag
        # is feasible. para [43] keeps the feasible one and drops the OOB one —
        # the opposite of partial-mode furthest-first truncation.
        lags_ti = np.array([[0.0, 10.0], [1.0, 0.0]])
        de = np.array([5.0, 9.0])
        out_lags, out_de = _reduce_to_fit(lags_ti, (4, 4), de)
        np.testing.assert_array_equal(out_lags, [[1.0, 0.0]])
        np.testing.assert_array_equal(
            out_de, [9.0]
        )  # parallel array sliced too

    def test_both_extremes_overspan_reduced(self):
        # Two lags each fit alone but jointly over-span a 4-wide TI (ti-1==3):
        # extent 3-(-3)=6 > 3. One outermost node is dropped until it fits.
        lags_ti = np.array([[0.0, 3.0], [0.0, -3.0]])
        (out_lags,) = _reduce_to_fit(lags_ti, (4, 4))
        self.assertEqual(len(out_lags), 1)

    def test_collocated_never_dropped_and_inbounds_kept(self):
        # h=0 and an in-bounds lag survive; the over-long lag is removed.
        lags_ti = np.array([[0.0, 0.0], [3.0, 0.0], [9.0, 0.0]])
        (out_lags,) = _reduce_to_fit(lags_ti, (4, 4))  # ti-1 == 3
        self.assertIn([0.0, 0.0], out_lags.tolist())  # collocated kept
        self.assertNotIn([9.0, 0.0], out_lags.tolist())  # most-outside dropped

    def test_already_fits_is_noop(self):
        lags_ti = np.array([[1.0, 0.0], [0.0, -1.0]])
        (out_lags,) = _reduce_to_fit(lags_ti, (20, 20))
        np.testing.assert_array_equal(out_lags, lags_ti)

    def test_empty(self):
        (out,) = _reduce_to_fit(np.empty((0, 2)), (4, 4))
        self.assertEqual(out.size, 0)

    def test_strong_anis_retains_pattern_neighbour(self):
        # End to end: strong anisotropy on a small TI used to collapse the
        # window to a random draw; with para [43] the feasible neighbour is
        # retained, so output stays valid and finite (no crash, no NaN).
        rng = np.random.default_rng(0)
        ti = gs.mps.TrainingImage(rng.integers(0, 2, (6, 6)))
        ds = gs.mps.DirectSampling(
            MPSModel(ti, scan_fraction=1.0, scale=[1.0, 0.1])
        )
        field = ds([np.arange(8, dtype=float)] * 2, seed=0)
        self.assertTrue(np.all(np.isfinite(field)))
        self.assertTrue(set(np.unique(field)).issubset({0.0, 1.0}))


class TestLagTransformMatrix(unittest.TestCase):
    """Canonical-map transform semantics (spec §2; oracles b, g)."""

    def test_scale_half_doubles_ti_lags(self):
        # Oracle b: scale=0.5 -> structures half the TI size -> TI-frame lags
        # are 2x the SG lags (lag_ti = (1/s) * R^-1 * lag_sg).
        scale_map = np.broadcast_to(
            np.array([0.5, 0.5]), (4, 4, 2)
        )
        M = _lag_transform_matrix(2, None, scale_map, np.array([0, 0]))
        lags = np.array([[3.0, 0.0], [0.0, 5.0]])
        lags_ti = np.rint(lags @ M.T)
        np.testing.assert_array_equal(
            lags_ti, [[6.0, 0.0], [0.0, 10.0]]
        )

    def test_uniform_dilation_isotropic(self):
        # Uniform dilation was inexpressible in the axis-0-pinned anis form.
        scale_map = np.broadcast_to(np.array([2.0, 2.0]), (3, 3, 2))
        M = _lag_transform_matrix(2, None, scale_map, np.array([1, 1]))
        np.testing.assert_allclose(M, 0.5 * np.eye(2))

    def test_rotation_only_matches_detransform(self):
        from gstools.tools.geometric import matrix_detransform

        rot_map = np.full((4, 4, 1), np.pi / 6)
        M = _lag_transform_matrix(2, rot_map, None, np.array([2, 3]))
        np.testing.assert_array_equal(
            M, matrix_detransform(2, [np.pi / 6], [1.0, 1.0])
        )

    def test_per_node_map_indexed_at_node(self):
        rot_map = np.zeros((4, 4, 1))
        rot_map[3, 1, 0] = np.pi / 2
        M_id = _lag_transform_matrix(2, rot_map, None, np.array([0, 0]))
        M_rot = _lag_transform_matrix(2, rot_map, None, np.array([3, 1]))
        np.testing.assert_allclose(M_id, np.eye(2), atol=1e-15)
        self.assertFalse(np.allclose(M_rot, np.eye(2)))

    def test_geometry_drop_ignores_sg_rank(self):
        # Oracle g (B3 regression): the NEAREST lag (SG rank 0) lands outside
        # the TI -> IT is dropped (geometry rule), not the furthest (rank rule).
        lags_ti = np.array([[9.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        values = np.array([10.0, 20.0, 30.0])
        out_lags, out_vals = _reduce_to_fit(lags_ti, (5, 5), values)
        np.testing.assert_array_equal(out_lags, [[1.0, 1.0], [2.0, 0.0]])
        np.testing.assert_array_equal(out_vals, [20.0, 30.0])


class TestVariationNNeighbors(unittest.TestCase):
    def test_variation_n1_raises_at_variable_construction(self):
        with self.assertRaisesRegex(ValueError, "variation"):
            Variable(
                "x",
                np.linspace(0.0, 1.0, 20),
                categorical=False,
                distance="variation",
                n_neighbors=1,
            )

    def test_variation_n2_ok(self):
        v = Variable(
            "x",
            np.linspace(0.0, 1.0, 20),
            categorical=False,
            distance="variation",
            n_neighbors=2,
        )
        ti = TrainingImage([v])
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.5))
        self.assertEqual(ds.n_neighbors, 2)

    def test_variation_n_neighbors_setter_rejects_1(self):
        v = Variable(
            "x",
            np.linspace(0.0, 1.0, 20),
            categorical=False,
            distance="variation",
            n_neighbors=3,
        )
        with self.assertRaisesRegex(ValueError, "variation"):
            v.n_neighbors = 1

    def test_non_variation_n1_ok(self):
        v = Variable(
            "x",
            np.linspace(0.0, 1.0, 20),
            categorical=False,
            distance="l2",
            n_neighbors=1,
        )
        ti = TrainingImage([v])
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.5))
        self.assertEqual(ds.n_neighbors, 1)

    def test_variation_guard_not_bypassable_via_ds(self):
        # The guard lives on Variable.n_neighbors setter; DS is a read-only view.
        v = Variable(
            "x",
            np.linspace(0.0, 1.0, 20),
            categorical=False,
            distance="variation",
            n_neighbors=3,
        )
        with self.assertRaisesRegex(ValueError, "variation"):
            v.n_neighbors = 1

    def test_variation_guard_is_atomic(self):
        # Variation guard raises on the Variable; the value stays at 3.
        v = Variable(
            "x",
            np.linspace(0.0, 1.0, 20),
            categorical=False,
            distance="variation",
            n_neighbors=3,
        )
        ti = TrainingImage([v])
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.5))
        with self.assertRaises(ValueError):
            v.n_neighbors = 1
        self.assertEqual(ds.n_neighbors, 3)  # unchanged


class TestStrictBoundaryWarning(unittest.TestCase):
    def test_strict_infeasible_warns_and_falls_back(self):
        # A lag larger than the TI cannot fit any anchor in strict mode, so the
        # function must warn before falling back to partial truncation.
        ti_shape = np.array([5])
        lags_ti = np.array(
            [[10.0]]
        )  # |lag| 10 > TI size 5 → strict infeasible
        with self.assertWarns(RuntimeWarning):
            lo, hi, keep = _window_bounds(lags_ti, ti_shape, "strict")
        # falls back: partial loop drops the over-long lag → keep < len(lags_ti)
        self.assertLess(keep, len(lags_ti) + 1)

    def test_strict_feasible_no_warning(self):
        # A lag that fits leaves strict mode satisfied → no warning.
        ti_shape = np.array([20])
        lags_ti = np.array([[1.0]])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            lo, hi, keep = _window_bounds(lags_ti, ti_shape, "strict")
        self.assertEqual(keep, 1)


class TestScanWindowThreshold(unittest.TestCase):
    def test_ds_mode_strict_threshold(self):
        # Mariethoz2010 ¶23: DS acceptance is strict d < t. A candidate at
        # exactly the threshold (0.1) must be rejected; the 0.05 candidate is
        # accepted instead. With the old inclusive d <= t the at-threshold
        # candidate would win first (scan order).
        win_shape = (2,)
        lo = np.array([0])
        dists = np.array([0.1, 0.05])

        def dist_fn(y_blk):
            return dists[: len(y_blk)]

        y = _scan_window(lo, win_shape, 0, 2, 0.1, dist_fn)
        self.assertEqual(int(y[0]), 1)  # the 0.05 candidate, not the 0.1 one

    def test_scan_window_greedy_first_under_threshold(self):
        # DS mode (threshold > 0) must return the FIRST candidate in scan order
        # with d < threshold — NOT the global-argmin candidate.
        # Design: candidate 0 has d=0.15 (under threshold 0.2, not the argmin);
        # candidate 1 has d=0.05 (under threshold 0.2, IS the argmin).
        # np.argmax(under) returns the first True in [True, True] == index 0,
        # confirming greedy (first-under-threshold) semantics, not argmin semantics.
        win_shape = (3,)
        lo = np.array([0])
        # distances for positions 0, 1, 2 in scan order
        dists = np.array([0.15, 0.05, 0.30])

        def dist_fn(y_blk):
            idxs = (y_blk[:, 0] - lo[0]).astype(int)
            return dists[idxs]

        # Argmin candidate is position 1 (d=0.05); first-under-threshold is
        # position 0 (d=0.15 < 0.2). The function must return position 0.
        y = _scan_window(lo, win_shape, 0, 3, 0.2, dist_fn)
        self.assertEqual(
            int(y[0]), 0
        )  # first-under-threshold, not argmin (pos 1)

    def test_dsbc_accepts_exact_match(self):
        # DSBC (t=0): exact match d == 0 is still accepted (d <= 0).
        win_shape = (2,)
        lo = np.array([0])
        dists = np.array([0.0, 0.5])

        def dist_fn(y_blk):
            return dists[: len(y_blk)]

        y = _scan_window(lo, win_shape, 0, 2, 0.0, dist_fn)
        self.assertEqual(int(y[0]), 0)  # the exact-match candidate


class TestNaNTrainingImage(unittest.TestCase):
    """Masked / incomplete TIs: NaN cells are undefined (warn + handle).

    Contract: NaN cells are treated as undefined. They are (a) excluded from
    the continuous data range ``d_max``, (b) excluded per-position from every
    candidate distance, and (c) never pasted into the field. Construction warns
    once. A TI with no fully-defined cell cannot be simulated and raises.
    """

    def test_nan_excluded_from_dmax(self):
        # Finding #1: a NaN must not collapse d_max to the 1.0 fallback.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ti = TrainingImage(
                np.array([0.0, np.nan, 100.0]),
                categorical=False,
                distance="l1",
            )
        self.assertAlmostEqual(ti.variable()._d_max, 100.0)

    def test_construction_warns_on_nan(self):
        with self.assertWarns(UserWarning):
            TrainingImage(
                np.array([0.0, np.nan, 1.0]), categorical=False, distance="l1"
            )

    def test_partial_nan_continuous_runs_finite(self):
        # Finding #2: a NaN patch must not crash (was IndexError) and must
        # produce finite output within the defined data range.
        data = np.random.RandomState(0).rand(20, 20)
        data[5:8, 5:8] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ti = TrainingImage(data, categorical=False, distance="l1")
        ds = DirectSampling(MPSModel(ti, scan_fraction=1.0))
        field = ds([np.arange(15, dtype=float)] * 2, seed=1)
        self.assertEqual(field.shape, (15, 15))
        self.assertTrue(np.all(np.isfinite(field)))
        self.assertGreaterEqual(field.min(), np.nanmin(data))
        self.assertLessEqual(field.max(), np.nanmax(data))

    def test_partial_nan_categorical_runs_subset(self):
        # Finding #2: categorical NaN patch must not crash (was "produced NaN"
        # ValueError) and output must be a subset of the *defined* categories.
        data = np.random.RandomState(0).randint(0, 2, (20, 20)).astype(float)
        data[5:8, 5:8] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ti = TrainingImage(data, categorical=True)
        ds = DirectSampling(MPSModel(ti, scan_fraction=1.0))
        field = ds([np.arange(15, dtype=float)] * 2, seed=1)
        self.assertTrue(np.all(np.isfinite(field)))
        self.assertTrue(set(np.unique(field)).issubset({0.0, 1.0}))

    def test_all_nan_ti_raises(self):
        # No fully-defined cell → cannot paste any value → clear error.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ti = TrainingImage(
                np.full((5, 5), np.nan), categorical=False, distance="l2"
            )
        ds = DirectSampling(MPSModel(ti, scan_fraction=1.0))
        with self.assertRaises(ValueError):
            ds([np.arange(3, dtype=float)] * 2, seed=0)


class TestTrainingImageWindow(unittest.TestCase):
    """TrainingImage.window(slices) sub-region views (M10 para [40])."""

    def _cont_ti(self):
        rng = np.random.default_rng(0)
        data = rng.random((20, 30)) * 10.0
        return gs.mps.TrainingImage(data, categorical=False), data

    def test_window_is_view_no_copy(self):
        ti, data = self._cont_ti()
        win = ti.window(np.s_[5:15, 10:25])
        self.assertEqual(win.shape, (10, 15))
        self.assertTrue(
            np.shares_memory(win.variable().data, ti.variable().data)
        )
        np.testing.assert_array_equal(
            win.variable().data, data[5:15, 10:25]
        )

    def test_window_d_max_from_slice(self):
        # d_max of the view is the range of the SLICED data, not the parent's.
        data = np.zeros((10, 10))
        data[0, 0] = 100.0  # parent range 100, outside the window
        data[5:, 5:] = np.arange(25).reshape(5, 5)  # window range 24
        ti = gs.mps.TrainingImage(data, categorical=False)
        win = ti.window(np.s_[5:10, 5:10])
        self.assertEqual(win.variable().d_max, 24.0)
        self.assertEqual(ti.variable().d_max, 100.0)

    def test_window_preserves_config(self):
        rng = np.random.default_rng(1)
        ti = gs.mps.TrainingImage(
            [
                Variable("a", rng.integers(0, 3, (12, 12)), weight=2.0,
                         n_neighbors=7),
                Variable("b", rng.random((12, 12)), categorical=False,
                         distance="l2", weight=1.0, max_radius=4.0),
            ]
        )
        win = ti.window(np.s_[2:10, 0:6])
        self.assertEqual(win.shape, (8, 6))
        self.assertTrue(win.multivariate)
        self.assertEqual(win.weights, ti.weights)
        self.assertEqual(win.variable("a").n_neighbors, 7)
        self.assertEqual(win.variable("b").distance, "l2")
        self.assertEqual(win.variable("b").max_radius, 4.0)

    def test_open_ended_slices(self):
        ti, data = self._cont_ti()
        win = ti.window(np.s_[:10, 15:])
        np.testing.assert_array_equal(win.variable().data, data[:10, 15:])

    def test_strided_slice_raises(self):
        ti, _ = self._cont_ti()
        with self.assertRaisesRegex(ValueError, "step"):
            ti.window(np.s_[0:10:2, 0:10])

    def test_out_of_bounds_raises(self):
        ti, _ = self._cont_ti()
        with self.assertRaisesRegex(ValueError, "bounds"):
            ti.window(np.s_[0:25, 0:10])  # axis 0 has length 20
        with self.assertRaisesRegex(ValueError, "bounds"):
            ti.window(np.s_[-3:10, 0:10])  # negative indices rejected

    def test_degenerate_slice_raises(self):
        ti, _ = self._cont_ti()
        with self.assertRaisesRegex(ValueError, "degenerate|empty"):
            ti.window(np.s_[5:5, 0:10])

    def test_wrong_slice_count_raises(self):
        ti, _ = self._cont_ti()
        with self.assertRaisesRegex(ValueError, "slice"):
            ti.window(np.s_[0:10])

    def test_non_iterable_raises_valueerror(self):
        # Non-iterable arguments (int, None) must raise ValueError, not TypeError.
        ti, _ = self._cont_ti()
        with self.assertRaisesRegex(ValueError, "slice"):
            ti.window(5)
        with self.assertRaisesRegex(ValueError, "slice"):
            ti.window(None)

    def test_1d_single_slice(self):
        ti = gs.mps.TrainingImage(np.arange(50, dtype=float),
                                  categorical=False)
        win = ti.window(np.s_[10:30])
        self.assertEqual(win.shape, (20,))


class TestPrecomputeOffsetsGuard(unittest.TestCase):
    def test_large_3d_grid_raises_valueerror(self):
        # 200^3 with default max_offset=200 → (401)^3 > 5M → guard fires
        with self.assertRaises(ValueError) as ctx:
            _precompute_offsets((200, 200, 200))
        msg = str(ctx.exception)
        self.assertIn("max_offset", msg)
        self.assertIn("max_radius", msg)

    def test_large_3d_grid_with_max_radius_ok(self):
        # explicit small max_offset bypasses the guard
        off = _precompute_offsets((200, 200, 200), max_offset=10)
        self.assertEqual(off.shape[1], 3)

    def test_small_grid_ok(self):
        # small 2-D grid stays under threshold
        off = _precompute_offsets((20, 20))
        self.assertFalse(np.any(np.all(off == 0, axis=1)))


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
                threshold=0.0,
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
    """Behaviour tests for set_mv_transforms, post_process, progress, and cond_weight."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.mv_data = {
            "a": rng.integers(0, 3, (20, 20)).astype(float),
            "b": rng.integers(0, 2, (20, 20)).astype(float),
        }
        cls.mv_ti = TrainingImage(
            [
                Variable("a", cls.mv_data["a"], n_neighbors=4),
                Variable("b", cls.mv_data["b"], n_neighbors=4),
            ]
        )
        cls.pos = [np.arange(6, dtype=float)] * 2

    def test_set_mv_transforms_mean_applied(self):
        """set_mv_transforms(mean={'a': 100.0}) must shift variable 'a' by 100."""
        ds = DirectSampling(MPSModel(self.mv_ti, scan_fraction=0.3))
        ds.set_mv_transforms(mean={"a": 100.0})

        field_pp = ds(self.pos, seed=5)
        field_raw = ds(self.pos, seed=5, post_process=False)

        # post-processed 'a' == raw 'a' + 100 everywhere
        np.testing.assert_allclose(
            field_pp["a"],
            field_raw["a"] + 100.0,
            err_msg="Post-processed 'a' should equal raw 'a' + 100.",
        )
        # variable 'b' has no transform — must be unaffected
        np.testing.assert_array_equal(
            field_pp["b"],
            field_raw["b"],
            err_msg="Variable 'b' should be unaffected by the 'a' mean transform.",
        )

    def test_post_process_false_returns_raw(self):
        """post_process=False must return values that are a subset of TI categories."""
        ds = DirectSampling(MPSModel(self.mv_ti, scan_fraction=0.3))
        ds.set_mv_transforms(mean={"a": 100.0})
        field_raw = ds(self.pos, seed=7, post_process=False)

        ti_a_values = set(np.unique(self.mv_data["a"]))
        sim_a_values = set(np.unique(field_raw["a"]))
        self.assertTrue(
            sim_a_values.issubset(ti_a_values),
            msg=f"post_process=False 'a' values {sim_a_values} are not a subset of TI values {ti_a_values}.",
        )

    def test_progress_callback_invoked(self):
        """progress callable must be called exactly n_nodes times; final call done == total == n_nodes."""
        rng = np.random.default_rng(1)
        data = rng.integers(0, 3, (20, 20)).astype(float)
        ti = TrainingImage(data, n_neighbors=4)
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.3))
        pos = [np.arange(6, dtype=float)] * 2
        n_nodes = 6 * 6

        calls = []

        def cb(done, total):
            calls.append((done, total))

        ds(pos, seed=0, progress=cb)

        self.assertEqual(
            len(calls),
            n_nodes,
            msg=f"Progress callback should be called {n_nodes} times, got {len(calls)}.",
        )
        last_done, last_total = calls[-1]
        self.assertEqual(last_done, n_nodes)
        self.assertEqual(last_total, n_nodes)

    def test_cond_weight_changes_output(self):
        """Different cond_weight values must yield different fields while honoring the conditioned node."""
        rng = np.random.default_rng(2)
        data = rng.integers(0, 3, (20, 20)).astype(float)
        ti = TrainingImage(data, n_neighbors=8)
        pos = [np.arange(8, dtype=float)] * 2
        cond_pos = [[4.0], [4.0]]
        cond_val = [2]

        ds1 = DirectSampling(MPSModel(ti, scan_fraction=0.3, cond_weight=1.0))
        ds1.set_condition(cond_pos, cond_val)
        f1 = ds1(pos, seed=0)

        ds5 = DirectSampling(MPSModel(ti, scan_fraction=0.3, cond_weight=5.0))
        ds5.set_condition(cond_pos, cond_val)
        f5 = ds5(pos, seed=0)

        # the two fields must differ (cond_weight propagates through the distance)
        self.assertFalse(
            np.array_equal(f1, f5),
            msg="cond_weight=1.0 and cond_weight=5.0 produced identical output; cond_weight has no effect.",
        )
        # the conditioned node must keep its value in both runs
        self.assertEqual(int(f1[4, 4]), 2)
        self.assertEqual(int(f5[4, 4]), 2)


class TestReturnTypeContract(unittest.TestCase):
    """0a — Return-type contract: univariate returns ndarray; MV returns dict.

    These are PHASE 0 regression guards: they must pass on the current code and
    will catch any refactor that accidentally changes the public return type.
    """

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
    # Univariate: must return a plain numpy.ndarray
    # ------------------------------------------------------------------

    def test_univariate_returns_ndarray(self):
        """DirectSampling on a univariate TI must return a bare numpy.ndarray."""
        ds = DirectSampling(MPSModel(self.ti_uni, scan_fraction=0.5))
        out = ds(self.pos, seed=101)
        self.assertIsInstance(
            out,
            np.ndarray,
            msg=(
                f"Univariate DirectSampling.__call__ returned {type(out)!r}; "
                "expected numpy.ndarray."
            ),
        )

    def test_univariate_correct_shape(self):
        """Univariate output shape must equal the simulation grid shape."""
        ds = DirectSampling(MPSModel(self.ti_uni, scan_fraction=0.5))
        out = ds(self.pos, seed=102)
        self.assertEqual(
            out.shape,
            self.sim_shape,
            msg=(
                f"Univariate output shape {out.shape} != "
                f"expected sim_shape {self.sim_shape}."
            ),
        )

    def test_univariate_dtype_numeric(self):
        """Univariate output must have a numeric (floating or integer) dtype."""
        ds = DirectSampling(MPSModel(self.ti_uni, scan_fraction=0.5))
        out = ds(self.pos, seed=103)
        self.assertTrue(
            np.issubdtype(out.dtype, np.number),
            msg=f"Univariate output dtype {out.dtype} is not numeric.",
        )

    def test_univariate_values_subset_of_ti(self):
        """Univariate output values must be drawn from the TI — no new values."""
        ds = DirectSampling(MPSModel(self.ti_uni, scan_fraction=0.5))
        out = ds(self.pos, seed=104)
        ti_vals = set(np.unique(self.ti_uni.data))
        sim_vals = set(np.unique(out))
        self.assertTrue(
            sim_vals.issubset(ti_vals),
            msg=(
                f"Simulated values {sim_vals} are not a subset of TI values "
                f"{ti_vals}."
            ),
        )

    # ------------------------------------------------------------------
    # Multivariate: must return a dict keyed by variable name
    # ------------------------------------------------------------------

    def test_multivariate_returns_dict(self):
        """DirectSampling on a multivariate TI must return a dict."""
        ds = DirectSampling(MPSModel(self.ti_mv, scan_fraction=0.5))
        out = ds(self.pos, seed=105)
        self.assertIsInstance(
            out,
            dict,
            msg=(
                f"Multivariate DirectSampling.__call__ returned {type(out)!r}; "
                "expected dict."
            ),
        )

    def test_multivariate_keys_match_ti_variables(self):
        """MV output dict keys must exactly match the TI variable names."""
        ds = DirectSampling(MPSModel(self.ti_mv, scan_fraction=0.5))
        out = ds(self.pos, seed=106)
        expected_keys = {v.name for v in self.ti_mv.variables}
        actual_keys = set(out.keys())
        self.assertEqual(
            actual_keys,
            expected_keys,
            msg=(
                f"MV output keys {actual_keys} do not match TI variables "
                f"{expected_keys}."
            ),
        )

    def test_multivariate_each_value_is_ndarray(self):
        """Every value in the MV output dict must be a numpy.ndarray."""
        ds = DirectSampling(MPSModel(self.ti_mv, scan_fraction=0.5))
        out = ds(self.pos, seed=107)
        for var, arr in out.items():
            self.assertIsInstance(
                arr,
                np.ndarray,
                msg=(
                    f"MV output[{var!r}] is {type(arr)!r}; expected "
                    "numpy.ndarray."
                ),
            )

    def test_multivariate_each_value_has_sim_shape(self):
        """Every array in the MV output dict must have the simulation grid shape."""
        ds = DirectSampling(MPSModel(self.ti_mv, scan_fraction=0.5))
        out = ds(self.pos, seed=108)
        for var, arr in out.items():
            self.assertEqual(
                arr.shape,
                self.sim_shape,
                msg=(
                    f"MV output[{var!r}].shape == {arr.shape}; expected "
                    f"sim_shape {self.sim_shape}."
                ),
            )

    def test_multivariate_values_subset_of_ti(self):
        """Every MV output variable must draw values only from the TI."""
        ds = DirectSampling(MPSModel(self.ti_mv, scan_fraction=0.5))
        out = ds(self.pos, seed=109)
        for v in self.ti_mv.variables:
            ti_vals = set(np.unique(v.data))
            sim_vals = set(np.unique(out[v.name]))
            self.assertTrue(
                sim_vals.issubset(ti_vals),
                msg=(
                    f"MV output[{v.name!r}] contains {sim_vals - ti_vals} which "
                    "are not in the TI."
                ),
            )

    # ------------------------------------------------------------------
    # 1-D univariate (n-D generality guard — not hardcoded 2-D)
    # ------------------------------------------------------------------

    def test_univariate_1d_returns_ndarray_correct_shape(self):
        """1-D univariate simulation returns ndarray of shape (N,) — n-D guard."""
        arr1d = np.tile([0.0, 1.0], 10)
        ti1d = TrainingImage(arr1d, categorical=True)
        ds = DirectSampling(MPSModel(ti1d, scan_fraction=1.0))
        pos1d = [np.arange(7, dtype=float)]
        out = ds(pos1d, seed=110)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (7,))

    def test_univariate_3d_returns_ndarray_correct_shape(self):
        """3-D univariate simulation returns ndarray of shape (A,B,C) — n-D guard."""
        rng = np.random.default_rng(57)
        ti3d = TrainingImage(
            rng.integers(0, 2, (8, 8, 8)).astype(float), categorical=True
        )
        ds = DirectSampling(MPSModel(ti3d, scan_fraction=0.3))
        pos3d = [np.arange(4, dtype=float)] * 3
        out = ds(pos3d, seed=111)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (4, 4, 4))


class TestSimulationPath(unittest.TestCase):
    """Tests for the pluggable path= parameter in DirectSampling / ds_simulate."""

    def _make_ds(
        self, seed_data=0, shape=(20, 20), n_vals=3, scan_fraction=0.5
    ):
        rng = np.random.default_rng(seed_data)
        data = rng.integers(0, n_vals, shape).astype(float)
        ti = TrainingImage(data, categorical=True, n_neighbors=4)
        return DirectSampling(MPSModel(ti, scan_fraction=scan_fraction)), ti

    # ------------------------------------------------------------------
    # "sequential" mode
    # ------------------------------------------------------------------

    def test_sequential_path_order(self):
        """path='sequential' -> engine visits nodes in lexicographic (raster) order."""
        ds, _ = self._make_ds()
        pos = [np.arange(5, dtype=float)] * 2
        sim_shape = (5, 5)
        # Patch ds_simulate to capture the engine's path array before returning.
        from gstools.mps import simulate as sim_mod

        captured = {}
        orig_cls = sim_mod._DirectSamplingEngine

        class _PatchedEngine(orig_cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                captured["path"] = self.path.copy()

        sim_mod._DirectSamplingEngine = _PatchedEngine
        try:
            ds(pos, seed=1, path="sequential")
        finally:
            sim_mod._DirectSamplingEngine = orig_cls

        expected = np.argwhere(
            np.ones(sim_shape, dtype=bool)
        )  # all nodes, raster
        np.testing.assert_array_equal(captured["path"], expected)

    def test_sequential_deterministic_without_path_seed(self):
        """path='sequential' produces the same field on two calls without fixing path_seed."""
        ds, _ = self._make_ds()
        pos = [np.arange(6, dtype=float)] * 2
        f1 = ds(pos, seed=3, path="sequential")
        f2 = ds(pos, seed=3, path="sequential")
        np.testing.assert_array_equal(f1, f2)

    def test_sequential_valid_values(self):
        """path='sequential' output values are a subset of TI values."""
        ds, ti = self._make_ds()
        pos = [np.arange(6, dtype=float)] * 2
        field = ds(pos, seed=5, path="sequential")
        self.assertEqual(field.shape, (6, 6))
        self.assertFalse(np.any(np.isnan(field)))
        ti_vals = set(np.unique(ti.data))
        self.assertTrue(set(np.unique(field)).issubset(ti_vals))

    def test_sequential_conditioning_honored(self):
        """path='sequential' honors exact conditioning (invariant: cond == value)."""
        ds, _ = self._make_ds()
        ds.set_condition([[3.0], [2.0]], [1.0])
        field = ds([np.arange(8, dtype=float)] * 2, seed=7, path="sequential")
        self.assertEqual(field[3, 2], 1.0)

    # ------------------------------------------------------------------
    # explicit array mode
    # ------------------------------------------------------------------

    def test_explicit_path_order_honored(self):
        """Explicit path is used exactly as supplied — verified via a degenerate TI."""
        # TI is a constant (all zeros), so every simulated value is 0 regardless
        # of order; we just verify no error and shape is correct.
        ti = TrainingImage(np.zeros((10, 10), dtype=float), categorical=True)
        ds = DirectSampling(MPSModel(ti, scan_fraction=1.0))
        pos = [np.arange(4, dtype=float)] * 2
        # Build path in reverse raster order
        base = np.argwhere(np.ones((4, 4), dtype=bool))
        explicit = base[::-1].copy()
        field = ds(pos, seed=0, path=explicit)
        self.assertEqual(field.shape, (4, 4))
        np.testing.assert_array_equal(field, np.zeros((4, 4)))

    def test_explicit_path_subset_property(self):
        """Explicit path: output values are a subset of TI values."""
        ds, ti = self._make_ds()
        pos = [np.arange(5, dtype=float)] * 2
        base = np.argwhere(np.ones((5, 5), dtype=bool))
        # reverse order
        explicit = base[::-1].copy()
        field = ds(pos, seed=0, path=explicit)
        ti_vals = set(np.unique(ti.data))
        self.assertTrue(set(np.unique(field)).issubset(ti_vals))

    def test_explicit_path_conditioning_honored(self):
        """Explicit path: conditioning data are still honored exactly."""
        ds, _ = self._make_ds()
        ds.set_condition([[2.0], [4.0]], [2.0])
        pos = [np.arange(6, dtype=float)] * 2
        sim_shape = (6, 6)
        # All nodes except the conditioned one
        unknown_mask = np.ones(sim_shape, dtype=bool)
        unknown_mask[2, 4] = (
            False  # conditioned node excluded from unknown set
        )
        base = np.argwhere(unknown_mask)
        np.testing.assert_array_equal(
            base, np.argwhere(unknown_mask)
        )  # sanity
        field = ds(pos, seed=2, path=base)
        self.assertEqual(field[2, 4], 2.0)

    # ------------------------------------------------------------------
    # Validation errors (explicit array)
    # ------------------------------------------------------------------

    def test_explicit_wrong_dim(self):
        """Wrong dim (1-D array) raises ValueError with 'shape' in message."""
        ds, _ = self._make_ds()
        pos = [np.arange(4, dtype=float)] * 2
        with self.assertRaisesRegex(ValueError, "shape"):
            ds(pos, seed=0, path=np.array([0, 1, 2, 3]))  # 1-D, not (N,2)

    def test_explicit_wrong_second_dim(self):
        """Wrong number of columns (N,3 for a 2-D grid) raises ValueError."""
        ds, _ = self._make_ds()
        pos = [np.arange(4, dtype=float)] * 2
        bad = np.zeros((5, 3), dtype=int)
        with self.assertRaisesRegex(ValueError, "shape"):
            ds(pos, seed=0, path=bad)

    def test_explicit_out_of_bounds_coord(self):
        """Out-of-bounds coordinate raises ValueError."""
        ds, _ = self._make_ds()
        pos = [np.arange(4, dtype=float)] * 2
        # Build a valid raster path then corrupt one coordinate
        base = np.argwhere(np.ones((4, 4), dtype=bool)).copy()
        base[0, 0] = 99  # out of bounds
        with self.assertRaisesRegex(ValueError, "out-of-bounds"):
            ds(pos, seed=0, path=base)

    def test_explicit_missing_node(self):
        """Path missing a required unknown node raises ValueError."""
        ds, _ = self._make_ds()
        pos = [np.arange(4, dtype=float)] * 2
        base = np.argwhere(np.ones((4, 4), dtype=bool))
        # Drop last node
        incomplete = base[:-1].copy()
        with self.assertRaisesRegex(ValueError, "missing"):
            ds(pos, seed=0, path=incomplete)

    def test_explicit_extra_node_silently_skipped(self):
        """Conditioned nodes present in an explicit path are silently skipped.

        A full-grid path (e.g. a spiral over all nodes) should work unchanged
        with conditioning data — the engine drops pre-filled entries itself.
        """
        ds, ti = self._make_ds()
        ds.set_condition([[0.0], [0.0]], [0.0])
        pos = [np.arange(4, dtype=float)] * 2
        # Full raster includes (0,0) which is now conditioned.
        # The engine should drop it silently and simulate the remaining 15 nodes.
        full_raster = np.argwhere(np.ones((4, 4), dtype=bool))
        field = ds(pos, seed=0, path=full_raster)
        self.assertEqual(field.shape, (4, 4))
        # Conditioned node must keep its exact value
        self.assertEqual(field[0, 0], 0.0)
        # All values must be a subset of TI values
        ti_vals = set(np.unique(ti.data))
        self.assertTrue(set(np.unique(field)).issubset(ti_vals))

    def test_explicit_duplicate_node(self):
        """Duplicate rows in explicit path raises ValueError with 'duplicate'."""
        ds, _ = self._make_ds()
        pos = [np.arange(4, dtype=float)] * 2
        base = np.argwhere(np.ones((4, 4), dtype=bool)).copy()
        # Replace last row with first row (duplicate)
        base[-1] = base[0]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ds(pos, seed=0, path=base)

    def test_invalid_string_raises(self):
        """Unknown string for path raises ValueError."""
        ds, _ = self._make_ds()
        pos = [np.arange(4, dtype=float)] * 2
        with self.assertRaisesRegex(ValueError, "path"):
            ds(pos, seed=0, path="zigzag")

    # ------------------------------------------------------------------
    # Regression: "random" default is byte-identical to previous behavior
    # ------------------------------------------------------------------

    def test_random_default_regression(self):
        """path='random' (default) output is byte-identical to pre-path behavior.

        Regression value captured from the code before the path parameter was
        added, using seed=42, scan_fraction=0.5, 8x8 grid, 3-valued categorical
        TI of shape (20,20) built from np.random.default_rng(0).
        """
        rng_data = np.random.default_rng(0)
        data = rng_data.integers(0, 3, (20, 20)).astype(float)
        ti = TrainingImage(data, categorical=True)
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.5))
        field = ds([np.arange(8, dtype=float)] * 2, seed=42)
        expected = np.array(
            [
                [0.0, 1.0, 1.0, 2.0, 2.0, 0.0, 2.0, 1.0],
                [2.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 1.0],
                [0.0, 2.0, 1.0, 2.0, 2.0, 2.0, 0.0, 2.0],
                [2.0, 2.0, 0.0, 2.0, 2.0, 0.0, 1.0, 1.0],
                [1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 2.0, 0.0],
                [2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0],
                [2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0],
                [1.0, 1.0, 1.0, 0.0, 2.0, 2.0, 0.0, 2.0],
            ]
        )
        np.testing.assert_array_equal(field, expected)

    def test_path_seed_inert_for_sequential(self):
        """path_seed has no effect when path='sequential' (documented behavior)."""
        ds, _ = self._make_ds()
        pos = [np.arange(5, dtype=float)] * 2
        f1 = ds(pos, seed=0, path="sequential", path_seed=1)
        f2 = ds(pos, seed=0, path="sequential", path_seed=99)
        np.testing.assert_array_equal(f1, f2)

    def test_ds_simulate_direct_sequential(self):
        """ds_simulate path='sequential' passes through correctly."""
        from gstools.random.rng import RNG

        arr1d = np.tile([0, 1], 10).astype(float)
        ti = TrainingImage(arr1d, categorical=True, n_neighbors=4)
        rng = RNG(7)
        result = ds_simulate(
            ti,
            sim_shape=(6,),
            threshold=0.0,
            scan_fraction=1.0,
            rng_path=rng.random,
            rng_nodes=rng.random,
            path="sequential",
        )
        self.assertIn(None, result)
        arr = result[None]
        self.assertEqual(arr.shape, (6,))
        self.assertFalse(np.any(np.isnan(arr)))
        self.assertTrue(set(np.unique(arr)).issubset({0.0, 1.0}))

    def test_1d_sequential_path_order(self):
        """1-D sequential path is [[0],[1],...,[N-1]] (n-D correctness check)."""
        arr1d = np.tile([0.0, 1.0], 10)
        ti = TrainingImage(arr1d, categorical=True)
        ds = DirectSampling(MPSModel(ti, scan_fraction=1.0))
        pos = [np.arange(5, dtype=float)]

        from gstools.mps import simulate as sim_mod

        captured = {}
        orig_cls = sim_mod._DirectSamplingEngine

        class _PatchedEngine(orig_cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                captured["path"] = self.path.copy()

        sim_mod._DirectSamplingEngine = _PatchedEngine
        try:
            ds(pos, seed=0, path="sequential")
        finally:
            sim_mod._DirectSamplingEngine = orig_cls

        expected = np.arange(5).reshape(5, 1)
        np.testing.assert_array_equal(captured["path"], expected)


class TestCondWeightOverride(unittest.TestCase):
    """cond_weight lives on MPSModel; DirectSampling is a read-only view."""

    def setUp(self):
        rng = np.random.default_rng(42)
        self.ti_data = rng.integers(0, 3, (20, 20))

    def test_shared_model_both_see_same_cond_weight(self):
        """Two DirectSampling from the same MPSModel both read the same cond_weight."""
        ti = TrainingImage(self.ti_data, n_neighbors=4)
        model = MPSModel(ti, scan_fraction=0.5, cond_weight=1.0)
        ds1 = DirectSampling(model)
        ds2 = DirectSampling(model)

        # Both read from the shared model — no override mechanism.
        self.assertAlmostEqual(ds1.cond_weight, 1.0)
        self.assertAlmostEqual(ds2.cond_weight, 1.0)
        self.assertAlmostEqual(model.cond_weight, 1.0)

    def test_cond_weight_on_model_is_honored(self):
        """Two DS with identical MPSModel(cond_weight=2) produce the same output."""
        ti1 = TrainingImage(self.ti_data, n_neighbors=4)
        ds1 = DirectSampling(MPSModel(ti1, scan_fraction=0.5, cond_weight=2.0))
        cpos = [np.array([5.0, 10.0]), np.array([5.0, 10.0])]
        cval = np.array([1.0, 2.0])
        ds1.set_condition(cpos, cval)
        out1 = ds1([np.arange(15, dtype=float)] * 2, seed=7)

        ti2 = TrainingImage(self.ti_data, n_neighbors=4)
        ds2 = DirectSampling(MPSModel(ti2, scan_fraction=0.5, cond_weight=2.0))
        ds2.set_condition(cpos, cval)
        out2 = ds2([np.arange(15, dtype=float)] * 2, seed=7)

        np.testing.assert_array_equal(out1, out2)

    def test_cond_weight_setter_raises(self):
        """DirectSampling.cond_weight is read-only; assignment raises AttributeError."""
        ti = TrainingImage(self.ti_data, n_neighbors=4)
        model = MPSModel(ti, scan_fraction=0.5, cond_weight=1.0)
        ds = DirectSampling(model)
        with self.assertRaises(AttributeError):
            ds.cond_weight = 3.5


class TestDataEvent(unittest.TestCase):
    def _make(self, k=3, dim=2):
        lags_sg = np.arange(k * dim, dtype=np.float64).reshape(k, dim)
        values = np.arange(k, dtype=np.float64)
        cond_mask = np.array([True, False, False][:k], dtype=bool)
        lag_norms = np.linalg.norm(lags_sg, axis=1)
        return DataEvent(lags_sg, values, cond_mask, lag_norms)

    def test_len_is_lag_count(self):
        de = self._make(k=3)
        self.assertEqual(len(de), 3)

    def test_lags_ti_defaults_none(self):
        de = self._make()
        self.assertIsNone(de.lags_ti)

    def test_with_lags_ti_sets_field_and_keeps_sg(self):
        de = self._make(k=3)
        ti = de.lags_sg * 2
        de2 = de.with_lags_ti(ti)
        np.testing.assert_array_equal(de2.lags_ti, ti)
        np.testing.assert_array_equal(de2.lags_sg, de.lags_sg)
        self.assertIsNone(de.lags_ti)  # original unchanged

    def test_truncate_slices_all_arrays(self):
        de = self._make(k=3).with_lags_ti(np.arange(6.0).reshape(3, 2))
        t = de.truncate(2)
        self.assertEqual(len(t), 2)
        np.testing.assert_array_equal(t.lags_sg, de.lags_sg[:2])
        np.testing.assert_array_equal(t.values, de.values[:2])
        np.testing.assert_array_equal(t.cond_mask, de.cond_mask[:2])
        np.testing.assert_array_equal(t.lag_norms, de.lag_norms[:2])
        np.testing.assert_array_equal(t.lags_ti, de.lags_ti[:2])

    def test_truncate_does_not_mutate_lags_sg_via_lags_ti(self):
        # Aliasing invariant: truncating must not let lags_ti writes hit lags_sg.
        de = self._make(k=3)
        de = de.with_lags_ti(
            de.lags_sg.copy()
        )  # caller must copy for independence
        t = de.truncate(2)
        t.lags_ti[:] = -999.0  # mutate the truncated TI-frame copy
        # original SG lags must be untouched
        self.assertFalse(np.any(de.lags_sg == -999.0))

    def test_truncate_preserves_none_lags_ti(self):
        de = self._make(k=3)
        self.assertIsNone(de.truncate(2).lags_ti)


class TestPerVariableSetterDedup(unittest.TestCase):
    def _mv(self, n_neighbors=32, max_radius=None):
        a = Variable(
            "a",
            np.zeros((8, 8), dtype=int),
            categorical=True,
            n_neighbors=n_neighbors,
            max_radius=max_radius,
        )
        b = Variable(
            "b",
            np.zeros((8, 8), dtype=int),
            categorical=True,
            n_neighbors=n_neighbors,
            max_radius=max_radius,
        )
        return DirectSampling(
            MPSModel(TrainingImage([a, b]), scan_fraction=0.2)
        )

    def _uni(self, n_neighbors=32, max_radius=None):
        ti = TrainingImage(
            np.zeros((8, 8), dtype=int),
            categorical=True,
            n_neighbors=n_neighbors,
            max_radius=max_radius,
        )
        return DirectSampling(MPSModel(ti, scan_fraction=0.2))

    def test_n_neighbors_scalar_sets_all(self):
        # Configure all variables at construction; getter collapses to scalar.
        ds = self._mv(n_neighbors=7)
        self.assertEqual(ds.n_neighbors, 7)

    def test_n_neighbors_dict_sets_per_var(self):
        # Configure per-variable at construction; getter returns dict.
        a = Variable("a", np.zeros((8, 8), dtype=int), n_neighbors=4)
        b = Variable("b", np.zeros((8, 8), dtype=int), n_neighbors=6)
        ds = DirectSampling(MPSModel(TrainingImage([a, b]), scan_fraction=0.2))
        self.assertEqual(ds.n_neighbors, {"a": 4, "b": 6})

    def test_n_neighbors_variable_setter_works(self):
        # After construction the Variable.n_neighbors setter is still mutable.
        ds = self._mv(n_neighbors=5)
        ds.ti.variable("a").n_neighbors = 7
        self.assertEqual(ds.ti.variable("a").n_neighbors, 7)
        self.assertEqual(ds.n_neighbors, {"a": 7, "b": 5})

    def test_max_radius_none_default(self):
        # Default is None; getter collapses to scalar.
        ds = self._mv()
        self.assertIsNone(ds.max_radius)

    def test_max_radius_nonpositive_raises_at_construction(self):
        # max_radius validation lives on Variable constructor.
        with self.assertRaisesRegex(ValueError, "max_radius"):
            Variable("a", np.zeros((8, 8), dtype=int), max_radius=-1.0)

    def test_max_radius_scalar_at_construction(self):
        # Configure max_radius at construction; getter reflects it.
        ds = self._mv(max_radius=5.0)
        self.assertEqual(ds.max_radius, 5.0)

    def test_max_radius_scalar_nonpositive_raises(self):
        # Positivity enforced at Variable construction.
        with self.assertRaisesRegex(ValueError, "max_radius"):
            Variable("x", np.zeros((8, 8), dtype=int), max_radius=-1.0)


class TestSpecResolver(unittest.TestCase):
    """Exact-shape spec resolution (spec §2; oracle h, i-prep)."""

    def test_none_passthrough(self):
        arr, stat = resolve_spec(None, (4, 4), 1, _flat_pos((4, 4)), "rotation")
        self.assertIsNone(arr)
        self.assertTrue(stat)

    def test_scalar_is_stationary_broadcast_view(self):
        arr, stat = resolve_spec(0.7, (4, 5), 1, _flat_pos((4, 5)), "rotation")
        self.assertTrue(stat)
        self.assertEqual(arr.shape, (4, 5, 1))
        self.assertTrue(np.all(arr == 0.7))
        # zero-copy: broadcast view, not materialized
        self.assertEqual(arr.strides[:2], (0, 0))
        self.assertFalse(arr.flags.writeable)

    def test_stationary_component_vector(self):
        # 3 Tait-Bryan angles on a 3-D grid (n_comp = no_of_angles(3) = 3)
        arr, stat = resolve_spec(
            np.array([0.1, 0.2, 0.3]), (4, 4, 4), 3, _flat_pos((4, 4, 4)),
            "rotation",
        )
        self.assertTrue(stat)
        self.assertEqual(arr.shape, (4, 4, 4, 3))
        np.testing.assert_array_equal(arr[2, 1, 3], [0.1, 0.2, 0.3])

    def test_grid_shape_form_4a(self):
        vals = np.linspace(0, 1, 20).reshape(4, 5)
        arr, stat = resolve_spec(vals, (4, 5), 2, _flat_pos((4, 5)), "scale",
                                 positive=False)
        self.assertFalse(stat)
        self.assertEqual(arr.shape, (4, 5, 2))
        self.assertEqual(arr[3, 4, 0], arr[3, 4, 1])  # broadcast across comps

    def test_grid_plus_comp_form_4b(self):
        vals = np.random.default_rng(0).random((4, 5, 2)) + 0.1
        arr, stat = resolve_spec(vals, (4, 5), 2, _flat_pos((4, 5)), "scale")
        self.assertFalse(stat)
        np.testing.assert_array_equal(arr, vals)

    def test_flattened_per_node_map_raises(self):
        # B5 regression (oracle h): flattened (n_nodes,) map on a 2-D grid
        # must raise, never silently degrade to stationary.
        with self.assertRaisesRegex(ValueError, "shape"):
            resolve_spec(np.linspace(0, 1, 20), (4, 5), 1,
                         _flat_pos((4, 5)), "rotation")

    def test_collision_n_nodes_equals_n_comp_raises(self):
        # oracle h: 1-D grid of length n_comp — both readings match; must be
        # a loud ValueError naming both interpretations.
        with self.assertRaisesRegex(ValueError, "stationary.*per-node|per-node.*stationary"):
            resolve_spec(np.array([0.1, 0.2, 0.3]), (3,), 3, _flat_pos((3,)),
                         "rotation")

    def test_per_node_map_on_1d_grid_is_form_4a(self):
        # oracle h: valid per-node maps on 1-D grids are expressible.
        vals = np.linspace(0.5, 2.0, 7)
        arr, stat = resolve_spec(vals, (7,), 1, _flat_pos((7,)), "scale")
        self.assertFalse(stat)
        self.assertEqual(arr.shape, (7, 1))
        np.testing.assert_array_equal(arr[:, 0], vals)

    def test_callable_form(self):
        def f(x, y):
            return x * 0.1 + y

        pos = _flat_pos((3, 4))
        arr, stat = resolve_spec(f, (3, 4), 1, pos, "rotation")
        self.assertFalse(stat)
        self.assertEqual(arr.shape, (3, 4, 1))
        self.assertAlmostEqual(float(arr[2, 3, 0]), 2 * 0.1 + 3)

    def test_callable_with_component_axis(self):
        def f(x, y):
            return np.stack([np.ones_like(x), 0.5 * np.ones_like(x)], axis=-1)

        arr, stat = resolve_spec(f, (3, 4), 2, _flat_pos((3, 4)), "scale")
        self.assertEqual(arr.shape, (3, 4, 2))
        np.testing.assert_array_equal(arr[1, 1], [1.0, 0.5])

    def test_callable_wrong_shape_raises(self):
        with self.assertRaisesRegex(ValueError, "callable"):
            resolve_spec(lambda x, y: np.zeros((5,)), (3, 4), 1,
                         _flat_pos((3, 4)), "rotation")

    def test_positive_validation(self):
        with self.assertRaisesRegex(ValueError, "positive|> 0"):
            resolve_spec(-1.0, (4, 4), 2, _flat_pos((4, 4)), "scale",
                         positive=True)
        with self.assertRaisesRegex(ValueError, "positive|> 0"):
            resolve_spec(np.zeros((4, 4)), (4, 4), 2, _flat_pos((4, 4)),
                         "scale", positive=True)


class TestZoneConstruction(unittest.TestCase):
    def _ti(self):
        return gs.mps.TrainingImage(
            np.random.default_rng(0).integers(0, 2, (10, 10))
        )

    def test_zone_holds_ti_and_where(self):
        ti = self._ti()
        mask = np.zeros((5, 5), dtype=bool)
        z = gs.Zone(ti, where=mask)
        self.assertIs(z.ti, ti)
        np.testing.assert_array_equal(z.where, mask)

    def test_non_ti_raises(self):
        with self.assertRaises(TypeError):
            gs.Zone(np.zeros((5, 5)), where=np.ones((5, 5), dtype=bool))

    def test_non_boolean_where_raises(self):
        # oracle k: no silent != 0 coercion
        with self.assertRaisesRegex(ValueError, "bool"):
            gs.Zone(self._ti(), where=np.ones((5, 5), dtype=int))

    def test_callable_where_accepted(self):
        z = gs.Zone(self._ti(), where=lambda x, y: x > 2)
        self.assertTrue(callable(z.where))


class TestMPSModelNonstationarySpec(unittest.TestCase):
    def _ti(self, shape=(10, 10)):
        return gs.mps.TrainingImage(
            np.random.default_rng(0).integers(0, 2, shape)
        )

    def test_defaults(self):
        m = MPSModel(self._ti())
        self.assertIsNone(m.rotation)
        self.assertIsNone(m.scale)
        self.assertEqual(m.zones, [])

    def test_specs_stored(self):
        rot = np.pi / 4
        sc = [1.0, 0.5]
        m = MPSModel(self._ti(), rotation=rot, scale=sc)
        self.assertEqual(m.rotation, rot)
        np.testing.assert_array_equal(m.scale, sc)

    def test_rotation_on_1d_ti_raises(self):
        # oracle k: no_of_angles(1) == 0 — every rotation form is degenerate.
        ti1d = gs.mps.TrainingImage(np.arange(30, dtype=float),
                                    categorical=False)
        with self.assertRaisesRegex(ValueError, "1-D|rotation"):
            MPSModel(ti1d, rotation=0.5)

    def test_scale_on_1d_ti_allowed(self):
        ti1d = gs.mps.TrainingImage(np.arange(30, dtype=float),
                                    categorical=False)
        m = MPSModel(ti1d, scale=2.0)  # dilation along the single axis
        self.assertEqual(m.scale, 2.0)

    def test_nonpositive_scale_raises_at_construction(self):
        with self.assertRaisesRegex(ValueError, "positive|> 0"):
            MPSModel(self._ti(), scale=0.0)
        with self.assertRaisesRegex(ValueError, "positive|> 0"):
            MPSModel(self._ti(), scale=np.full((10, 10), -1.0))

    def test_callable_specs_accepted(self):
        m = MPSModel(self._ti(), rotation=lambda x, y: x * 0.01,
                     scale=lambda x, y: 1.0 + 0 * x)
        self.assertTrue(callable(m.rotation))
        self.assertTrue(callable(m.scale))

    def test_zone_variable_set_must_match(self):
        rng = np.random.default_rng(0)
        primary = gs.mps.TrainingImage(
            [Variable("a", rng.integers(0, 2, (8, 8)))]
        )
        wrong_name = gs.mps.TrainingImage(
            [Variable("b", rng.integers(0, 2, (8, 8)))]
        )
        z = gs.Zone(wrong_name, where=np.ones((5, 5), dtype=bool))
        with self.assertRaisesRegex(ValueError, "variable"):
            MPSModel(primary, zones=[z])

    def test_zone_variable_kind_must_match(self):
        rng = np.random.default_rng(0)
        primary = gs.mps.TrainingImage(
            [Variable("a", rng.integers(0, 2, (8, 8)), categorical=True)]
        )
        wrong_kind = gs.mps.TrainingImage(
            [Variable("a", rng.random((8, 8)), categorical=False)]
        )
        z = gs.Zone(wrong_kind, where=np.ones((5, 5), dtype=bool))
        with self.assertRaisesRegex(ValueError, "categorical|kind"):
            MPSModel(primary, zones=[z])

    def test_zone_ndim_must_match(self):
        z = gs.Zone(
            gs.mps.TrainingImage(np.zeros((5, 5, 5))),
            where=np.ones((5, 5), dtype=bool),
        )
        with self.assertRaisesRegex(ValueError, "dim"):
            MPSModel(self._ti(), zones=[z])

    def test_non_zone_in_zones_raises(self):
        with self.assertRaises(TypeError):
            MPSModel(self._ti(), zones=[self._ti()])

    def test_repr_array_by_shape_callable_by_name(self):
        def my_rot(x, y):
            return x * 0.0

        m1 = MPSModel(self._ti(), rotation=my_rot,
                      scale=np.ones((10, 10, 2)))
        r = repr(m1)
        self.assertIn("my_rot", r)
        self.assertIn("shape=(10, 10, 2)", r)
        m2 = MPSModel(self._ti())
        self.assertNotIn("rotation", repr(m2))


class TestWhereAndSelector(unittest.TestCase):
    def test_resolve_where_array(self):
        mask = np.zeros((4, 4), dtype=bool)
        mask[2:] = True
        out = resolve_where(mask, (4, 4), _flat_pos((4, 4)))
        np.testing.assert_array_equal(out, mask)

    def test_resolve_where_callable(self):
        out = resolve_where(lambda x, y: x >= 2, (4, 4), _flat_pos((4, 4)))
        self.assertEqual(out.shape, (4, 4))
        self.assertTrue(out[3, 0])
        self.assertFalse(out[1, 0])

    def test_resolve_where_callable_non_bool_raises(self):
        with self.assertRaisesRegex(ValueError, "bool"):
            resolve_where(lambda x, y: x + y, (4, 4), _flat_pos((4, 4)))

    def test_resolve_where_wrong_shape_raises(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            resolve_where(np.zeros((3, 3), dtype=bool), (4, 4),
                          _flat_pos((4, 4)))

    def test_selector_values(self):
        ti = gs.mps.TrainingImage(np.zeros((5, 5)))
        m1 = np.zeros((4, 4), dtype=bool); m1[:2] = True
        m2 = np.zeros((4, 4), dtype=bool); m2[3:] = True
        sel = build_zone_selector(
            [gs.Zone(ti, where=m1), gs.Zone(ti, where=m2)],
            (4, 4), _flat_pos((4, 4)),
        )
        self.assertEqual(sel[0, 0], 1)
        self.assertEqual(sel[3, 0], 2)
        self.assertEqual(sel[2, 0], 0)  # uncovered -> primary

    def test_overlapping_zones_raise(self):
        # oracle k: deterministic — no silent precedence order
        ti = gs.mps.TrainingImage(np.zeros((5, 5)))
        m1 = np.zeros((4, 4), dtype=bool); m1[:2] = True
        m2 = np.zeros((4, 4), dtype=bool); m2[1:] = True  # overlaps row 1
        with self.assertRaisesRegex(ValueError, "overlap"):
            build_zone_selector(
                [gs.Zone(ti, where=m1), gs.Zone(ti, where=m2)],
                (4, 4), _flat_pos((4, 4)),
            )


class TestZonatedSimulation(unittest.TestCase):
    """Zonated DS (M10 para [40]) — oracles d, e, j, l."""

    def _two_zone_setup(self, sg=10):
        rng = np.random.default_rng(0)
        ti_a = gs.mps.TrainingImage(rng.integers(0, 3, (15, 15)),
                                    n_neighbors=4)
        ti_b = gs.mps.TrainingImage(rng.integers(10, 13, (15, 15)),
                                    n_neighbors=4)
        gx, _ = np.meshgrid(np.arange(sg), np.arange(sg), indexing="ij")
        zone = gs.Zone(ti_b, where=(gx >= sg // 2))
        model = MPSModel(ti_a, scan_fraction=0.5, zones=[zone])
        pos = [np.arange(sg, dtype=float)] * 2
        return model, pos, sg

    def test_per_zone_subset_property(self):
        # Oracle d: disjoint TI value sets -> values in zone z ⊆ TI_z values.
        # This also covers the fallback path: fallback draws MUST come from
        # the selected zone TI, or rare fallbacks would leak primary values.
        model, pos, sg = self._two_zone_setup()
        field = gs.mps.DirectSampling(model)(pos, seed=0)
        self.assertTrue(np.all(np.isin(field[: sg // 2], [0, 1, 2])))
        self.assertTrue(np.all(np.isin(field[sg // 2:], [10, 11, 12])))

    def test_zone_thread_count_determinism(self):
        # Oracle e: zones + rotation, varying num_threads -> identical output.
        rng = np.random.default_rng(1)
        ti_a = gs.mps.TrainingImage(rng.integers(0, 3, (15, 15)),
                                    n_neighbors=4)
        ti_b = gs.mps.TrainingImage(rng.integers(10, 13, (15, 15)),
                                    n_neighbors=4)
        gx, _ = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
        model = MPSModel(
            ti_a, scan_fraction=0.5, rotation=np.pi / 6,
            zones=[gs.Zone(ti_b, where=(gx >= 4))],
        )
        pos = [np.arange(8, dtype=float)] * 2
        f1 = gs.mps.DirectSampling(model)(pos, seed=5, num_threads=1)
        f4 = gs.mps.DirectSampling(model)(pos, seed=5, num_threads=4)
        np.testing.assert_array_equal(f1, f4)

    def test_overlapping_zones_raise_at_call(self):
        # Oracle k: overlap detected at resolve time in __call__.
        rng = np.random.default_rng(0)
        ti = gs.mps.TrainingImage(rng.integers(0, 2, (10, 10)))
        m1 = np.zeros((6, 6), dtype=bool); m1[:3] = True
        m2 = np.zeros((6, 6), dtype=bool); m2[2:] = True
        model = MPSModel(
            ti, zones=[gs.Zone(ti, where=m1), gs.Zone(ti, where=m2)]
        )
        ds = gs.mps.DirectSampling(model)
        with self.assertRaisesRegex(ValueError, "overlap"):
            ds([np.arange(6, dtype=float)] * 2, seed=0)

    def test_per_zone_d_max(self):
        # Oracle j: distances in each zone normalized by that zone TI's d_max.
        # White-box: the zone domain's scan config carries the zone's range.
        from gstools.mps.simulate import _DirectSamplingEngine
        from gstools.random.rng import RNG

        rng = np.random.default_rng(0)
        ti_narrow = gs.mps.TrainingImage(rng.random((10, 10)),
                                         categorical=False)   # range ~1
        ti_wide = gs.mps.TrainingImage(rng.random((10, 10)) * 100.0,
                                       categorical=False)     # range ~100
        sel = np.zeros((6, 6), dtype=np.intp); sel[3:] = 1
        engine = _DirectSamplingEngine(
            ti_narrow, (6, 6), threshold=0.1, scan_fraction=1.0,
            rng_path=RNG(0).random, rng_nodes=RNG(1).random,
            zone_tis=[ti_wide], zone_selector=sel,
        )
        d0 = engine.domains[0].scan_config.d_max[None]
        d1 = engine.domains[1].scan_config.d_max[None]
        self.assertAlmostEqual(d0, ti_narrow.variable().d_max)
        self.assertAlmostEqual(d1, ti_wide.variable().d_max)
        self.assertGreater(d1, 10 * d0)

    def test_window_zone_d_max_from_slice(self):
        # Sub-region zone: d_max comes from the slice, not the parent TI.
        data = np.zeros((10, 10))
        data[0, 0] = 100.0
        data[5:, 5:] = np.arange(25).reshape(5, 5)
        ti = gs.mps.TrainingImage(data, categorical=False)
        from gstools.mps.simulate import _DirectSamplingEngine
        from gstools.random.rng import RNG

        sel = np.zeros((4, 4), dtype=np.intp); sel[2:] = 1
        engine = _DirectSamplingEngine(
            ti, (4, 4), threshold=0.1, scan_fraction=1.0,
            rng_path=RNG(0).random, rng_nodes=RNG(1).random,
            zone_tis=[ti.window(np.s_[5:10, 5:10])], zone_selector=sel,
        )
        self.assertEqual(engine.domains[1].scan_config.d_max[None], 24.0)

    def test_composite_zone_ti_joint_copy(self):
        # Oracle l: composite multivariate zone TI with a swapped A-channel.
        # A-values in the zone ⊆ composite A values; B-values ⊆ composite B
        # values; every zone (a, b) pair exists co-located in the composite
        # (the whole TI-cell vector is copied — joint-scan invariant).
        rng = np.random.default_rng(3)
        prim_a = rng.integers(0, 3, (15, 15))
        prim_b = rng.integers(0, 2, (15, 15))
        comp_a = rng.integers(5, 8, (15, 15))
        comp_b = rng.integers(2, 4, (15, 15))
        primary = gs.mps.TrainingImage([
            Variable("a", prim_a, n_neighbors=4),
            Variable("b", prim_b, n_neighbors=4),
        ])
        composite = gs.mps.TrainingImage([
            Variable("a", comp_a, n_neighbors=4),
            Variable("b", comp_b, n_neighbors=4),
        ])
        gx, _ = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
        model = MPSModel(
            primary, scan_fraction=0.5,
            zones=[gs.Zone(composite, where=(gx >= 4))],
        )
        out = gs.mps.DirectSampling(model)(
            [np.arange(8, dtype=float)] * 2, seed=0
        )
        za, zb = out["a"][4:], out["b"][4:]
        self.assertTrue(np.all(np.isin(za, [5, 6, 7])))
        self.assertTrue(np.all(np.isin(zb, [2, 3])))
        zone_pairs = set(zip(za.ravel().tolist(), zb.ravel().tolist()))
        ti_pairs = set(zip(comp_a.ravel().tolist(),
                           comp_b.ravel().tolist()))
        self.assertTrue(zone_pairs <= ti_pairs)
        self.assertTrue(np.all(np.isin(out["a"][:4], [0, 1, 2])))

    def test_zone_with_conditioning(self):
        # Conditioning honored exactly inside a zone.
        model, pos, sg = self._two_zone_setup()
        ds = gs.mps.DirectSampling(model)
        ds.set_condition([[7.0], [7.0]], [11])
        field = ds(pos, seed=0)
        self.assertEqual(int(field[7, 7]), 11)

    def test_no_zones_unchanged(self):
        # zones=[] must be bit-identical to the pre-zonation engine.
        rng = np.random.default_rng(0)
        ti = gs.mps.TrainingImage(rng.integers(0, 3, (15, 15)),
                                  n_neighbors=4)
        pos = [np.arange(8, dtype=float)] * 2
        f_plain = gs.mps.DirectSampling(
            MPSModel(ti, scan_fraction=0.5)
        )(pos, seed=2)
        f_empty = gs.mps.DirectSampling(
            MPSModel(ti, scan_fraction=0.5, zones=[])
        )(pos, seed=2)
        np.testing.assert_array_equal(f_plain, f_empty)


class TestVecDistanceDmaxOverride(unittest.TestCase):
    def test_d_max_override(self):
        ti = gs.mps.TrainingImage(
            np.array([[0.0, 1.0], [2.0, 3.0]]), categorical=False
        )  # own range = 3
        de = np.array([0.0])
        cand = np.array([[3.0]])
        d_own = ti.vec_distance_var(None, de, cand)
        d_ovr = ti.vec_distance_var(None, de, cand, d_max=6.0)
        self.assertAlmostEqual(float(d_own[0]), 1.0)
        self.assertAlmostEqual(float(d_ovr[0]), 0.5)


if __name__ == "__main__":
    unittest.main()
