#!/usr/bin/env python
"""Unittest for the MPS module (TrainingImage and DirectSampling)."""

import unittest

import numpy as np

import gstools as gs
from gstools import config as gs_config
from gstools.mps.direct_sampling import (
    DirectSampling,
    _precompute_offsets,
    ds_simulate,
)
from gstools.mps.distance import (
    categorical_dist,
    compute_node_weights,
    l1_dist,
    l2_dist,
    lp_dist,
    variation_dist,
)
from gstools.mps.training_image import TrainingImage

class TestDirectSamplingParallel(unittest.TestCase):
    def test_valid_values(self):
        rng = np.random.default_rng(0)
        data = rng.integers(0, 3, (20, 20))
        ti = TrainingImage(data)
        ds = DirectSampling(ti, n_neighbors=8, scan_fraction=0.2, num_threads=2)
        field = ds([np.arange(8, dtype=float)] * 2, seed=0)
        self.assertEqual(field.shape, (8, 8))
        self.assertTrue(np.all(np.isin(field, [0, 1, 2])))

    def test_reproducible(self):
        # DAG parallelism is deterministic: same seed → same parallel result
        rng = np.random.default_rng(0)
        data = rng.integers(0, 3, (20, 20))
        ti = TrainingImage(data)
        ds = DirectSampling(ti, n_neighbors=8, scan_fraction=0.2, num_threads=2)
        pos = [np.arange(8, dtype=float)] * 2
        self.assertTrue(np.array_equal(ds(pos, seed=7), ds(pos, seed=7)))

    def test_conditioning_preserved(self):
        rng = np.random.default_rng(0)
        data = rng.integers(0, 3, (20, 20))
        ti = TrainingImage(data)
        ds = DirectSampling(ti, n_neighbors=4, scan_fraction=0.2, num_threads=2)
        ds.set_condition([[5.0], [5.0]], [2])
        field = ds([np.arange(10, dtype=float)] * 2, seed=0)
        self.assertEqual(field[5, 5], 2)

    def test_global_config(self):
        # num_threads=None reads gs_config.NUM_THREADS
        rng = np.random.default_rng(0)
        data = rng.integers(0, 3, (20, 20))
        ti = TrainingImage(data)
        pos = [np.arange(8, dtype=float)] * 2
        old = gs_config.NUM_THREADS
        try:
            gs_config.NUM_THREADS = 2
            field = DirectSampling(ti, n_neighbors=8, scan_fraction=0.2)(
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
        ti = TrainingImage(data)
        pos = [np.arange(12, dtype=float)] * 2
        ds = DirectSampling(ti, n_neighbors=2, scan_fraction=0.3, num_threads=4)
        field = ds(pos, seed=42)
        self.assertEqual(field.shape, (12, 12))
        self.assertTrue(np.all(np.isin(field, [0.0, 1.0])))

    def test_stress(self):
        # large grid, sparse DAG, conditioning, varying thread counts
        rng = np.random.default_rng(3)
        data = rng.integers(0, 4, (40, 40))
        ti = TrainingImage(data)
        pos = [np.arange(25, dtype=float)] * 2
        cond_pos = [
            rng.integers(0, 25, 20).astype(float),
            rng.integers(0, 25, 20).astype(float),
        ]
        cond_val = rng.integers(0, 4, 20).astype(float)
        for nt in (2, 4, 8):
            ds = DirectSampling(
                ti, n_neighbors=3, scan_fraction=0.4, num_threads=nt
            )
            ds.set_condition(cond_pos, cond_val)
            field = ds(pos, seed=11)
            self.assertEqual(field.shape, (25, 25))
            self.assertTrue(np.all(np.isin(field, [0, 1, 2, 3])))


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
        dist = self.ti_cat.distance(a, a)
        self.assertAlmostEqual(dist, 0.0)

        # Completely mismatched, uniform weights → 1.0
        b = np.array([1.0, 0.0, 1.0])
        dist = self.ti_cat.distance(a, b)
        self.assertAlmostEqual(dist, 1.0)

        # One of three mismatched → 1/3
        c = np.array([1.0, 1.0, 0.0])
        dist = self.ti_cat.distance(a, c)
        self.assertAlmostEqual(dist, 1.0 / 3.0)

        # Two of four mismatched → 0.5 (spec-required half-mismatch case)
        a4 = np.array([0.0, 1.0, 0.0, 1.0])
        c4 = np.array([1.0, 0.0, 0.0, 1.0])
        dist = self.ti_cat.distance(a4, c4)
        self.assertAlmostEqual(dist, 0.5)

    def test_distance_continuous(self):
        x = np.array([0.0, 0.5, 1.0])
        y = np.array([0.2, 0.3, 0.8])

        # l1
        ti_l1 = TrainingImage(
            np.linspace(0.0, 1.0, 10), categorical=False, distance="l1"
        )
        self.assertAlmostEqual(ti_l1.distance(x, x), 0.0)
        self.assertAlmostEqual(ti_l1.distance(x, y), 0.2, places=6)

        # l2
        ti_l2 = TrainingImage(
            np.linspace(0.0, 1.0, 10), categorical=False, distance="l2"
        )
        self.assertAlmostEqual(ti_l2.distance(x, x), 0.0)
        self.assertAlmostEqual(ti_l2.distance(x, y), 0.2, places=6)

        # lp (p=3.5) — non-uniform diffs [0.3, 0.1, 0.3] distinguish lp from l1/l2
        y_lp = np.array([0.3, 0.4, 0.7])
        ti_lp = TrainingImage(
            np.linspace(0.0, 1.0, 10), categorical=False, distance="l3.5"
        )
        self.assertAlmostEqual(ti_lp.distance(x, x), 0.0)
        self.assertAlmostEqual(ti_lp.distance(x, y_lp), 0.2680, places=3)
        self.assertGreater(ti_lp.distance(x, y_lp), ti_l1.distance(x, y_lp))

        # variation (default p=2)
        ti_var = TrainingImage(
            np.linspace(0.0, 1.0, 10), categorical=False, distance="variation"
        )
        self.assertAlmostEqual(ti_var.distance(x, x), 0.0)
        self.assertAlmostEqual(ti_var.distance(x, y), 0.094281, places=5)
        # constant shift → distance = 0 (key behavioral property of variation distance)
        self.assertAlmostEqual(ti_var.distance(x, x + 0.15), 0.0, places=10)

        # variation1 (L^1 aggregation)
        ti_var1 = TrainingImage(
            np.linspace(0.0, 1.0, 10), categorical=False, distance="variation1"
        )
        self.assertAlmostEqual(ti_var1.distance(x, x), 0.0)
        self.assertAlmostEqual(ti_var1.distance(x, y), 0.08889, places=4)
        self.assertAlmostEqual(ti_var1.distance(x, x + 0.15), 0.0, places=10)
        # L^1 < L^2 for non-uniform diffs
        self.assertLess(ti_var1.distance(x, y), ti_var.distance(x, y))

        # variation2 explicit matches variation (regression guard)
        ti_var2 = TrainingImage(
            np.linspace(0.0, 1.0, 10), categorical=False, distance="variation2"
        )
        self.assertAlmostEqual(
            ti_var2.distance(x, y), ti_var.distance(x, y), places=10
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
        d1 = self.ti_cat.distance(
            a, b, cond_mask=[True, False, False], cond_weight=1.0
        )
        d2 = self.ti_cat.distance(
            a, b, cond_mask=[True, False, False], cond_weight=2.0
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
        d_power = ti_p.distance(x, z, lag_norms=lags)
        d_flat = ti_flat.distance(x, z, lag_norms=lags)
        # power=1 weights far neighbours less → smaller distance for far mismatch
        self.assertLess(d_power, d_flat)

    def test_distance_empty_event(self):
        dist = self.ti_cat.distance(np.array([]), np.array([]))
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

        # categorical: identical → 0, opposite → 1
        self.assertAlmostEqual(categorical_dist(a, a, w), 0.0)
        self.assertAlmostEqual(categorical_dist(a, b, w), 1.0)

        # continuous: identical → 0
        x = np.array([0.0, 0.5, 1.0])
        d_max = 1.0
        self.assertAlmostEqual(l1_dist(x, x, w, d_max), 0.0)
        self.assertAlmostEqual(l2_dist(x, x, w, d_max), 0.0)
        self.assertAlmostEqual(lp_dist(x, x, w, d_max, 3.5), 0.0)
        self.assertAlmostEqual(variation_dist(x, x, w, d_max), 0.0)
        self.assertAlmostEqual(variation_dist(x, x, w, d_max, p=1.0), 0.0)

        # distances in [0, 1]
        y = np.array([0.2, 0.3, 0.8])
        self.assertAlmostEqual(l1_dist(x, y, w, d_max), 0.2, places=6)
        self.assertAlmostEqual(l2_dist(x, y, w, d_max), 0.2, places=6)
        self.assertAlmostEqual(
            variation_dist(x, y, w, d_max), 0.094281, places=5
        )
        self.assertAlmostEqual(
            variation_dist(x, y, w, d_max, p=1.0), 0.08889, places=4
        )
        # p=2 explicit matches default
        self.assertAlmostEqual(
            variation_dist(x, y, w, d_max, p=2.0),
            variation_dist(x, y, w, d_max),
            places=10,
        )

        # lp: non-uniform diffs [0.3, 0.1, 0.3] verify the p-norm exponent is used
        y_lp = np.array([0.3, 0.4, 0.7])
        self.assertAlmostEqual(
            lp_dist(x, y_lp, w, d_max, 3.5), 0.2680, places=3
        )
        self.assertGreater(
            lp_dist(x, y_lp, w, d_max, 3.5), l1_dist(x, y_lp, w, d_max)
        )

    def test_variation_dist_bounded(self):
        """variation_dist with distance_power > 0 must stay in [0, 1]."""
        # Adversarial: weight concentrated on maximally anti-correlated element
        x = np.array([0.0, 1.0, 0.0])
        y = np.array([1.0, 0.0, 1.0])
        lags = np.array([10.0, 0.1, 10.0])
        w = compute_node_weights(3, lags, 1.0)
        d = variation_dist(x, y, w, 1.0)
        self.assertGreaterEqual(d, 0.0)
        self.assertLessEqual(d, 1.0)
        self.assertAlmostEqual(d, 0.661747, places=5)
        # Also via TrainingImage.distance()
        ti = TrainingImage(
            np.linspace(0.0, 1.0, 10),
            categorical=False,
            distance="variation",
            distance_power=1.0,
        )
        self.assertLessEqual(ti.distance(x, y, lag_norms=lags), 1.0)

    def test_variation_dist_out_of_range_clamped(self):
        """Out-of-range SG values (from conditioning / mean-shift) must clamp to [0, 1]."""
        ti = TrainingImage(
            np.linspace(0.0, 1.0, 10),  # d_max == 1.0
            categorical=False,
            distance="variation",
        )
        de_sim = np.array([5.0, 0.0])  # 5.0 is far outside the TI range
        de_ti = np.array([0.0, 1.0])
        self.assertLessEqual(ti.distance(de_sim, de_ti), 1.0)
        vec = ti.vec_distance(de_sim, de_ti[np.newaxis, :])
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
        """All-conditioning event with cond_weight=0 must not yield NaN weights."""
        w = compute_node_weights(
            3,
            lag_norms=None,
            distance_power=0.0,
            cond_mask=np.array([True, True, True]),
            cond_weight=0.0,
        )
        self.assertTrue(np.all(np.isfinite(w)))
        self.assertAlmostEqual(w.sum(), 1.0)
        np.testing.assert_allclose(w, np.full(3, 1.0 / 3.0))

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

    def test_weights_positional_arg_guard(self):
        # Passing a string as the 3rd positional arg (the old `distance` slot)
        # must raise TypeError with a helpful message, not silently use "l1".
        with self.assertRaises(TypeError):
            TrainingImage(np.zeros((4, 4), dtype=int), True, "variation")


class TestDirectSampling(unittest.TestCase):
    def setUp(self):
        # 1-D categorical TI: alternating 0/1, length 20
        arr1d = np.tile([0, 1], 10).astype(float)
        self.ti1d = TrainingImage(arr1d, categorical=True)

        # 2-D categorical TI: 8×8 checkerboard
        self.ti2d = TrainingImage(
            (np.indices((8, 8)).sum(axis=0) % 2).astype(float),
            categorical=True,
        )

        rng = np.random.default_rng(0)
        self.ti2d_rand = TrainingImage(
            rng.integers(0, 2, size=(20, 20)).astype(float),
            categorical=True,
        )

        # 1-D continuous TI
        self.ti1d_cont = TrainingImage(
            np.linspace(0.0, 1.0, 20), categorical=False, distance="l1"
        )

        self.x1d = np.arange(10, dtype=float)
        self.x2d = np.arange(6, dtype=float)
        self.y2d = np.arange(6, dtype=float)

    def test_raise(self):
        with self.assertRaises(ValueError):
            DirectSampling(self.ti1d, boundary="bad")
        with self.assertRaises(ValueError):
            DirectSampling(self.ti1d, max_radius=0)
        with self.assertRaises(ValueError):
            DirectSampling(self.ti1d, max_radius=-1.0)
        ds = DirectSampling(self.ti1d)
        with self.assertRaises(ValueError):
            ds([self.x1d], seed=42, mesh_type="unstructured")

    def test_repr(self):
        ds = DirectSampling(self.ti1d)
        r = repr(ds)
        self.assertIsInstance(r, str)
        self.assertIn("DirectSampling", r)

    def test_properties_and_setters(self):
        ds = DirectSampling(
            self.ti1d,
            n_neighbors=16,
            scan_fraction=0.5,
            threshold=0.05,
            cond_weight=2.0,
            boundary="partial",
            max_radius=3.0,
        )
        self.assertIs(ds.ti, self.ti1d)
        self.assertEqual(ds.n_neighbors, 16)
        self.assertAlmostEqual(ds.scan_fraction, 0.5)
        self.assertAlmostEqual(ds.threshold, 0.05)
        self.assertAlmostEqual(ds.cond_weight, 2.0)
        self.assertEqual(ds.boundary, "partial")
        self.assertAlmostEqual(ds.max_radius, 3.0)

        ds.n_neighbors = 8
        self.assertEqual(ds.n_neighbors, 8)
        ds.scan_fraction = 1.0
        self.assertAlmostEqual(ds.scan_fraction, 1.0)
        ds.threshold = 0.0
        self.assertAlmostEqual(ds.threshold, 0.0)
        ds.cond_weight = 1.0
        self.assertAlmostEqual(ds.cond_weight, 1.0)

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
        ds = DirectSampling(self.ti1d, n_neighbors=4, scan_fraction=1.0)
        field = ds([self.x1d], seed=42)
        self.assertEqual(field.shape, (10,))
        self.assertFalse(np.any(np.isnan(field)))

    def test_shape_2d(self):
        ds = DirectSampling(self.ti2d, n_neighbors=4, scan_fraction=1.0)
        field = ds([self.x2d, self.y2d], seed=42)
        self.assertEqual(field.shape, (6, 6))
        self.assertFalse(np.any(np.isnan(field)))
        # All output values must be in the TI value set {0, 1}
        unique_vals = set(np.unique(field))
        self.assertTrue(unique_vals.issubset({0.0, 1.0}))

    def test_regression_1d(self):
        ds = DirectSampling(self.ti1d, n_neighbors=4, scan_fraction=1.0)
        field = ds([self.x1d], seed=42)
        self.assertAlmostEqual(field[0], 1.0)
        self.assertAlmostEqual(field[5], 0.0)
        self.assertAlmostEqual(field[9], 0.0)

    def test_regression_2d(self):
        ds = DirectSampling(self.ti2d, n_neighbors=4, scan_fraction=1.0)
        field = ds([self.x2d, self.y2d], seed=42)
        self.assertAlmostEqual(field[0, 0], 1.0)
        self.assertAlmostEqual(field[2, 3], 0.0)
        self.assertAlmostEqual(field[5, 5], 1.0)

    def test_seeded_reproducibility(self):
        ds = DirectSampling(self.ti2d_rand, n_neighbors=8, scan_fraction=0.5)
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
        ds = DirectSampling(self.ti1d, n_neighbors=4, scan_fraction=1.0)
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
            self.ti2d, n_neighbors=4, scan_fraction=1.0, boundary="partial"
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
            ti_tiny, n_neighbors=32, scan_fraction=1.0, boundary="partial"
        )
        field = ds([np.arange(30, dtype=float)] * 2, seed=1)
        self.assertEqual(field.shape, (30, 30))
        self.assertFalse(np.any(np.isnan(field)))

    def test_threshold_above_one_warns_in_constructor(self):
        with self.assertWarns(UserWarning):
            DirectSampling(self.ti1d, threshold=5.0)

    def test_scan_fraction_window_semantics(self):
        """scan_fraction=0.1 applies to the window, not the TI — no crash, valid output."""
        rng = np.random.default_rng(0)
        ti = TrainingImage(
            rng.integers(0, 2, (20, 20)).astype(float), categorical=True
        )
        ds = DirectSampling(ti, n_neighbors=4, scan_fraction=0.1)
        field = ds([np.arange(6, dtype=float)] * 2, seed=0)
        self.assertEqual(field.shape, (6, 6))
        self.assertFalse(np.any(np.isnan(field)))
        self.assertTrue(set(np.unique(field)).issubset({0.0, 1.0}))

    def test_max_radius(self):
        ds = DirectSampling(
            self.ti2d, n_neighbors=4, scan_fraction=1.0, max_radius=2.0
        )
        field = ds([self.x2d, self.y2d], seed=42)
        self.assertEqual(field.shape, (6, 6))
        self.assertFalse(np.any(np.isnan(field)))

    def test_continuous_ti(self):
        ds = DirectSampling(
            self.ti1d_cont, n_neighbors=4, scan_fraction=1.0, threshold=0.05
        )
        field = ds([np.arange(8, dtype=float)], seed=42)
        self.assertEqual(field.shape, (8,))
        self.assertFalse(np.any(np.isnan(field)))
        self.assertTrue(np.all(field >= 0.0))
        self.assertTrue(np.all(field <= 1.0))

    def test_ds_simulate_direct(self):
        result = ds_simulate(
            self.ti1d,
            sim_shape=(8,),
            n_neighbors=4,
            threshold=0.0,
            scan_fraction=1.0,
            rng=np.random.RandomState(7),
        )
        self.assertEqual(result.shape, (8,))
        self.assertFalse(np.any(np.isnan(result)))
        # Check values — seeded values for ds_simulate(seed=7) with ti1d
        self.assertTrue(set(np.unique(result)).issubset({0.0, 1.0}))

    def test_empty_search_window_recovery(self):
        # n_neighbors >> TI size collapses search windows → must recover silently
        ti_tiny = TrainingImage(np.array([0.0, 1.0, 0.0]), categorical=True)
        ds = DirectSampling(ti_tiny, n_neighbors=10, scan_fraction=1.0)
        field = ds([np.arange(5, dtype=float)], seed=1)
        self.assertEqual(field.shape, (5,))
        self.assertFalse(np.any(np.isnan(field)))
        self.assertTrue(set(np.unique(field)).issubset({0.0, 1.0}))

    def test_gstools_namespace(self):
        self.assertIs(gs.DirectSampling, DirectSampling)
        self.assertIs(gs.TrainingImage, TrainingImage)
        self.assertIs(gs.mps.DirectSampling, DirectSampling)
        self.assertIs(gs.mps.TrainingImage, TrainingImage)


class TestMultivariateTrainingImage(unittest.TestCase):
    def test_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "same shape"):
            TrainingImage({"a": np.zeros((4, 4)), "b": np.zeros((3, 3))})

    def test_equal_weights(self):
        ti = TrainingImage(
            {"a": np.zeros((4, 4), dtype=int), "b": np.ones((4, 4), dtype=int)}
        )
        self.assertTrue(ti.multivariate)
        self.assertEqual(ti.variables, ["a", "b"])
        self.assertAlmostEqual(ti.weights["a"], 0.5)
        self.assertAlmostEqual(ti.weights["b"], 0.5)
        self.assertEqual(ti.shape, (4, 4))
        self.assertEqual(ti.ndim, 2)

    def test_weights_must_sum_to_one(self):
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            TrainingImage(
                {"a": np.zeros((4, 4), dtype=int), "b": np.zeros((4, 4), dtype=int)},
                weights={"a": 0.3, "b": 0.3},
            )

    def test_distance_weighted_sum(self):
        # w=0.5 each; d_a=1.0 (full mismatch), d_b=0.0 -> joint = 0.5
        ti = TrainingImage(
            {"a": np.zeros((4, 4), dtype=int), "b": np.zeros((4, 4), dtype=int)},
            weights={"a": 0.5, "b": 0.5},
        )
        de_sg = {"a": np.array([1, 1]), "b": np.array([0, 0])}
        de_ti = {"a": np.array([0, 0]), "b": np.array([0, 0])}
        self.assertAlmostEqual(ti.distance(de_sg, de_ti), 0.5)

    def test_per_variable_categorical_and_distance(self):
        ti = TrainingImage(
            {"cat": np.zeros((4, 4), dtype=int),
             "cont": np.linspace(0, 100, 16).reshape(4, 4)},
            categorical={"cat": True, "cont": False},
            distance={"cat": "l1", "cont": "l2"},
        )
        self.assertTrue(ti.categorical["cat"])
        self.assertFalse(ti.categorical["cont"])
        self.assertAlmostEqual(ti._d_max["cont"], 100.0)

    def test_univariate_variation_p_preserved(self):
        # Regression guard: the shared _parse_distance refactor must keep
        # variation<p> support and the _variation_p_norm attribute intact.
        ti = TrainingImage(
            np.linspace(0, 100, 16).reshape(4, 4),
            categorical=False,
            distance="variation1.5",
        )
        self.assertFalse(ti.multivariate)
        self.assertIsNone(ti.variables)
        self.assertIsNone(ti.weights)
        self.assertEqual(ti._variation_p_norm, 1.5)
        self.assertIsNone(ti._p_norm)

    def test_variable_accessor(self):
        a = np.arange(16).reshape(4, 4)
        ti = TrainingImage({"a": a, "b": np.zeros((4, 4), dtype=int)})
        np.testing.assert_array_equal(ti.variable("a"), a)
        # univariate TIs have no named variables
        uni = TrainingImage(np.zeros((4, 4), dtype=int))
        with self.assertRaises(TypeError):
            uni.variable("a")

    def test_adjust_value_var(self):
        ti = TrainingImage(
            {"v": np.linspace(0, 100, 16).reshape(4, 4),
             "c": np.zeros((4, 4), dtype=int)},
            categorical={"v": False, "c": True},
            distance={"v": "variation", "c": "l1"},
        )
        # variation: Z(y) - mean(de_ti) + mean(de_sim) = 50 - 50 + 20 = 20
        self.assertAlmostEqual(
            ti.adjust_value_var("v", 50.0, np.array([10.0, 20.0, 30.0]),
                                np.array([40.0, 50.0, 60.0])),
            20.0,
        )
        # categorical variable: returned unchanged
        self.assertEqual(
            ti.adjust_value_var("c", 1.0, np.array([0, 1]), np.array([1, 0])), 1.0
        )
        # empty data event: returned unchanged even for variation
        self.assertAlmostEqual(
            ti.adjust_value_var("v", 7.0, np.array([]), np.array([])), 7.0
        )

    def test_weights_unknown_variable(self):
        with self.assertRaisesRegex(ValueError, "unknown"):
            TrainingImage(
                {"a": np.zeros((4, 4), dtype=int),
                 "b": np.zeros((4, 4), dtype=int)},
                weights={"a": 0.5, "b": 0.5, "ghost": 0.0},
            )


class TestMultivariateDirectSampling(unittest.TestCase):
    def test_fills_all_nodes(self):
        # Node-wise path must leave no NaN in any variable.
        rng = np.random.default_rng(42)
        data = {"a": rng.integers(0, 2, (20, 20)),
                "b": rng.integers(0, 2, (20, 20))}
        ti = TrainingImage(data)
        ds = DirectSampling(ti, n_neighbors=4, scan_fraction=0.3)
        field = ds([np.arange(6, dtype=float)] * 2, seed=1)
        self.assertFalse(np.any(np.isnan(field["a"])))
        self.assertFalse(np.any(np.isnan(field["b"])))

    def test_output_shapes(self):
        rng = np.random.default_rng(0)
        data = {"primary": rng.integers(0, 3, (20, 20)),
                "secondary": rng.integers(0, 2, (20, 20))}
        ti = TrainingImage(data)
        ds = DirectSampling(ti, n_neighbors=8, scan_fraction=0.1)
        field = ds([np.arange(10, dtype=float)] * 2, seed=0)
        self.assertEqual(field["primary"].shape, (10, 10))
        self.assertIn("secondary", field)
        self.assertEqual(field["secondary"].shape, (10, 10))

    def test_values_valid(self):
        rng = np.random.default_rng(1)
        data = {"a": rng.integers(0, 4, (20, 20)),
                "b": rng.integers(0, 2, (20, 20))}
        ti = TrainingImage(data)
        ds = DirectSampling(ti, n_neighbors=4, scan_fraction=0.2)
        field = ds([np.arange(8, dtype=float)] * 2, seed=5)
        self.assertTrue(np.all(np.isin(field["a"], [0, 1, 2, 3])))
        self.assertTrue(np.all(np.isin(field["b"], [0, 1])))

    def test_per_variable_n_neighbors(self):
        rng = np.random.default_rng(3)
        data = {"primary": rng.integers(0, 3, (20, 20)),
                "secondary": rng.integers(0, 2, (20, 20))}
        ti = TrainingImage(data)
        ds = DirectSampling(ti, n_neighbors={"primary": 8, "secondary": 2},
                            scan_fraction=0.2)
        field = ds([np.arange(8, dtype=float)] * 2, seed=7)
        self.assertEqual(field["primary"].shape, (8, 8))
        self.assertFalse(np.any(np.isnan(field["primary"])))
        self.assertFalse(np.any(np.isnan(field["secondary"])))

    def test_3d_runs(self):
        rng = np.random.default_rng(8)
        data = {"a": rng.integers(0, 2, (10, 10, 10)),
                "b": rng.integers(0, 2, (10, 10, 10))}
        ti = TrainingImage(data)
        ds = DirectSampling(ti, n_neighbors=4, scan_fraction=0.2)
        field = ds([np.arange(4, dtype=float)] * 3, seed=0)
        self.assertEqual(field["a"].shape, (4, 4, 4))
        self.assertFalse(np.any(np.isnan(field["a"])))
        self.assertFalse(np.any(np.isnan(field["b"])))

    def test_partial_boundary_runs(self):
        rng = np.random.default_rng(11)
        data = {"a": rng.integers(0, 2, (20, 20)),
                "b": rng.integers(0, 2, (20, 20))}
        ti = TrainingImage(data)
        ds = DirectSampling(ti, n_neighbors=6, scan_fraction=0.3,
                            boundary="partial")
        field = ds([np.arange(8, dtype=float)] * 2, seed=2)
        self.assertFalse(np.any(np.isnan(field["a"])))
        self.assertFalse(np.any(np.isnan(field["b"])))
        self.assertTrue(np.all(np.isin(field["a"], [0, 1])))

    def test_ds_mode_threshold(self):
        rng = np.random.default_rng(12)
        data = {"a": rng.integers(0, 3, (20, 20)),
                "b": rng.integers(0, 2, (20, 20))}
        ti = TrainingImage(data)
        ds = DirectSampling(ti, n_neighbors=4, scan_fraction=0.5, threshold=0.3)
        field = ds([np.arange(8, dtype=float)] * 2, seed=4)
        self.assertFalse(np.any(np.isnan(field["a"])))
        self.assertTrue(np.all(np.isin(field["a"], [0, 1, 2])))
        self.assertTrue(np.all(np.isin(field["b"], [0, 1])))

    def test_continuous_variation_variable(self):
        # A continuous variable with variation distance exercises the
        # mean-shift adjust_value_var path end to end.
        rng = np.random.default_rng(13)
        data = {"cont": rng.random((20, 20)) * 10.0,
                "cat": rng.integers(0, 2, (20, 20))}
        ti = TrainingImage(
            data,
            categorical={"cont": False, "cat": True},
            distance={"cont": "variation", "cat": "l1"},
        )
        ds = DirectSampling(ti, n_neighbors=6, scan_fraction=0.3)
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
            {"a": ids, "b": ids + 100},
            categorical={"a": True, "b": True},
        )
        ds = DirectSampling(ti, n_neighbors=8, scan_fraction=1.0, threshold=0.0)
        field = ds([np.arange(6, dtype=float)] * 2, seed=0)
        np.testing.assert_array_equal(field["b"], field["a"] + 100)

    def test_equal_treatment_named_fields(self):
        # No privileged primary: all variables are first-class named fields.
        rng = np.random.default_rng(20)
        ti = TrainingImage({"x": rng.integers(0, 2, (15, 15)),
                            "y": rng.integers(0, 2, (15, 15))})
        ds = DirectSampling(ti, n_neighbors=4, scan_fraction=0.3)
        field = ds([np.arange(6, dtype=float)] * 2, seed=0)
        self.assertEqual(set(field), {"x", "y"})
        self.assertIn("x", ds.field_names)
        self.assertIn("y", ds.field_names)
        np.testing.assert_array_equal(ds["x"], field["x"])
        np.testing.assert_array_equal(ds["y"], field["y"])

    def test_invalid_variable_name_raises(self):
        ti = TrainingImage({"a b": np.zeros((5, 5), dtype=int),
                            "c": np.zeros((5, 5), dtype=int)})
        with self.assertRaisesRegex(ValueError, "field name"):
            DirectSampling(ti, n_neighbors=4)

    def test_n_neighbors_dict_requires_multivariate(self):
        ti = TrainingImage(np.zeros((10, 10), dtype=int))
        with self.assertRaisesRegex(ValueError, "multivariate"):
            DirectSampling(ti, n_neighbors={"a": 4})

    def test_n_neighbors_dict_keys_must_match(self):
        ti = TrainingImage({"a": np.zeros((10, 10), dtype=int),
                            "b": np.zeros((10, 10), dtype=int)})
        with self.assertRaisesRegex(ValueError, "keys must match"):
            DirectSampling(ti, n_neighbors={"a": 4})

    def test_set_condition_basic(self):
        rng = np.random.default_rng(0)
        data = {"a": rng.integers(0, 2, (20, 20)),
                "b": rng.integers(0, 2, (20, 20))}
        ti = TrainingImage(data)
        ds = DirectSampling(ti, n_neighbors=4, scan_fraction=0.3)
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
        data = {"a": rng.integers(0, 2, (20, 20)),
                "b": rng.integers(0, 2, (20, 20))}
        ti = TrainingImage(data)
        ds = DirectSampling(ti, n_neighbors=4, scan_fraction=0.3)
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
        data = {"a": rng.integers(0, 2, (20, 20)),
                "b": rng.integers(0, 2, (20, 20))}
        ti = TrainingImage(data)
        ds = DirectSampling(ti, n_neighbors=4, scan_fraction=0.3)
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
        ti = TrainingImage({"a": na, "b": nb})
        ds = DirectSampling(ti, n_neighbors=4, scan_fraction=1.0, threshold=0.0)
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
        ds = DirectSampling(ti, n_neighbors=8, scan_fraction=0.2)
        ds.set_condition([[5.0], [5.0]], [2])
        field = ds([np.arange(10, dtype=float)] * 2, seed=0)
        self.assertEqual(field[5, 5], 2)

    def test_set_condition_length_mismatch(self):
        ti = TrainingImage({"a": np.zeros((10, 10), dtype=int),
                            "b": np.zeros((10, 10), dtype=int)})
        ds = DirectSampling(ti, n_neighbors=4, scan_fraction=0.3)
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
        data = {"a": rng.integers(0, 3, (20, 20)),
                "b": rng.integers(0, 2, (20, 20))}
        ti = TrainingImage(data)
        ds = DirectSampling(ti, n_neighbors=4, scan_fraction=0.2, num_threads=2)
        field = ds([np.arange(8, dtype=float)] * 2, seed=0)
        self.assertTrue(np.all(np.isin(field["a"], [0, 1, 2])))
        self.assertTrue(np.all(np.isin(field["b"], [0, 1])))

    def test_parallel_matches_serial(self):
        # The node-vertex DAG commits every per-variable neighbour before a node
        # runs, so num_threads > 1 is bit-identical to serial.
        rng = np.random.default_rng(5)
        data = {"a": rng.integers(0, 3, (20, 20)),
                "b": rng.integers(0, 2, (20, 20))}
        ti = TrainingImage(data)
        pos = [np.arange(8, dtype=float)] * 2
        ds_s = DirectSampling(ti, n_neighbors=4, scan_fraction=0.2, num_threads=1)
        ds_p = DirectSampling(ti, n_neighbors=4, scan_fraction=0.2, num_threads=2)
        f_s = ds_s(pos, seed=7)
        f_p = ds_p(pos, seed=7)
        np.testing.assert_array_equal(f_s["a"], f_p["a"])
        np.testing.assert_array_equal(f_s["b"], f_p["b"])

    def test_parallel_conditioning_preserved(self):
        rng = np.random.default_rng(6)
        data = {"a": rng.integers(0, 2, (20, 20)),
                "b": rng.integers(0, 2, (20, 20))}
        ti = TrainingImage(data)
        ds = DirectSampling(ti, n_neighbors=4, scan_fraction=0.3, num_threads=2)
        ds.set_condition(
            cond_pos=[[3.0], [3.0]],
            cond_val={"a": np.array([1]), "b": np.array([0])},
        )
        field = ds([np.arange(8, dtype=float)] * 2, seed=9)
        self.assertEqual(field["a"][3, 3], 1)
        self.assertEqual(field["b"][3, 3], 0)

    def test_set_condition_array_on_multivariate_raises(self):
        ti = TrainingImage({"a": np.zeros((10, 10), dtype=int),
                            "b": np.zeros((10, 10), dtype=int)})
        ds = DirectSampling(ti, n_neighbors=4, scan_fraction=0.3)
        with self.assertRaisesRegex(ValueError, "dict"):
            ds.set_condition([[3.0], [3.0]], np.array([1]))

    def test_set_condition_collision_nan_drops_variable(self):
        # Closer point {a:1, b:nan} wins over farther {a:0, b:0}; a is pinned to
        # the closer value and b is left to the simulation (filled, finite).
        rng = np.random.default_rng(7)
        data = {"a": rng.integers(0, 2, (20, 20)),
                "b": rng.integers(0, 2, (20, 20))}
        ti = TrainingImage(data)
        ds = DirectSampling(ti, n_neighbors=4, scan_fraction=0.3)
        ds.set_condition(
            cond_pos=[[4.4, 4.1], [4.4, 4.1]],
            cond_val={"a": np.array([0, 1]), "b": np.array([0, np.nan])},
        )
        field = ds([np.arange(8, dtype=float)] * 2, seed=1)
        self.assertEqual(field["a"][4, 4], 1)
        self.assertFalse(np.isnan(field["b"][4, 4]))

    def test_distance_power_multivariate(self):
        # distance_power > 0 exercises the per-variable lag-norm weighting path.
        rng = np.random.default_rng(8)
        data = {"cont": rng.random((20, 20)) * 10.0,
                "cat": rng.integers(0, 2, (20, 20))}
        ti = TrainingImage(
            data,
            categorical={"cont": False, "cat": True},
            distance={"cont": "l2", "cat": "l1"},
            distance_power=1.0,
        )
        ds = DirectSampling(ti, n_neighbors=6, scan_fraction=0.3)
        field = ds([np.arange(8, dtype=float)] * 2, seed=3)
        self.assertEqual(field["cont"].shape, (8, 8))
        self.assertTrue(np.all(np.isfinite(field["cont"])))
        self.assertTrue(np.all(np.isin(field["cat"], [0, 1])))

    def test_single_variable_multivariate(self):
        rng = np.random.default_rng(9)
        ti = TrainingImage({"only": rng.integers(0, 3, (20, 20))})
        ds = DirectSampling(ti, n_neighbors=4, scan_fraction=0.3)
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
            {"a": ids, "b": ids + 100},
            categorical={"a": True, "b": True},
        )
        # scan_fraction=1.0, threshold=0.01: very strict but must still complete
        # without NaN and must reproduce the joint relationship everywhere.
        ds = DirectSampling(ti, n_neighbors=8, scan_fraction=1.0, threshold=0.01)
        field = ds([np.arange(6, dtype=float)] * 2, seed=0)
        self.assertFalse(np.any(np.isnan(field["a"])))
        self.assertFalse(np.any(np.isnan(field["b"])))
        np.testing.assert_array_equal(field["b"], field["a"] + 100)


class TestNonstationarity(unittest.TestCase):
    """Geometric non-stationarity for DirectSampling (set_nonstationary)."""

    def _make_ds(self, ti_shape=(20, 20), **kw):
        rng = np.random.default_rng(0)
        data = rng.integers(0, 3, ti_shape)
        ti = gs.mps.TrainingImage(data)
        defaults = dict(n_neighbors=4, scan_fraction=0.2)
        defaults.update(kw)
        return gs.mps.DirectSampling(ti, **defaults), ti

    def test_scalar_rotation_valid_values(self):
        ds, ti = self._make_ds()
        ds.set_nonstationary(rotation=np.pi / 4)
        pos = [np.arange(8, dtype=float)] * 2
        field = ds(pos, seed=0)
        self.assertEqual(field.shape, (8, 8))
        self.assertTrue(np.all(np.isin(field, [0, 1, 2])))

    def test_rotation_changes_output(self):
        ds_plain, ti = self._make_ds(ti_shape=(30, 30), n_neighbors=8)
        pos = [np.arange(10, dtype=float)] * 2
        f_plain = ds_plain(pos, seed=7)

        ds_rot = gs.mps.DirectSampling(ti, n_neighbors=8, scan_fraction=0.2)
        ds_rot.set_nonstationary(rotation=np.pi / 4)
        f_rot = ds_rot(pos, seed=7)
        self.assertFalse(np.array_equal(f_plain, f_rot))

    def test_array_rotation_map_runs(self):
        ds, _ = self._make_ds()
        angle_map = np.linspace(0, np.pi / 2, 64).reshape(8, 8)
        ds.set_nonstationary(rotation=angle_map)
        field = ds([np.arange(8, dtype=float)] * 2, seed=1)
        self.assertEqual(field.shape, (8, 8))
        self.assertTrue(np.all(np.isin(field, [0, 1, 2])))

    def test_anis_changes_output(self):
        ds_plain, ti = self._make_ds(ti_shape=(30, 30), n_neighbors=8)
        pos = [np.arange(10, dtype=float)] * 2
        f_plain = ds_plain(pos, seed=3)

        ds_anis = gs.mps.DirectSampling(ti, n_neighbors=8, scan_fraction=0.2)
        ds_anis.set_nonstationary(anis=0.5)
        f_anis = ds_anis(pos, seed=3)
        self.assertFalse(np.array_equal(f_plain, f_anis))

    def test_combined_rotation_anis_runs(self):
        ds, _ = self._make_ds()
        ds.set_nonstationary(rotation=np.pi / 6, anis=0.5)
        field = ds([np.arange(8, dtype=float)] * 2, seed=2)
        self.assertEqual(field.shape, (8, 8))
        self.assertTrue(np.all(np.isin(field, [0, 1, 2])))

    def test_conditioning_preserved(self):
        ds, _ = self._make_ds(scan_fraction=0.3)
        ds.set_nonstationary(rotation=np.pi / 4)
        ds.set_condition([[4.0], [4.0]], [2])
        field = ds([np.arange(8, dtype=float)] * 2, seed=0)
        self.assertEqual(int(field[4, 4]), 2)

    def test_partial_boundary_runs(self):
        ds, _ = self._make_ds(boundary="partial")
        ds.set_nonstationary(rotation=np.pi / 4)
        field = ds([np.arange(8, dtype=float)] * 2, seed=0)
        self.assertTrue(np.all(np.isin(field, [0, 1, 2])))

    def test_3d_rotation_runs(self):
        rng = np.random.default_rng(0)
        data = rng.integers(0, 2, (12, 12, 12))
        ti = gs.mps.TrainingImage(data)
        ds = gs.mps.DirectSampling(ti, n_neighbors=4, scan_fraction=0.1)
        ds.set_nonstationary(rotation=np.pi / 4)
        field = ds([np.arange(5, dtype=float)] * 3, seed=0)
        self.assertEqual(field.shape, (5, 5, 5))
        self.assertTrue(np.all(np.isin(field, [0, 1])))

    def test_collapsed_window_fallback(self):
        # 90° rotation on a tiny TI can force all windows to collapse.
        # Output must be finite and within TI values — no crash, no NaN.
        rng = np.random.default_rng(0)
        data = rng.integers(0, 2, (4, 4))
        ti = gs.mps.TrainingImage(data)
        ds = gs.mps.DirectSampling(ti, n_neighbors=8, scan_fraction=1.0)
        ds.set_nonstationary(rotation=np.pi / 2)
        field = ds([np.arange(6, dtype=float)] * 2, seed=0)
        self.assertTrue(np.all(np.isfinite(field)))
        self.assertTrue(np.all(np.isin(field, [0, 1])))

    def test_identity_matches_no_transform(self):
        # θ=0, anis=1 must produce bit-identical output to the plain path.
        ds_plain, ti = self._make_ds(ti_shape=(20, 20), n_neighbors=8)
        pos = [np.arange(8, dtype=float)] * 2
        f_plain = ds_plain(pos, seed=5)

        ds_id = gs.mps.DirectSampling(ti, n_neighbors=8, scan_fraction=0.2)
        ds_id.set_nonstationary(rotation=0.0, anis=1.0)
        f_id = ds_id(pos, seed=5)
        np.testing.assert_array_equal(f_plain, f_id)

    def test_multivariate_rotation_valid_output(self):
        # Exercises the ds_simulate_mv transform path: shape, value subsets,
        # and that rotation changes output vs no rotation (same seed).
        rng = np.random.default_rng(0)
        ti = gs.mps.TrainingImage(
            {"a": rng.integers(0, 3, (20, 20)), "b": rng.random((20, 20))},
            categorical={"a": True, "b": False},
        )
        pos = [np.arange(8, dtype=float)] * 2

        ds_plain = gs.mps.DirectSampling(ti, n_neighbors=4, scan_fraction=0.2)
        result_plain = ds_plain(pos, seed=9)

        ds_rot = gs.mps.DirectSampling(ti, n_neighbors=4, scan_fraction=0.2)
        ds_rot.set_nonstationary(rotation=np.pi / 4)
        result_rot = ds_rot(pos, seed=9)

        self.assertEqual(result_rot["a"].shape, (8, 8))
        self.assertEqual(result_rot["b"].shape, (8, 8))
        self.assertTrue(np.all(np.isin(result_rot["a"], [0, 1, 2])))
        self.assertTrue(np.all(np.isfinite(result_rot["b"])))
        self.assertFalse(np.array_equal(result_plain["a"], result_rot["a"]))


if __name__ == "__main__":
    unittest.main()
