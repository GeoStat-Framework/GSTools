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
        ds = DirectSampling(
            ti, n_neighbors=8, scan_fraction=0.2, num_threads=2
        )
        field = ds([np.arange(8, dtype=float)] * 2, seed=0)
        self.assertEqual(field.shape, (8, 8))
        self.assertTrue(np.all(np.isin(field, [0, 1, 2])))

    def test_reproducible(self):
        # DAG parallelism is deterministic: same seed → same parallel result
        rng = np.random.default_rng(0)
        data = rng.integers(0, 3, (20, 20))
        ti = TrainingImage(data)
        ds = DirectSampling(
            ti, n_neighbors=8, scan_fraction=0.2, num_threads=2
        )
        pos = [np.arange(8, dtype=float)] * 2
        self.assertTrue(np.array_equal(ds(pos, seed=7), ds(pos, seed=7)))

    def test_conditioning_preserved(self):
        rng = np.random.default_rng(0)
        data = rng.integers(0, 3, (20, 20))
        ti = TrainingImage(data)
        ds = DirectSampling(
            ti, n_neighbors=4, scan_fraction=0.2, num_threads=2
        )
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
        ds = DirectSampling(
            ti, n_neighbors=2, scan_fraction=0.3, num_threads=4
        )
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

    def test_serial_equals_parallel(self):
        rng = np.random.default_rng(0)
        cases = [
            TrainingImage(
                rng.integers(0, 3, (25, 25)).astype(float), categorical=True
            ),
            TrainingImage(
                rng.random((25, 25)), categorical=False, distance="l2"
            ),
            TrainingImage(
                rng.random((25, 25)), categorical=False, distance="variation"
            ),
        ]
        pos = [np.arange(16, dtype=float)] * 2
        for ti in cases:
            serial = DirectSampling(
                ti, n_neighbors=8, scan_fraction=0.5, num_threads=1
            )(pos, seed=7)
            for nt in (2, 4, 8):
                parallel = DirectSampling(
                    ti, n_neighbors=8, scan_fraction=0.5, num_threads=nt
                )(pos, seed=7)
                self.assertTrue(
                    np.array_equal(serial, parallel),
                    msg=f"serial != parallel (num_threads={nt})",
                )


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
        with self.assertRaises(ValueError):
            DirectSampling(self.ti1d, n_neighbors=0)
        with self.assertRaises(ValueError):
            DirectSampling(self.ti1d, scan_fraction=0)
        with self.assertRaises(ValueError):
            DirectSampling(self.ti1d, scan_fraction=1.5)
        with self.assertRaises(ValueError):
            DirectSampling(self.ti1d, threshold=-0.1)
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

    def test_post_process_categorical_noop(self):
        # Default construction leaves mean/normalizer/trend unset, so
        # post_process must not alter categorical output (only float cast).
        ds = DirectSampling(self.ti2d, n_neighbors=4, scan_fraction=1.0)
        processed = ds([self.x2d, self.y2d], seed=7, post_process=True)
        raw = ds([self.x2d, self.y2d], seed=7, post_process=False)
        np.testing.assert_array_equal(processed, raw)
        self.assertTrue(set(np.unique(processed)).issubset({0.0, 1.0}))

    def test_conditioning_outside_grid_snaps_to_boundary(self):
        # A conditioning point beyond the domain snaps to the nearest grid node
        # (current behaviour: silent boundary snap, no error).
        ds = DirectSampling(self.ti2d, n_neighbors=4, scan_fraction=1.0)
        ds.set_condition([[100.0], [100.0]], [1.0])  # far outside the 6x6 grid
        field = ds([self.x2d, self.y2d], seed=7)
        self.assertAlmostEqual(field[5, 5], 1.0)

    def test_max_radius_below_one_random_fill(self):
        # max_radius in (0, 1) excludes every neighbour (nearest cell is at
        # distance 1.0), so each node falls back to a random TI sample. Must
        # still run and produce valid values.
        ds = DirectSampling(
            self.ti2d, n_neighbors=4, scan_fraction=1.0, max_radius=0.5
        )
        field = ds([self.x2d, self.y2d], seed=7)
        self.assertEqual(field.shape, (6, 6))
        self.assertFalse(np.any(np.isnan(field)))
        self.assertTrue(set(np.unique(field)).issubset({0.0, 1.0}))

    def test_simulation_3d(self):
        rng = np.random.default_rng(0)
        ti = TrainingImage(
            rng.integers(0, 2, (10, 10, 10)).astype(float), categorical=True
        )
        ds = DirectSampling(ti, n_neighbors=6, scan_fraction=0.5)
        field = ds([np.arange(6, dtype=float)] * 3, seed=7)
        self.assertEqual(field.shape, (6, 6, 6))
        self.assertFalse(np.any(np.isnan(field)))
        self.assertTrue(set(np.unique(field)).issubset({0.0, 1.0}))

    def test_continuous_2d_simulation(self):
        rng = np.random.default_rng(0)
        ti = TrainingImage(
            rng.random((20, 20)), categorical=False, distance="l2"
        )
        ds = DirectSampling(
            ti, n_neighbors=6, scan_fraction=0.5, threshold=0.05
        )
        field = ds([np.arange(10, dtype=float)] * 2, seed=7)
        self.assertEqual(field.shape, (10, 10))
        self.assertFalse(np.any(np.isnan(field)))
        # values are copied from the TI, so they stay within the TI range
        self.assertGreaterEqual(field.min(), ti.data.min())
        self.assertLessEqual(field.max(), ti.data.max())

    def test_lp_distance_simulation(self):
        # Exercises the vectorized Lp scan path through a full simulation
        # (p != 1, 2), which the unit-level distance tests do not cover.
        rng = np.random.default_rng(2)
        ti = TrainingImage(
            rng.random((20, 20)), categorical=False, distance="l3"
        )
        ds = DirectSampling(
            ti, n_neighbors=6, scan_fraction=0.5, threshold=0.05
        )
        field = ds([np.arange(10, dtype=float)] * 2, seed=7)
        self.assertEqual(field.shape, (10, 10))
        self.assertFalse(np.any(np.isnan(field)))
        self.assertGreaterEqual(field.min(), ti.data.min())
        self.assertLessEqual(field.max(), ti.data.max())

    def test_variation_distance_simulation(self):
        # The variation metric applies the mean-shift on assignment, so values
        # may leave the raw TI range; they must stay finite and non-NaN.
        rng = np.random.default_rng(1)
        ti = TrainingImage(
            rng.random((20, 20)), categorical=False, distance="variation"
        )
        ds = DirectSampling(
            ti, n_neighbors=6, scan_fraction=0.5, threshold=0.05
        )
        field = ds([np.arange(10, dtype=float)] * 2, seed=7)
        self.assertEqual(field.shape, (10, 10))
        self.assertTrue(np.all(np.isfinite(field)))

    def test_gstools_namespace(self):
        self.assertIs(gs.DirectSampling, DirectSampling)
        self.assertIs(gs.TrainingImage, TrainingImage)
        self.assertIs(gs.mps.DirectSampling, DirectSampling)
        self.assertIs(gs.mps.TrainingImage, TrainingImage)


if __name__ == "__main__":
    unittest.main()
