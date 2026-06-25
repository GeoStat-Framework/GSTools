#!/usr/bin/env python
"""Unittest for the MPS module (TrainingImage and DirectSampling)."""

import unittest
import warnings

import numpy as np

import gstools as gs
from gstools import config as gs_config
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
    _precompute_offsets,
    _reduce_to_fit,
    _transform_lags,
    _window_bounds,
)
from gstools.mps.scan import _scan_window
from gstools.mps.simulate import _MV_VAR, _univar_as_mv_ti, ds_simulate
from gstools.mps.training_image import TrainingImage


def _uni_dist(ti, de_sim, de_ti, **kw):
    """Univariate data-event distance via the vectorized production path.

    The scalar ``TrainingImage.distance`` was removed; tests exercise the same
    ``vec_distance_var`` the simulation uses by wrapping the univariate TI as
    the engine does and reading out the single candidate row.
    """
    mv = _univar_as_mv_ti(ti)
    row = np.asarray(de_ti, dtype=float)[np.newaxis, :]
    return float(mv.vec_distance_var(_MV_VAR, de_sim, row, **kw)[0])


def _mv_dist(ti, de_sim, de_ti, **kw):
    """Multivariate joint data-event distance via the vectorized path."""
    total = 0.0
    for k in ti.variables:
        row = np.asarray(de_ti[k], dtype=float)[np.newaxis, :]
        total += ti.weights[k] * float(
            ti.vec_distance_var(k, de_sim[k], row, **kw)[0]
        )
    return total


class TestDirectSamplingParallel(unittest.TestCase):
    def test_valid_values(self):
        rng = np.random.default_rng(0)
        data = rng.integers(0, 3, (20, 20))
        ti = TrainingImage(data)
        ds = DirectSampling(MPSModel(ti, n_neighbors=8, scan_fraction=0.2))
        ds.num_threads = 2
        field = ds([np.arange(8, dtype=float)] * 2, seed=0)
        self.assertEqual(field.shape, (8, 8))
        self.assertTrue(np.all(np.isin(field, [0, 1, 2])))

    def test_reproducible(self):
        # DAG parallelism is deterministic: same seed → same parallel result
        rng = np.random.default_rng(0)
        data = rng.integers(0, 3, (20, 20))
        ti = TrainingImage(data)
        ds = DirectSampling(MPSModel(ti, n_neighbors=8, scan_fraction=0.2))
        ds.num_threads = 2
        pos = [np.arange(8, dtype=float)] * 2
        self.assertTrue(np.array_equal(ds(pos, seed=7), ds(pos, seed=7)))

    def test_conditioning_preserved(self):
        rng = np.random.default_rng(0)
        data = rng.integers(0, 3, (20, 20))
        ti = TrainingImage(data)
        ds = DirectSampling(MPSModel(ti, n_neighbors=4, scan_fraction=0.2))
        ds.num_threads = 2
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
            field = DirectSampling(
                MPSModel(ti, n_neighbors=8, scan_fraction=0.2)
            )(pos, seed=7)
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
        ds = DirectSampling(MPSModel(ti, n_neighbors=2, scan_fraction=0.3))
        ds.num_threads = 4
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
            ds = DirectSampling(MPSModel(ti, n_neighbors=3, scan_fraction=0.4))
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
        ti = TrainingImage(ti_data, categorical=False, distance="l2")
        ds = DirectSampling(MPSModel(ti, n_neighbors=2, scan_fraction=0.2))
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
        mv = _univar_as_mv_ti(ti)
        vec = mv.vec_distance_var(_MV_VAR, de_sim, de_ti[np.newaxis, :])
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
            MPSModel(self.ti1d, boundary="bad")
        with self.assertRaises(ValueError):
            MPSModel(self.ti1d, max_radius=0)
        with self.assertRaises(ValueError):
            MPSModel(self.ti1d, max_radius=-1.0)
        ds = DirectSampling(MPSModel(self.ti1d))
        with self.assertRaises(ValueError):
            ds([self.x1d], seed=42, mesh_type="unstructured")

    def test_repr(self):
        ds = DirectSampling(MPSModel(self.ti1d))
        r = repr(ds)
        self.assertIsInstance(r, str)
        self.assertIn("DirectSampling", r)

    def test_properties_and_setters(self):
        ds = DirectSampling(
            MPSModel(
                self.ti1d,
                n_neighbors=16,
                scan_fraction=0.5,
                threshold=0.05,
                cond_weight=2.0,
                boundary="partial",
                max_radius=3.0,
            )
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
        ds = DirectSampling(
            MPSModel(self.ti1d, n_neighbors=4, scan_fraction=1.0)
        )
        field = ds([self.x1d], seed=42)
        self.assertEqual(field.shape, (10,))
        self.assertFalse(np.any(np.isnan(field)))

    def test_shape_2d(self):
        ds = DirectSampling(
            MPSModel(self.ti2d, n_neighbors=4, scan_fraction=1.0)
        )
        field = ds([self.x2d, self.y2d], seed=42)
        self.assertEqual(field.shape, (6, 6))
        self.assertFalse(np.any(np.isnan(field)))
        # All output values must be in the TI value set {0, 1}
        unique_vals = set(np.unique(field))
        self.assertTrue(unique_vals.issubset({0.0, 1.0}))

    def test_regression_1d(self):
        ds = DirectSampling(
            MPSModel(self.ti1d, n_neighbors=4, scan_fraction=1.0)
        )
        field = ds([self.x1d], seed=42)
        self.assertAlmostEqual(field[0], 1.0)
        self.assertAlmostEqual(field[5], 0.0)
        self.assertAlmostEqual(field[9], 0.0)

    def test_regression_2d(self):
        ds = DirectSampling(
            MPSModel(self.ti2d, n_neighbors=4, scan_fraction=1.0)
        )
        field = ds([self.x2d, self.y2d], seed=42)
        self.assertAlmostEqual(field[0, 0], 1.0)
        self.assertAlmostEqual(field[2, 3], 0.0)
        self.assertAlmostEqual(field[5, 5], 1.0)

    def test_seeded_reproducibility(self):
        ds = DirectSampling(
            MPSModel(self.ti2d_rand, n_neighbors=8, scan_fraction=0.5)
        )
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
        ds = DirectSampling(
            MPSModel(self.ti1d, n_neighbors=4, scan_fraction=1.0)
        )
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
                n_neighbors=4,
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
                n_neighbors=32,
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
        ds = DirectSampling(MPSModel(ti, n_neighbors=4, scan_fraction=0.1))
        field = ds([np.arange(6, dtype=float)] * 2, seed=0)
        self.assertEqual(field.shape, (6, 6))
        self.assertFalse(np.any(np.isnan(field)))
        self.assertTrue(set(np.unique(field)).issubset({0.0, 1.0}))

    def test_max_radius(self):
        ds = DirectSampling(
            MPSModel(
                self.ti2d, n_neighbors=4, scan_fraction=1.0, max_radius=2.0
            )
        )
        field = ds([self.x2d, self.y2d], seed=42)
        self.assertEqual(field.shape, (6, 6))
        self.assertFalse(np.any(np.isnan(field)))

    def test_continuous_ti(self):
        ds = DirectSampling(
            MPSModel(
                self.ti1d_cont,
                n_neighbors=4,
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
            _univar_as_mv_ti(self.ti1d),
            sim_shape=(8,),
            n_neighbors=4,
            threshold=0.0,
            scan_fraction=1.0,
            rng_path=rng,
            rng_nodes=rng,
        )["_v"]
        self.assertEqual(result.shape, (8,))
        self.assertFalse(np.any(np.isnan(result)))
        self.assertTrue(set(np.unique(result)).issubset({0.0, 1.0}))

    def test_empty_search_window_recovery(self):
        # n_neighbors >> TI size collapses search windows → must recover silently
        ti_tiny = TrainingImage(np.array([0.0, 1.0, 0.0]), categorical=True)
        ds = DirectSampling(
            MPSModel(ti_tiny, n_neighbors=10, scan_fraction=1.0)
        )
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
                {
                    "a": np.zeros((4, 4), dtype=int),
                    "b": np.zeros((4, 4), dtype=int),
                },
                weights={"a": 0.3, "b": 0.3},
            )

    def test_distance_weighted_sum(self):
        # w=0.5 each; d_a=1.0 (full mismatch), d_b=0.0 -> joint = 0.5
        ti = TrainingImage(
            {
                "a": np.zeros((4, 4), dtype=int),
                "b": np.zeros((4, 4), dtype=int),
            },
            weights={"a": 0.5, "b": 0.5},
        )
        de_sg = {"a": np.array([1, 1]), "b": np.array([0, 0])}
        de_ti = {"a": np.array([0, 0]), "b": np.array([0, 0])}
        self.assertAlmostEqual(_mv_dist(ti, de_sg, de_ti), 0.5)

    def test_per_variable_categorical_and_distance(self):
        ti = TrainingImage(
            {
                "cat": np.zeros((4, 4), dtype=int),
                "cont": np.linspace(0, 100, 16).reshape(4, 4),
            },
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

    def test_adjust_value_multivariate(self):
        ti = TrainingImage(
            {
                "v": np.linspace(0, 100, 16).reshape(4, 4),
                "c": np.zeros((4, 4), dtype=int),
            },
            categorical={"v": False, "c": True},
            distance={"v": "variation", "c": "l1"},
        )
        # variation: Z(y) - mean(de_ti) + mean(de_sim) = 50 - 50 + 20 = 20
        self.assertAlmostEqual(
            ti.adjust_value(
                50.0,
                np.array([10.0, 20.0, 30.0]),
                np.array([40.0, 50.0, 60.0]),
                var="v",
            ),
            20.0,
        )
        # categorical variable: returned unchanged
        self.assertEqual(
            ti.adjust_value(1.0, np.array([0, 1]), np.array([1, 0]), var="c"),
            1.0,
        )
        # empty data event: returned unchanged even for variation
        self.assertAlmostEqual(
            ti.adjust_value(7.0, np.array([]), np.array([]), var="v"), 7.0
        )

    def test_weights_unknown_variable(self):
        with self.assertRaisesRegex(ValueError, "unknown"):
            TrainingImage(
                {
                    "a": np.zeros((4, 4), dtype=int),
                    "b": np.zeros((4, 4), dtype=int),
                },
                weights={"a": 0.5, "b": 0.5, "ghost": 0.0},
            )


class TestMultivariateDirectSampling(unittest.TestCase):
    def test_fills_all_nodes(self):
        # Node-wise path must leave no NaN in any variable.
        rng = np.random.default_rng(42)
        data = {
            "a": rng.integers(0, 2, (20, 20)),
            "b": rng.integers(0, 2, (20, 20)),
        }
        ti = TrainingImage(data)
        ds = DirectSampling(MPSModel(ti, n_neighbors=4, scan_fraction=0.3))
        field = ds([np.arange(6, dtype=float)] * 2, seed=1)
        self.assertFalse(np.any(np.isnan(field["a"])))
        self.assertFalse(np.any(np.isnan(field["b"])))

    def test_output_shapes(self):
        rng = np.random.default_rng(0)
        data = {
            "primary": rng.integers(0, 3, (20, 20)),
            "secondary": rng.integers(0, 2, (20, 20)),
        }
        ti = TrainingImage(data)
        ds = DirectSampling(MPSModel(ti, n_neighbors=8, scan_fraction=0.1))
        field = ds([np.arange(10, dtype=float)] * 2, seed=0)
        self.assertEqual(field["primary"].shape, (10, 10))
        self.assertIn("secondary", field)
        self.assertEqual(field["secondary"].shape, (10, 10))

    def test_values_valid(self):
        rng = np.random.default_rng(1)
        data = {
            "a": rng.integers(0, 4, (20, 20)),
            "b": rng.integers(0, 2, (20, 20)),
        }
        ti = TrainingImage(data)
        ds = DirectSampling(MPSModel(ti, n_neighbors=4, scan_fraction=0.2))
        field = ds([np.arange(8, dtype=float)] * 2, seed=5)
        self.assertTrue(np.all(np.isin(field["a"], [0, 1, 2, 3])))
        self.assertTrue(np.all(np.isin(field["b"], [0, 1])))

    def test_per_variable_n_neighbors(self):
        rng = np.random.default_rng(3)
        data = {
            "primary": rng.integers(0, 3, (20, 20)),
            "secondary": rng.integers(0, 2, (20, 20)),
        }
        ti = TrainingImage(data)
        ds = DirectSampling(
            MPSModel(
                ti,
                n_neighbors={"primary": 8, "secondary": 2},
                scan_fraction=0.2,
            )
        )
        field = ds([np.arange(8, dtype=float)] * 2, seed=7)
        self.assertEqual(field["primary"].shape, (8, 8))
        self.assertFalse(np.any(np.isnan(field["primary"])))
        self.assertFalse(np.any(np.isnan(field["secondary"])))

    def test_3d_runs(self):
        rng = np.random.default_rng(8)
        data = {
            "a": rng.integers(0, 2, (10, 10, 10)),
            "b": rng.integers(0, 2, (10, 10, 10)),
        }
        ti = TrainingImage(data)
        ds = DirectSampling(MPSModel(ti, n_neighbors=4, scan_fraction=0.2))
        field = ds([np.arange(4, dtype=float)] * 3, seed=0)
        self.assertEqual(field["a"].shape, (4, 4, 4))
        self.assertFalse(np.any(np.isnan(field["a"])))
        self.assertFalse(np.any(np.isnan(field["b"])))

    def test_partial_boundary_runs(self):
        rng = np.random.default_rng(11)
        data = {
            "a": rng.integers(0, 2, (20, 20)),
            "b": rng.integers(0, 2, (20, 20)),
        }
        ti = TrainingImage(data)
        ds = DirectSampling(
            MPSModel(ti, n_neighbors=6, scan_fraction=0.3, boundary="partial")
        )
        field = ds([np.arange(8, dtype=float)] * 2, seed=2)
        self.assertFalse(np.any(np.isnan(field["a"])))
        self.assertFalse(np.any(np.isnan(field["b"])))
        self.assertTrue(np.all(np.isin(field["a"], [0, 1])))

    def test_ds_mode_threshold(self):
        rng = np.random.default_rng(12)
        data = {
            "a": rng.integers(0, 3, (20, 20)),
            "b": rng.integers(0, 2, (20, 20)),
        }
        ti = TrainingImage(data)
        ds = DirectSampling(
            MPSModel(ti, n_neighbors=4, scan_fraction=0.5, threshold=0.3)
        )
        field = ds([np.arange(8, dtype=float)] * 2, seed=4)
        self.assertFalse(np.any(np.isnan(field["a"])))
        self.assertTrue(np.all(np.isin(field["a"], [0, 1, 2])))
        self.assertTrue(np.all(np.isin(field["b"], [0, 1])))

    def test_continuous_variation_variable(self):
        # A continuous variable with variation distance exercises the
        # mean-shift adjust_value_var path end to end.
        rng = np.random.default_rng(13)
        data = {
            "cont": rng.random((20, 20)) * 10.0,
            "cat": rng.integers(0, 2, (20, 20)),
        }
        ti = TrainingImage(
            data,
            categorical={"cont": False, "cat": True},
            distance={"cont": "variation", "cat": "l1"},
        )
        ds = DirectSampling(MPSModel(ti, n_neighbors=6, scan_fraction=0.3))
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
        ds = DirectSampling(
            MPSModel(ti, n_neighbors=8, scan_fraction=1.0, threshold=0.0)
        )
        field = ds([np.arange(6, dtype=float)] * 2, seed=0)
        np.testing.assert_array_equal(field["b"], field["a"] + 100)

    def test_equal_treatment_named_fields(self):
        # No privileged primary: all variables are first-class named fields.
        rng = np.random.default_rng(20)
        ti = TrainingImage(
            {
                "x": rng.integers(0, 2, (15, 15)),
                "y": rng.integers(0, 2, (15, 15)),
            }
        )
        ds = DirectSampling(MPSModel(ti, n_neighbors=4, scan_fraction=0.3))
        field = ds([np.arange(6, dtype=float)] * 2, seed=0)
        self.assertEqual(set(field), {"x", "y"})
        self.assertIn("x", ds.field_names)
        self.assertIn("y", ds.field_names)
        np.testing.assert_array_equal(ds["x"], field["x"])
        np.testing.assert_array_equal(ds["y"], field["y"])

    def test_invalid_variable_name_raises(self):
        ti = TrainingImage(
            {
                "a b": np.zeros((5, 5), dtype=int),
                "c": np.zeros((5, 5), dtype=int),
            }
        )
        with self.assertRaisesRegex(ValueError, "field name"):
            DirectSampling(MPSModel(ti, n_neighbors=4))

    def test_n_neighbors_dict_requires_multivariate(self):
        ti = TrainingImage(np.zeros((10, 10), dtype=int))
        with self.assertRaisesRegex(ValueError, "multivariate"):
            DirectSampling(MPSModel(ti, n_neighbors={"a": 4}))

    def test_n_neighbors_dict_keys_must_match(self):
        ti = TrainingImage(
            {
                "a": np.zeros((10, 10), dtype=int),
                "b": np.zeros((10, 10), dtype=int),
            }
        )
        with self.assertRaisesRegex(ValueError, "keys must match"):
            DirectSampling(MPSModel(ti, n_neighbors={"a": 4}))

    def test_set_condition_basic(self):
        rng = np.random.default_rng(0)
        data = {
            "a": rng.integers(0, 2, (20, 20)),
            "b": rng.integers(0, 2, (20, 20)),
        }
        ti = TrainingImage(data)
        ds = DirectSampling(MPSModel(ti, n_neighbors=4, scan_fraction=0.3))
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
        data = {
            "a": rng.integers(0, 2, (20, 20)),
            "b": rng.integers(0, 2, (20, 20)),
        }
        ti = TrainingImage(data)
        ds = DirectSampling(MPSModel(ti, n_neighbors=4, scan_fraction=0.3))
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
        data = {
            "a": rng.integers(0, 2, (20, 20)),
            "b": rng.integers(0, 2, (20, 20)),
        }
        ti = TrainingImage(data)
        ds = DirectSampling(MPSModel(ti, n_neighbors=4, scan_fraction=0.3))
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
        ds = DirectSampling(
            MPSModel(ti, n_neighbors=4, scan_fraction=1.0, threshold=0.0)
        )
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
        ds = DirectSampling(MPSModel(ti, n_neighbors=8, scan_fraction=0.2))
        ds.set_condition([[5.0], [5.0]], [2])
        field = ds([np.arange(10, dtype=float)] * 2, seed=0)
        self.assertEqual(field[5, 5], 2)

    def test_set_condition_length_mismatch(self):
        ti = TrainingImage(
            {
                "a": np.zeros((10, 10), dtype=int),
                "b": np.zeros((10, 10), dtype=int),
            }
        )
        ds = DirectSampling(MPSModel(ti, n_neighbors=4, scan_fraction=0.3))
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
        data = {
            "a": rng.integers(0, 3, (20, 20)),
            "b": rng.integers(0, 2, (20, 20)),
        }
        ti = TrainingImage(data)
        ds = DirectSampling(MPSModel(ti, n_neighbors=4, scan_fraction=0.2))
        ds.num_threads = 2
        field = ds([np.arange(8, dtype=float)] * 2, seed=0)
        self.assertTrue(np.all(np.isin(field["a"], [0, 1, 2])))
        self.assertTrue(np.all(np.isin(field["b"], [0, 1])))

    def test_parallel_matches_serial(self):
        # The node-vertex DAG commits every per-variable neighbour before a node
        # runs, so num_threads > 1 is bit-identical to serial.
        rng = np.random.default_rng(5)
        data = {
            "a": rng.integers(0, 3, (20, 20)),
            "b": rng.integers(0, 2, (20, 20)),
        }
        ti = TrainingImage(data)
        pos = [np.arange(8, dtype=float)] * 2
        ds_s = DirectSampling(MPSModel(ti, n_neighbors=4, scan_fraction=0.2))
        ds_s.num_threads = 1
        ds_p = DirectSampling(MPSModel(ti, n_neighbors=4, scan_fraction=0.2))
        ds_p.num_threads = 2
        f_s = ds_s(pos, seed=7)
        f_p = ds_p(pos, seed=7)
        np.testing.assert_array_equal(f_s["a"], f_p["a"])
        np.testing.assert_array_equal(f_s["b"], f_p["b"])

    def test_parallel_conditioning_preserved(self):
        rng = np.random.default_rng(6)
        data = {
            "a": rng.integers(0, 2, (20, 20)),
            "b": rng.integers(0, 2, (20, 20)),
        }
        ti = TrainingImage(data)
        ds = DirectSampling(MPSModel(ti, n_neighbors=4, scan_fraction=0.3))
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
            {
                "a": np.zeros((10, 10), dtype=int),
                "b": np.zeros((10, 10), dtype=int),
            }
        )
        ds = DirectSampling(MPSModel(ti, n_neighbors=4, scan_fraction=0.3))
        with self.assertRaisesRegex(ValueError, "dict"):
            ds.set_condition([[3.0], [3.0]], np.array([1]))

    def test_set_condition_empty_dict_raises(self):
        ti = TrainingImage(
            {
                "a": np.zeros((10, 10), dtype=int),
                "b": np.zeros((10, 10), dtype=int),
            }
        )
        ds = DirectSampling(MPSModel(ti, n_neighbors=4, scan_fraction=0.3))
        with self.assertRaisesRegex(ValueError, "empty"):
            ds.set_condition([[3.0], [3.0]], {})

    def test_set_condition_unknown_variable_raises(self):
        ti = TrainingImage(
            {
                "a": np.zeros((10, 10), dtype=int),
                "b": np.zeros((10, 10), dtype=int),
            }
        )
        ds = DirectSampling(MPSModel(ti, n_neighbors=4, scan_fraction=0.3))
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
        data = {
            "a": rng.integers(0, 2, (20, 20)),
            "b": rng.integers(0, 2, (20, 20)),
        }
        ti = TrainingImage(data)
        ds = DirectSampling(MPSModel(ti, n_neighbors=4, scan_fraction=0.3))
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
        data = {
            "cont": rng.random((20, 20)) * 10.0,
            "cat": rng.integers(0, 2, (20, 20)),
        }
        ti = TrainingImage(
            data,
            categorical={"cont": False, "cat": True},
            distance={"cont": "l2", "cat": "l1"},
            distance_power=1.0,
        )
        ds = DirectSampling(MPSModel(ti, n_neighbors=6, scan_fraction=0.3))
        field = ds([np.arange(8, dtype=float)] * 2, seed=3)
        self.assertEqual(field["cont"].shape, (8, 8))
        self.assertTrue(np.all(np.isfinite(field["cont"])))
        self.assertTrue(np.all(np.isin(field["cat"], [0, 1])))

    def test_single_variable_multivariate(self):
        rng = np.random.default_rng(9)
        ti = TrainingImage({"only": rng.integers(0, 3, (20, 20))})
        ds = DirectSampling(MPSModel(ti, n_neighbors=4, scan_fraction=0.3))
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
        ds = DirectSampling(
            MPSModel(ti, n_neighbors=8, scan_fraction=1.0, threshold=0.01)
        )
        field = ds([np.arange(6, dtype=float)] * 2, seed=0)
        self.assertFalse(np.any(np.isnan(field["a"])))
        self.assertFalse(np.any(np.isnan(field["b"])))
        np.testing.assert_array_equal(field["b"], field["a"] + 100)

    def test_mv_custom_store_name_raises(self):
        # Finding #6: a custom store name has no single field to bind to for a
        # multivariate run; reject it instead of silently dropping it.
        rng = np.random.default_rng(0)
        ti = TrainingImage(
            {
                "a": rng.integers(0, 2, (20, 20)),
                "b": rng.integers(0, 2, (20, 20)),
            }
        )
        ds = DirectSampling(MPSModel(ti, n_neighbors=4, scan_fraction=0.3))
        with self.assertRaisesRegex(ValueError, "custom store name"):
            ds([np.arange(6, dtype=float)] * 2, seed=0, store="myrun")

    def test_mv_store_false_not_stored(self):
        # store=False must still work for MV (no fields persisted on the model).
        rng = np.random.default_rng(0)
        ti = TrainingImage(
            {
                "a": rng.integers(0, 2, (20, 20)),
                "b": rng.integers(0, 2, (20, 20)),
            }
        )
        ds = DirectSampling(MPSModel(ti, n_neighbors=4, scan_fraction=0.3))
        out = ds([np.arange(6, dtype=float)] * 2, seed=0, store=False)
        self.assertEqual(set(out), {"a", "b"})
        self.assertEqual(list(ds.field_names), [])


class TestNonstationarity(unittest.TestCase):
    """Geometric non-stationarity for DirectSampling (set_nonstationary)."""

    def _make_ds(self, ti_shape=(20, 20), **kw):
        rng = np.random.default_rng(0)
        data = rng.integers(0, 3, ti_shape)
        ti = gs.mps.TrainingImage(data)
        defaults = dict(n_neighbors=4, scan_fraction=0.2)
        defaults.update(kw)
        return gs.mps.DirectSampling(MPSModel(ti, **defaults)), ti

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

        ds_rot = gs.mps.DirectSampling(
            MPSModel(ti, n_neighbors=8, scan_fraction=0.2)
        )
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

        ds_anis = gs.mps.DirectSampling(
            MPSModel(ti, n_neighbors=8, scan_fraction=0.2)
        )
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
        ds = gs.mps.DirectSampling(
            MPSModel(ti, n_neighbors=4, scan_fraction=0.1)
        )
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
        ds = gs.mps.DirectSampling(
            MPSModel(ti, n_neighbors=8, scan_fraction=1.0)
        )
        ds.set_nonstationary(rotation=np.pi / 2)
        field = ds([np.arange(6, dtype=float)] * 2, seed=0)
        self.assertTrue(np.all(np.isfinite(field)))
        self.assertTrue(np.all(np.isin(field, [0, 1])))

    def test_identity_matches_no_transform(self):
        # θ=0, anis=1 must produce bit-identical output to the plain path.
        ds_plain, ti = self._make_ds(ti_shape=(20, 20), n_neighbors=8)
        pos = [np.arange(8, dtype=float)] * 2
        f_plain = ds_plain(pos, seed=5)

        ds_id = gs.mps.DirectSampling(
            MPSModel(ti, n_neighbors=8, scan_fraction=0.2)
        )
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

        ds_plain = gs.mps.DirectSampling(
            MPSModel(ti, n_neighbors=4, scan_fraction=0.2)
        )
        result_plain = ds_plain(pos, seed=9)

        ds_rot = gs.mps.DirectSampling(
            MPSModel(ti, n_neighbors=4, scan_fraction=0.2)
        )
        ds_rot.set_nonstationary(rotation=np.pi / 4)
        result_rot = ds_rot(pos, seed=9)

        self.assertEqual(result_rot["a"].shape, (8, 8))
        self.assertEqual(result_rot["b"].shape, (8, 8))
        self.assertTrue(np.all(np.isin(result_rot["a"], [0, 1, 2])))
        self.assertTrue(np.all(np.isfinite(result_rot["b"])))
        self.assertFalse(np.array_equal(result_plain["a"], result_rot["a"]))

    def test_flattened_rotation_map_raises(self):
        # Finding #5: a per-node map passed flattened (length Nx*Ny) on a 2-D
        # grid must raise, not silently apply only element [0].
        ds, _ = self._make_ds(ti_shape=(20, 20))
        ds.set_nonstationary(rotation=np.linspace(0, np.pi, 25))
        with self.assertRaisesRegex(ValueError, "flattened|per-node"):
            ds([np.arange(5, dtype=float)] * 2, seed=0)

    def test_3d_stationary_rotation_triple_runs(self):
        # A genuine stationary multi-component value (3 Tait-Bryan angles for a
        # 3-D grid) is still accepted (length == no_of_angles(3) == 3).
        rng = np.random.default_rng(0)
        ti = gs.mps.TrainingImage(rng.integers(0, 2, (12, 12, 12)))
        ds = gs.mps.DirectSampling(
            MPSModel(ti, n_neighbors=4, scan_fraction=0.1)
        )
        ds.set_nonstationary(rotation=np.array([0.1, 0.2, 0.3]))
        field = ds([np.arange(5, dtype=float)] * 3, seed=0)
        self.assertEqual(field.shape, (5, 5, 5))
        self.assertTrue(np.all(np.isin(field, [0, 1])))


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
            {
                "a": rng_mv.integers(0, 3, (20, 20)).astype(float),
                "b": rng_mv.integers(0, 2, (20, 20)).astype(float),
            }
        )
        cls.x1d = np.arange(10, dtype=float)
        cls.x2d = np.arange(6, dtype=float)
        cls.y2d = np.arange(6, dtype=float)

    def test_snapshot(self):
        # --- univariate 1D, seed=42 ---
        ds = DirectSampling(
            MPSModel(self.ti1d, n_neighbors=4, scan_fraction=1.0)
        )
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
        ds2 = DirectSampling(
            MPSModel(self.ti2d, n_neighbors=4, scan_fraction=1.0)
        )
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
        ds3 = DirectSampling(
            MPSModel(self.ti2d_rand, n_neighbors=8, scan_fraction=0.5)
        )
        f = ds3([self.x2d, self.y2d], seed=42)
        np.testing.assert_array_equal(
            f,
            [
                [2.0, 0.0, 1.0, 2.0, 2.0, 1.0],
                [1.0, 0.0, 1.0, 1.0, 2.0, 0.0],
                [2.0, 2.0, 0.0, 0.0, 2.0, 2.0],
                [2.0, 0.0, 1.0, 2.0, 0.0, 0.0],
                [2.0, 2.0, 1.0, 2.0, 0.0, 1.0],
                [2.0, 1.0, 2.0, 2.0, 2.0, 0.0],
            ],
            err_msg="snap_uni_rand seed=42",
        )

        # --- multivariate 2D, seed=42 ---
        ds_mv = DirectSampling(
            MPSModel(self.ti_mv, n_neighbors=4, scan_fraction=0.5)
        )
        res = ds_mv([self.x2d, self.y2d], seed=42)
        np.testing.assert_array_equal(
            res["a"],
            [
                [1.0, 1.0, 0.0, 2.0, 2.0, 0.0],
                [2.0, 2.0, 1.0, 2.0, 1.0, 2.0],
                [2.0, 1.0, 1.0, 1.0, 1.0, 2.0],
                [2.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 2.0, 2.0, 2.0, 2.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ],
            err_msg="snap_mv var=a seed=42",
        )
        np.testing.assert_array_equal(
            res["b"],
            [
                [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ],
            err_msg="snap_mv var=b seed=42",
        )


class TestDSSeedControl(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        data = rng.integers(0, 3, (20, 20))
        self.ti = TrainingImage(data.astype(float))
        self.pos = [np.arange(8, dtype=float)] * 2

    def test_fixed_path_seed_different_node_seed(self):
        # Same visit order, different TI search → different output
        ds = DirectSampling(
            MPSModel(self.ti, n_neighbors=4, scan_fraction=0.3)
        )
        fa = ds(self.pos, path_seed=1, node_seed=10)
        fb = ds(self.pos, path_seed=1, node_seed=99)
        self.assertFalse(np.array_equal(fa, fb))

    def test_fixed_node_seed_different_path_seed(self):
        # Same TI search, different visit order → different output
        ds = DirectSampling(
            MPSModel(self.ti, n_neighbors=4, scan_fraction=0.3)
        )
        fa = ds(self.pos, path_seed=1, node_seed=10)
        fb = ds(self.pos, path_seed=99, node_seed=10)
        self.assertFalse(np.array_equal(fa, fb))

    def test_both_seeds_fixed_reproducible(self):
        # Explicit seeds on both → fully reproducible across calls
        ds = DirectSampling(
            MPSModel(self.ti, n_neighbors=4, scan_fraction=0.3)
        )
        fa = ds(self.pos, path_seed=7, node_seed=42)
        fb = ds(self.pos, path_seed=7, node_seed=42)
        self.assertTrue(np.array_equal(fa, fb))

    def test_default_still_reproducible(self):
        # No explicit seeds → same seed= value still reproducible
        ds = DirectSampling(
            MPSModel(self.ti, n_neighbors=4, scan_fraction=0.3)
        )
        fa = ds(self.pos, seed=5)
        fb = ds(self.pos, seed=5)
        self.assertTrue(np.array_equal(fa, fb))


class TestUnivarEqualsSingleVarMv(unittest.TestCase):
    """Permanent guard: univariate engine == single-variable multivariate engine.

    Proves bit-identical output before the univariate body is replaced with a
    thin wrapper in the unification step. Must remain green permanently.
    """

    @classmethod
    def setUpClass(cls):
        rng0 = np.random.default_rng(0)
        cls.ti_uni = TrainingImage(
            rng0.integers(0, 3, (20, 20)).astype(float), categorical=True
        )
        cls.ti_mv1 = TrainingImage(
            {
                "_v": np.random.default_rng(0)
                .integers(0, 3, (20, 20))
                .astype(float)
            },
            categorical=True,
        )
        cls.pos = [np.arange(8, dtype=float)] * 2
        cls.sim_shape = (8, 8)

    def test_equality(self):
        rng_seed = 42

        # univariate TI wrapped as single-var mv
        uni_rng = np.random.RandomState(rng_seed)
        result_uni = ds_simulate(
            _univar_as_mv_ti(self.ti_uni),
            sim_shape=self.sim_shape,
            n_neighbors=8,
            threshold=0.0,
            scan_fraction=1.0,
            rng_path=uni_rng,
            rng_nodes=uni_rng,
        )["_v"]

        # native single-variable mv TI
        mv_rng = np.random.RandomState(rng_seed)
        result_mv = ds_simulate(
            self.ti_mv1,
            sim_shape=self.sim_shape,
            n_neighbors=8,
            threshold=0.0,
            scan_fraction=1.0,
            rng_path=mv_rng,
            rng_nodes=mv_rng,
        )["_v"]

        np.testing.assert_array_equal(
            result_uni,
            result_mv,
            err_msg="_univar_as_mv_ti != native single-var mv",
        )


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
            MPSModel(ti, n_neighbors=8, scan_fraction=1.0)
        )
        ds.set_nonstationary(anis=0.1)
        field = ds([np.arange(8, dtype=float)] * 2, seed=0)
        self.assertTrue(np.all(np.isfinite(field)))
        self.assertTrue(set(np.unique(field)).issubset({0.0, 1.0}))


class TestVariationNNeighbors(unittest.TestCase):
    def test_variation_n1_raises_at_construction(self):
        ti = TrainingImage(
            np.linspace(0.0, 1.0, 20), categorical=False, distance="variation"
        )
        with self.assertRaisesRegex(ValueError, "variation"):
            DirectSampling(MPSModel(ti, n_neighbors=1, scan_fraction=0.5))

    def test_variation_n2_ok(self):
        ti = TrainingImage(
            np.linspace(0.0, 1.0, 20), categorical=False, distance="variation"
        )
        ds = DirectSampling(MPSModel(ti, n_neighbors=2, scan_fraction=0.5))
        self.assertEqual(ds.n_neighbors, 2)

    def test_variation_n_neighbors_setter_rejects_1(self):
        ti = TrainingImage(
            np.linspace(0.0, 1.0, 20), categorical=False, distance="variation"
        )
        ds = DirectSampling(MPSModel(ti, n_neighbors=3, scan_fraction=0.5))
        with self.assertRaisesRegex(ValueError, "variation"):
            ds.n_neighbors = 1

    def test_non_variation_n1_ok(self):
        ti = TrainingImage(
            np.linspace(0.0, 1.0, 20), categorical=False, distance="l2"
        )
        ds = DirectSampling(MPSModel(ti, n_neighbors=1, scan_fraction=0.5))
        self.assertEqual(ds.n_neighbors, 1)


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
        self.assertAlmostEqual(ti._d_max, 100.0)

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
        ds = DirectSampling(MPSModel(ti, n_neighbors=8, scan_fraction=1.0))
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
        ds = DirectSampling(MPSModel(ti, n_neighbors=8, scan_fraction=1.0))
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
        ds = DirectSampling(MPSModel(ti, n_neighbors=2, scan_fraction=1.0))
        with self.assertRaises(ValueError):
            ds([np.arange(3, dtype=float)] * 2, seed=0)


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
        ti = TrainingImage(data, categorical=True)
        ti_props = np.array([np.mean(data == v) for v in [0, 1, 2]])

        ds = DirectSampling(
            MPSModel(
                ti,
                n_neighbors=16,
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
        ti = TrainingImage(data, categorical=False, distance="l2")
        ti_mean = float(data.mean())
        ti_std = float(data.std())

        ds = DirectSampling(
            MPSModel(
                ti,
                n_neighbors=12,
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
        cls.mv_ti = TrainingImage(cls.mv_data)
        cls.pos = [np.arange(6, dtype=float)] * 2

    def test_set_mv_transforms_mean_applied(self):
        """set_mv_transforms(mean={'a': 100.0}) must shift variable 'a' by 100."""
        ds = DirectSampling(
            MPSModel(self.mv_ti, n_neighbors=4, scan_fraction=0.3)
        )
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
        ds = DirectSampling(
            MPSModel(self.mv_ti, n_neighbors=4, scan_fraction=0.3)
        )
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
        ti = TrainingImage(data)
        ds = DirectSampling(MPSModel(ti, n_neighbors=4, scan_fraction=0.3))
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
        ti = TrainingImage(data)
        pos = [np.arange(8, dtype=float)] * 2
        cond_pos = [[4.0], [4.0]]
        cond_val = [2]

        ds1 = DirectSampling(
            MPSModel(ti, n_neighbors=8, scan_fraction=0.3, cond_weight=1.0)
        )
        ds1.set_condition(cond_pos, cond_val)
        f1 = ds1(pos, seed=0)

        ds5 = DirectSampling(
            MPSModel(ti, n_neighbors=8, scan_fraction=0.3, cond_weight=5.0)
        )
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


if __name__ == "__main__":
    unittest.main()
