"""Tests for gstools.mps.distance."""

import unittest

import numpy as np
from mps_fixtures import uni_dist

from gstools.mps.distance import (
    adjust_value,
    compute_node_weights,
    variable_distance,
    vec_categorical_dist,
    vec_categorical_penalty_dist,
    vec_l1_dist,
    vec_l2_dist,
    vec_lp_dist,
    vec_variation_dist,
)
from gstools.mps.training_image import TrainingImage, Variable


class TestTrainingImageDistanceHelpers(unittest.TestCase):
    """Distance functions and node weights used by the training image."""

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
        self.assertLessEqual(uni_dist(ti, x, y, lag_norms=lags), 1.0)

    def test_variation_dist_out_of_range_clamped(self):
        """Out-of-range SG values (from conditioning / mean-shift) must clamp to [0, 1]."""
        ti = TrainingImage(
            np.linspace(0.0, 1.0, 10),  # d_max == 1.0
            categorical=False,
            distance="variation",
        )
        de_sim = np.array([5.0, 0.0])  # 5.0 is far outside the TI range
        de_ti = np.array([0.0, 1.0])
        self.assertLessEqual(uni_dist(ti, de_sim, de_ti), 1.0)
        w = compute_node_weights(2, None, 0.0)
        vec = variable_distance(
            ti.variable(), de_sim, de_ti[np.newaxis, :], w, ti.variable().d_max
        )
        self.assertEqual(vec.shape, (1,))
        self.assertLessEqual(vec[0], 1.0)

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
        self.assertLess(w[0], 0.9)  # far from the ~1.0 blow-up
        # cond_weight scales the collocated entry (it carries cond_mask=True).
        w_c = compute_node_weights(
            3,
            [0.0, 1.0, 2.0],
            2.0,
            cond_mask=np.array([True, False, False]),
            cond_weight=5.0,
        )
        self.assertGreater(w_c[0], w[0])


class TestVecCategoricalPenaltyDist(unittest.TestCase):
    """Unit tests for vec_categorical_penalty_dist."""

    def setUp(self):
        # Asymmetric 3-category penalty matrix: T[u,u]=0, T[u,v] in (0,1].
        self.T = np.array(
            [
                [0.0, 0.5, 0.8],
                [0.3, 0.0, 0.6],
                [0.9, 0.4, 0.0],
            ]
        )
        self.sim = np.array([0.0, 1.0, 2.0])
        self.w = np.array([0.2, 0.3, 0.5])

    def test_hand_computed_no_nan(self):
        ti = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 2.0]])
        d = vec_categorical_penalty_dist(self.sim, ti, self.w, self.T)
        # row0: exact match -> 0
        # row1: 0.2*T[0,1] + 0.3*T[1,0] + 0.5*T[2,2] = 0.2*0.5+0.3*0.3+0 = 0.19
        np.testing.assert_allclose(d, [0.0, 0.19])

    def test_hand_computed_has_nan(self):
        ti = np.array([[0.0, 1.0, 2.0], [1.0, np.nan, 2.0]])
        d = vec_categorical_penalty_dist(
            self.sim, ti, self.w, self.T, has_nan=True
        )
        # row1: valid=[T,F,T], we=[0.2,0,0.5], wsum=0.7
        # mism = 0.2*T[0,1] + 0 + 0.5*T[2,2] = 0.1
        # dist = 0.1 / 0.7
        np.testing.assert_allclose(d, [0.0, 0.1 / 0.7])

    def test_all_nan_row_is_inf(self):
        ti = np.array([[np.nan, np.nan, np.nan]])
        d = vec_categorical_penalty_dist(
            self.sim, ti, self.w, self.T, has_nan=True
        )
        self.assertTrue(np.isinf(d[0]))

    def test_reduction_to_binary_mismatch_no_nan(self):
        # T = 1 - eye(C) collapses exactly to vec_categorical_dist.
        rng = np.random.default_rng(0)
        c, n, max_scan = 4, 5, 20
        sim = rng.integers(0, c, n).astype(float)
        ti = rng.integers(0, c, (max_scan, n)).astype(float)
        w = rng.random(n)
        w /= w.sum()
        t_binary = 1.0 - np.eye(c)
        d_pen = vec_categorical_penalty_dist(sim, ti, w, t_binary)
        d_bin = vec_categorical_dist(sim, ti, w)
        np.testing.assert_allclose(d_pen, d_bin)

    def test_reduction_to_binary_mismatch_has_nan(self):
        rng = np.random.default_rng(1)
        c, n, max_scan = 4, 5, 20
        sim = rng.integers(0, c, n).astype(float)
        ti = rng.integers(0, c, (max_scan, n)).astype(float)
        ti[3, 2] = np.nan
        ti[7, :] = np.nan  # fully undefined row -> +inf in both
        w = rng.random(n)
        w /= w.sum()
        t_binary = 1.0 - np.eye(c)
        d_pen = vec_categorical_penalty_dist(
            sim, ti, w, t_binary, has_nan=True
        )
        d_bin = vec_categorical_dist(sim, ti, w, has_nan=True)
        np.testing.assert_allclose(d_pen, d_bin, equal_nan=True)


class TestRenormZeroWeightRows(unittest.TestCase):
    """Renormalisation of distances over zero-weight rows."""

    def test_renorm_zero_weight_rows(self):
        # Zero-weight rows are +inf, except 0.0 for a defined row when the
        # whole event has zero weight.
        from gstools.mps.distance import _renorm

        num = np.array([0.0, 2.0, 0.0])
        wsum = np.array([0.0, 4.0, 0.0])
        defined = np.array([True, True, False])
        np.testing.assert_array_equal(
            _renorm(num, wsum, defined, 1.0), [np.inf, 0.5, np.inf]
        )
        np.testing.assert_array_equal(
            _renorm(num, wsum, defined, 0.0), [0.0, 0.5, np.inf]
        )


def _reference_weights(
    n, lag_norms, distance_power, cond_mask=None, cond_weight=1.0
):
    """Pre-change ``compute_node_weights`` body, used as the bit-exact oracle."""
    raw_w = np.ones(n, dtype=np.float64)
    if lag_norms is not None and distance_power != 0.0:
        norms = np.asarray(lag_norms, dtype=np.float64)
        nz = norms != 0.0
        raw_w[nz] = norms[nz] ** (-distance_power)
    if cond_mask is not None:
        raw_w[np.asarray(cond_mask, dtype=bool)] *= cond_weight
    total = raw_w.sum()
    if total == 0.0:
        return np.zeros(n, dtype=np.float64)
    if not np.isfinite(total):
        return np.full(n, 1.0 / n, dtype=np.float64)
    return raw_w / total


class TestComputeNodeWeightsExtremes(unittest.TestCase):
    """Node weights when the raw weight sum overflows or underflows."""

    def test_overflow_keeps_conditioning_dominant(self):
        w = compute_node_weights(
            3, np.ones(3), 0.0, np.array([True, True, False]), 1e308
        )
        np.testing.assert_allclose(w, [0.5, 0.5, 0.0], atol=1e-12)

    def test_underflow_keeps_nearest_node(self):
        w = compute_node_weights(2, np.array([2.0, 3.0]), 2048.0)
        np.testing.assert_allclose(w, [1.0, 0.0])

    def test_log_path_agrees_with_direct_path(self):
        # Normalised weights are invariant under a uniform lag scaling, so
        # scaling the lags until the raw weights underflow (log path) must
        # reproduce the direct-path result of the unscaled lags.
        lags = np.array([2.0, 3.0, 4.0])
        for mask, cond_weight in (
            (None, 1.0),
            (np.array([1, 0, 0], bool), 3.0),
        ):
            with self.subTest(cond_mask=mask):
                direct = compute_node_weights(3, lags, 2.0, mask, cond_weight)
                scaled = compute_node_weights(
                    3, lags * 1e200, 2.0, mask, cond_weight
                )
                np.testing.assert_allclose(scaled, direct, rtol=1e-12)

    def test_log_path_keeps_structural_zeros_exact(self):
        # Conditioning nodes with cond_weight=0 stay exactly 0; the underflowing
        # remainder is still normalised relative to each other.
        lags = np.array([1.0, 2.0, 3.0, 4.0])
        mask = np.array([True, True, False, False])
        w = compute_node_weights(4, lags * 1e200, 2.0, mask, 0.0)
        np.testing.assert_array_equal(w[:2], [0.0, 0.0])
        self.assertAlmostEqual(w.sum(), 1.0, places=12)
        expected = compute_node_weights(2, lags[2:], 2.0)
        np.testing.assert_allclose(w[2:], expected, rtol=1e-12)

    def test_log_path_zero_lag_node_stays_finite_and_positive(self):
        # Overflowing cond_weight forces the log path; the collocated h=0 node
        # keeps the unit-lag baseline weight (equal to a unit-lag node).
        w = compute_node_weights(
            3,
            np.array([0.0, 1.0, 2.0]),
            1.0,
            np.array([True, True, False]),
            1e308,
        )
        self.assertTrue(np.all(np.isfinite(w)))
        self.assertGreater(w[0], 0.0)
        self.assertAlmostEqual(w[0], w[1], places=12)
        self.assertAlmostEqual(w.sum(), 1.0, places=12)

    def test_all_zero_conditioning_weight_returns_zeros(self):
        w = compute_node_weights(
            2, np.array([1.0, 2.0]), 0.0, np.array([True, True]), 0.0
        )
        np.testing.assert_array_equal(w, [0.0, 0.0])

    def test_ordinary_inputs_unchanged(self):
        rng = np.random.default_rng(0)
        powers = [0.0, 0.5, 1.0, 2.0]
        cond_weights = [0.0, 1.0, 5.0]
        for case in range(200):
            n = int(rng.integers(1, 41))
            power = powers[int(rng.integers(0, len(powers)))]
            cond_weight = cond_weights[int(rng.integers(0, len(cond_weights)))]
            lags = rng.uniform(1.0, 20.0, n)
            cond_mask = rng.random(n) < 0.5 if case % 2 else None
            with self.subTest(case=case):
                np.testing.assert_array_equal(
                    compute_node_weights(
                        n, lags, power, cond_mask, cond_weight
                    ),
                    _reference_weights(n, lags, power, cond_mask, cond_weight),
                )


class TestAdjustValue(unittest.TestCase):
    def test_passthrough_categorical_and_lp(self):
        ti_cat = TrainingImage(
            np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]), categorical=True
        )
        ti_l1 = TrainingImage(
            np.linspace(0.0, 1.0, 20), categorical=False, distance="l1"
        )
        for ti in (ti_cat, ti_l1):
            self.assertAlmostEqual(
                adjust_value(
                    ti.variable(),
                    0.7,
                    np.array([0.1, 0.3]),
                    np.array([0.4, 0.6]),
                ),
                0.7,
            )

    def test_variation_mean_shift(self):
        # Z(y) - Z_bar(y) + Z_bar(x) = 0.7 - 0.6 + 0.3 = 0.4
        ti = TrainingImage(
            np.linspace(0.0, 1.0, 20), categorical=False, distance="variation"
        )
        result = adjust_value(
            ti.variable(),
            0.7,
            np.array([0.1, 0.3, 0.5]),
            np.array([0.4, 0.6, 0.8]),
        )
        self.assertAlmostEqual(result, 0.4, places=6)
        self.assertNotAlmostEqual(result, 0.7)  # must not be passthrough

    def test_variation_lp_variants(self):
        """The mean-shift applies for all variation<p> variants."""
        de_sim = np.array([0.1, 0.3, 0.5])  # mean = 0.3
        de_ti = np.array([0.4, 0.6, 0.8])  # mean = 0.6
        for spec in ("variation", "variation1", "variation1.5"):
            ti = TrainingImage(
                np.linspace(0.0, 1.0, 20), categorical=False, distance=spec
            )
            self.assertAlmostEqual(
                adjust_value(ti.variable(), 0.7, de_sim, de_ti), 0.4, places=6
            )

    def test_multivariate(self):
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
            adjust_value(
                ti.variable("v"),
                50.0,
                np.array([10.0, 20.0, 30.0]),
                np.array([40.0, 50.0, 60.0]),
            ),
            20.0,
        )
        self.assertEqual(
            adjust_value(
                ti.variable("c"), 1.0, np.array([0, 1]), np.array([1, 0])
            ),
            1.0,
        )

    def test_uses_common_support(self):
        # The shift is taken over the nodes defined in the TI event only:
        # 10 - mean([2, 4]) + mean([2, 3]) -- not nanmean of each event.
        ti = TrainingImage(
            np.random.default_rng(0).random((12, 12)),
            categorical=False,
            distance="variation",
        )
        shifted = adjust_value(
            ti.variable(), 10.0, [1.0, 2.0, 3.0], [np.nan, 2.0, 4.0]
        )
        self.assertAlmostEqual(shifted, 9.5)


class TestVariableDistanceDmax(unittest.TestCase):
    def test_d_max_override(self):
        ti = TrainingImage(
            np.array([[0.0, 1.0], [2.0, 3.0]]), categorical=False
        )  # own range = 3
        var = ti.variable()
        de = np.array([0.0])
        cand = np.array([[3.0]])
        w = compute_node_weights(1, None, var.distance_power)
        d_own = variable_distance(var, de, cand, w, var.d_max)
        d_ovr = variable_distance(var, de, cand, w, 6.0)
        self.assertAlmostEqual(float(d_own[0]), 1.0)
        self.assertAlmostEqual(float(d_ovr[0]), 0.5)

    def test_empty_data_event_is_zero(self):
        ti = TrainingImage(np.linspace(0.0, 1.0, 10), categorical=False)
        d = variable_distance(
            ti.variable(), np.array([]), np.zeros((3, 0)), np.array([]), 1.0
        )
        np.testing.assert_array_equal(d, np.zeros(3))

    def test_weights_from_variable_distance_power(self):
        ti = TrainingImage(
            np.linspace(0.0, 1.0, 10), categorical=False, distance_power=1.0
        )
        var = ti.variable()
        w = compute_node_weights(2, [0.0, 1.0], var.distance_power)
        d = variable_distance(
            var, np.array([0.0, 0.5]), np.array([[0.0, 0.5]]), w, var.d_max
        )
        self.assertEqual(d.shape, (1,))
        self.assertAlmostEqual(float(w.sum()), 1.0)


if __name__ == "__main__":
    unittest.main()


class TestIntegerDataEvents(unittest.TestCase):
    """Small integer dtypes must not wrap around in direct distance calls."""

    def setUp(self):
        self.w = np.array([0.5, 0.5])
        self.d_max = 255.0

    def test_vec_distances_match_float64(self):
        events = {
            np.uint8: ([10, 250], [[250, 10], [10, 250], [0, 255]]),
            np.int8: ([10, 120], [[120, 10], [10, 120], [-5, 127]]),
        }
        calls = {
            "l1": lambda s, t: vec_l1_dist(s, t, self.w, self.d_max),
            "l2": lambda s, t: vec_l2_dist(s, t, self.w, self.d_max),
            "lp": lambda s, t: vec_lp_dist(s, t, self.w, self.d_max, 3.0),
        }
        for dtype, (sim, ti) in events.items():
            sim, ti = np.array(sim, dtype), np.array(ti, dtype)
            for name, call in calls.items():
                with self.subTest(dtype=dtype.__name__, metric=name):
                    np.testing.assert_array_equal(
                        call(sim, ti),
                        call(sim.astype(float), ti.astype(float)),
                    )

    def test_variable_distance_continuous_integer_data(self):
        ti_data = np.arange(16, dtype=np.uint8).reshape(4, 4) * 16
        var = Variable("a", ti_data, categorical=False)
        sim = np.array([10, 250], dtype=np.uint8)
        events = np.array([[250, 10], [10, 250]], dtype=np.uint8)
        got = variable_distance(var, sim, events, self.w, 240.0)
        expected = variable_distance(
            var, sim.astype(float), events.astype(float), self.w, 240.0
        )
        np.testing.assert_array_equal(got, expected)
