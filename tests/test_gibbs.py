"""This is the unittest of the Gibbs sampler for inequality data."""

import unittest

import numpy as np
from scipy import stats

import gstools as gs
from gstools.geometry.gibbs import _trunc_norm, gibbs_sample


class TestGibbs(unittest.TestCase):
    def setUp(self):
        self.model = gs.Gaussian(dim=1, var=1.0, len_scale=10.0)
        self.pos = np.linspace(0, 40, 9)

    def test_all_exact(self):
        val = np.linspace(-1, 1, 9)
        np.testing.assert_array_equal(
            gibbs_sample(self.model, self.pos, val), val
        )

    def test_bounds(self):
        val = np.full(9, np.nan)
        val[0] = 0.0
        cases = [
            (np.full(9, 0.0), np.full(9, np.inf)),  # one-sided incl. low=0
            (np.full(9, -np.inf), np.full(9, -1.0)),
            (np.full(9, 0.5), np.full(9, 0.7)),  # two-sided
            (np.full(9, -np.inf), np.full(9, np.inf)),  # unbounded
        ]
        for low, upp in cases:
            res = gibbs_sample(self.model, self.pos, val, low, upp, seed=3)
            self.assertEqual(res[0], 0.0)
            self.assertTrue(np.all(res[1:] >= low[1:]))
            self.assertTrue(np.all(res[1:] <= upp[1:]))
            self.assertTrue(np.all(np.isfinite(res)))
        self.assertRaises(
            ValueError,
            gibbs_sample,
            self.model,
            self.pos,
            val,
            np.full(9, 1.0),
            np.full(9, 0.0),
        )

    def test_seed(self):
        val = np.full(9, np.nan)
        low = np.zeros(9)
        res = gibbs_sample(self.model, self.pos, val, low, seed=1)
        np.testing.assert_allclose(
            res,
            gibbs_sample(self.model, self.pos, val, low, seed=1),
            rtol=1e-10,
        )
        self.assertFalse(
            np.array_equal(
                res, gibbs_sample(self.model, self.pos, val, low, seed=2)
            )
        )

    def test_trunc_norm_tails(self):
        rng = np.random.RandomState(0)
        # one test per branch: Robert tail, narrow tail, inverse CDF
        for low, upp in [(6, np.inf), (8, np.inf), (10, 10.05), (-1, 2)]:
            smp = [_trunc_norm(0.0, 1.0, low, upp, rng) for _ in range(5000)]
            ref = stats.truncnorm(low, upp)
            tol = 4 * ref.std() / np.sqrt(len(smp))  # 4 standard errors
            self.assertAlmostEqual(np.mean(smp), ref.mean(), delta=tol)
            self.assertAlmostEqual(np.std(smp), ref.std(), delta=2 * tol)
        # mirrored lower tail and shifted/scaled
        smp = [_trunc_norm(2.0, 0.5, -np.inf, -2.0, rng) for _ in range(2000)]
        self.assertTrue(np.all(np.array(smp) <= -2.0))
        self.assertEqual(_trunc_norm(0.0, 1.0, 1.5, 1.5, rng), 1.5)

    def test_marginal(self):
        model = gs.Gaussian(dim=1, var=1.0, len_scale=1.0)
        draws = [
            gibbs_sample(
                model,
                [0.0, 50.0],
                [0.3, np.nan],
                [np.nan, 1.0],
                [np.nan, np.inf],
                seed=s,
                burn_in=5,
                sweeps=5,
            )[1]
            for s in range(400)
        ]
        ref = stats.truncnorm(1.0, np.inf).mean()
        self.assertAlmostEqual(np.mean(draws), ref, delta=0.06)

    def test_duplicates_warn(self):
        pos = np.array([0.0, 1.0, 1.0])
        with self.assertWarns(UserWarning):
            gibbs_sample(
                self.model, pos, [0.0, np.nan, np.nan], [0, 0, 0], seed=1
            )


if __name__ == "__main__":
    unittest.main()
