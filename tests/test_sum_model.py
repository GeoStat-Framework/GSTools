"""
This is the unittest of the SumModel class.
"""

import unittest
from pathlib import Path

import numpy as np

import gstools as gs


class TestSumModel(unittest.TestCase):
    def test_init(self):
        s1 = gs.SumModel(dim=3)
        s2 = gs.Nugget(dim=3)
        self.assertTrue(s1 == s2)

        m1 = gs.Nugget(nugget=1)
        s1 = m1 + m1 + m1
        s2 = gs.Nugget(nugget=3)
        self.assertTrue(s1 == s2)
        self.assertFalse(m1 == s2)

        # Nugget cant set positive var or len-scale
        with self.assertRaises(ValueError):
            m1.var = 1
        with self.assertRaises(ValueError):
            m1.len_scale = 1

    def test_attr_set(self):
        s1 = gs.Gaussian() + gs.Gaussian() + gs.Gaussian()
        with self.assertRaises(ValueError):
            s1.vars = [1, 2]
        with self.assertRaises(ValueError):
            s1.len_scales = [1, 2]

        s1.integral_scale = 10
        self.assertAlmostEqual(s1.integral_scale_0, 10)
        self.assertAlmostEqual(s1.integral_scale_1, 10)
        self.assertAlmostEqual(s1.integral_scale_2, 10)

        s1.var = 2
        s1.ratios = [0.2, 0.2, 0.6]
        self.assertAlmostEqual(s1.vars[0], 0.4)
        self.assertAlmostEqual(s1.vars[1], 0.4)
        self.assertAlmostEqual(s1.vars[2], 1.2)

        with self.assertRaises(ValueError):
            s1.ratios = [0.3, 0.2, 0.6]
        with self.assertRaises(ValueError):
            s1.ratios = [0.3, 0.2]

    def test_compare(self):
        s1 = gs.Gaussian(var=1) + gs.Exponential(var=2)
        s2 = gs.Exponential(var=1) + gs.Gaussian(var=2)
        self.assertFalse(s1 == gs.Nugget(dim=3))
        self.assertFalse(s1 == s2)
        self.assertFalse(gs.Exponential() == (gs.Exponential() + gs.Nugget()))
        self.assertFalse((gs.Exponential() + gs.Nugget()) == gs.Exponential())

    def test_copy(self):
        # check that models get copied
        m1 = gs.Gaussian()
        s1 = m1 + m1
        var = [1.0, 2.0]
        s1.vars = var
        np.testing.assert_array_almost_equal(s1.vars, var)

    def test_var_dist(self):
        s1 = gs.SumModel(gs.Exponential, gs.Exponential, var=3)
        np.testing.assert_array_almost_equal(s1.vars, [1.5, 1.5])

    def test_presence(self):
        s1 = gs.SumModel(gs.Exponential, gs.Exponential)
        self.assertFalse(gs.Gaussian() in s1)

    def test_len_dist(self):
        s1 = gs.SumModel(gs.Exponential, gs.Exponential, len_scale=10)
        np.testing.assert_array_almost_equal(s1.len_scales, [10, 10])

    def test_temporal(self):
        s1 = gs.SumModel(
            gs.Exponential, gs.Exponential, temporal=True, spatial_dim=2
        )
        self.assertTrue(all(mod.temporal for mod in s1))
        self.assertTrue(all(mod.dim == 3 for mod in s1))

    def test_magic(self):
        m1 = gs.Gaussian(dim=2, var=1.0, len_scale=1.0)
        m2 = gs.Matern(dim=2, var=2.0, len_scale=2.0, nu=2.0)
        m3 = gs.Integral(dim=2, var=3.0, len_scale=3.0, nu=3.0)
        s1 = gs.SumModel(m1, m2, m3)
        s2 = m1 + m2 + m3
        s3 = gs.SumModel(
            *(gs.Gaussian, gs.Matern, gs.Integral),
            dim=2,
            vars=[1.0, 2.0, 3.0],
            len_scales=[1.0, 2.0, 3.0],
            nu_1=2.0,
            nu_2=3.0,
        )
        self.assertTrue(s1 == s2)
        self.assertTrue(s2 == s3)
        self.assertTrue(s1 == s3)

    def test_exceptions(self):
        with self.assertRaises(ValueError):
            gs.Exponential() + 1

        with self.assertRaises(ValueError):
            1 + gs.Exponential()

        with self.assertRaises(ValueError):
            gs.SumModel(gs.Exponential, gs.Gaussian(dim=2))

        with self.assertRaises(ValueError):
            gs.SumModel(gs.Nugget, gs.Nugget(nugget=2))

        with self.assertRaises(ValueError):
            gs.SumModel(gs.Gaussian(dim=2), gs.Exponential)

        with self.assertRaises(ValueError):
            gs.SumModel(gs.Nugget(nugget=2), gs.Nugget)

        with self.assertRaises(ValueError):
            model = gs.Spherical() + gs.Spherical()
            model[0].dim = 2
            model.check()

        with self.assertRaises(ValueError):
            model = gs.Spherical() + gs.Spherical()
            model[0].nugget = 2
            model.check()

        with self.assertRaises(ValueError):
            gs.Spherical(latlon=True) + gs.Spherical()

        with self.assertRaises(ValueError):
            gs.Spherical(geo_scale=2) + gs.Spherical()

        with self.assertRaises(ValueError):
            gs.Spherical(temporal=True) + gs.Spherical()

        with self.assertRaises(ValueError):
            gs.Spherical(anis=0.5) + gs.Spherical()

        with self.assertRaises(ValueError):
            gs.Spherical(angles=0.5) + gs.Spherical()

    def test_generate(self):
        x = np.random.RandomState(19970221).rand(1000) * 100.0
        y = np.random.RandomState(20011012).rand(1000) * 100.0
        m1 = gs.Spherical(dim=2, var=2, len_scale=5)
        m2 = gs.Spherical(dim=2, var=1, len_scale=10)
        m3 = gs.Gaussian(dim=2, var=1, len_scale=20)
        model = m1 + m2 + m3
        srf = gs.SRF(model, mean=0, seed=199702212)
        field = srf((x, y))
        self.assertAlmostEqual(np.var(field), 3.7, places=1)
        # used for test_fit (see below)
        # bin_center, gamma = gs.vario_estimate((x, y), field, max_dist=50, bin_no=100)

    def test_fit(self):
        here = Path(__file__).parent
        bin_center, gamma = np.loadtxt(here / "data" / "variogram.txt")
        s2 = gs.SumModel(gs.Gaussian, gs.Spherical, gs.Spherical, dim=2)
        res1, _ = s2.fit_variogram(bin_center, gamma, nugget=False)
        res2, _ = s2.fit_variogram(
            bin_center, gamma, nugget=False, len_scale_2=5, var_0=1
        )
        res3, _ = s2.fit_variogram(bin_center, gamma, nugget=False, var=3.7)
        res4, _ = s2.fit_variogram(
            bin_center, gamma, nugget=False, len_scale=15
        )
        res5, _ = s2.fit_variogram(
            bin_center, gamma, len_scale=15, sill=3.7, var_1=1
        )
        res6, _ = s2.fit_variogram(
            bin_center, gamma, len_scale=15, sill=3.7, var_1=1, nugget=0.1
        )
        res7, _ = s2.fit_variogram(
            bin_center,
            gamma,
            len_scale=15,
            sill=3.7,
            var_1=1,
            var_2=1,
            nugget=0.1,
        )

        self.assertAlmostEqual(res1["var"], 3.7, places=1)
        self.assertAlmostEqual(res2["var"], 3.7, places=1)
        self.assertAlmostEqual(res3["var"], 3.7, places=5)
        self.assertAlmostEqual(res4["len_scale"], 15.0, places=5)
        self.assertAlmostEqual(res5["var"] + res5["nugget"], 3.7, places=2)
        self.assertAlmostEqual(res5["len_scale"], 15.0, places=5)
        self.assertAlmostEqual(res6["var"] + res6["nugget"], 3.7, places=2)
        self.assertAlmostEqual(res7["var_0"], 1.6, places=5)

        mod_n = gs.Nugget(dim=2)
        res, _ = mod_n.fit_variogram(bin_center, gamma)
        self.assertAlmostEqual(res["nugget"], 3.4, places=1)
        # nothing to fit
        with self.assertRaises(ValueError):
            mod_n.fit_variogram(bin_center, gamma, nugget=False)
        # fixed sub-vars greated than fixed total var
        with self.assertRaises(ValueError):
            s2.fit_variogram(bin_center, gamma, var=1, var_0=1, var_1=1)
        # fixed len_scale and sub-len-scale not possible
        with self.assertRaises(ValueError):
            s2.fit_variogram(bin_center, gamma, len_scale=15, len_scale_0=5)

    def test_sum_weights(self):
        mod = gs.Gaussian() + gs.Gaussian() + gs.Gaussian()
        # fixed sub-vars too big
        with self.assertRaises(ValueError):
            mod.set_var_weights([1], skip=[1, 2], var=1)
        # too many ids
        with self.assertRaises(ValueError):
            mod.set_var_weights([1, 1], skip=[1, 2])
        # wrong skip
        with self.assertRaises(ValueError):
            mod.set_var_weights([1, 1], skip=[10])

        mod = gs.Gaussian() + gs.Gaussian() + gs.Gaussian()
        # fixed sub-lens too big
        with self.assertRaises(ValueError):
            mod.set_len_weights([1], skip=[1, 2], len_scale=0.1)
        # too many ids
        with self.assertRaises(ValueError):
            mod.set_len_weights([1, 1], skip=[1, 2])
        # wrong skip
        with self.assertRaises(ValueError):
            mod.set_len_weights([1, 1], skip=[10])
        # check setting with skipping
        mod.set_len_weights([1, 1], skip=[0], len_scale=10)
        np.testing.assert_array_almost_equal(mod.len_scales, [1, 14.5, 14.5])
        # check setting
        mod.set_len_weights([1, 1, 2], len_scale=10)
        np.testing.assert_array_almost_equal(mod.len_scales, [7.5, 7.5, 15])


if __name__ == "__main__":
    unittest.main()
