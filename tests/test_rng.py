"""
This is the unittest of the RNG class.
"""

import unittest

import numpy as np
from scipy.stats import kurtosis, normaltest, skew

from gstools import Gaussian, TPLStable
from gstools.random.rng import RNG


class TestRNG(unittest.TestCase):
    def setUp(self):
        self.seed = 19031977
        self.rng = RNG(self.seed)
        self.many_modes = 1000000
        self.few_modes = 100

    def test_rng_normal_consistency(self):
        rng = RNG(21021997)
        z1_refs = [-1.93013270, 0.46330478]
        z2_refs = [-0.25536086, 0.98298696]

        z1 = self.rng.random.normal(size=self.few_modes)
        z2 = self.rng.random.normal(size=self.few_modes)
        self.assertAlmostEqual(z1[0], z1_refs[0])
        self.assertAlmostEqual(z1[1], z1_refs[1])
        self.assertAlmostEqual(z2[0], z2_refs[0])
        self.assertAlmostEqual(z2[1], z2_refs[1])
        self.rng.seed = self.seed
        z1 = self.rng.random.normal(size=self.few_modes)
        z2 = self.rng.random.normal(size=self.few_modes)
        self.assertAlmostEqual(z1[0], z1_refs[0])
        self.assertAlmostEqual(z1[1], z1_refs[1])
        self.assertAlmostEqual(z2[0], z2_refs[0])
        self.assertAlmostEqual(z2[1], z2_refs[1])

    def test_sample_sphere_1d(self):
        dim = 1
        sphere_coord = self.rng.sample_sphere(dim, self.few_modes)
        self.assertEqual(sphere_coord.shape, (dim, self.few_modes))
        sphere_coord = self.rng.sample_sphere(dim, self.many_modes)
        self.assertAlmostEqual(np.mean(sphere_coord), 0.0, places=3)

    def test_sample_sphere_2d(self):
        dim = 2
        sphere_coord = self.rng.sample_sphere(dim, self.few_modes)
        np.testing.assert_allclose(
            np.ones(self.few_modes),
            sphere_coord[0, :] ** 2 + sphere_coord[1, :] ** 2,
        )
        sphere_coord = self.rng.sample_sphere(dim, self.many_modes)
        self.assertAlmostEqual(np.mean(sphere_coord), 0.0, places=3)

    def test_sample_sphere_3d(self):
        dim = 3
        sphere_coord = self.rng.sample_sphere(dim, self.few_modes)
        self.assertEqual(sphere_coord.shape, (dim, self.few_modes))
        np.testing.assert_allclose(
            np.ones(self.few_modes),
            sphere_coord[0, :] ** 2
            + sphere_coord[1, :] ** 2
            + sphere_coord[2, :] ** 2,
        )
        sphere_coord = self.rng.sample_sphere(dim, self.many_modes)
        self.assertAlmostEqual(np.mean(sphere_coord), 0.0, places=3)

    def test_sample_dist(self):
        model = Gaussian(dim=1, var=3.5, len_scale=8.0)
        pdf, cdf, ppf = model.dist_func
        rad = self.rng.sample_dist(
            size=self.few_modes, pdf=pdf, cdf=cdf, ppf=ppf, a=0
        )
        self.assertEqual(rad.shape[0], self.few_modes)

        model = Gaussian(dim=2, var=3.5, len_scale=8.0)
        pdf, cdf, ppf = model.dist_func
        rad = self.rng.sample_dist(
            size=self.few_modes, pdf=pdf, cdf=cdf, ppf=ppf, a=0
        )
        self.assertEqual(rad.shape[0], self.few_modes)

        model = Gaussian(dim=3, var=3.5, len_scale=8.0)
        pdf, cdf, ppf = model.dist_func
        rad = self.rng.sample_dist(
            size=self.few_modes, pdf=pdf, cdf=cdf, ppf=ppf, a=0
        )
        self.assertEqual(rad.shape[0], self.few_modes)

        # model = Gaussian(dim=2, var=3.5, len_scale=8.)
        # pdf, cdf, ppf = model.dist_func
        # rad = self.rng.sample_dist(
        #    size=self.many_modes, pdf=pdf, cdf=cdf, ppf=ppf, a=0)
        # import matplotlib.pyplot as pt
        # pt.hist(rad, bins=30)
        # print(rad)
        # pt.show()

        # TODO test with different models

    # TODO rework this
    # def test_gau(self):
    #    for d in range(len(self.rngs)):
    #        Z, k = self.rngs[d]('gau', self.len_scale, self.many_modes)
    #        self.assertEqual(k.shape, (d+1, self.many_modes))
    #        self.assertAlmostEqual(np.mean(k), 0., places=2)
    #        self.assertAlmostEqual(np.std(k), 1/self.len_scale, places=2)
    #        self.assertAlmostEqual(skew(k[0, :]), 0., places=2)
    #        self.assertAlmostEqual(kurtosis(k[0, :]), 0., places=1)
    #        self.assertLess(normaltest(k[0, :])[1], 0.05)


if __name__ == "__main__":
    unittest.main()
