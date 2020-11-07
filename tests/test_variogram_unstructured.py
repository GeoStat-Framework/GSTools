# -*- coding: utf-8 -*-
"""
This is a unittest of the variogram module.
"""

import unittest
import numpy as np
from gstools import vario_estimate, Exponential, SRF
import gstools as gs


class TestVariogramUnstructured(unittest.TestCase):
    def setUp(self):

        # this code generates the testdata for the 2D rotation test cases
        x = np.random.RandomState(19970221).rand(20) * 10.0 - 5
        y = np.zeros_like(x)
        model = gs.Exponential(dim=2, var=2, len_scale=8)
        srf = gs.SRF(model, mean=0, seed=19970221)
        field = srf((x, y))

        bins = np.arange(10)
        bin_center, gamma = gs.vario_estimate((x,), field, bins)
        idx = np.argsort(x)
        self.test_data_rotation_1 = {
            "gamma": gamma,
            "x": x[idx],
            "field": field[idx],
            "bins": bins,
            "bin_center": bin_center,
        }

        # CODE ABOVE SHOULD GENERATE THIS DATA
        # x = np.array([
        # -4.86210059, -4.1984934 , -3.9246953 , -3.28490663, -2.16332379,
        # -1.87553275, -1.74125124, -1.27224687, -1.20931578, -0.2413368 ,
        #  0.03200921,  1.17099773,  1.53863105,  1.64478688,  2.75252136,
        #  3.3556915 ,  3.89828775,  4.21485964,  4.5364357 ,  4.79236969]),
        # field = np.array([
        # -1.10318365, -0.53566629, -0.41789049, -1.06167529,  0.38449961,
        # -0.36550477, -0.98905552, -0.19352766,  0.16264266,  0.26920833,
        #  0.05379665,  0.71275006,  0.36651935,  0.17366865,  1.20022343,
        #  0.79385446,  0.69456069,  1.0733393 ,  0.71191592,  0.71969766])
        # gamma_exp = np.array([
        # 0.14260989, 0.18301197, 0.25855841, 0.29990083, 0.67914526,
        # 0.60136535, 0.92875492, 1.46910435, 1.10165104])

    def test_doubles(self):
        x = np.arange(1, 11, 1, dtype=np.double)
        z = np.array(
            (41.2, 40.2, 39.7, 39.2, 40.1, 38.3, 39.1, 40.0, 41.1, 40.3),
            dtype=np.double,
        )
        bins = np.arange(1, 11, 1, dtype=np.double)
        bin_centres, gamma = vario_estimate([x], z, bins)
        self.assertAlmostEqual(gamma[0], 0.4917, places=4)

    def test_ints(self):
        x = np.arange(1, 5, 1, dtype=int)
        z = np.array((10, 20, 30, 40), dtype=int)
        bins = np.arange(1, 11, 1, dtype=int)
        bin_centres, gamma = vario_estimate([x], z, bins)
        self.assertAlmostEqual(gamma[0], 50.0, places=4)

    def test_np_int(self):
        x = np.arange(1, 5, 1, dtype=np.int)
        z = np.array((10, 20, 30, 40), dtype=np.int)
        bins = np.arange(1, 11, 1, dtype=np.int)
        bin_centres, gamma = vario_estimate([x], z, bins)
        self.assertAlmostEqual(gamma[0], 50.0, places=4)

    def test_mixed(self):
        x = np.arange(1, 11, 1, dtype=np.double)
        z = np.array(
            (41.2, 40.2, 39.7, 39.2, 40.1, 38.3, 39.1, 40.0, 41.1, 40.3),
            dtype=np.double,
        )
        bins = np.arange(1, 11, 1, dtype=int)
        bin_centres, gamma = vario_estimate([x], z, bins)
        self.assertAlmostEqual(gamma[0], 0.4917, places=4)

        x = np.arange(1, 5, 1, dtype=np.double)
        z = np.array((10, 20, 30, 40), dtype=int)
        bins = np.arange(1, 11, 1, dtype=int)
        bin_centres, gamma = vario_estimate([x], z, bins)
        self.assertAlmostEqual(gamma[0], 50.0, places=4)

        x = np.arange(1, 5, 1, dtype=np.double)
        z = np.array((10, 20, 30, 40), dtype=int)
        bins = np.arange(1, 11, 1, dtype=np.double)
        bin_centres, gamma = vario_estimate([x], z, bins)
        self.assertAlmostEqual(gamma[0], 50.0, places=4)

    def test_list(self):
        x = np.arange(1, 11, 1, dtype=np.double)
        z = [41.2, 40.2, 39.7, 39.2, 40.1, 38.3, 39.1, 40.0, 41.1, 40.3]
        bins = np.arange(1, 11, 1, dtype=np.double)
        bin_centres, gamma = vario_estimate([x], z, bins)
        self.assertAlmostEqual(gamma[1], 0.7625, places=4)

    def test_1d(self):
        x = np.arange(1, 11, 1, dtype=np.double)
        # literature values
        z = np.array(
            (41.2, 40.2, 39.7, 39.2, 40.1, 38.3, 39.1, 40.0, 41.1, 40.3),
            dtype=np.double,
        )
        bins = np.arange(1, 11, 1, dtype=np.double)
        bin_centres, gamma = vario_estimate([x], z, bins)
        self.assertAlmostEqual(gamma[0], 0.4917, places=4)
        self.assertAlmostEqual(gamma[1], 0.7625, places=4)

    def test_uncorrelated_2d(self):
        x_c = np.linspace(0.0, 100.0, 60)
        y_c = np.linspace(0.0, 100.0, 60)
        x, y = np.meshgrid(x_c, y_c)
        x = np.reshape(x, len(x_c) * len(y_c))
        y = np.reshape(y, len(x_c) * len(y_c))

        rng = np.random.RandomState(1479373475)
        field = rng.rand(len(x))

        bins = np.arange(0, 100, 10)

        bin_centres, gamma = vario_estimate((x, y), field, bins)

        var = 1.0 / 12.0
        self.assertAlmostEqual(gamma[0], var, places=2)
        self.assertAlmostEqual(gamma[len(gamma) // 2], var, places=2)
        self.assertAlmostEqual(gamma[-1], var, places=2)

    def test_uncorrelated_3d(self):
        x_c = np.linspace(0.0, 100.0, 15)
        y_c = np.linspace(0.0, 100.0, 15)
        z_c = np.linspace(0.0, 100.0, 15)
        x, y, z = np.meshgrid(x_c, y_c, z_c)
        x = np.reshape(x, len(x_c) * len(y_c) * len(z_c))
        y = np.reshape(y, len(x_c) * len(y_c) * len(z_c))
        z = np.reshape(z, len(x_c) * len(y_c) * len(z_c))

        rng = np.random.RandomState(1479373475)
        field = rng.rand(len(x))

        bins = np.arange(0, 100, 10)

        bin_centres, gamma = vario_estimate(
            (x, y, z), field, bins
        )

        var = 1.0 / 12.0
        self.assertAlmostEqual(gamma[0], var, places=2)
        self.assertAlmostEqual(gamma[len(gamma) // 2], var, places=2)
        self.assertAlmostEqual(gamma[-1], var, places=2)

    def test_sampling_1d(self):
        x = np.linspace(0.0, 100.0, 21000)

        rng = np.random.RandomState(1479373475)
        field = rng.rand(len(x))

        bins = np.arange(0, 100, 10)

        bin_centres, gamma = vario_estimate(
            [x], field, bins, sampling_size=5000, sampling_seed=1479373475
        )

        var = 1.0 / 12.0
        self.assertAlmostEqual(gamma[0], var, places=2)
        self.assertAlmostEqual(gamma[len(gamma) // 2], var, places=2)
        self.assertAlmostEqual(gamma[-1], var, places=2)

    def test_sampling_2d(self):
        x_c = np.linspace(0.0, 100.0, 600)
        y_c = np.linspace(0.0, 100.0, 600)
        x, y = np.meshgrid(x_c, y_c)
        x = np.reshape(x, len(x_c) * len(y_c))
        y = np.reshape(y, len(x_c) * len(y_c))

        rng = np.random.RandomState(1479373475)
        field = rng.rand(len(x))

        bins = np.arange(0, 100, 10)

        bin_centres, gamma = vario_estimate(
            (x, y), field, bins, sampling_size=2000, sampling_seed=1479373475
        )

        var = 1.0 / 12.0
        self.assertAlmostEqual(gamma[0], var, places=2)
        self.assertAlmostEqual(gamma[len(gamma) // 2], var, places=2)
        self.assertAlmostEqual(gamma[-1], var, places=2)

    def test_sampling_3d(self):
        x_c = np.linspace(0.0, 100.0, 100)
        y_c = np.linspace(0.0, 100.0, 100)
        z_c = np.linspace(0.0, 100.0, 100)
        x, y, z = np.meshgrid(x_c, y_c, z_c)
        x = np.reshape(x, len(x_c) * len(y_c) * len(z_c))
        y = np.reshape(y, len(x_c) * len(y_c) * len(z_c))
        z = np.reshape(z, len(x_c) * len(y_c) * len(z_c))

        rng = np.random.RandomState(1479373475)
        field = rng.rand(len(x))

        bins = np.arange(0, 100, 10)

        bin_centres, gamma = vario_estimate(
            (x, y, z),
            field,
            bins,
            sampling_size=2000,
            sampling_seed=1479373475,
        )
        var = 1.0 / 12.0
        self.assertAlmostEqual(gamma[0], var, places=2)
        self.assertAlmostEqual(gamma[len(gamma) // 2], var, places=2)
        self.assertAlmostEqual(gamma[-1], var, places=2)

    def test_assertions(self):
        x = np.arange(0, 10)
        x_e = np.arange(0, 11)
        y = np.arange(0, 11)
        y_e = np.arange(0, 12)
        z = np.arange(0, 12)
        z_e = np.arange(0, 15)
        bins = np.arange(0, 3)
        #        bins_e = np.arange(0, 1)
        field = np.arange(0, 10)
        field_e = np.arange(0, 9)

        self.assertRaises(
            ValueError, vario_estimate, [x_e], field, bins
        )
        self.assertRaises(
            ValueError, vario_estimate, (x, y_e), field, bins
        )
        self.assertRaises(
            ValueError, vario_estimate, (x, y_e, z), field, bins
        )
        self.assertRaises(
            ValueError, vario_estimate, (x, y, z_e), field, bins
        )
        self.assertRaises(
            ValueError, vario_estimate, (x_e, y, z), field, bins
        )
        self.assertRaises(
            ValueError, vario_estimate, (x, y, z), field_e, bins
        )
        self.assertRaises(
            ValueError, vario_estimate, [x], field_e, bins
        )

    def test_multi_field(self):
        x = np.random.RandomState(19970221).rand(100) * 100.0
        model = gs.Exponential(dim=1, var=2, len_scale=10)
        srf = gs.SRF(model)
        field1 = srf(x, seed=19970221)
        field2 = srf(x, seed=20011012)
        bins = np.arange(20) * 2
        bin_center, gamma1 = gs.vario_estimate(x, field1, bins)
        bin_center, gamma2 = gs.vario_estimate(x, field2, bins)
        bin_center, gamma = gs.vario_estimate(
            x, [field1, field2], bins
        )
        gamma_mean = 0.5 * (gamma1 + gamma2)
        for i in range(len(gamma)):
            self.assertAlmostEqual(gamma[i], gamma_mean[i], places=2)

    def test_no_data(self):
        x1 = np.random.RandomState(19970221).rand(100) * 100.0
        field1 = np.random.RandomState(20011012).rand(100) * 100.0
        field1[:10] = np.nan
        x2 = x1[10:]
        field2 = field1[10:]
        bins = np.arange(20) * 2
        bin_center, gamma1 = gs.vario_estimate(x1, field1, bins)
        bin_center, gamma2 = gs.vario_estimate(x2, field2, bins)
        for i in range(len(gamma1)):
            self.assertAlmostEqual(gamma1[i], gamma2[i], places=2)


if __name__ == "__main__":
    unittest.main()
