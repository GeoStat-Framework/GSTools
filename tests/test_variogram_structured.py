# -*- coding: utf-8 -*-
"""
This is a unittest of the variogram module.
"""

import unittest
import numpy as np
import gstools as gs


class TestVariogramstructured(unittest.TestCase):
    def setUp(self):
        pass

    def test_doubles(self):
        z = np.array(
            (41.2, 40.2, 39.7, 39.2, 40.1, 38.3, 39.1, 40.0, 41.1, 40.3),
            dtype=np.double,
        )
        gamma = gs.vario_estimate_axis(z)
        self.assertAlmostEqual(gamma[1], 0.4917, places=4)

    def test_ints(self):
        z = np.array((10, 20, 30, 40), dtype=int)
        gamma = gs.vario_estimate_axis(z)
        self.assertAlmostEqual(gamma[1], 50.0, places=4)

    def test_mixed(self):
        z = np.array(
            (41.2, 40.2, 39.7, 39.2, 40.1, 38.3, 39.1, 40.0, 41.1, 40.3),
            dtype=np.double,
        )
        gamma = gs.vario_estimate_axis(z)
        self.assertAlmostEqual(gamma[1], 0.4917, places=4)

        z = np.array((10, 20, 30, 40), dtype=int)

        gamma = gs.vario_estimate_axis(z)
        self.assertAlmostEqual(gamma[1], 50.0, places=4)

    def test_list(self):
        z = [41.2, 40.2, 39.7, 39.2, 40.1, 38.3, 39.1, 40.0, 41.1, 40.3]
        gamma = gs.vario_estimate_axis(z)
        self.assertAlmostEqual(gamma[1], 0.4917, places=4)

    def test_cressie_1d(self):
        z = [41.2, 40.2, 39.7, 39.2, 40.1, 38.3, 39.1, 40.0, 41.1, 40.3]
        gamma = gs.vario_estimate_axis(z, estimator="cressie")
        self.assertAlmostEqual(gamma[1], 1.546 / 2.0, places=3)

    def test_1d(self):
        # literature values
        z = np.array(
            (41.2, 40.2, 39.7, 39.2, 40.1, 38.3, 39.1, 40.0, 41.1, 40.3),
            dtype=np.double,
        )
        gamma = gs.vario_estimate_axis(z)
        self.assertAlmostEqual(gamma[0], 0.0000, places=4)
        self.assertAlmostEqual(gamma[1], 0.4917, places=4)
        self.assertAlmostEqual(gamma[2], 0.7625, places=4)

    def test_masked_1d(self):
        # literature values
        z = np.array(
            (41.2, 40.2, 39.7, 39.2, 40.1, 38.3, 39.1, 40.0, 41.1, 40.3),
            dtype=np.double,
        )
        z_ma = np.ma.masked_array(z, mask=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        gamma = gs.vario_estimate_axis(z_ma)
        self.assertAlmostEqual(gamma[0], 0.0000, places=4)
        self.assertAlmostEqual(gamma[1], 0.4917, places=4)
        self.assertAlmostEqual(gamma[2], 0.7625, places=4)
        z_ma = np.ma.masked_array(z, mask=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        gamma = gs.vario_estimate_axis(z_ma)
        self.assertAlmostEqual(gamma[0], 0.0000, places=4)
        self.assertAlmostEqual(gamma[1], 0.4906, places=4)
        self.assertAlmostEqual(gamma[2], 0.7107, places=4)

    def test_masked_2d(self):
        rng = np.random.RandomState(1479373475)
        field = rng.rand(80, 60)
        mask = np.zeros_like(field)
        field_ma = np.ma.masked_array(field, mask=mask)

        gamma_x = gs.vario_estimate_axis(field_ma, direction="x")
        gamma_y = gs.vario_estimate_axis(field_ma, direction="y")

        var = 1.0 / 12.0
        self.assertAlmostEqual(gamma_x[0], 0.0, places=2)
        self.assertAlmostEqual(gamma_x[len(gamma_x) // 2], var, places=2)
        self.assertAlmostEqual(gamma_x[-1], var, places=2)
        self.assertAlmostEqual(gamma_y[0], 0.0, places=2)
        self.assertAlmostEqual(gamma_y[len(gamma_y) // 2], var, places=2)
        self.assertAlmostEqual(gamma_y[-1], var, places=2)

        mask = np.zeros_like(field)
        mask[0, 0] = 1
        field = np.ma.masked_array(field, mask=mask)
        gamma_x = gs.vario_estimate_axis(field_ma, direction="x")
        gamma_y = gs.vario_estimate_axis(field_ma, direction="y")
        self.assertAlmostEqual(gamma_x[0], 0.0, places=2)
        self.assertAlmostEqual(gamma_y[0], 0.0, places=2)

    def test_masked_3d(self):
        rng = np.random.RandomState(1479373475)
        field = rng.rand(30, 30, 30)
        mask = np.zeros_like(field)
        field_ma = np.ma.masked_array(field, mask=mask)

        gamma_x = gs.vario_estimate_axis(field_ma, direction="x")
        gamma_y = gs.vario_estimate_axis(field_ma, direction="y")
        gamma_z = gs.vario_estimate_axis(field_ma, direction="z")

        var = 1.0 / 12.0
        self.assertAlmostEqual(gamma_x[0], 0.0, places=2)
        self.assertAlmostEqual(gamma_x[len(gamma_x) // 2], var, places=2)
        self.assertAlmostEqual(gamma_x[-1], var, places=2)
        self.assertAlmostEqual(gamma_y[0], 0.0, places=2)
        self.assertAlmostEqual(gamma_y[len(gamma_y) // 2], var, places=2)
        self.assertAlmostEqual(gamma_y[-1], var, places=2)
        self.assertAlmostEqual(gamma_z[0], 0.0, places=2)
        self.assertAlmostEqual(gamma_z[len(gamma_y) // 2], var, places=2)
        self.assertAlmostEqual(gamma_z[-1], var, places=2)

        mask = np.zeros_like(field)
        mask[0, 0, 0] = 1
        field = np.ma.masked_array(field, mask=mask)
        gamma_x = gs.vario_estimate_axis(field_ma, direction="x")
        gamma_y = gs.vario_estimate_axis(field_ma, direction="y")
        gamma_z = gs.vario_estimate_axis(field_ma, direction="z")
        self.assertAlmostEqual(gamma_x[0], 0.0, places=2)
        self.assertAlmostEqual(gamma_y[0], 0.0, places=2)
        self.assertAlmostEqual(gamma_z[0], 0.0, places=2)

    def test_uncorrelated_2d(self):
        x = np.linspace(0.0, 100.0, 80)
        y = np.linspace(0.0, 100.0, 60)

        rng = np.random.RandomState(1479373475)
        field = rng.rand(len(x), len(y))

        gamma_x = gs.vario_estimate_axis(field, direction="x")
        gamma_y = gs.vario_estimate_axis(field, direction="y")

        var = 1.0 / 12.0
        self.assertAlmostEqual(gamma_x[0], 0.0, places=2)
        self.assertAlmostEqual(gamma_x[len(gamma_x) // 2], var, places=2)
        self.assertAlmostEqual(gamma_x[-1], var, places=2)
        self.assertAlmostEqual(gamma_y[0], 0.0, places=2)
        self.assertAlmostEqual(gamma_y[len(gamma_y) // 2], var, places=2)
        self.assertAlmostEqual(gamma_y[-1], var, places=2)

    def test_uncorrelated_cressie_2d(self):
        x = np.linspace(0.0, 100.0, 80)
        y = np.linspace(0.0, 100.0, 60)

        rng = np.random.RandomState(1479373475)
        field = rng.rand(len(x), len(y))

        gamma_x = gs.vario_estimate_axis(
            field, direction="x", estimator="cressie"
        )
        gamma_y = gs.vario_estimate_axis(
            field, direction="y", estimator="cressie"
        )

        var = 1.0 / 12.0
        self.assertAlmostEqual(gamma_x[0], 0.0, places=1)
        self.assertAlmostEqual(gamma_x[len(gamma_x) // 2], var, places=1)
        self.assertAlmostEqual(gamma_y[0], 0.0, places=1)
        self.assertAlmostEqual(gamma_y[len(gamma_y) // 2], var, places=1)

    def test_uncorrelated_3d(self):
        x = np.linspace(0.0, 100.0, 30)
        y = np.linspace(0.0, 100.0, 30)
        z = np.linspace(0.0, 100.0, 30)

        rng = np.random.RandomState(1479373475)
        field = rng.rand(len(x), len(y), len(z))

        gamma = gs.vario_estimate_axis(field, "x")
        gamma = gs.vario_estimate_axis(field, "y")
        gamma = gs.vario_estimate_axis(field, "z")

        var = 1.0 / 12.0
        self.assertAlmostEqual(gamma[0], 0.0, places=2)
        self.assertAlmostEqual(gamma[len(gamma) // 2], var, places=2)
        self.assertAlmostEqual(gamma[-1], var, places=2)

    def test_directions_2d(self):
        x = np.linspace(0.0, 20.0, 100)
        y = np.linspace(0.0, 15.0, 80)
        rng = np.random.RandomState(1479373475)
        x_rand = rng.rand(len(x))
        y_rand = rng.rand(len(y))
        # random values repeated along y-axis
        field_x = np.tile(x_rand, (len(y), 1)).T
        # random values repeated along x-axis
        field_y = np.tile(y_rand, (len(x), 1))

        # gamma_x_x = gs.vario_estimate_axis(field_x, direction="x")
        gamma_x_y = gs.vario_estimate_axis(field_x, direction="y")

        gamma_y_x = gs.vario_estimate_axis(field_y, direction="x")
        # gamma_y_y = gs.vario_estimate_axis(field_y, direction="y")

        self.assertAlmostEqual(gamma_x_y[1], 0.0)
        self.assertAlmostEqual(gamma_x_y[len(gamma_x_y) // 2], 0.0)
        self.assertAlmostEqual(gamma_x_y[-1], 0.0)
        self.assertAlmostEqual(gamma_y_x[1], 0.0)
        self.assertAlmostEqual(gamma_y_x[len(gamma_x_y) // 2], 0.0)
        self.assertAlmostEqual(gamma_y_x[-1], 0.0)

    def test_directions_3d(self):
        x = np.linspace(0.0, 10.0, 20)
        y = np.linspace(0.0, 15.0, 25)
        z = np.linspace(0.0, 20.0, 30)
        rng = np.random.RandomState(1479373475)
        x_rand = rng.rand(len(x))
        y_rand = rng.rand(len(y))
        z_rand = rng.rand(len(z))

        field_x = np.tile(x_rand.reshape((len(x), 1, 1)), (1, len(y), len(z)))
        field_y = np.tile(y_rand.reshape((1, len(y), 1)), (len(x), 1, len(z)))
        field_z = np.tile(z_rand.reshape((1, 1, len(z))), (len(x), len(y), 1))

        # gamma_x_x = gs.vario_estimate_axis(field_x, direction="x")
        gamma_x_y = gs.vario_estimate_axis(field_x, direction="y")
        gamma_x_z = gs.vario_estimate_axis(field_x, direction="z")

        gamma_y_x = gs.vario_estimate_axis(field_y, direction="x")
        # gamma_y_y = gs.vario_estimate_axis(field_y, direction="y")
        gamma_y_z = gs.vario_estimate_axis(field_y, direction="z")

        gamma_z_x = gs.vario_estimate_axis(field_z, direction="x")
        gamma_z_y = gs.vario_estimate_axis(field_z, direction="y")
        # gamma_z_z = gs.vario_estimate_axis(field_z, direction="z")

        self.assertAlmostEqual(gamma_x_y[1], 0.0)
        self.assertAlmostEqual(gamma_x_y[len(gamma_x_y) // 2], 0.0)
        self.assertAlmostEqual(gamma_x_y[-1], 0.0)
        self.assertAlmostEqual(gamma_x_z[1], 0.0)
        self.assertAlmostEqual(gamma_x_z[len(gamma_x_y) // 2], 0.0)
        self.assertAlmostEqual(gamma_x_z[-1], 0.0)
        self.assertAlmostEqual(gamma_y_x[1], 0.0)
        self.assertAlmostEqual(gamma_y_x[len(gamma_x_y) // 2], 0.0)
        self.assertAlmostEqual(gamma_y_x[-1], 0.0)
        self.assertAlmostEqual(gamma_y_z[1], 0.0)
        self.assertAlmostEqual(gamma_y_z[len(gamma_x_y) // 2], 0.0)
        self.assertAlmostEqual(gamma_y_z[-1], 0.0)
        self.assertAlmostEqual(gamma_z_x[1], 0.0)
        self.assertAlmostEqual(gamma_z_x[len(gamma_x_y) // 2], 0.0)
        self.assertAlmostEqual(gamma_z_x[-1], 0.0)
        self.assertAlmostEqual(gamma_z_y[1], 0.0)
        self.assertAlmostEqual(gamma_z_y[len(gamma_x_y) // 2], 0.0)
        self.assertAlmostEqual(gamma_z_y[-1], 0.0)

    def test_exceptions(self):
        x = np.linspace(0.0, 10.0, 20)
        # rng = np.random.RandomState(1479373475)
        # x_rand = rng.rand(len(x))
        self.assertRaises(ValueError, gs.vario_estimate_axis, x, "a")

    def test_missing(self):
        x = np.linspace(0.0, 10.0, 10)
        x_nan = x.copy()
        x_nan[0] = np.nan
        x_mask = np.isnan(x_nan)
        x = np.ma.array(x, mask=x_mask)
        v1 = gs.vario_estimate_axis(x_nan)
        v2 = gs.vario_estimate_axis(x)
        for i in range(len(v1)):
            self.assertAlmostEqual(v1[i], v2[i])


if __name__ == "__main__":
    unittest.main()
