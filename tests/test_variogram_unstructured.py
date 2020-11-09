# -*- coding: utf-8 -*-
"""
This is a unittest of the variogram module.
"""

import unittest
import numpy as np
import gstools as gs


class TestVariogramUnstructured(unittest.TestCase):
    def setUp(self):
        model = gs.Exponential(dim=3, len_scale=[12, 6, 3])
        x = y = z = range(10)
        self.pos = (x, y, z)
        srf = gs.SRF(model, seed=123456)
        self.field = srf((x, y, z), mesh_type="structured")

    def test_doubles(self):
        x = np.arange(1, 11, 1, dtype=np.double)
        z = np.array(
            (41.2, 40.2, 39.7, 39.2, 40.1, 38.3, 39.1, 40.0, 41.1, 40.3),
            dtype=np.double,
        )
        bins = np.arange(1, 11, 1, dtype=np.double)
        bin_centres, gamma = gs.vario_estimate([x], z, bins)
        self.assertAlmostEqual(gamma[0], 0.4917, places=4)

    def test_ints(self):
        x = np.arange(1, 5, 1, dtype=int)
        z = np.array((10, 20, 30, 40), dtype=int)
        bins = np.arange(1, 11, 1, dtype=int)
        bin_centres, gamma = gs.vario_estimate([x], z, bins)
        self.assertAlmostEqual(gamma[0], 50.0, places=4)

    def test_np_int(self):
        x = np.arange(1, 5, 1, dtype=np.int)
        z = np.array((10, 20, 30, 40), dtype=np.int)
        bins = np.arange(1, 11, 1, dtype=np.int)
        bin_centres, gamma = gs.vario_estimate([x], z, bins)
        self.assertAlmostEqual(gamma[0], 50.0, places=4)

    def test_mixed(self):
        x = np.arange(1, 11, 1, dtype=np.double)
        z = np.array(
            (41.2, 40.2, 39.7, 39.2, 40.1, 38.3, 39.1, 40.0, 41.1, 40.3),
            dtype=np.double,
        )
        bins = np.arange(1, 11, 1, dtype=int)
        bin_centres, gamma = gs.vario_estimate([x], z, bins)
        self.assertAlmostEqual(gamma[0], 0.4917, places=4)

        x = np.arange(1, 5, 1, dtype=np.double)
        z = np.array((10, 20, 30, 40), dtype=int)
        bins = np.arange(1, 11, 1, dtype=int)
        bin_centres, gamma = gs.vario_estimate([x], z, bins)
        self.assertAlmostEqual(gamma[0], 50.0, places=4)

        x = np.arange(1, 5, 1, dtype=np.double)
        z = np.array((10, 20, 30, 40), dtype=int)
        bins = np.arange(1, 11, 1, dtype=np.double)
        bin_centres, gamma = gs.vario_estimate([x], z, bins)
        self.assertAlmostEqual(gamma[0], 50.0, places=4)

    def test_list(self):
        x = np.arange(1, 11, 1, dtype=np.double)
        z = [41.2, 40.2, 39.7, 39.2, 40.1, 38.3, 39.1, 40.0, 41.1, 40.3]
        bins = np.arange(1, 11, 1, dtype=np.double)
        bin_centres, gamma = gs.vario_estimate([x], z, bins)
        self.assertAlmostEqual(gamma[1], 0.7625, places=4)

    def test_1d(self):
        x = np.arange(1, 11, 1, dtype=np.double)
        # literature values
        z = np.array(
            (41.2, 40.2, 39.7, 39.2, 40.1, 38.3, 39.1, 40.0, 41.1, 40.3),
            dtype=np.double,
        )
        bins = np.arange(1, 11, 1, dtype=np.double)
        bin_centres, gamma = gs.vario_estimate([x], z, bins)
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

        bin_centres, gamma = gs.vario_estimate((x, y), field, bins)

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

        bin_centres, gamma = gs.vario_estimate((x, y, z), field, bins)

        var = 1.0 / 12.0
        self.assertAlmostEqual(gamma[0], var, places=2)
        self.assertAlmostEqual(gamma[len(gamma) // 2], var, places=2)
        self.assertAlmostEqual(gamma[-1], var, places=2)

    def test_sampling_1d(self):
        x = np.linspace(0.0, 100.0, 21000)

        rng = np.random.RandomState(1479373475)
        field = rng.rand(len(x))

        bins = np.arange(0, 100, 10)

        bin_centres, gamma = gs.vario_estimate(
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

        bin_centres, gamma = gs.vario_estimate(
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

        bin_centres, gamma = gs.vario_estimate(
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

        self.assertRaises(ValueError, gs.vario_estimate, [x_e], field, bins)
        self.assertRaises(ValueError, gs.vario_estimate, (x, y_e), field, bins)
        self.assertRaises(
            ValueError, gs.vario_estimate, (x, y_e, z), field, bins
        )
        self.assertRaises(
            ValueError, gs.vario_estimate, (x, y, z_e), field, bins
        )
        self.assertRaises(
            ValueError, gs.vario_estimate, (x_e, y, z), field, bins
        )
        self.assertRaises(
            ValueError, gs.vario_estimate, (x, y, z), field_e, bins
        )
        self.assertRaises(ValueError, gs.vario_estimate, [x], field_e, bins)

    def test_multi_field(self):
        x = np.random.RandomState(19970221).rand(100) * 100.0
        model = gs.Exponential(dim=1, var=2, len_scale=10)
        srf = gs.SRF(model)
        field1 = srf(x, seed=19970221)
        field2 = srf(x, seed=20011012)
        bins = np.arange(20) * 2
        bin_center, gamma1 = gs.vario_estimate(x, field1, bins)
        bin_center, gamma2 = gs.vario_estimate(x, field2, bins)
        bin_center, gamma = gs.vario_estimate(x, [field1, field2], bins)
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

    def test_direction_axis(self):
        field = np.ma.array(self.field)
        field.mask = np.abs(field) < 0.1
        bins = range(10)
        __, vario_u = gs.vario_estimate(
            *(self.pos, field, bins),
            direction=((1, 0, 0), (0, 1, 0), (0, 0, 1)),  # x-, y- and z-axis
            bandwidth=0.25,  # bandwith small enough to only match lines
            mesh_type="structured",
        )
        vario_s_x = gs.vario_estimate_axis(field, "x")
        vario_s_y = gs.vario_estimate_axis(field, "y")
        vario_s_z = gs.vario_estimate_axis(field, "z")
        for i in range(len(bins) - 1):
            self.assertAlmostEqual(vario_u[0][i], vario_s_x[i])
            self.assertAlmostEqual(vario_u[1][i], vario_s_y[i])
            self.assertAlmostEqual(vario_u[2][i], vario_s_z[i])

    def test_direction_angle(self):
        bins = range(0, 10, 2)
        __, v2, c2 = gs.vario_estimate(
            *(self.pos[:2], self.field[0], bins),
            angles=np.pi / 4,  # 45 deg
            mesh_type="structured",
            return_counts=True,
        )
        __, v1, c1 = gs.vario_estimate(
            *(self.pos[:2], self.field[0], bins),
            direction=(1, 1),  # 45 deg
            mesh_type="structured",
            return_counts=True,
        )
        for i in range(len(bins) - 1):
            self.assertAlmostEqual(v1[i], v2[i])
            self.assertAlmostEqual(c1[i], c2[i])


if __name__ == "__main__":
    unittest.main()
