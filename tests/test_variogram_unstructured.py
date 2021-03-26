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
        self.assertRaises(
            ValueError, gs.vario_estimate, [x], field, bins, estimator="bla"
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
            self.assertEqual(c1[i], c2[i])

    def test_direction_assertion(self):
        pos = [[1, 2, 3], [1, 2, 3]]
        bns = [1, 2]
        fld = np.ma.array([1, 2, 3])
        self.assertRaises(  # degenerated direction
            ValueError, gs.vario_estimate, pos, fld, bns, direction=[0, 0]
        )
        self.assertRaises(  # wrong shape of direction
            ValueError, gs.vario_estimate, pos, fld, bns, direction=[[[3, 1]]]
        )
        self.assertRaises(  # wrong dimension of direction
            ValueError, gs.vario_estimate, pos, fld, bns, direction=[[3, 1, 2]]
        )
        self.assertRaises(  # wrong shape of angles
            ValueError, gs.vario_estimate, pos, fld, bns, angles=[[[1]]]
        )
        self.assertRaises(  # wrong dimension of angles
            ValueError, gs.vario_estimate, pos, fld, bns, angles=[[1, 1]]
        )
        self.assertRaises(  # direction on latlon
            ValueError,
            gs.vario_estimate,
            pos,
            fld,
            bns,
            direction=[1, 0],
            latlon=True,
        )

    def test_mask_no_data(self):
        pos = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
        bns = [0, 4]
        fld1 = np.ma.array([1, 2, 3, 4, 5])
        fld2 = np.ma.array([np.nan, 2, 3, 4, 5])
        fld3 = np.ma.array([1, 2, 3, 4, 5])
        mask = [False, False, True, False, False]
        fld1.mask = [True, False, False, False, False]
        fld2.mask = mask
        __, v1, c1 = gs.vario_estimate(
            *(pos, fld1, bns),
            mask=mask,
            return_counts=True,
        )
        __, v2, c2 = gs.vario_estimate(*(pos, fld2, bns), return_counts=True)
        __, v3, c3 = gs.vario_estimate(
            *(pos, fld3, bns),
            no_data=1,
            mask=mask,
            return_counts=True,
        )
        __, v4, c4 = gs.vario_estimate(
            *(pos, fld3, bns),
            mask=True,
            return_counts=True,
        )
        __, v5 = gs.vario_estimate(*(pos, fld3, bns), mask=True)

        self.assertAlmostEqual(v1[0], v2[0])
        self.assertAlmostEqual(v1[0], v3[0])
        self.assertEqual(c1[0], c2[0])
        self.assertEqual(c1[0], c3[0])
        self.assertAlmostEqual(v4[0], 0.0)
        self.assertEqual(c4[0], 0)
        self.assertAlmostEqual(v5[0], 0.0)

    def test_fit_directional(self):
        model = gs.Stable(dim=3)
        bins = [0, 3, 6, 9, 12]
        model.len_scale_bounds = [0, 20]
        bin_center, emp_vario, counts = gs.vario_estimate(
            *(self.pos, self.field, bins),
            direction=model.main_axes(),
            mesh_type="structured",
            return_counts=True,
        )
        # check if this succeeds
        model.fit_variogram(bin_center, emp_vario, sill=1, return_r2=True)
        self.assertTrue(1 > model.anis[0] > model.anis[1])
        model.fit_variogram(bin_center, emp_vario, sill=1, anis=[0.5, 0.25])
        self.assertTrue(15 > model.len_scale)
        model.fit_variogram(bin_center, emp_vario, sill=1, weights=counts)
        len_save = model.len_scale
        model.fit_variogram(bin_center, emp_vario, sill=1, weights=counts[0])
        self.assertAlmostEqual(len_save, model.len_scale)
        # catch wrong dim for dir.-vario
        with self.assertRaises(ValueError):
            model.fit_variogram(bin_center, emp_vario[:2])

    def test_auto_binning(self):
        # structured mesh
        bin_center, emp_vario = gs.vario_estimate(
            self.pos,
            self.field,
            mesh_type="structured",
        )
        self.assertEqual(len(bin_center), 21)
        self.assertTrue(np.all(bin_center[1:] > bin_center[:-1]))
        self.assertTrue(np.all(bin_center > 0))
        # unstructured mesh
        bin_center, emp_vario = gs.vario_estimate(
            self.pos,
            self.field[:, 0, 0],
        )
        self.assertEqual(len(bin_center), 8)
        self.assertTrue(np.all(bin_center[1:] > bin_center[:-1]))
        self.assertTrue(np.all(bin_center > 0))
        # latlon coords
        bin_center, emp_vario = gs.vario_estimate(
            self.pos[:2],
            self.field[..., 0],
            mesh_type="structured",
            latlon=True,
        )
        self.assertEqual(len(bin_center), 15)
        self.assertTrue(np.all(bin_center[1:] > bin_center[:-1]))
        self.assertTrue(np.all(bin_center > 0))

    def test_standard_bins(self):
        # structured mesh
        bins = gs.standard_bins(self.pos, dim=3, mesh_type="structured")
        self.assertEqual(len(bins), 22)
        self.assertTrue(np.all(bins[1:] > bins[:-1]))
        self.assertTrue(np.all(bins[1:] > 0))
        # no pos given
        self.assertRaises(ValueError, gs.standard_bins)

    def test_raise(self):
        # 1d field given for latlon estimation -> needs 2d
        self.assertRaises(
            ValueError, gs.vario_estimate, [[1, 2]], [1, 2], latlon=True
        )


if __name__ == "__main__":
    unittest.main()
