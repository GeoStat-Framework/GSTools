# -*- coding: utf-8 -*-
"""
This is a unittest of the variogram module.
"""

import unittest
import numpy as np
from gstools import vario_estimate_unstructured, Exponential, SRF
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
        bin_center, gamma = gs.vario_estimate_unstructured((x,), field, bins)
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
        bin_centres, gamma = vario_estimate_unstructured([x], z, bins)
        self.assertAlmostEqual(gamma[0], 0.4917, places=4)

    def test_ints(self):
        x = np.arange(1, 5, 1, dtype=int)
        z = np.array((10, 20, 30, 40), dtype=int)
        bins = np.arange(1, 11, 1, dtype=int)
        bin_centres, gamma = vario_estimate_unstructured([x], z, bins)
        self.assertAlmostEqual(gamma[0], 50.0, places=4)

    def test_np_int(self):
        x = np.arange(1, 5, 1, dtype=np.int)
        z = np.array((10, 20, 30, 40), dtype=np.int)
        bins = np.arange(1, 11, 1, dtype=np.int)
        bin_centres, gamma = vario_estimate_unstructured([x], z, bins)
        self.assertAlmostEqual(gamma[0], 50.0, places=4)

    def test_mixed(self):
        x = np.arange(1, 11, 1, dtype=np.double)
        z = np.array(
            (41.2, 40.2, 39.7, 39.2, 40.1, 38.3, 39.1, 40.0, 41.1, 40.3),
            dtype=np.double,
        )
        bins = np.arange(1, 11, 1, dtype=int)
        bin_centres, gamma = vario_estimate_unstructured([x], z, bins)
        self.assertAlmostEqual(gamma[0], 0.4917, places=4)

        x = np.arange(1, 5, 1, dtype=np.double)
        z = np.array((10, 20, 30, 40), dtype=int)
        bins = np.arange(1, 11, 1, dtype=int)
        bin_centres, gamma = vario_estimate_unstructured([x], z, bins)
        self.assertAlmostEqual(gamma[0], 50.0, places=4)

        x = np.arange(1, 5, 1, dtype=np.double)
        z = np.array((10, 20, 30, 40), dtype=int)
        bins = np.arange(1, 11, 1, dtype=np.double)
        bin_centres, gamma = vario_estimate_unstructured([x], z, bins)
        self.assertAlmostEqual(gamma[0], 50.0, places=4)

    def test_list(self):
        x = np.arange(1, 11, 1, dtype=np.double)
        z = [41.2, 40.2, 39.7, 39.2, 40.1, 38.3, 39.1, 40.0, 41.1, 40.3]
        bins = np.arange(1, 11, 1, dtype=np.double)
        bin_centres, gamma = vario_estimate_unstructured([x], z, bins)
        self.assertAlmostEqual(gamma[1], 0.7625, places=4)

    def test_1d(self):
        x = np.arange(1, 11, 1, dtype=np.double)
        # literature values
        z = np.array(
            (41.2, 40.2, 39.7, 39.2, 40.1, 38.3, 39.1, 40.0, 41.1, 40.3),
            dtype=np.double,
        )
        bins = np.arange(1, 11, 1, dtype=np.double)
        bin_centres, gamma = vario_estimate_unstructured([x], z, bins)
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

        bin_centres, gamma = vario_estimate_unstructured((x, y), field, bins)

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

        bin_centres, gamma = vario_estimate_unstructured(
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

        bin_centres, gamma = vario_estimate_unstructured(
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

        bin_centres, gamma = vario_estimate_unstructured(
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

        bin_centres, gamma = vario_estimate_unstructured(
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
            ValueError, vario_estimate_unstructured, [x_e], field, bins
        )
        self.assertRaises(
            ValueError, vario_estimate_unstructured, (x, y_e), field, bins
        )
        self.assertRaises(
            ValueError, vario_estimate_unstructured, (x, y_e, z), field, bins
        )
        self.assertRaises(
            ValueError, vario_estimate_unstructured, (x, y, z_e), field, bins
        )
        self.assertRaises(
            ValueError, vario_estimate_unstructured, (x_e, y, z), field, bins
        )
        self.assertRaises(
            ValueError, vario_estimate_unstructured, (x, y, z), field_e, bins
        )
        self.assertRaises(
            ValueError, vario_estimate_unstructured, [x], field_e, bins
        )

    # def test_angles_2D_x2x(self):

    #     x = self.test_data_rotation_1["x"]
    #     field = self.test_data_rotation_1["field"]
    #     gamma_exp = self.test_data_rotation_1["gamma"]
    #     bins = self.test_data_rotation_1["bins"]
    #     y = np.zeros_like(x)

    #     # test case 1.)
    #     #    all along x axis on x axis

    #     bin_centres, gamma = vario_estimate_unstructured(
    #         (x, y), field, bins, angles=[0]
    #     )

    #     for i in range(gamma.size):
    #         self.assertAlmostEqual(gamma_exp[i], gamma[i], places=3)

    # def test_angles_2D_y2x(self):

    #     x = self.test_data_rotation_1["x"]
    #     field = self.test_data_rotation_1["field"]
    #     gamma_exp = self.test_data_rotation_1["gamma"]
    #     bins = self.test_data_rotation_1["bins"]
    #     y = np.zeros_like(x)

    #     # test case 2.)
    #     #    all along y axis on y axis but calculation for x axis

    #     bin_centres, gamma = vario_estimate_unstructured(
    #         (y, x), field, bins, angles=[0]
    #     )

    #     for i in range(gamma.size):
    #         self.assertAlmostEqual(0, gamma[i], places=3)

    # def test_angles_2D_y2y(self):

    #     x = self.test_data_rotation_1["x"]
    #     field = self.test_data_rotation_1["field"]
    #     gamma_exp = self.test_data_rotation_1["gamma"]
    #     bins = self.test_data_rotation_1["bins"]
    #     y = np.zeros_like(x)

    #     # test case 3.)
    #     #    all along y axis on y axis and calculation for y axis

    #     bin_centres, gamma = vario_estimate_unstructured(
    #         (y, x), field, bins, angles=[np.pi / 2.0]
    #     )

    #     for i in range(gamma.size):
    #         self.assertAlmostEqual(gamma_exp[i], gamma[i], places=3)

    # def test_angles_2D_xy2x(self):

    #     x = self.test_data_rotation_1["x"]
    #     field = self.test_data_rotation_1["field"]
    #     gamma_exp = self.test_data_rotation_1["gamma"]
    #     bins = self.test_data_rotation_1["bins"]
    #     y = np.zeros_like(x)

    #     # test case 4.)
    #     #    data along 45deg axis but calculation for x axis

    #     ccos, csin = np.cos(np.pi / 4.0), np.sin(np.pi / 4.0)

    #     xr = [xx * ccos - yy * csin for xx, yy in zip(x, y)]
    #     yr = [xx * csin + yy * ccos for xx, yy in zip(x, y)]

    #     bin_centres, gamma = vario_estimate_unstructured(
    #         (xr, yr), field, bins, angles=[0]
    #     )

    #     for i in range(gamma.size):
    #         self.assertAlmostEqual(0, gamma[i], places=3)

    # def test_angles_2D_estim(self):

    #     seed = gs.random.MasterRNG(19970221)
    #     rng = np.random.RandomState(seed())
    #     rng = np.random
    #     x = rng.randint(0, 100, size=3000)
    #     y = rng.randint(0, 100, size=3000)

    #     model = gs.Exponential(
    #         dim=2, var=1, len_scale=[12, 3], angles=np.pi / 8
    #     )
    #     model_maj = gs.Exponential(dim=1, var=1, len_scale=[12])
    #     model_min = gs.Exponential(dim=1, var=1, len_scale=[3])

    #     srf = gs.SRF(model, seed=20170519)
    #     field = srf((x, y))

    #     bins = np.arange(0, 50, 2.5)
    #     angle_mask = 22.5
    #     angle_tol = 22.5

    #     bin_centers_maj, gamma_maj = gs.vario_estimate_unstructured(
    #         (x, y),
    #         field,
    #         bins,
    #         angles=[np.deg2rad(angle_mask)],
    #         angles_tol=np.deg2rad(angle_tol),
    #     )

    #     bin_centers_min, gamma_min = gs.vario_estimate_unstructured(
    #         (x, y),
    #         field,
    #         bins,
    #         angles=[np.deg2rad(angle_mask + 90.0)],
    #         angles_tol=np.deg2rad(angle_tol),
    #     )

    #     gamma_maj_real = model_maj.variogram(bin_centers_maj)
    #     gamma_min_real = model_min.variogram(bin_centers_min)

    #     # we have no real way of testing values, but we can test some basic properties which definitelly need to be true
    #     # test that the major estimate aligns better with major real than minor real
    #     self.assertTrue(
    #         np.sum((gamma_maj_real - gamma_maj) ** 2)
    #         < np.sum((gamma_min_real - gamma_maj) ** 2)
    #     )
    #     # test that the minor estimate aligns better with minor real than major real
    #     self.assertTrue(
    #         np.sum((gamma_min_real - gamma_min) ** 2)
    #         < np.sum((gamma_maj_real - gamma_min) ** 2)
    #     )
    #     # test that both variograms converge within reasonable closeness (less than 10% rel error) to the actual field variance
    #     self.assertTrue(
    #         (np.mean(gamma_min[-5:]) - np.var(field)) / np.var(field) < 0.1
    #     )
    #     self.assertTrue(
    #         (np.mean(gamma_maj[-5:]) - np.var(field)) / np.var(field) < 0.1
    #     )

    # def test_angles_line_3D(self):

    #     x = self.test_data_rotation_1["x"]
    #     field = self.test_data_rotation_1["field"]
    #     gamma_exp = self.test_data_rotation_1["gamma"]
    #     bins = self.test_data_rotation_1["bins"]

    #     def test_xyz(x, y, z, angles, gamma_exp):
    #         bin_centres, gamma = vario_estimate_unstructured(
    #             (x, y, z), field, bins, angles=angles
    #         )

    #         if np.ndim(gamma_exp) == 0:
    #             gamma_exp = np.ones_like(gamma) * gamma_exp

    #         for i in range(gamma.size):
    #             self.assertAlmostEqual(gamma_exp[i], gamma[i], places=3)

    #     # all along x axis and calculation for x axis
    #     test_xyz(x, np.zeros_like(x), np.zeros_like(x), [0], gamma_exp)
    #     # all along y axis and calculation for x axis
    #     test_xyz(np.zeros_like(x), x, np.zeros_like(x), [0], 0)
    #     # all along z axis and calculation for x axis
    #     test_xyz(np.zeros_like(x), np.zeros_like(x), x, [0], 0)

    #     angles_rot_azim_90 = [np.pi / 2]
    #     # all along x axis and calculation for y axis
    #     test_xyz(x, np.zeros_like(x), np.zeros_like(x), angles_rot_azim_90, 0)
    #     # all along y axis and calculation for y axis
    #     test_xyz(
    #         np.zeros_like(x),
    #         x,
    #         np.zeros_like(x),
    #         angles_rot_azim_90,
    #         gamma_exp,
    #     )
    #     # all along z axis and calculation for y axis
    #     test_xyz(np.zeros_like(x), np.zeros_like(x), x, angles_rot_azim_90, 0)

    #     # for elevation it is important to check, that IF elevation is 90° or 270° it does
    #     # not matter how we rotated before, since any rotation around z (in XY plane)
    #     # followed by a rotation around x' (in YZ' plane) by 90° will result in the same
    #     # coordinates, (when the structure is two points with zero extend)

    #     # test with [0, 90]
    #     angles_rot_azim_90_elev_90 = [0, np.pi / 2]
    #     # all along x axis and calculation for z axis
    #     test_xyz(
    #         x,
    #         np.zeros_like(x),
    #         np.zeros_like(x),
    #         angles_rot_azim_90_elev_90,
    #         0,
    #     )
    #     # all along y axis and calculation for z axis
    #     test_xyz(
    #         np.zeros_like(x),
    #         x,
    #         np.zeros_like(x),
    #         angles_rot_azim_90_elev_90,
    #         0,
    #     )
    #     # all along z axis and calculation for z axis
    #     test_xyz(
    #         np.zeros_like(x),
    #         np.zeros_like(x),
    #         x,
    #         angles_rot_azim_90_elev_90,
    #         gamma_exp,
    #     )

    #     # test with [90, 90]
    #     angles_rot_azim_90_elev_90 = [np.pi / 2, np.pi / 2]
    #     # all along x axis and calculation for z axis
    #     test_xyz(
    #         x,
    #         np.zeros_like(x),
    #         np.zeros_like(x),
    #         angles_rot_azim_90_elev_90,
    #         0,
    #     )
    #     # all along y axis and calculation for z axis
    #     test_xyz(
    #         np.zeros_like(x),
    #         x,
    #         np.zeros_like(x),
    #         angles_rot_azim_90_elev_90,
    #         0,
    #     )
    #     # all along z axis and calculation for z axis
    #     test_xyz(
    #         np.zeros_like(x),
    #         np.zeros_like(x),
    #         x,
    #         angles_rot_azim_90_elev_90,
    #         gamma_exp,
    #     )

    # def test_angles_3D_estim(self):

    #     seed = gs.random.MasterRNG(19970221)
    #     rng = np.random.RandomState(seed())
    #     rng = np.random
    #     x = rng.randint(0, 50, size=1000)
    #     y = rng.randint(0, 50, size=1000)
    #     z = rng.randint(0, 50, size=1000)

    #     model = gs.Exponential(
    #         dim=3, var=1, len_scale=[10, 5, 2], angles=[np.pi / 8, np.pi / 16]
    #     )
    #     model_maj = gs.Exponential(dim=1, var=1, len_scale=[10])
    #     model_min1 = gs.Exponential(dim=1, var=1, len_scale=[5])
    #     model_min2 = gs.Exponential(dim=1, var=1, len_scale=[2])

    #     srf = gs.SRF(model, seed=20170519)
    #     field = srf((x, y, z))

    #     bins = np.arange(0, 25, 5)
    #     angle_mask = [np.pi / 8, np.pi / 16]
    #     angle_tol = 22.5

    #     # in x'
    #     bin_centers_maj, gamma_maj = gs.vario_estimate_unstructured(
    #         (x, y, z),
    #         field,
    #         bins,
    #         angles=[angle_mask[0], angle_mask[1]],
    #         angles_tol=np.deg2rad(angle_tol),
    #     )

    #     # in y'
    #     bin_centers_min1, gamma_min1 = gs.vario_estimate_unstructured(
    #         (x, y, z),
    #         field,
    #         bins,
    #         angles=[angle_mask[0] + 0.5 * np.pi, angle_mask[1]],
    #         angles_tol=np.deg2rad(angle_tol),
    #     )

    #     # in z'
    #     bin_centers_min2, gamma_min2 = gs.vario_estimate_unstructured(
    #         (x, y, z),
    #         field,
    #         bins,
    #         angles=[angle_mask[0], angle_mask[1] + 0.5 * np.pi],
    #         angles_tol=np.deg2rad(angle_tol),
    #     )

    #     gamma_maj_real = model_maj.variogram(bin_centers_maj)
    #     gamma_min1_real = model_min1.variogram(bin_centers_min1)
    #     gamma_min2_real = model_min2.variogram(bin_centers_min2)

    #     # we have no real way of testing values, but we can test some basic properties which definitelly need to be true
    #     # test that the major estimate aligns better with major real than the minor reals
    #     self.assertTrue(
    #         np.sum((gamma_maj_real - gamma_maj) ** 2)
    #         < np.sum((gamma_min1_real - gamma_maj) ** 2)
    #     )
    #     self.assertTrue(
    #         np.sum((gamma_maj_real - gamma_maj) ** 2)
    #         < np.sum((gamma_min2_real - gamma_maj) ** 2)
    #     )
    #     # test that the minor1 estimate aligns better with minor1 real than major real and minor2 real
    #     self.assertTrue(
    #         np.sum((gamma_min1_real - gamma_min1) ** 2)
    #         < np.sum((gamma_maj_real - gamma_min1) ** 2)
    #     )
    #     self.assertTrue(
    #         np.sum((gamma_min1_real - gamma_min1) ** 2)
    #         < np.sum((gamma_min2_real - gamma_min1) ** 2)
    #     )
    #     # test that the minor2 estimate aligns better with minor2 real than major real and minor1 real
    #     self.assertTrue(
    #         np.sum((gamma_min2_real - gamma_min2) ** 2)
    #         < np.sum((gamma_maj_real - gamma_min2) ** 2)
    #     )
    #     self.assertTrue(
    #         np.sum((gamma_min2_real - gamma_min2) ** 2)
    #         < np.sum((gamma_min1_real - gamma_min2) ** 2)
    #     )
    #     # test that all variograms converge within reasonable closeness (less than 10% rel error) to the actual field variance
    #     self.assertTrue(
    #         (np.mean(gamma_maj[-5:]) - np.var(field)) / np.var(field) < 0.1
    #     )
    #     self.assertTrue(
    #         (np.mean(gamma_min1[-5:]) - np.var(field)) / np.var(field) < 0.1
    #     )
    #     self.assertTrue(
    #         (np.mean(gamma_min2[-5:]) - np.var(field)) / np.var(field) < 0.1
    #     )


if __name__ == "__main__":
    unittest.main()
