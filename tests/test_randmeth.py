# -*- coding: utf-8 -*-
"""
This is the unittest of the RandMeth class.
"""
from __future__ import division, absolute_import, print_function

import copy
import unittest
import numpy as np
from gstools import Gaussian
from gstools.field.generator import RandMeth
import emcee as mc


MC_VER = int(mc.__version__.split('.')[0])


class TestRandMeth(unittest.TestCase):
    def setUp(self):
        self.cov_model_1d = Gaussian(
            dim=1, var=1.5, len_scale=3.5, mode_no=100
        )
        self.cov_model_2d = copy.deepcopy(self.cov_model_1d)
        self.cov_model_2d.dim = 2
        self.cov_model_3d = copy.deepcopy(self.cov_model_1d)
        self.cov_model_3d.dim = 3
        self.seed = 19031977
        self.x_grid = np.linspace(0.0, 10.0, 9)
        self.y_grid = np.linspace(-5.0, 5.0, 16)
        self.z_grid = np.linspace(-6.0, 7.0, 8)
        self.x_grid = np.reshape(self.x_grid, (len(self.x_grid), 1, 1, 1))
        self.y_grid = np.reshape(self.y_grid, (1, len(self.y_grid), 1, 1))
        self.z_grid = np.reshape(self.z_grid, (1, 1, len(self.z_grid), 1))
        self.x_tuple = np.linspace(0.0, 10.0, 10)
        self.y_tuple = np.linspace(-5.0, 5.0, 10)
        self.z_tuple = np.linspace(-6.0, 8.0, 10)
        self.x_tuple = np.reshape(self.x_tuple, (len(self.x_tuple), 1))
        self.y_tuple = np.reshape(self.y_tuple, (len(self.y_tuple), 1))
        self.z_tuple = np.reshape(self.z_tuple, (len(self.z_tuple), 1))

        self.rm_1d = RandMeth(self.cov_model_1d, 100, self.seed)
        self.rm_2d = RandMeth(self.cov_model_2d, 100, self.seed)
        self.rm_3d = RandMeth(self.cov_model_3d, 100, self.seed)

    def test_unstruct_1d(self):
        modes = self.rm_1d(self.x_tuple)
        self.assertAlmostEqual(modes[0], 3.19799030)
        self.assertAlmostEqual(modes[1], 2.44848295)

    def test_unstruct_2d(self):
        modes = self.rm_2d(self.x_tuple, self.y_tuple)
        self.assertAlmostEqual(modes[0], 1.67318010)
        self.assertAlmostEqual(modes[1], 2.12310269)

    def test_unstruct_3d(self):
        modes = self.rm_3d(self.x_tuple, self.y_tuple, self.z_tuple)
        if MC_VER < 3:
            self.assertAlmostEqual(modes[0], 0.58808155)
            self.assertAlmostEqual(modes[1], 0.54844907)
        else:
            self.assertAlmostEqual(modes[0], 0.55488481)
            self.assertAlmostEqual(modes[1], 1.18506639)

    def test_struct_1d(self):
        modes = self.rm_1d(self.x_grid)
        self.assertAlmostEqual(modes[0], 3.19799030)
        self.assertAlmostEqual(modes[1], 2.34788923)

    def test_struct_2d(self):
        modes = self.rm_2d(self.x_grid, self.y_grid)
        self.assertAlmostEqual(modes[0, 0], 1.67318010)
        self.assertAlmostEqual(modes[1, 0], 1.54740003)
        self.assertAlmostEqual(modes[0, 1], 2.02106551)
        self.assertAlmostEqual(modes[1, 1], 1.86883255)

    def test_struct_3d(self):
        modes = self.rm_3d(self.x_grid, self.y_grid, self.z_grid)
        if MC_VER < 3:
            self.assertAlmostEqual(modes[0, 0, 0], 0.58808155)
            self.assertAlmostEqual(modes[1, 0, 0], 0.91479114)
            self.assertAlmostEqual(modes[0, 1, 0], 0.61639899)
            self.assertAlmostEqual(modes[0, 0, 1], 0.83769551)
            self.assertAlmostEqual(modes[1, 1, 0], 0.81599044)
            self.assertAlmostEqual(modes[0, 1, 1], 0.95702504)
            self.assertAlmostEqual(modes[1, 0, 1], 0.49079625)
            self.assertAlmostEqual(modes[1, 1, 1], 0.51527539)
        else:
            self.assertAlmostEqual(modes[0, 0, 0], 0.55488481)
            self.assertAlmostEqual(modes[1, 0, 0], 1.17684277)
            self.assertAlmostEqual(modes[0, 1, 0], 0.41858766)
            self.assertAlmostEqual(modes[0, 0, 1], 0.69300397)
            self.assertAlmostEqual(modes[1, 1, 0], 0.95133855)
            self.assertAlmostEqual(modes[0, 1, 1], 0.65475042)
            self.assertAlmostEqual(modes[1, 0, 1], 1.45393842)
            self.assertAlmostEqual(modes[1, 1, 1], 1.40915120)

    def test_reset(self):
        modes = self.rm_2d(self.x_tuple, self.y_tuple)
        self.assertAlmostEqual(modes[0], 1.67318010)
        self.assertAlmostEqual(modes[1], 2.12310269)

        self.rm_2d.seed = self.rm_2d.seed
        modes = self.rm_2d(self.x_tuple, self.y_tuple)
        self.assertAlmostEqual(modes[0], 1.67318010)
        self.assertAlmostEqual(modes[1], 2.12310269)

        self.rm_2d.seed = 74893621
        modes = self.rm_2d(self.x_tuple, self.y_tuple)
        self.assertAlmostEqual(modes[0], -1.94278053)
        self.assertAlmostEqual(modes[1], -1.12401651)

        self.rm_1d.model = self.cov_model_3d
        modes = self.rm_1d(self.x_tuple, self.y_tuple, self.z_tuple)
        if MC_VER < 3:
            self.assertAlmostEqual(modes[0], 0.58808155)
            self.assertAlmostEqual(modes[1], 0.54844907)
        else:
            self.assertAlmostEqual(modes[0], 0.55488481)
            self.assertAlmostEqual(modes[1], 1.18506639)

        self.rm_2d.mode_no = 800
        modes = self.rm_2d(self.x_tuple, self.y_tuple)
        self.assertAlmostEqual(modes[0], -3.20809251)
        self.assertAlmostEqual(modes[1], -2.62032778)


if __name__ == "__main__":
    unittest.main()
