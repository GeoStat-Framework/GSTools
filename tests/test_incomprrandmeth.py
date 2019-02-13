# -*- coding: utf-8 -*-
"""
This is the unittest of the RandMeth class.
"""
from __future__ import division, absolute_import, print_function

import copy
import unittest
import numpy as np
from gstools import Gaussian
from gstools.field.generator import IncomprRandMeth
import emcee as mc


MC_VER = int(mc.__version__.split('.')[0])


class TestIncomprRandMeth(unittest.TestCase):
    def setUp(self):
        self.cov_model_2d = Gaussian(
            dim=2, var=1.5, len_scale=2.5, mode_no=100
        )
        self.cov_model_3d = copy.deepcopy(self.cov_model_2d)
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

        self.rm_2d = IncomprRandMeth(self.cov_model_2d, mode_no=100, seed=self.seed)
        self.rm_3d = IncomprRandMeth(self.cov_model_3d, mode_no=100, seed=self.seed)

    def test_unstruct_2d(self):
        modes = self.rm_2d(self.x_tuple, self.y_tuple)
        self.assertAlmostEqual(modes[0, 0], 1.84891292)
        self.assertAlmostEqual(modes[0, 1], 2.21703433)
        self.assertAlmostEqual(modes[1, 1], -0.10643386)

    def test_unstruct_3d(self):
        modes = self.rm_3d(self.x_tuple, self.y_tuple, self.z_tuple)
        if MC_VER < 3:
            self.assertAlmostEqual(modes[0, 1], 2.09358400)
            self.assertAlmostEqual(modes[1, 0], -0.33468344)
            self.assertAlmostEqual(modes[1, 1], -0.56828020)
        else:
            self.assertAlmostEqual(modes[0, 0], 2.31331290)
            self.assertAlmostEqual(modes[0, 1], 2.35153925)
            self.assertAlmostEqual(modes[1, 0], 0.16866230)

    def test_struct_2d(self):
        modes = self.rm_2d(self.x_grid, self.y_grid)
        self.assertAlmostEqual(modes[0, 0, 0], 1.84891292)
        self.assertAlmostEqual(modes[0, 1, 0], 1.44383168)
        self.assertAlmostEqual(modes[1, 1, 1], -0.36931603)

    def test_struct_3d(self):
        modes = self.rm_3d(self.x_grid, self.y_grid, self.z_grid)
        if MC_VER < 3:
            self.assertAlmostEqual(modes[0, 1, 0, 0], 2.31453214)
            self.assertAlmostEqual(modes[0, 0, 1, 0], 1.88215044)
            self.assertAlmostEqual(modes[0, 0, 0, 1], 1.58948541)
            self.assertAlmostEqual(modes[1, 1, 1, 0], -0.58578527)
        else:
            self.assertAlmostEqual(modes[0, 0, 0, 0], 2.31331290)
            self.assertAlmostEqual(modes[1, 0, 1, 1], -0.21625276)
            self.assertAlmostEqual(modes[1, 1, 0, 1], 0.04596176)
            self.assertAlmostEqual(modes[1, 1, 1, 1], 0.33077794)

    def test_assertions(self):
        cov_model_1d = Gaussian(
            dim=1, var=1.5, len_scale=2.5, mode_no=100
        )
        self.assertRaises(ValueError, IncomprRandMeth, cov_model_1d)


if __name__ == "__main__":
    unittest.main()
