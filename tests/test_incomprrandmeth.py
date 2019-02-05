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
        self.x_grid = np.linspace(0.0, 10.0, 90)
        self.y_grid = np.linspace(-5.0, 5.0, 160)
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

        self.rm_2d = IncomprRandMeth(self.cov_model_2d, 100, self.seed)
        self.rm_3d = IncomprRandMeth(self.cov_model_3d, 100, self.seed)

    def test_unstruct_2d(self):
        modes = self.rm_2d(self.x_tuple, self.y_tuple)
        self.assertAlmostEqual(modes[0, 0], 0.84891292)
        self.assertAlmostEqual(modes[0, 1], 1.21703433)
        self.assertAlmostEqual(modes[1, 0], -0.32636970)
        self.assertAlmostEqual(modes[1, 1], -0.10643386)

    def test_unstruct_3d(self):
        modes = self.rm_3d(self.x_tuple, self.y_tuple, self.z_tuple)
        if MC_VER < 3:
            self.assertAlmostEqual(modes[0, 0], 0.88667725)
            self.assertAlmostEqual(modes[0, 1], 1.09358400)
            self.assertAlmostEqual(modes[1, 0], -0.33468344)
            self.assertAlmostEqual(modes[1, 1], -0.56828020)
        else:
            self.assertAlmostEqual(modes[0, 0], 1.31331290)
            self.assertAlmostEqual(modes[0, 1], 1.35153925)
            self.assertAlmostEqual(modes[1, 0], 0.16866230)
            self.assertAlmostEqual(modes[1, 1], 0.35681837)

    def test_struct_2d(self):
        modes = self.rm_2d(self.x_grid, self.y_grid)
        self.assertAlmostEqual(modes[0, 0, 0], 0.84891292)
        self.assertAlmostEqual(modes[0, 1, 0], 0.86320815)
        self.assertAlmostEqual(modes[0, 0, 1], 0.87560642)
        self.assertAlmostEqual(modes[0, 1, 1], 0.89107176)
        self.assertAlmostEqual(modes[1, 0, 0], -0.32636970)
        self.assertAlmostEqual(modes[1, 1, 0], -0.37348097)
        self.assertAlmostEqual(modes[1, 0, 1], -0.29447634)
        self.assertAlmostEqual(modes[1, 1, 1], -0.34103203)

    def test_struct_3d(self):
        modes = self.rm_3d(self.x_grid, self.y_grid, self.z_grid)
        if MC_VER < 3:
            self.assertAlmostEqual(modes[0, 0, 0, 0], 0.84742804)
            self.assertAlmostEqual(modes[0, 1, 0, 0], 0.89935101)
            self.assertAlmostEqual(modes[0, 0, 1, 0], 0.84378030)
            self.assertAlmostEqual(modes[0, 0, 0, 1], 0.56595163)
            self.assertAlmostEqual(modes[1, 1, 1, 0], -0.38718441)
            self.assertAlmostEqual(modes[1, 0, 1, 1], -0.67655891)
            self.assertAlmostEqual(modes[1, 1, 0, 1], -0.75725040)
            self.assertAlmostEqual(modes[1, 1, 1, 1], -0.74871210)
        else:
            self.assertAlmostEqual(modes[0, 0, 0, 0], 1.26963134)
            self.assertAlmostEqual(modes[0, 1, 0, 0], 1.32551211)
            self.assertAlmostEqual(modes[0, 0, 1, 0], 1.29541995)
            self.assertAlmostEqual(modes[0, 0, 0, 1], 0.67662399)
            self.assertAlmostEqual(modes[1, 1, 1, 0], 0.19680564)
            self.assertAlmostEqual(modes[1, 0, 1, 1], -0.31931331)
            self.assertAlmostEqual(modes[1, 1, 0, 1], -0.30627922)
            self.assertAlmostEqual(modes[1, 1, 1, 1], -0.28800715)

if __name__ == "__main__":
    unittest.main()
