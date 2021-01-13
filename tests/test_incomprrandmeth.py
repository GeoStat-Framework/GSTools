# -*- coding: utf-8 -*-
"""
This is the unittest of the RandMeth class.
"""

import copy
import unittest
import numpy as np
from gstools import Gaussian
from gstools.field.generator import IncomprRandMeth


class TestIncomprRandMeth(unittest.TestCase):
    def setUp(self):
        self.cov_model_2d = Gaussian(dim=2, var=1.5, len_scale=2.5)
        self.cov_model_3d = copy.deepcopy(self.cov_model_2d)
        self.cov_model_3d.dim = 3
        self.seed = 19031977
        self.x_grid = np.linspace(0.0, 10.0, 9)
        self.y_grid = np.linspace(-5.0, 5.0, 16)
        self.z_grid = np.linspace(-6.0, 7.0, 8)
        self.x_tuple = np.linspace(0.0, 10.0, 10)
        self.y_tuple = np.linspace(-5.0, 5.0, 10)
        self.z_tuple = np.linspace(-6.0, 8.0, 10)

        self.rm_2d = IncomprRandMeth(
            self.cov_model_2d, mode_no=100, seed=self.seed
        )
        self.rm_3d = IncomprRandMeth(
            self.cov_model_3d, mode_no=100, seed=self.seed
        )

    def test_unstruct_2d(self):
        modes = self.rm_2d((self.x_tuple, self.y_tuple))
        self.assertAlmostEqual(modes[0, 0], 0.50751115)
        self.assertAlmostEqual(modes[0, 1], 1.03291018)
        self.assertAlmostEqual(modes[1, 1], -0.22003005)

    def test_unstruct_3d(self):
        modes = self.rm_3d((self.x_tuple, self.y_tuple, self.z_tuple))
        self.assertAlmostEqual(modes[0, 0], 0.7924546333550331)
        self.assertAlmostEqual(modes[0, 1], 1.660747056686244)
        self.assertAlmostEqual(modes[1, 0], -0.28049855754819514)

    def test_assertions(self):
        cov_model_1d = Gaussian(dim=1, var=1.5, len_scale=2.5)
        self.assertRaises(ValueError, IncomprRandMeth, cov_model_1d)


if __name__ == "__main__":
    unittest.main()
