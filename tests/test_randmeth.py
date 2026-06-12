"""
This is the unittest of the RandMeth class.
"""

import copy
import unittest

import numpy as np

from gstools import Gaussian
from gstools.field.generator import RandMeth


class TestRandMeth(unittest.TestCase):
    def setUp(self):
        self.cov_model_1d = Gaussian(dim=1, var=1.5, len_scale=3.5)
        self.cov_model_2d = copy.deepcopy(self.cov_model_1d)
        self.cov_model_2d.dim = 2
        self.cov_model_3d = copy.deepcopy(self.cov_model_1d)
        self.cov_model_3d.dim = 3
        self.seed = 19031977
        self.x_grid = np.linspace(0.0, 10.0, 9)
        self.y_grid = np.linspace(-5.0, 5.0, 16)
        self.z_grid = np.linspace(-6.0, 7.0, 8)
        self.x_tuple = np.linspace(0.0, 10.0, 10)
        self.y_tuple = np.linspace(-5.0, 5.0, 10)
        self.z_tuple = np.linspace(-6.0, 8.0, 10)

        self.rm_1d = RandMeth(self.cov_model_1d, mode_no=100, seed=self.seed)
        self.rm_2d = RandMeth(self.cov_model_2d, mode_no=100, seed=self.seed)
        self.rm_3d = RandMeth(self.cov_model_3d, mode_no=100, seed=self.seed)

    def test_unstruct_1d(self):
        modes = self.rm_1d((self.x_tuple,))
        self.assertAlmostEqual(modes[0], 3.19799030)
        self.assertAlmostEqual(modes[1], 2.44848295)

    def test_unstruct_2d(self):
        modes = self.rm_2d((self.x_tuple, self.y_tuple))
        self.assertAlmostEqual(modes[0], 1.67318010)
        self.assertAlmostEqual(modes[1], 2.12310269)

    def test_unstruct_3d(self):
        modes = self.rm_3d((self.x_tuple, self.y_tuple, self.z_tuple))
        self.assertAlmostEqual(modes[0], 1.3240234883187239)
        self.assertAlmostEqual(modes[1], 1.6367244277732766)

    def test_reset(self):
        modes = self.rm_2d((self.x_tuple, self.y_tuple))
        self.assertAlmostEqual(modes[0], 1.67318010)
        self.assertAlmostEqual(modes[1], 2.12310269)

        self.rm_2d.seed = self.rm_2d.seed
        modes = self.rm_2d((self.x_tuple, self.y_tuple))
        self.assertAlmostEqual(modes[0], 1.67318010)
        self.assertAlmostEqual(modes[1], 2.12310269)

        self.rm_2d.seed = 74893621
        modes = self.rm_2d((self.x_tuple, self.y_tuple))
        self.assertAlmostEqual(modes[0], -1.94278053)
        self.assertAlmostEqual(modes[1], -1.12401651)

        self.rm_1d.model = self.cov_model_3d
        modes = self.rm_1d((self.x_tuple, self.y_tuple, self.z_tuple))
        self.assertAlmostEqual(modes[0], 1.3240234883187239)
        self.assertAlmostEqual(modes[1], 1.6367244277732766)

        self.rm_2d.mode_no = 800
        modes = self.rm_2d((self.x_tuple, self.y_tuple))
        self.assertAlmostEqual(modes[0], -3.20809251)
        self.assertAlmostEqual(modes[1], -2.62032778)


if __name__ == "__main__":
    unittest.main()
