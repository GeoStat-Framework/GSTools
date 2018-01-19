
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This is the unittest of the RandMeth class.
"""
from __future__ import division, absolute_import, print_function

import unittest
import numpy as np
from gstools.field import RandMeth


class TestRandMeth(unittest.TestCase):
    def setUp(self):
        self.cov_model = 'gau'
        self.len_scale = 3.5
        self.seed = 19031977
        self.x_grid = np.linspace(0., 10., 9)
        self.y_grid = np.linspace(-5., 5., 16)
        self.z_grid = np.linspace(-6., 7., 8)
        self.x_grid = np.reshape(self.x_grid, (len(self.x_grid), 1, 1, 1))
        self.y_grid = np.reshape(self.y_grid, (1, len(self.y_grid), 1, 1))
        self.z_grid = np.reshape(self.z_grid, (1, 1, len(self.z_grid), 1))
        self.x_tuple = np.linspace(0., 10., 10)
        self.y_tuple = np.linspace(-5., 5., 10)
        self.z_tuple = np.linspace(-6., 8., 10)
        self.x_tuple = np.reshape(self.x_tuple, (len(self.x_tuple), 1))
        self.y_tuple = np.reshape(self.y_tuple, (len(self.y_tuple), 1))
        self.z_tuple = np.reshape(self.z_tuple, (len(self.z_tuple), 1))

        self.rm_1d = RandMeth(1, self.cov_model, self.len_scale, 100, self.seed)
        self.rm_2d = RandMeth(2, self.cov_model, self.len_scale, 100, self.seed)
        self.rm_3d = RandMeth(3, self.cov_model, self.len_scale, 100, self.seed)

    def test_unstruct_1d(self):
        modes = self.rm_1d(self.x_tuple)
        self.assertAlmostEqual(modes[0], 2.61114815)
        self.assertAlmostEqual(modes[1],-1.09762656)

    def test_unstruct_2d(self):
        modes = self.rm_2d(self.x_tuple, self.y_tuple)
        self.assertAlmostEqual(modes[0], 0.66896403)
        self.assertAlmostEqual(modes[1], 1.40165745)

    def test_unstruct_3d(self):
        modes = self.rm_3d(self.x_tuple, self.y_tuple, self.z_tuple)
        self.assertAlmostEqual(modes[0], 0.28444828)
        self.assertAlmostEqual(modes[1], 0.52285547)

    def test_struct_1d(self):
        modes = self.rm_1d(self.x_grid)
        self.assertAlmostEqual(modes[0], 2.61114815)
        self.assertAlmostEqual(modes[1],-0.98901582)

    def test_struct_2d(self):
        modes = self.rm_2d(self.x_grid, self.y_grid)
        self.assertAlmostEqual(modes[0,0], 0.66896403)
        self.assertAlmostEqual(modes[1,0],-0.66976332)
        self.assertAlmostEqual(modes[0,1],-2.08766420)
        self.assertAlmostEqual(modes[1,1], 0.71573459)

    def test_struct_3d(self):
        modes = self.rm_3d(self.x_grid, self.y_grid, self.z_grid)
        self.assertAlmostEqual(modes[0,0,0], 0.28444828)
        self.assertAlmostEqual(modes[1,0,0], 0.62812678)
        self.assertAlmostEqual(modes[0,1,0], 2.90484916)
        self.assertAlmostEqual(modes[0,0,1], 0.06608837)
        self.assertAlmostEqual(modes[1,1,0],-0.49580849)
        self.assertAlmostEqual(modes[0,1,1], 0.81327753)
        self.assertAlmostEqual(modes[1,0,1], 1.53838263)
        self.assertAlmostEqual(modes[1,1,1], 0.64161150)

    def test_reset(self):
        modes = self.rm_2d(self.x_tuple, self.y_tuple)
        self.assertAlmostEqual(modes[0], 0.66896403)
        self.assertAlmostEqual(modes[1], 1.40165745)

        self.rm_2d.reset(2, self.cov_model, self.len_scale, 100, self.seed)
        modes = self.rm_2d(self.x_tuple, self.y_tuple)
        self.assertAlmostEqual(modes[0], 0.66896403)
        self.assertAlmostEqual(modes[1], 1.40165745)

        self.rm_2d.reset(2, self.cov_model, self.len_scale, 100, 74893621)
        modes = self.rm_2d(self.x_tuple, self.y_tuple)
        self.assertAlmostEqual(modes[0], 0.93183497)
        self.assertAlmostEqual(modes[1],-1.46225851)

    def test_scalar(self):
        mode = self.rm_1d(10.)
        mode_ref = 0.98212715
        self.assertAlmostEqual(mode, mode_ref)
        self.rm_1d.reset(2, self.cov_model, self.len_scale, 100, self.seed)
        mode = self.rm_1d(10., 0.)
        self.assertAlmostEqual(mode, mode_ref)
        self.rm_1d.reset(3, self.cov_model, self.len_scale, 100, self.seed)
        mode = self.rm_1d(10., 0., 0.)
        self.assertAlmostEqual(mode, mode_ref)


if __name__ == '__main__':
    unittest.main()
