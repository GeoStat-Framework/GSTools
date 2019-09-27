# -*- coding: utf-8 -*-
"""
This is the unittest of CovModel class.
"""
from __future__ import division, absolute_import, print_function

import numpy as np
import unittest
from gstools import Gaussian, Exponential, Spherical, krige


class TestKrige(unittest.TestCase):
    def setUp(self):
        self.cov_models = [Gaussian, Exponential, Spherical]
        self.dims = range(1, 4)
        self.data = np.array(
            [
                [0.3, 1.2, 0.5, 0.47],
                [1.9, 0.6, 1.0, 0.56],
                [1.1, 3.2, 1.5, 0.74],
                [3.3, 4.4, 2.0, 1.47],
                [4.7, 3.8, 2.5, 1.74],
            ]
        )
        self.cond_pos = (self.data[:, 0], self.data[:, 1], self.data[:, 2])
        self.cond_val = self.data[:, 3]
        self.mean = np.mean(self.cond_val)
        self.grid_x = np.concatenate((self.cond_pos[0], np.linspace(5, 20)))
        self.grid_y = np.concatenate((self.cond_pos[1], np.linspace(5, 20)))
        self.grid_z = np.concatenate((self.cond_pos[2], np.linspace(5, 20)))
        self.pos = (self.grid_x, self.grid_y, self.grid_z)

    def test_simple(self):
        for Model in self.cov_models:
            model = Model(
                dim=1, var=0.5, len_scale=2, anis=[0.1, 1], angles=[0.5, 0, 0]
            )
            simple = krige.Simple(
                model, self.mean, self.cond_pos[0], self.cond_val
            )
            field_1, __ = simple.unstructured(self.pos[0])
            field_2, __ = simple.structured(self.pos[0])
            for i, val in enumerate(self.cond_val):
                self.assertAlmostEqual(val, field_1[i], places=2)
                self.assertAlmostEqual(val, field_2[(i,)], places=2)
            self.assertAlmostEqual(self.mean, field_1[-1], places=2)
            self.assertAlmostEqual(self.mean, field_2[(-1,)], places=2)

            for dim in self.dims[1:]:
                model = Model(
                    dim=dim,
                    var=0.5,
                    len_scale=2,
                    anis=[0.1, 1],
                    angles=[0.5, 0, 0],
                )
                simple = krige.Simple(
                    model, self.mean, self.cond_pos[:dim], self.cond_val
                )
                field_1, __ = simple.unstructured(self.pos[:dim])
                field_2, __ = simple.structured(self.pos[:dim])
                for i, val in enumerate(self.cond_val):
                    self.assertAlmostEqual(val, field_1[i], places=2)
                    self.assertAlmostEqual(val, field_2[dim * (i,)], places=2)
                self.assertAlmostEqual(self.mean, field_1[-1], places=2)
                self.assertAlmostEqual(
                    self.mean, field_2[dim * (-1,)], places=2
                )

    def test_ordinary(self):
        for Model in self.cov_models:
            model = Model(
                dim=1, var=0.5, len_scale=2, anis=[0.1, 1], angles=[0.5, 0, 0]
            )
            ordinary = krige.Ordinary(model, self.cond_pos[0], self.cond_val)
            field_1, __ = ordinary.unstructured(self.pos[0])
            field_2, __ = ordinary.structured(self.pos[0])
            for i, val in enumerate(self.cond_val):
                self.assertAlmostEqual(val, field_1[i], places=2)
                self.assertAlmostEqual(val, field_2[(i,)], places=2)

            for dim in self.dims[1:]:
                model = Model(
                    dim=dim,
                    var=0.5,
                    len_scale=2,
                    anis=[0.1, 1],
                    angles=[0.5, 0, 0],
                )
                ordinary = krige.Ordinary(
                    model, self.cond_pos[:dim], self.cond_val
                )
                field_1, __ = ordinary.unstructured(self.pos[:dim])
                field_2, __ = ordinary.structured(self.pos[:dim])
                for i, val in enumerate(self.cond_val):
                    self.assertAlmostEqual(val, field_1[i], places=2)
                    self.assertAlmostEqual(val, field_2[dim * (i,)], places=2)


if __name__ == "__main__":
    unittest.main()
