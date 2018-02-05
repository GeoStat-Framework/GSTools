#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This is a unittest of the variogram module.
"""
from __future__ import division, absolute_import, print_function

import unittest
import numpy as np
from gstools import variogram


class TestVariogramUnstructured(unittest.TestCase):
    def setUp(self):
        pass

    def test_doubles(self):
        x = np.arange(1, 11, 1, dtype=np.double)
        z = np.array((41.2, 40.2, 39.7, 39.2, 40.1,
                      38.3, 39.1, 40.0, 41.1, 40.3), dtype=np.double)
        bins = np.arange(1, 11, 1, dtype=np.double)
        gamma = variogram.estimate_unstructured(z, bins, x)
        self.assertAlmostEqual(gamma[0], .4917, places=4)

    def test_ints(self):
        x = np.arange(1, 5, 1, dtype=int)
        z = np.array((10, 20, 30, 40), dtype=int)
        bins = np.arange(1, 11, 1, dtype=int)
        gamma = variogram.estimate_unstructured(z, bins, x)
        self.assertAlmostEqual(gamma[0], 50., places=4)

    def test_longs(self):
        x = np.arange(1, 5, 1, dtype=long)
        z = np.array((10, 20, 30, 40), dtype=long)
        bins = np.arange(1, 11, 1, dtype=long)
        gamma = variogram.estimate_unstructured(z, bins, x)
        self.assertAlmostEqual(gamma[0], 50., places=4)

    def test_np_int(self):
        x = np.arange(1, 5, 1, dtype=np.int)
        z = np.array((10, 20, 30, 40), dtype=np.int)
        bins = np.arange(1, 11, 1, dtype=np.int)
        gamma = variogram.estimate_unstructured(z, bins, x)
        self.assertAlmostEqual(gamma[0], 50., places=4)

    def test_mixed(self):
        x = np.arange(1, 11, 1, dtype=np.double)
        z = np.array((41.2, 40.2, 39.7, 39.2, 40.1,
                      38.3, 39.1, 40.0, 41.1, 40.3), dtype=np.double)
        bins = np.arange(1, 11, 1, dtype=int)
        gamma = variogram.estimate_unstructured(z, bins, x)
        self.assertAlmostEqual(gamma[0], .4917, places=4)

        x = np.arange(1, 5, 1, dtype=np.double)
        z = np.array((10, 20, 30, 40), dtype=long)
        bins = np.arange(1, 11, 1, dtype=int)
        gamma = variogram.estimate_unstructured(z, bins, x)
        self.assertAlmostEqual(gamma[0], 50., places=4)

        x = np.arange(1, 5, 1, dtype=np.double)
        z = np.array((10, 20, 30, 40), dtype=long)
        bins = np.arange(1, 11, 1, dtype=np.double)
        gamma = variogram.estimate_unstructured(z, bins, x)
        self.assertAlmostEqual(gamma[0], 50., places=4)

    def test_1d(self):
        x = np.arange(1, 11, 1, dtype=np.double)
        #literature values
        z = np.array((41.2, 40.2, 39.7, 39.2, 40.1,
                      38.3, 39.1, 40.0, 41.1, 40.3), dtype=np.double)
        bins = np.arange(1, 11, 1, dtype=np.double)
        gamma = variogram.estimate_unstructured(z, bins, x)
        self.assertAlmostEqual(gamma[0], .4917, places=4)
        self.assertAlmostEqual(gamma[1], .7625, places=4)

    def test_uncorrelated_2d(self):
        x_c = np.linspace(0., 100., 60)
        y_c = np.linspace(0., 100., 60)
        x, y = np.meshgrid(x_c, y_c)
        x = np.reshape(x, len(x_c)*len(y_c))
        y = np.reshape(y, len(x_c)*len(y_c))

        rng = np.random.RandomState(1479373475)
        field = rng.rand(len(x))

        bins = np.arange(0, 100, 10)

        gamma = variogram.estimate_unstructured(field, bins, x, y)

        var = 1. / 12.
        self.assertAlmostEqual(gamma[0], var, places=2)
        self.assertAlmostEqual(gamma[len(gamma)//2], var, places=2)
        self.assertAlmostEqual(gamma[-1], var, places=2)

    def test_uncorrelated_3d(self):
        x_c = np.linspace(0., 100., 15)
        y_c = np.linspace(0., 100., 15)
        z_c = np.linspace(0., 100., 15)
        x, y, z = np.meshgrid(x_c, y_c, z_c)
        x = np.reshape(x, len(x_c)*len(y_c)*len(z_c))
        y = np.reshape(y, len(x_c)*len(y_c)*len(z_c))
        z = np.reshape(z, len(x_c)*len(y_c)*len(z_c))

        rng = np.random.RandomState(1479373475)
        field = rng.rand(len(x))

        bins = np.arange(0, 100, 10)

        gamma = variogram.estimate_unstructured(field, bins, x, y, z)

        var = 1. / 12.
        self.assertAlmostEqual(gamma[0], var, places=2)
        self.assertAlmostEqual(gamma[len(gamma)//2], var, places=2)
        self.assertAlmostEqual(gamma[-1], var, places=2)

    def test_assertions(self):
        x = np.arange(0, 10)
        x_e = np.arange(0, 11)
        y = np.arange(0, 11)
        y_e = np.arange(0, 12)
        z = np.arange(0, 12)
        z_e = np.arange(0, 15)
        bins = np.arange(0, 3)
        bins_e = np.arange(0, 1)
        field = np.arange(0, 10)
        field_e = np.arange(0, 9)

        self.assertRaises(ValueError, variogram.estimate_unstructured, field, bins, x_e)
        self.assertRaises(ValueError, variogram.estimate_unstructured, field, bins, x, y_e)
        self.assertRaises(ValueError, variogram.estimate_unstructured, field, bins, x, y_e, z)
        self.assertRaises(ValueError, variogram.estimate_unstructured, field, bins, x, y, z_e)
        self.assertRaises(ValueError, variogram.estimate_unstructured, field, bins, x_e, y, z)
        self.assertRaises(ValueError, variogram.estimate_unstructured, field_e, bins, x, y, z)
        self.assertRaises(ValueError, variogram.estimate_unstructured, field_e, bins, x)


if __name__ == '__main__':
    unittest.main()
