#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This is a unittest of the variogram module.
"""
from __future__ import division, absolute_import, print_function

import unittest
import numpy as np
from gstools import variogram


class TestVariogramstructured(unittest.TestCase):
    def setUp(self):
        pass

    def test_doubles(self):
        x = np.arange(1, 11, 1, dtype=np.double)
        z = np.array((41.2, 40.2, 39.7, 39.2, 40.1,
                      38.3, 39.1, 40.0, 41.1, 40.3), dtype=np.double)
        gamma = variogram.estimate_structured(x, z)
        self.assertAlmostEqual(gamma[1], .4917, places=4)

    def test_ints(self):
        x = np.arange(1, 5, 1, dtype=int)
        z = np.array((10, 20, 30, 40), dtype=int)
        gamma = variogram.estimate_structured(x, z)
        self.assertAlmostEqual(gamma[1], 50., places=4)

    def test_longs(self):
        x = np.arange(1, 5, 1, dtype=long)
        z = np.array((10, 20, 30, 40), dtype=long)
        gamma = variogram.estimate_structured(x, z)
        self.assertAlmostEqual(gamma[1], 50., places=4)

    def test_np_int(self):
        x = np.arange(1, 5, 1, dtype=np.int)
        z = np.array((10, 20, 30, 40), dtype=np.int)
        gamma = variogram.estimate_structured(x, z)
        self.assertAlmostEqual(gamma[1], 50., places=4)

    def test_mixed(self):
        x = np.arange(1, 11, 1, dtype=np.double)
        z = np.array((41.2, 40.2, 39.7, 39.2, 40.1,
                      38.3, 39.1, 40.0, 41.1, 40.3), dtype=np.double)
        gamma = variogram.estimate_structured(x, z)
        self.assertAlmostEqual(gamma[1], .4917, places=4)

        x = np.arange(1, 5, 1, dtype=np.double)
        z = np.array((10, 20, 30, 40), dtype=long)
        gamma = variogram.estimate_structured(x, z)
        self.assertAlmostEqual(gamma[1], 50., places=4)

        x = np.arange(1, 5, 1, dtype=np.double)
        z = np.array((10, 20, 30, 40), dtype=long)
        gamma = variogram.estimate_structured(x, z)
        self.assertAlmostEqual(gamma[1], 50., places=4)

    def test_1d(self):
        x = np.arange(1, 11, 1, dtype=np.double)
        #literature values
        z = np.array((41.2, 40.2, 39.7, 39.2, 40.1,
                      38.3, 39.1, 40.0, 41.1, 40.3), dtype=np.double)
        gamma = variogram.estimate_structured(x, z)
        self.assertAlmostEqual(gamma[0], .0000, places=4)
        self.assertAlmostEqual(gamma[1], .4917, places=4)
        self.assertAlmostEqual(gamma[2], .7625, places=4)

    def test_masked_1d(self):
        x = np.arange(1, 11, 1, dtype=np.double)
        #literature values
        z = np.array((41.2, 40.2, 39.7, 39.2, 40.1,
                      38.3, 39.1, 40.0, 41.1, 40.3), dtype=np.double)
        z_ma = np.ma.masked_array(z, mask=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        gamma = variogram.estimate_structured(x, z_ma)
        self.assertAlmostEqual(gamma[0], .0000, places=4)
        self.assertAlmostEqual(gamma[1], .4917, places=4)
        self.assertAlmostEqual(gamma[2], .7625, places=4)
        z_ma = np.ma.masked_array(z, mask=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        gamma = variogram.estimate_structured(x, z_ma)
        self.assertAlmostEqual(gamma[0], .0000, places=4)
        self.assertAlmostEqual(gamma[1], .4906, places=4)
        self.assertAlmostEqual(gamma[2], .7107, places=4)

    def test_uncorrelated_2d(self):
        x = np.linspace(0., 100., 80)
        y = np.linspace(0., 100., 60)

        rng = np.random.RandomState(1479373475)
        field = rng.rand(len(x), len(y))

        gamma_x = variogram.estimate_structured((x, y), field, direction='x')
        gamma_y = variogram.estimate_structured((x, y), field, direction='y')

        var = 1. / 12.
        self.assertAlmostEqual(gamma_x[0], 0., places=2)
        self.assertAlmostEqual(gamma_x[len(gamma_x)//2], var, places=2)
        self.assertAlmostEqual(gamma_x[-1], var, places=2)
        self.assertAlmostEqual(gamma_y[0], 0., places=2)
        self.assertAlmostEqual(gamma_y[len(gamma_y)//2], var, places=2)
        self.assertAlmostEqual(gamma_y[-1], var, places=2)

    def test_uncorrelated_3d(self):
        x = np.linspace(0., 100., 30)
        y = np.linspace(0., 100., 30)
        z = np.linspace(0., 100., 30)

        rng = np.random.RandomState(1479373475)
        field = rng.rand(len(x), len(y), len(z))

        gamma = variogram.estimate_structured((x, y, z), field, 'x')
        gamma = variogram.estimate_structured((x, y, z), field, 'y')
        gamma = variogram.estimate_structured((x, y, z), field, 'z')

        var = 1. / 12.
        self.assertAlmostEqual(gamma[0], 0., places=2)
        self.assertAlmostEqual(gamma[len(gamma)//2], var, places=2)
        self.assertAlmostEqual(gamma[-1], var, places=2)

    def test_directions_2d(self):
        x = np.linspace(0., 20., 100)
        y = np.linspace(0., 15., 80)
        rng = np.random.RandomState(1479373475)
        x_rand = rng.rand(len(x))
        y_rand = rng.rand(len(y))
        #random values repeated along y-axis
        field_x = np.tile(x_rand, (len(y), 1)).T
        #random values repeated along x-axis
        field_y = np.tile(y_rand, (len(x), 1))

        gamma_x_x = variogram.estimate_structured((x, y), field_x,
                                                  direction='x')
        gamma_x_y = variogram.estimate_structured((x, y), field_x,
                                                  direction='y')

        gamma_y_x = variogram.estimate_structured((x, y), field_y,
                                                  direction='x')
        gamma_y_y = variogram.estimate_structured((x, y), field_y,
                                                  direction='y')

        self.assertAlmostEqual(gamma_x_y[1], 0.)
        self.assertAlmostEqual(gamma_x_y[len(gamma_x_y)//2], 0.)
        self.assertAlmostEqual(gamma_x_y[-1], 0.)
        self.assertAlmostEqual(gamma_y_x[1], 0.)
        self.assertAlmostEqual(gamma_y_x[len(gamma_x_y)//2], 0.)
        self.assertAlmostEqual(gamma_y_x[-1], 0.)

    def test_directions_3d(self):
        x = np.linspace(0., 10., 20)
        y = np.linspace(0., 15., 25)
        z = np.linspace(0., 20., 30)
        rng = np.random.RandomState(1479373475)
        x_rand = rng.rand(len(x))
        y_rand = rng.rand(len(y))
        z_rand = rng.rand(len(z))

        field_x = np.tile(x_rand.reshape((len(x), 1, 1)), (1, len(y), len(z)))
        field_y = np.tile(y_rand.reshape((1, len(y), 1)), (len(x), 1, len(z)))
        field_z = np.tile(z_rand.reshape((1, 1, len(z))), (len(x), len(y), 1))

        gamma_x_x = variogram.estimate_structured((x, y, z), field_x,
                                                  direction='x')
        gamma_x_y = variogram.estimate_structured((x, y, z), field_x,
                                                  direction='y')
        gamma_x_z = variogram.estimate_structured((x, y, z), field_x,
                                                  direction='z')

        gamma_y_x = variogram.estimate_structured((x, y, z), field_y,
                                                  direction='x')
        gamma_y_y = variogram.estimate_structured((x, y, z), field_y,
                                                  direction='y')
        gamma_y_z = variogram.estimate_structured((x, y, z), field_y,
                                                  direction='z')

        gamma_z_x = variogram.estimate_structured((x, y, z), field_z,
                                                  direction='x')
        gamma_z_y = variogram.estimate_structured((x, y, z), field_z,
                                                  direction='y')
        gamma_z_z = variogram.estimate_structured((x, y, z), field_z,
                                                  direction='z')
        self.assertAlmostEqual(gamma_x_y[1], 0.)
        self.assertAlmostEqual(gamma_x_y[len(gamma_x_y)//2], 0.)
        self.assertAlmostEqual(gamma_x_y[-1], 0.)
        self.assertAlmostEqual(gamma_x_z[1], 0.)
        self.assertAlmostEqual(gamma_x_z[len(gamma_x_y)//2], 0.)
        self.assertAlmostEqual(gamma_x_z[-1], 0.)
        self.assertAlmostEqual(gamma_y_x[1], 0.)
        self.assertAlmostEqual(gamma_y_x[len(gamma_x_y)//2], 0.)
        self.assertAlmostEqual(gamma_y_x[-1], 0.)
        self.assertAlmostEqual(gamma_y_z[1], 0.)
        self.assertAlmostEqual(gamma_y_z[len(gamma_x_y)//2], 0.)
        self.assertAlmostEqual(gamma_y_z[-1], 0.)
        self.assertAlmostEqual(gamma_z_x[1], 0.)
        self.assertAlmostEqual(gamma_z_x[len(gamma_x_y)//2], 0.)
        self.assertAlmostEqual(gamma_z_x[-1], 0.)
        self.assertAlmostEqual(gamma_z_y[1], 0.)
        self.assertAlmostEqual(gamma_z_y[len(gamma_x_y)//2], 0.)
        self.assertAlmostEqual(gamma_z_y[-1], 0.)


if __name__ == '__main__':
    unittest.main()
