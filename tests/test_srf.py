#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This is the unittest of SRF class.
"""
from __future__ import division, absolute_import, print_function

import unittest
import numpy as np
from gstools.field import SRF


class TestSRF(unittest.TestCase):
    def setUp(self):
        self.cov_model = {
                     'dim': 2,
                     'mean': .3,
                     'var': 1.5,
                     'len_scale': 4.,
                     'model': 'gau',
                     'mode_no': 100,
                     }

        self.seed = 825718662
        self.x_grid = np.linspace(0., 12., 48)
        self.y_grid = np.linspace(0., 10., 46)
        self.z_grid = np.linspace(0., 10., 40)

        self.x_grid_c = np.linspace(-6., 6., 8)
        self.y_grid_c = np.linspace(-6., 6., 8)
        self.z_grid_c = np.linspace(-6., 6., 8)

        rng = np.random.RandomState(123018)
        self.x_tuple = rng.uniform(0., 10, 100)
        self.y_tuple = rng.uniform(0., 10, 100)
        self.z_tuple = rng.uniform(0., 10, 100)

    def test_shape_1d(self):
        self.cov_model['dim'] = 1
        srf = SRF(**self.cov_model)
        field_str = srf(self.x_grid, seed=self.seed, mesh_type='structured')
        field_unstr = srf(self.x_tuple, seed=self.seed, mesh_type='unstructured')
        self.assertEqual(field_str.shape, (len(self.x_grid), ))
        self.assertEqual(field_unstr.shape, (len(self.x_tuple), ))

    def test_shape_2d(self):
        self.cov_model['dim'] = 2
        srf = SRF(**self.cov_model)
        field_str = srf(self.x_grid, self.y_grid, seed=self.seed,
                        mesh_type='structured')
        field_unstr = srf(self.x_tuple, self.y_tuple, seed=self.seed,
                          mesh_type='unstructured')
        self.assertEqual(field_str.shape, (len(self.x_grid), len(self.y_grid)))
        self.assertEqual(field_unstr.shape, (len(self.x_tuple), ))

    def test_shape_3d(self):
        self.cov_model['dim'] = 3
        srf = SRF(**self.cov_model)
        field_str = srf(self.x_grid, self.y_grid, self.z_grid, seed=self.seed,
                        mesh_type='structured')
        field_unstr = srf(self.x_tuple, self.y_tuple, self.z_tuple,
                          seed=987654, mesh_type='unstructured')
        self.assertEqual(field_str.shape, (len(self.x_grid), len(self.y_grid),
                                       len(self.z_grid)))
        self.assertEqual(field_unstr.shape, (len(self.x_tuple), ))

    def test_anisotropy_2d(self):
        srf = SRF(**self.cov_model)
        field_iso = srf(self.x_grid, self.y_grid, seed=self.seed, mesh_type='structured')
        self.cov_model['anis'] = 2.
        srf = SRF(**self.cov_model)
        field_aniso = srf(self.x_grid, self.y_grid, seed=self.seed, mesh_type='structured')
        self.assertAlmostEqual(field_iso[0,0], field_aniso[0,0])
        self.assertAlmostEqual(field_iso[0,4], field_aniso[0,2])
        self.assertAlmostEqual(field_iso[0,10], field_aniso[0,5])

    def test_anisotropy_3d(self):
        self.cov_model['dim'] = 3
        srf = SRF(**self.cov_model)
        field_iso = srf(self.x_grid, self.y_grid, self.z_grid, seed=self.seed,
                        mesh_type='structured')
        self.cov_model['anis'] = (2., .25)
        srf = SRF(**self.cov_model)
        field_aniso = srf(self.x_grid, self.y_grid, self.z_grid,
                          seed=self.seed, mesh_type='structured')
        self.assertAlmostEqual(field_iso[0,0,0], field_aniso[0,0,0])
        self.assertAlmostEqual(field_iso[0,4,0], field_aniso[0,2,0])
        self.assertAlmostEqual(field_iso[0,10,0], field_aniso[0,5,0])
        self.assertAlmostEqual(field_iso[0,0,0], field_aniso[0,0,0])
        self.assertAlmostEqual(field_iso[0,0,1], field_aniso[0,0,4])
        self.assertAlmostEqual(field_iso[0,0,3], field_aniso[0,0,12])

    def test_rotation_2d(self):
        x_len = len(self.x_grid_c)
        y_len = len(self.y_grid_c)
        xu, yu = np.meshgrid(self.x_grid_c, self.y_grid_c)
        xu = np.reshape(xu, x_len*y_len)
        yu = np.reshape(yu, x_len*y_len)

        self.cov_model['anis'] = 4.
        srf = SRF(**self.cov_model)

        field = srf(xu, yu, seed=self.seed, mesh_type='unstructured')
        field_str = np.reshape(field, (y_len, x_len))

        self.cov_model['angles'] = np.pi/2.
        srf = SRF(**self.cov_model)
        field_rot = srf(xu, yu, seed=self.seed, mesh_type='unstructured')
        field_rot_str = np.reshape(field_rot, (y_len, x_len))

        self.assertAlmostEqual(field_str[0,0], field_rot_str[-1,0])
        self.assertAlmostEqual(field_str[1,2], field_rot_str[-3,1])

    def test_rotation_3d(self):
        x_len = len(self.x_grid_c)
        y_len = len(self.y_grid_c)
        z_len = len(self.z_grid_c)
        xu, yu, zu = np.meshgrid(self.x_grid_c, self.y_grid_c, self.z_grid_c)
        xu = np.reshape(xu, x_len*y_len*z_len)
        yu = np.reshape(yu, x_len*y_len*z_len)
        zu = np.reshape(zu, x_len*y_len*z_len)

        self.cov_model['dim'] = 3
        self.cov_model['anis'] = (4., 2.)
        srf = SRF(**self.cov_model)
        field = srf(xu, yu, zu, seed=self.seed, mesh_type='unstructured')
        field_str = np.reshape(field, (y_len, x_len, z_len))

        self.cov_model['angles'] = (np.pi/2., np.pi/2.)
        srf = SRF(**self.cov_model)
        field_rot = srf(xu, yu, zu, seed=self.seed, mesh_type='unstructured')
        field_rot_str = np.reshape(field_rot, (y_len, x_len, z_len))

        self.assertAlmostEqual(field_str[ 0, 0, 0], field_rot_str[-1,-1, 0])
        self.assertAlmostEqual(field_str[ 1, 2, 0], field_rot_str[-3,-1, 1])
        self.assertAlmostEqual(field_str[ 0, 0, 1], field_rot_str[-1,-2, 0])

    def test_calls(self):
        srf = SRF(**self.cov_model)
        field = srf(self.x_tuple, self.y_tuple, seed=self.seed)
        srf.generate(self.x_tuple, self.y_tuple, seed=self.seed)
        field2 = srf.unstructured(self.x_tuple, self.y_tuple, seed=self.seed)
        self.assertAlmostEqual(field[0], srf.field[0])
        self.assertAlmostEqual(field[0], field2[0])
        field = srf(self.x_tuple, self.y_tuple, seed=self.seed,
                    mesh_type='structured')
        srf.generate(self.x_tuple, self.y_tuple, seed=self.seed,
                     mesh_type='structured')
        field2 = srf.structured(self.x_tuple, self.y_tuple, seed=self.seed)
        self.assertAlmostEqual(field[0,0], srf.field[0,0])
        self.assertAlmostEqual(field[0,0], field2[0,0])

    def test_assertions(self):
        self.cov_model['dim'] = 0
        self.assertRaises(ValueError, SRF, **self.cov_model)
        self.cov_model['dim'] = 4
        self.assertRaises(ValueError, SRF, **self.cov_model)
        self.cov_model['dim'] = 2
        srf = SRF(**self.cov_model)
        self.assertRaises(ValueError, srf, self.x_tuple)
        self.assertRaises(ValueError, srf, self.x_grid, self.y_grid)
        self.cov_model['dim'] = 3
        srf = SRF(**self.cov_model)
        self.assertRaises(ValueError, srf, self.x_tuple, self.y_tuple)
        self.assertRaises(ValueError, srf, self.x_grid, self.y_grid, self.z_grid)
        self.assertRaises(ValueError, srf, self.x_tuple, self.y_tuple,
                          self.z_tuple, self.seed, 'hyper_mesh')


if __name__ == '__main__':
    unittest.main()
