"""
This is the unittest of the Fourier class.
"""

import copy
import unittest

import numpy as np

import gstools as gs
from gstools.field.generator import Fourier


class TestFourier(unittest.TestCase):
    def setUp(self):
        self.seed = 19900408
        self.cov_model_1d = gs.Gaussian(dim=1, var=0.5, len_scale=10.)
        self.cov_model_2d = gs.Gaussian(dim=2, var=2.0, len_scale=30.)
        self.cov_model_3d = gs.Gaussian(dim=3, var=2.1, len_scale=21.)
        self.x = np.linspace(0, 80, 11)
        self.y = np.linspace(0, 30, 31)
        self.z = np.linspace(0, 91, 13)

        self.modes_no_1d = 20
        self.trunc_1d = 8
        self.modes_no_2d = [16, 7]
        self.trunc_2d = [16, 7]
        self.modes_no_3d = [16, 7, 11]
        self.trunc_3d = [16, 7, 12]

        self.srf_1d = gs.SRF(
            self.cov_model_1d,
            generator="Fourier",
            modes_no=self.modes_no_1d,
            modes_truncation=self.trunc_1d,
            seed=self.seed,
        )
        self.srf_2d = gs.SRF(
            self.cov_model_2d,
            generator="Fourier",
            modes_no=self.modes_no_2d,
            modes_truncation=self.trunc_2d,
            seed=self.seed,
        )
        self.srf_3d = gs.SRF(
            self.cov_model_3d,
            generator="Fourier",
            modes_no=self.modes_no_3d,
            modes_truncation=self.trunc_3d,
            seed=self.seed,
        )

    def test_1d(self):
        field = self.srf_1d((self.x,), mesh_type="structured")
        self.assertAlmostEqual(field[0], 0.9009981010688789)

    def test_2d(self):
        field = self.srf_2d((self.x, self.y), mesh_type="structured")
        self.assertAlmostEqual(field[0, 0], 1.1085370190533947)

    def test_3d(self):
        field = self.srf_3d((self.x, self.y, self.z), mesh_type="structured")
        self.assertAlmostEqual(field[0, 0, 0], 1.7648407965681794)

    def test_periodicity(self):
        field = self.srf_2d((self.x, self.y), mesh_type="structured")
        self.assertAlmostEqual(field[0, len(self.y)//2], field[-1, len(self.y)//2])

    def test_assertions(self):
        # unstructured grids not supported
        self.assertRaises(ValueError, self.srf_2d, (self.x, self.y))
        self.assertRaises(
            ValueError, self.srf_2d, (self.x, self.y), mesh_type="unstructured"
        )
