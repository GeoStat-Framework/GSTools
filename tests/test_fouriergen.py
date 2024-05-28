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
        self.cov_model_1d = gs.Gaussian(dim=1, var=0.5, len_scale=10.0)
        self.cov_model_2d = gs.Gaussian(dim=2, var=2.0, len_scale=30.0)
        self.cov_model_3d = gs.Gaussian(dim=3, var=2.1, len_scale=21.0)
        L = [80, 30, 91]
        self.x = np.linspace(0, L[0], 11)
        self.y = np.linspace(0, L[1], 31)
        self.z = np.linspace(0, L[2], 13)

        cutoff_rel = 0.999
        cutoff_abs = 1
        dk = [2 * np.pi / l for l in L]

        self.modes_1d = [np.arange(0, cutoff_abs, dk[0])]
        self.modes_2d = self.modes_1d + [np.arange(0, cutoff_abs, dk[1])]
        self.modes_3d = self.modes_2d + [np.arange(0, cutoff_abs, dk[2])]

        self.srf_1d = gs.SRF(
            self.cov_model_1d,
            generator="Fourier",
            modes=self.modes_1d,
            seed=self.seed,
        )
        self.srf_2d = gs.SRF(
            self.cov_model_2d,
            generator="Fourier",
            modes=self.modes_2d,
            seed=self.seed,
        )
        self.srf_3d = gs.SRF(
            self.cov_model_3d,
            generator="Fourier",
            mode_rel_cutoff=cutoff_rel,
            period=L,
            seed=self.seed,
        )

    def test_1d(self):
        field = self.srf_1d((self.x,), mesh_type="structured")
        self.assertAlmostEqual(field[0], 0.40939877176695477)

    def test_2d(self):
        field = self.srf_2d((self.x, self.y), mesh_type="structured")
        self.assertAlmostEqual(field[0, 0], 1.6338790313270515)

    def test_3d(self):
        field = self.srf_3d((self.x, self.y, self.z), mesh_type="structured")
        self.assertAlmostEqual(field[0, 0, 0], 0.2613561098408796)

    def test_periodicity(self):
        field = self.srf_2d((self.x, self.y), mesh_type="structured")
        self.assertAlmostEqual(
            field[0, len(self.y) // 2], field[-1, len(self.y) // 2]
        )

    def test_assertions(self):
        # unstructured grids not supported
        self.assertRaises(ValueError, self.srf_2d, (self.x, self.y))
        self.assertRaises(
            ValueError, self.srf_2d, (self.x, self.y), mesh_type="unstructured"
        )
        with self.assertRaises(ValueError):
            gs.SRF(self.cov_model_2d, generator="Fourier")
