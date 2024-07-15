"""
This is the unittest of the Fourier class.
"""

import unittest

import numpy as np

import gstools as gs


class TestFourier(unittest.TestCase):
    def setUp(self):
        self.seed = 19900408
        self.cov_model_1d = gs.Gaussian(dim=1, var=0.5, len_scale=10.0)
        self.cov_model_2d = gs.Gaussian(dim=2, var=2.0, len_scale=30.0)
        self.cov_model_3d = gs.Gaussian(dim=3, var=2.1, len_scale=21.0)
        self.L = [80, 30, 91]
        self.x = np.linspace(0, self.L[0], 11)
        self.y = np.linspace(0, self.L[1], 31)
        self.z = np.linspace(0, self.L[2], 13)

        self.mode_no = [12, 6, 14]

        self.srf_1d = gs.SRF(
            self.cov_model_1d,
            generator="Fourier",
            mode_no=[self.mode_no[0]],
            period=[self.L[0]],
            seed=self.seed,
        )
        self.srf_2d = gs.SRF(
            self.cov_model_2d,
            generator="Fourier",
            mode_no=self.mode_no[:2],
            period=self.L[:2],
            seed=self.seed,
        )
        self.srf_3d = gs.SRF(
            self.cov_model_3d,
            generator="Fourier",
            mode_no=self.mode_no,
            period=self.L,
            seed=self.seed,
        )

    def test_1d(self):
        field = self.srf_1d((self.x,), mesh_type="structured")
        self.assertAlmostEqual(field[0], 0.6236929351309081)

    def test_2d(self):
        field = self.srf_2d((self.x, self.y), mesh_type="structured")
        self.assertAlmostEqual(field[0, 0], -0.1431996611581266)

    def test_3d(self):
        field = self.srf_3d((self.x, self.y, self.z), mesh_type="structured")
        self.assertAlmostEqual(field[0, 0, 0], -1.0433325279452803)

    def test_periodicity_1d(self):
        field = self.srf_1d((self.x,), mesh_type="structured")
        self.assertAlmostEqual(field[0], field[-1])

    def test_periodicity_2d(self):
        field = self.srf_2d((self.x, self.y), mesh_type="structured")
        self.assertAlmostEqual(
            field[0, len(self.y) // 2], field[-1, len(self.y) // 2]
        )
        self.assertAlmostEqual(
            field[len(self.x) // 2, 0], field[len(self.x) // 2, -1]
        )

    def test_periodicity_3d(self):
        field = self.srf_3d((self.x, self.y, self.z), mesh_type="structured")
        self.assertAlmostEqual(
            field[0, len(self.y) // 2, 0], field[-1, len(self.y) // 2, 0]
        )
        self.assertAlmostEqual(field[0, 0, 0], field[0, -1, 0])
        self.assertAlmostEqual(
            field[len(self.x) // 2, len(self.y) // 2, 0],
            field[len(self.x) // 2, len(self.y) // 2, -1],
        )

    def test_setters(self):
        new_period = [5, 10]
        self.srf_2d.generator.period = new_period
        np.testing.assert_almost_equal(
            self.srf_2d.generator.period,
            np.array(new_period),
        )
        new_mode_no = [6, 6]
        self.srf_2d.generator.mode_no = new_mode_no
        np.testing.assert_almost_equal(
            self.srf_2d.generator.mode_no,
            np.array(new_mode_no),
        )

    def test_assertions(self):
        # unstructured grids not supported
        self.assertRaises(ValueError, self.srf_2d, (self.x, self.y))
        self.assertRaises(
            ValueError, self.srf_2d, (self.x, self.y), mesh_type="unstructured"
        )
        self.assertRaises(
            ValueError,
            gs.SRF,
            self.cov_model_2d,
            generator="Fourier",
            mode_no=[13, 50],
            period=self.L[:2],
            seed=self.seed,
        )
