"""
This is the unittest of the RandMeth class.
"""

import copy
import unittest

import numpy as np

from gstools import Gaussian, Matern, Stable
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
        # the 3D reference values below were created with mcmc sampling
        self.rm_3d = RandMeth(
            self.cov_model_3d, mode_no=100, seed=self.seed, sampling="mcmc"
        )

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

        # the 3D reference values below were created with mcmc sampling
        self.rm_1d.sampling = "mcmc"
        self.rm_1d.model = self.cov_model_3d
        modes = self.rm_1d((self.x_tuple, self.y_tuple, self.z_tuple))
        self.assertAlmostEqual(modes[0], 1.3240234883187239)
        self.assertAlmostEqual(modes[1], 1.6367244277732766)

        self.rm_2d.mode_no = 800
        modes = self.rm_2d((self.x_tuple, self.y_tuple))
        self.assertAlmostEqual(modes[0], -3.20809251)
        self.assertAlmostEqual(modes[1], -2.62032778)

    def test_sampling_auto(self):
        pos = (self.x_tuple, self.y_tuple, self.z_tuple)

        def modes(model, sampling):
            rm = RandMeth(
                model, mode_no=100, seed=self.seed, sampling=sampling
            )
            return rm(pos[: model.dim])

        # inversion if the model has a ppf (Gaussian in 2D)
        np.testing.assert_allclose(
            modes(self.cov_model_2d, "auto"),
            modes(self.cov_model_2d, "inversion"),
        )
        # direct sampling if there is no ppf, but a direct sampler
        np.testing.assert_allclose(
            modes(self.cov_model_3d, "auto"),
            modes(self.cov_model_3d, "direct"),
        )
        matern = Matern(dim=2, nu=1.5, len_scale=3.5)
        np.testing.assert_allclose(
            modes(matern, "auto"), modes(matern, "direct")
        )
        # mcmc otherwise
        stable = Stable(dim=2, len_scale=3.5)
        np.testing.assert_allclose(
            modes(stable, "auto"), modes(stable, "mcmc")
        )

    def test_sampling_direct(self):
        model = Matern(dim=2, var=1.5, len_scale=3.5, nu=1.5)
        rm = RandMeth(model, mode_no=100, seed=self.seed, sampling="direct")
        modes = rm((self.x_tuple, self.y_tuple))
        self.assertAlmostEqual(modes[0], 0.06124152755960492)
        self.assertAlmostEqual(modes[1], 0.5247458125683446)

    def test_sampling_direct_not_available(self):
        with self.assertRaises(ValueError):
            RandMeth(Stable(dim=2), seed=self.seed, sampling="direct")
        with self.assertRaises(ValueError):
            RandMeth(Stable(dim=2, var=0), seed=self.seed, sampling="direct")
        rm = RandMeth(Stable(dim=2), mode_no=100, seed=self.seed)
        with self.assertRaises(ValueError):
            rm.sampling = "direct"
        self.assertEqual(rm.sampling, "auto")

    def test_sampling_unknown(self):
        with self.assertRaises(ValueError):
            RandMeth(self.cov_model_2d, seed=self.seed, sampling="wrong")

    def test_sampling_direct_error_keeps_state(self):
        rm = RandMeth(Matern(dim=2), seed=self.seed, sampling="direct")
        modes = rm._cov_sample.copy()
        stable = Stable(dim=2, var=3.0)
        with self.assertRaises(ValueError):
            rm.update(stable)
        # the failed update did not change the generator
        self.assertEqual(rm.model, Matern(dim=2))
        np.testing.assert_array_equal(rm._cov_sample, modes)
        # with another sampling, the new model is used and sampled
        rm.sampling = "mcmc"
        rm.update(stable)
        self.assertEqual(rm.model, stable)


if __name__ == "__main__":
    unittest.main()
