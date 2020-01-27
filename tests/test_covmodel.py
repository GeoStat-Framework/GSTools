# -*- coding: utf-8 -*-
"""
This is the unittest of CovModel class.
"""
import numpy as np
import unittest
from gstools import (
    CovModel,
    Gaussian,
    Exponential,
    Rational,
    Stable,
    Matern,
    Linear,
    Circular,
    Spherical,
    TPLGaussian,
    TPLExponential,
    TPLStable,
)


class TestCovModel(unittest.TestCase):
    def setUp(self):
        self.cov_models = [
            Gaussian,
            Exponential,
            Rational,
            Stable,
            Matern,
            Linear,
            Circular,
            Spherical,
            TPLGaussian,
            TPLExponential,
            TPLStable,
        ]
        self.std_cov_models = [
            Gaussian,
            Exponential,
            Rational,
            Stable,
            Matern,
            Linear,
            Circular,
            Spherical,
        ]
        self.dims = range(1, 4)
        self.lens = [[10, 5, 2]]
        self.anis = [[0.5, 0.2]]
        self.nuggets = [0, 1]
        self.vars = [1, 2]
        self.angles = [[1, 2, 3]]

        self.gamma_x = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0]
        self.gamma_y = [0.2, 0.5, 0.6, 0.8, 0.8, 0.9]

    def test_creation(self):
        with self.assertRaises(TypeError):
            CovModel()

        class User(CovModel):
            def cor(self, h):
                return np.exp(-h ** 2)

        user = User(len_scale=2)
        self.assertAlmostEqual(user.correlation(1), np.exp(-0.25))

        for Model in self.cov_models:
            for dim in self.dims:
                for angles in self.angles:
                    for nugget in self.nuggets:
                        for len_scale, anis in zip(self.lens, self.anis):
                            model = Model(
                                dim=dim, len_scale=len_scale, angles=angles
                            )
                            model1 = Model(
                                dim=dim, len_scale=10, anis=anis, angles=angles
                            )
                            self.assertTrue(model == model1)
                            self.assertAlmostEqual(
                                model.variogram(1),
                                model.var + model.nugget - model.covariance(1),
                            )
                            self.assertAlmostEqual(
                                model.covariance(1),
                                model.var * model.correlation(1),
                            )
                            self.assertAlmostEqual(
                                model.covariance(1),
                                model.var * model.correlation(1),
                            )
                            self.assertAlmostEqual(
                                model.vario_spatial(([1], [2], [3]))[0],
                                model.var
                                + model.nugget
                                - model.cov_spatial(([1], [2], [3]))[0],
                            )
                            self.assertAlmostEqual(
                                model.cov_nugget(0), model.sill
                            )
                            self.assertAlmostEqual(model.vario_nugget(0), 0.0)
                            self.assertAlmostEqual(
                                model.cov_nugget(1), model.covariance(1)
                            )
                            self.assertAlmostEqual(model.vario_nugget(0), 0.0)
                            self.assertAlmostEqual(
                                model.vario_nugget(1), model.variogram(1)
                            )
                            # check if callable
                            model.vario_spatial((1, 2, 3))
                            model.spectral_density([0, 1])
                            model.spectrum([0, 1])
                            model.spectral_rad_pdf([0, 1])
                            model.ln_spectral_rad_pdf([0, 1])
                            model.integral_scale_vec
                            model.percentile_scale(0.9)
                            if model.has_cdf:
                                model.spectral_rad_cdf([0, 1])
                            if model.has_ppf:
                                model.spectral_rad_ppf([0.0, 0.99])
                            model.pykrige_kwargs

    def test_fitting(self):
        for Model in self.std_cov_models:
            for dim in self.dims:
                model = Model(dim=dim)
                model.fit_variogram(self.gamma_x, self.gamma_y, nugget=False)
                self.assertAlmostEqual(model.nugget, 0.0)


if __name__ == "__main__":
    unittest.main()
