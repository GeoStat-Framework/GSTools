"""
This is the unittest of CovModel class.
"""

import unittest

import numpy as np

from gstools import (
    Circular,
    CovModel,
    Cubic,
    Exponential,
    Gaussian,
    HyperSpherical,
    Integral,
    JBessel,
    Linear,
    Matern,
    Rational,
    Spherical,
    Stable,
    SuperSpherical,
    TPLExponential,
    TPLGaussian,
    TPLSimple,
    TPLStable,
)
from gstools.covmodel.tools import (
    AttributeWarning,
    check_arg_in_bounds,
    check_bounds,
)


class Gau_var(CovModel):
    def variogram(self, r):
        h = np.abs(r) / self.len_rescaled
        return self.var * (1.0 - np.exp(-(h**2))) + self.nugget


class Gau_cov(CovModel):
    def covariance(self, r):
        h = np.abs(r) / self.len_rescaled
        return self.var * np.exp(-(h**2))


class Gau_cor(CovModel):
    def correlation(self, r):
        h = np.abs(r) / self.len_rescaled
        return np.exp(-(h**2))


class Gau_fix(CovModel):
    def cor(self, h):
        return np.exp(-(h**2))

    def fix_dim(self):
        return 2


class Mod_add(CovModel):
    def cor(self, h):
        return 1.0

    def default_opt_arg(self):
        return {"alpha": 1}


class TestCovModel(unittest.TestCase):
    def setUp(self):
        self.std_cov_models = [
            Gaussian,
            Exponential,
            Stable,
            Rational,
            Cubic,
            Matern,
            Linear,
            Circular,
            Spherical,
            HyperSpherical,
            SuperSpherical,
            JBessel,
            TPLSimple,
            Integral,
        ]
        self.tpl_cov_models = [
            TPLGaussian,
            TPLExponential,
            TPLStable,
        ]
        self.cov_models = self.std_cov_models + self.tpl_cov_models
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
                return np.exp(-(h**2))

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
                                model.vario_spatial(([1], [2], [3])[:dim])[0],
                                model.var
                                + model.nugget
                                - model.cov_spatial(([1], [2], [3])[:dim])[0],
                            )
                            self.assertAlmostEqual(
                                model.cor_spatial(([1], [2], [3])[:dim])[0],
                                model.cov_spatial(([1], [2], [3])[:dim])[0]
                                / model.var,
                            )
                            for d in range(dim):
                                self.assertAlmostEqual(
                                    model.vario_axis(1, axis=d),
                                    model.var
                                    + model.nugget
                                    - model.cov_axis(1, axis=d),
                                )
                                self.assertAlmostEqual(
                                    model.cor_axis(1, axis=d),
                                    model.cov_axis(1, axis=d) / model.var,
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
                            model.vario_spatial((1, 2, 3)[:dim])
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
                            # check arg bound setting
                            model.set_arg_bounds(
                                var=[2, np.inf], nugget=[1, 2]
                            )
                            self.assertAlmostEqual(model.var, 3)
                            self.assertAlmostEqual(model.nugget, 1.5)

    def test_tpl_models(self):
        for Model in self.tpl_cov_models:
            for dim in self.dims:
                model = Model(dim=dim, len_scale=9, len_low=1, rescale=2)
                self.assertAlmostEqual(model.len_up_rescaled, 5)
                model.len_low = 0.0
                self.assertAlmostEqual(model.cor(2), model.correlation(9))
                # also check resetting of var when sill is given lower
                model.fit_variogram(
                    self.gamma_x, self.gamma_y, sill=1.1, nugget=False
                )
                self.assertAlmostEqual(model.var, 1.1, delta=1e-5)
                # check var_raw handling
                model = Model(var_raw=1, len_low=0, integral_scale=10)
                var_save = model.var
                model.var_raw = 1.1
                self.assertAlmostEqual(model.var, var_save * 1.1)
                self.assertAlmostEqual(model.integral_scale, 10)
                # integral scale is not setable when len_low is not 0
                with self.assertRaises(ValueError):
                    Model(var_raw=1, len_low=5, integral_scale=10)

    def test_fitting(self):
        for Model in self.std_cov_models:
            for dim in self.dims:
                model = Model(dim=dim)
                model.fit_variogram(self.gamma_x, self.gamma_y, nugget=False)
                self.assertAlmostEqual(model.nugget, 0.0)
                model = Model(dim=dim)
                # also check resetting of var when sill is given lower
                model.fit_variogram(self.gamma_x, self.gamma_y, sill=0.9)
                self.assertAlmostEqual(model.nugget + model.var, 0.9)
                model = Model(dim=dim)
                # more detailed checks
                model.fit_variogram(
                    self.gamma_x, self.gamma_y, sill=2, nugget=False
                )
                self.assertAlmostEqual(model.var, 2.0)
                model = Model(dim=dim)
                model.fit_variogram(
                    self.gamma_x, self.gamma_y, sill=2, nugget=1
                )
                self.assertAlmostEqual(model.var, 1)
                model = Model(dim=dim)
                ret = model.fit_variogram(
                    self.gamma_x,
                    self.gamma_y,
                    loss="linear",
                    return_r2=True,
                    weights="inv",
                    init_guess="current",
                )
                self.assertEqual(len(ret), 3)

        # treatment of sill/var/nugget by fitting
        model = Stable()
        model.fit_variogram(
            self.gamma_x, self.gamma_y, nugget=False, var=False, sill=2
        )
        self.assertAlmostEqual(model.var, 1)
        self.assertAlmostEqual(model.nugget, 1)
        model.fit_variogram(self.gamma_x, self.gamma_y, var=2, sill=3)
        self.assertAlmostEqual(model.var, 2)
        self.assertAlmostEqual(model.nugget, 1)
        model.var = 3
        model.fit_variogram(
            self.gamma_x, self.gamma_y, nugget=False, var=False, sill=2
        )
        self.assertAlmostEqual(model.var, 2)
        self.assertAlmostEqual(model.nugget, 0)
        model.fit_variogram(self.gamma_x, self.gamma_y, weights="inv")
        len_save = model.len_scale
        model.fit_variogram(
            self.gamma_x, self.gamma_y, weights=lambda x: 1 / (1 + x)
        )
        self.assertAlmostEqual(model.len_scale, len_save, places=6)
        # check ValueErrors
        with self.assertRaises(ValueError):
            model.fit_variogram(self.gamma_x, self.gamma_y, sill=2, var=3)
        with self.assertRaises(ValueError):
            model.fit_variogram(self.gamma_x, self.gamma_y, sill=2, nugget=3)
        with self.assertRaises(ValueError):
            model.fit_variogram(self.gamma_x, self.gamma_y, method="wrong")
        with self.assertRaises(ValueError):
            model.fit_variogram(self.gamma_x, self.gamma_y, wrong=False)
        model.var_bounds = [0, 1]
        model.nugget_bounds = [0, 1]
        with self.assertRaises(ValueError):
            model.fit_variogram(self.gamma_x, self.gamma_y, sill=3)
        # init guess
        with self.assertRaises(ValueError):
            model.fit_variogram(self.gamma_x, self.gamma_y, init_guess="wrong")
        with self.assertRaises(ValueError):
            model.fit_variogram(
                self.gamma_x, self.gamma_y, init_guess={"wrong": 1}
            )
        # sill fixing
        model.var_bounds = [0, np.inf]
        model.fit_variogram(
            self.gamma_x, np.array(self.gamma_y) + 1, sill=2, alpha=False
        )
        self.assertAlmostEqual(model.var + model.nugget, 2)
        # check isotropicity for latlon models
        model = Stable(latlon=True)
        with self.assertRaises(ValueError):
            model.fit_variogram(self.gamma_x, 3 * [self.gamma_y])

    def test_covmodel_class(self):
        model_std = Gaussian(rescale=3, var=1.1, nugget=1.2, len_scale=1.3)
        model_var = Gau_var(rescale=3, var=1.1, nugget=1.2, len_scale=1.3)
        model_cov = Gau_cov(rescale=3, var=1.1, nugget=1.2, len_scale=1.3)
        model_cor = Gau_cor(rescale=3, var=1.1, nugget=1.2, len_scale=1.3)
        var = model_std.variogram(2.5)
        cov = model_std.covariance(2.5)
        corr = model_std.correlation(2.5)
        cor = model_std.cor(2.5)

        self.assertFalse(check_bounds(bounds=[0]))
        self.assertFalse(check_bounds(bounds=[1, -1]))
        self.assertFalse(check_bounds(bounds=[0, 1, 2, 3]))
        self.assertFalse(check_bounds(bounds=[0, 1, "kk"]))
        self.assertRaises(ValueError, model_std.set_arg_bounds, wrong_arg=[1])
        self.assertRaises(
            ValueError, model_std.set_arg_bounds, wrong_arg=[-1, 1]
        )

        # checking some properties
        model_par = Stable()
        self.assertFalse(model_par.do_rotation)
        self.assertEqual(len(model_par.arg), len(model_par.arg_list))
        self.assertEqual(len(model_par.iso_arg), len(model_par.iso_arg_list))
        self.assertEqual(len(model_par.arg), len(model_par.iso_arg) + 2)
        self.assertEqual(len(model_par.len_scale_vec), model_par.dim)
        self.assertFalse(Gaussian() == Stable())
        model_par.hankel_kw = {"N": 300}
        self.assertEqual(model_par.hankel_kw["N"], 300)

        # arg in bounds check
        model_std.set_arg_bounds(var=[0.5, 1.5])
        with self.assertRaises(ValueError):
            model_std.var = 0.4
        with self.assertRaises(ValueError):
            model_std.var = 1.6
        model_std.set_arg_bounds(var=[0.5, 1.5, "oo"])
        with self.assertRaises(ValueError):
            model_std.var = 0.5
        with self.assertRaises(ValueError):
            model_std.var = 1.5
        with self.assertRaises(ValueError):
            model_std.var_bounds = [1, -1]
        with self.assertRaises(ValueError):
            model_std.len_scale_bounds = [1, -1]
        with self.assertRaises(ValueError):
            model_std.nugget_bounds = [1, -1]
        with self.assertRaises(ValueError):
            model_std.anis_bounds = [1, -1]
        # reset the standard model
        model_std = Gaussian(rescale=3, var=1.1, nugget=1.2, len_scale=1.3)
        # std value from bounds with neg. inf and finit bound
        model_add = Mod_add()
        model_add.set_arg_bounds(alpha=[-np.inf, 0])
        self.assertAlmostEqual(model_add.alpha, -1)
        # special treatment of anis check
        model_std.set_arg_bounds(anis=[2, 4, "oo"])
        self.assertTrue(np.all(np.isclose(model_std.anis, 3)))
        # dim specific checks
        with self.assertWarns(AttributeWarning):
            Gau_fix(dim=1)
        self.assertRaises(ValueError, Gaussian, dim=0)
        self.assertRaises(ValueError, Gau_fix, latlon=True)
        # check inputs
        self.assertRaises(ValueError, model_std.percentile_scale, per=-1.0)
        self.assertRaises(ValueError, Gaussian, anis=-1.0)
        self.assertRaises(ValueError, Gaussian, len_scale=[1, -1])
        self.assertRaises(ValueError, check_arg_in_bounds, model_std, "wrong")
        self.assertWarns(AttributeWarning, Gaussian, wrong_arg=1.0)
        with self.assertWarns(AttributeWarning):
            self.assertRaises(ValueError, Gaussian, len_rescaled=1.0)

        # check correct subclassing
        with self.assertRaises(TypeError):

            class Gau_err(CovModel):
                pass

        self.assertAlmostEqual(var, model_var.variogram(2.5))
        self.assertAlmostEqual(var, model_cov.variogram(2.5))
        self.assertAlmostEqual(var, model_cor.variogram(2.5))
        self.assertAlmostEqual(cov, model_var.covariance(2.5))
        self.assertAlmostEqual(cov, model_cov.covariance(2.5))
        self.assertAlmostEqual(cov, model_cor.covariance(2.5))
        self.assertAlmostEqual(corr, model_var.correlation(2.5))
        self.assertAlmostEqual(corr, model_cov.correlation(2.5))
        self.assertAlmostEqual(corr, model_cor.correlation(2.5))
        self.assertAlmostEqual(cor, model_var.cor(2.5))
        self.assertAlmostEqual(cor, model_cov.cor(2.5))
        self.assertAlmostEqual(cor, model_cor.cor(2.5))

    def test_rescale(self):
        model1 = Exponential()
        model2 = Exponential(rescale=2.1)
        model3 = Exponential(rescale=2.1, len_scale=2.1)

        self.assertAlmostEqual(
            model1.integral_scale, 2.1 * model2.integral_scale
        )
        self.assertAlmostEqual(model1.integral_scale, model3.integral_scale)

    def test_special_models(self):
        # Matern and Integral converge to gaussian
        model0 = Integral(rescale=0.5)
        model0.set_arg_bounds(nu=[0, 1001])
        model0.nu = 1000
        model1 = Matern()
        model1.set_arg_bounds(nu=[0, 101])
        model1.nu = 100
        model2 = Gaussian(rescale=0.5)
        self.assertAlmostEqual(model0.variogram(1), model2.variogram(1), 2)
        self.assertAlmostEqual(model0.spectrum(1), model2.spectrum(1), 2)
        self.assertAlmostEqual(model1.variogram(1), model2.variogram(1))
        self.assertAlmostEqual(model1.spectrum(1), model2.spectrum(1), 2)
        # stable model gets unstable for alpha < 0.3
        with self.assertWarns(AttributeWarning):
            Stable(alpha=0.2)
        with self.assertWarns(AttributeWarning):
            TPLStable(alpha=0.2)
        # corner case for JBessel model
        with self.assertWarns(AttributeWarning):
            JBessel(dim=3, nu=0.5)


if __name__ == "__main__":
    unittest.main()
