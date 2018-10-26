# -*- coding: utf-8 -*-
"""
GStools subpackage providing different covariance models.

.. currentmodule:: gstools.field.cov

The following classes and functions are provided

.. autosummary::
   Gau
   Exp
   Sph
   SphRescaled
   Rat
   Mat
   MatRescaled
"""
# pylint: disable=no-member
from __future__ import print_function, division, absolute_import

import warnings
import numpy as np
from scipy import special as sps
from gstools.field.cov_base import CovModel

__all__ = [
    "Gau",
    "Exp",
    "Sph",
    "SphRescaled",
    "Rat",
    "Mat",
    "MatRescaled",
]


# Gaussian Model ##############################################################

class Gau(CovModel):
    r'''The Gaussian covariance model

    Notes
    -----
    This model is given by the following normalized covariance function:

    .. math::
       \tilde{C}(r) =
       \exp\left(- \frac{\pi}{4} \cdot \left(\frac{r}{\ell}\right)^2\right)
    '''
    def covariance_normed(self, r):
        r = np.abs(np.array(r, dtype=float))
        return np.exp(-np.pi/4*(r/self.len_scale)**2)

    def spectrum(self, k):
        return (self.var*(self.len_scale/np.pi)**self.dim *
                np.exp(-(k*self.len_scale)**2/np.pi))

    def spectral_rad_cdf(self, r):
        r'''The cdf of the radial spectral density

        Notes
        -----
        Since the spectrum is radial-symmetric, we can calculate, the pdf and
        cdf of the radii-distribution according to the spectral density

        .. math::
           \mathrm{CDF}(r) = \intop_0^r \mathrm{PDF}(\tau) d\tau
        '''
        if self.dim == 1:
            return sps.erf(self.len_scale*r/np.sqrt(np.pi))
        elif self.dim == 2:
            return 1. - np.exp(-(r*self.len_scale)**2/np.pi)
        elif self.dim == 3:
            return (sps.erf(self.len_scale*r/np.sqrt(np.pi)) -
                    2*r*self.len_scale/np.pi *
                    np.exp(-(r*self.len_scale)**2/np.pi))
        return None

    def spectral_rad_ppf(self, u):
        r'''The ppf of the radial spectral density

        Notes
        -----
        To sample the radii of the given spectral density we can calculate
        the PPF (Percent Point Function), to sample from a uniform distribution

        .. math::
           \mathrm{PPF}(u) = \mathrm{CDF}^{-1}(u)
        '''
        if self.dim == 1:
            return sps.erfinv(u)*np.sqrt(np.pi)/self.len_scale
        elif self.dim == 2:
            return (np.sqrt(np.pi)/self.len_scale *
                    np.sqrt(-np.log(1.-u)))
        return None

    def _has_ppf(self):
        """ppf for 3 dimensions is not analytical"""
        # since the ppf is not analytical for dim=3, we have to state that
        if self.dim == 3:
            return False
        return True

    def calc_integral_scale(self):
        '''The integral scale of the gaussian model is the length scale'''
        return self.len_scale


# Exponential Model ###########################################################

class Exp(CovModel):
    r'''The Exponential covariance model

    Notes
    -----
    This model is given by the following normalized covariance function:

    .. math::
       \tilde{C}(r) =
       \exp\left(- \frac{r}{\ell} \right)
    '''
    def covariance_normed(self, r):
        r = np.abs(np.array(r, dtype=float))
        return np.exp(-r/self.len_scale)

    def spectrum(self, k):
        return (
            self.var*self.len_scale**self.dim*sps.gamma((self.dim+1)/2) /
            (np.pi*(1. + (k*self.len_scale)**2))**((self.dim+1)/2))

    def spectral_rad_cdf(self, r):
        r'''The cdf of the radial spectral density

        Notes
        -----
        Since the spectrum is radial-symmetric, we can calculate, the pdf and
        cdf of the radii-distribution according to the spectral density

        .. math::
           \mathrm{CDF}(r) = \intop_0^r \mathrm{PDF}(\tau) d\tau
        '''
        if self.dim == 1:
            return np.arctan(r*self.len_scale)*2/np.pi
        elif self.dim == 2:
            return 1. - 1/np.sqrt(1 + (r*self.len_scale)**2)
        elif self.dim == 3:
            return (np.arctan(r*self.len_scale) -
                    r*self.len_scale/(1 + (r*self.len_scale)**2))*2/np.pi
        return None

    def spectral_rad_ppf(self, u):
        r'''The ppf of the radial spectral density

        Notes
        -----
        To sample the radii of the given spectral density we can calculate
        the PPF (Percent Point Function), to sample from a uniform distribution

        .. math::
           \mathrm{PPF}(u) = \mathrm{CDF}^{-1}(u)
        '''
        if self.dim == 1:
            return np.tan(np.pi/2*u)/self.len_scale
        elif self.dim == 2:
            return np.sqrt(1/u**2 - 1.)/self.len_scale
        return None

    def _has_ppf(self):
        """ppf for 3 dimensions is not analytical"""
        # since the ppf is not analytical for dim=3, we have to state that
        if self.dim == 3:
            return False
        return True

    def calc_integral_scale(self):
        '''The integral scale of the exponential model is the length scale'''
        return self.len_scale


# Spherical Model #############################################################

class Sph(CovModel):
    r'''The Spherical covariance model

    Notes
    -----
    This model is given by the following normalized covariance function:

    .. math::
       \tilde{C}(r) =
       \begin{cases}
       1-\frac{3}{2}\cdot\frac{r}{\ell} +
       \frac{1}{2}\cdot\left(\frac{r}{\ell}\right)^{3}
       & r<\ell\\
       0 & r\geq\ell
       \end{cases}
    '''
    def covariance_normed(self, r):
        r = np.atleast_1d(np.abs(np.array(r, dtype=float)))
        res = 1. - 3./2.*r/self.len_scale + 1./2.*(r/self.len_scale)**3
        res[r > self.len_scale] = 0.
        return res


class SphRescaled(CovModel):
    r'''The rescaled Spherical covariance model

    Notes
    -----
    This model is given by the following normalized covariance function:

    .. math::
       \tilde{C}(r) =
       \begin{cases}
       1-\frac{9}{16}\cdot\frac{r}{\ell} +
       \frac{27}{1024}\cdot\left(\frac{r}{\ell}\right)^{3}
       & r<\frac{8}{3}\ell\\
       0 & r\geq\frac{8}{3}\ell
       \end{cases}
    '''
    def covariance_normed(self, r):
        r = np.atleast_1d(np.abs(np.array(r, dtype=float)))
        res = 1. - 9./16.*r/self.len_scale + 27./1024.*(r/self.len_scale)**3
        res[r > 8./3.*self.len_scale] = 0.
        return res

    def calc_integral_scale(self):
        '''The integral scale of the spherical model is the length scale'''
        return self.len_scale


# Rational Model ##############################################################

class Rat(CovModel):
    r'''The rational quadratic covariance model

    Notes
    -----
    This model is given by the following normalized covariance function:

    .. math::
       \tilde{C}(r) =
       \left(1 + \frac{1}{2\alpha} \cdot
       \left(\frac{r}{\ell}\right)^2\right)^{-\alpha}

    :math:`\alpha` is a shape parameter and should be > 0.5.
    '''
    def default_opt_arg(self):
        return {"alpha": 1.}

    def default_opt_arg_bounds(self):
        return {"alpha": [0.5, np.inf]}

    def covariance_normed(self, r):
        r = np.abs(np.array(r, dtype=float))
        return np.power(1 + 0.5/self.alpha*(r/self.len_scale)**2, -self.alpha)


# Stable Model ################################################################

class Stab(CovModel):
    r'''The stable covariance model

    Notes
    -----
    This model is given by the following normalized covariance function:

    .. math::
       \tilde{C}(r) =
       \exp\left(- \left(\frac{r}{\ell}\right)^{\alpha}\right)

    :math:`\alpha` is a shape parameter with :math:`\alpha\in(0,2]`
    '''
    def default_opt_arg(self):
        return {"alpha": 1.5}

    def default_opt_arg_bounds(self):
        return {"alpha": [0, 2, "oc"]}

    def covariance_normed(self, r):
        r = np.abs(np.array(r, dtype=float))
        return np.exp(-np.power(r/self.len_scale, self.alpha))


# Matérn Model ################################################################

class Mat(CovModel):
    r'''The Matérn covariance model

    Notes
    -----
    This model is given by the following normalized covariance function:

    .. math::
       \tilde{C}(r) =
       \frac{2^{1-\nu}}{\Gamma\left(\nu\right)} \cdot
       \left(\sqrt{2\nu}\cdot\frac{r}{\ell}\right)^{\nu} \cdot
       \mathrm{K}_{\nu}\left(\sqrt{2\nu}\cdot\frac{r}{\ell}\right)

    Where :math:`\Gamma` is the gamma function and :math:`\mathrm{K}_{\nu}`
    is the modified Bessel function of the second kind.

    :math:`\nu` is a shape parameter and should be >= 0.5.
    '''
    def default_opt_arg(self):
        return {"nu": 1.}

    def default_opt_arg_bounds(self):
        return {"nu": [0.5, 60., "cc"]}

    def check_opt_arg(self):
        if self.nu > 50.:
            warnings.warn("Mat: parameter 'nu' is > 50, " +
                          "calculations most likely get unstable here")

    def covariance_normed(self, r):
        r = np.abs(np.array(r, dtype=float))
        r_gz = r[r > 0.]
        res = np.ones_like(r)
        with np.errstate(over='ignore', invalid='ignore'):
            res[r > 0.] = (
                np.power(2., 1.-self.nu) / sps.gamma(self.nu) *
                np.power(np.sqrt(2.*self.nu) * r_gz/self.len_scale, self.nu) *
                sps.kv(self.nu, np.sqrt(2.*self.nu) * r_gz/self.len_scale))
        # if nu >> 1 we get errors for the farfield, there 0 is approached
        res[np.logical_not(np.isfinite(res))] = 0.0
        # covariance is positiv
        res = np.maximum(res, 0.)
        return res


class MatRescaled(CovModel):
    r'''The rescaled Matérn covariance model

    Notes
    -----
    This model is given by the following normalized covariance function:

    .. math::
       \tilde{C}(r) =
       \frac{2^{1-\nu}}{\Gamma\left(\nu\right)} \cdot
       \left(\frac{\pi}{B\left(\nu,\frac{1}{2}\right)} \cdot
       \frac{r}{\ell}\right)^{\nu} \cdot
       \mathrm{K}_{\nu}\left(\frac{\pi}{B\left(\nu,\frac{1}{2}\right)} \cdot
       \frac{r}{\ell}\right)

    Where :math:`\Gamma` is the gamma function,
    :math:`\mathrm{K}_{\nu}` is the modified Bessel function
    of the second kind and :math:`B` is the Euler beta function.

    :math:`\nu` is a shape parameter and should be > 0.5.
    '''
    def default_opt_arg(self):
        return {"nu": 1.}

    def default_opt_arg_bounds(self):
        return {"nu": [0.5, 60., "cc"]}

    def check_opt_arg(self):
        if self.nu > 50.:
            warnings.warn("Mat: parameter 'nu' is > 50, " +
                          "calculations most likely get unstable here")

    def covariance_normed(self, r):
        r = np.abs(np.array(r, dtype=float))
        r_gz = r[r > 0.]
        res = np.ones_like(r)
        with np.errstate(over='ignore', invalid='ignore'):
            res[r > 0.] = (
                np.power(2, 1.-self.nu) / sps.gamma(self.nu) *
                np.power(np.pi/sps.beta(self.nu, .5) * r_gz/self.len_scale,
                         self.nu) *
                sps.kv(self.nu,
                       np.pi/sps.beta(self.nu, .5) * r_gz/self.len_scale))
        # if nu >> 1 we get errors for the farfield, there 0 is approached
        res[np.logical_not(np.isfinite(res))] = 0.0
        # covariance is positiv
        res = np.maximum(res, 0.)
        return res


# Tests #######################################################################

if __name__ == "__main__":

    test_init = 0
    test_sub = 0

    if test_init:
        Crazy1 = CovModel()
    if test_sub:
        class Crazy(CovModel):
            pass

    plot_vario = 1
    plot_spec = 1
    plot_fit = 1

    c = Mat(nu=1, integral_scale=1, len_scale_bounds=[0.5, 50., "oo"])
    d = Gau(integral_scale=1)
    e = Exp(integral_scale=1)
    f = Rat(alpha=1, integral_scale=1)
    g = Sph(integral_scale=1)

    print("Mat: attributes, properties and methods")
    print(c.arg_bounds)
    c.set_arg_bounds(len_scale=[0.5, 100., "cc"])
    print(c.arg_bounds, "arg_bounds")
    print(c.variogram(1.), "variogram(1.)")
    print(c.variogram_normed(1.), "variogram_normed(1.)")
    print(c.covariance(1.), "covariance(1.)")
    print(c.covariance_normed(1.), "covariance_normed(1.)")
    print(c.spectrum(1.), "spectrum(1.)")
    print(c.spectral_density(1.), "spectral_density(1.)")
    print(c.spectral_rad_pdf(1.), "spectral_rad_pdf(1.)")
    print(c.ln_spectral_rad_pdf(1.), "ln_spectral_rad_pdf(1.)")
    print(c.len_scale_bounds, "len_scale_bounds")
    print(c.var_bounds, "var_bounds")
    print(c.arg_bounds, "arg_bounds")
    print(c.opt_arg, "opt_arg")
    print(c.opt_arg_bounds, "opt_arg_bounds")
    print(c.has_cdf, "has_cdf")
    print(c.has_ppf, "has_ppf")
    print(c.dim, "dim")
    print(c.var, "var")
    print(c.len_scale, "len_scale")
    print(c.len_scale_vec, "len_scale_vec")
    print(c.anis, "anis")
    print(c.angles, "angles")
    print(c.integral_scale, "integral_scale")
    print(c.integral_scale_vec, "integral_scale_vec")
    print()

    print(c.percentile_scale())
    print(d.percentile_scale())
    print(e.percentile_scale())
    print(f.percentile_scale())
    print(g.percentile_scale())

    if plot_vario:
        c.plot_variogram()
        d.plot_variogram()
        e.plot_variogram()
        f.plot_variogram()
        g.plot_variogram()

    if plot_spec:
        c.plot_spectral_rad_pdf()
        d.plot_spectral_rad_pdf()
        e.plot_spectral_rad_pdf()
        f.plot_spectral_rad_pdf()
        g.plot_spectral_rad_pdf()

    model1 = Exp(var=1, len_scale=1)
    print(model1.integral_scale)
    a = np.linspace(0, 10, 20)
    b = model1.variogram(a)
    model2 = Mat()
    print(model2.fit_variogram(a, b)[0])

    if plot_fit:
        model1.plot_variogram()
        model2.plot_variogram()
