# -*- coding: utf-8 -*-
"""
GStools subpackage providing different covariance models.

.. currentmodule:: gstools.field.cov

The following classes and functions are provided

.. autosummary::
   Gaussian
   Exponential
   Matern
   Rational
   Stable
   Spherical
   Linear
   MaternRescal
   SphericalRescal
   TPLGaussian
   TPLExponential
   TPLStable
"""
# pylint: disable=no-member
from __future__ import print_function, division, absolute_import

import warnings
import numpy as np
from scipy import special as sps
from gstools.covmodel.base import CovModel
from gstools.tools.special import exp_int

__all__ = [
    "Gaussian",
    "Exponential",
    "Spherical",
    "SphericalRescal",
    "Rational",
    "Stable",
    "Matern",
    "MaternRescal",
    "Linear",
    "TPLGaussian",
    "TPLExponential",
    "TPLStable",
]


# Gaussian Model ##############################################################


class Gaussian(CovModel):
    r"""The Gaussian covariance model

    Note
    ----
    This model is given by the following normalized covariance function:

    .. math::
       \tilde{C}(r) =
       \exp\left(- \frac{\pi}{4} \cdot \left(\frac{r}{\ell}\right)^2\right)
    """

    def covariance_normed(self, r):
        r"""Gaussian normalized covariance

        .. math::
           \tilde{C}(r) =
           \exp\left(- \frac{\pi}{4} \cdot \left(\frac{r}{\ell}\right)^2\right)
       """
        r = np.array(np.abs(r), dtype=float)
        return np.exp(-np.pi / 4 * (r / self.len_scale) ** 2)

    def spectrum(self, k):
        return (
            self.var
            * (self.len_scale / np.pi) ** self.dim
            * np.exp(-(k * self.len_scale) ** 2 / np.pi)
        )

    def spectral_rad_cdf(self, r):
        r"""The cdf of the radial spectral density

        Note
        ----
        Since the spectrum is radial-symmetric, we can calculate, the pdf and
        cdf of the radii-distribution according to the spectral density

        .. math::
           \mathrm{CDF}(r) = \intop_0^r \mathrm{PDF}(\tau) d\tau
        """
        if self.dim == 1:
            return sps.erf(self.len_scale * r / np.sqrt(np.pi))
        if self.dim == 2:
            return 1.0 - np.exp(-(r * self.len_scale) ** 2 / np.pi)
        if self.dim == 3:
            return sps.erf(
                self.len_scale * r / np.sqrt(np.pi)
            ) - 2 * r * self.len_scale / np.pi * np.exp(
                -(r * self.len_scale) ** 2 / np.pi
            )
        return None

    def spectral_rad_ppf(self, u):
        r"""The ppf of the radial spectral density

        Note
        ----
        To sample the radii of the given spectral density we can calculate
        the PPF (Percent Point Function), to sample from a uniform distribution

        .. math::
           \mathrm{PPF}(u) = \mathrm{CDF}^{-1}(u)
        """
        if self.dim == 1:
            return sps.erfinv(u) * np.sqrt(np.pi) / self.len_scale
        if self.dim == 2:
            return np.sqrt(np.pi) / self.len_scale * np.sqrt(-np.log(1.0 - u))
        return None

    def _has_ppf(self):
        """ppf for 3 dimensions is not analytical given"""
        # since the ppf is not analytical for dim=3, we have to state that
        if self.dim == 3:
            return False
        return True

    def calc_integral_scale(self):
        """The integral scale of the gaussian model is the length scale"""
        return self.len_scale


# Exponential Model ###########################################################


class Exponential(CovModel):
    r"""The Exponential covariance model

    Note
    ----
    This model is given by the following normalized covariance function:

    .. math::
       \tilde{C}(r) =
       \exp\left(- \frac{r}{\ell} \right)
    """

    def covariance_normed(self, r):
        r"""Exponential normalized covariance

        .. math::
           \tilde{C}(r) =
           \exp\left(- \frac{r}{\ell} \right)
       """
        r = np.array(np.abs(r), dtype=float)
        return np.exp(-1 * r / self.len_scale)

    def spectrum(self, k):
        return (
            self.var
            * self.len_scale ** self.dim
            * sps.gamma((self.dim + 1) / 2)
            / (np.pi * (1.0 + (k * self.len_scale) ** 2))
            ** ((self.dim + 1) / 2)
        )

    def spectral_rad_cdf(self, r):
        r"""The cdf of the radial spectral density

        Note
        ----
        Since the spectrum is radial-symmetric, we can calculate, the pdf and
        cdf of the radii-distribution according to the spectral density

        .. math::
           \mathrm{CDF}(r) = \intop_0^r \mathrm{PDF}(\tau) d\tau
        """
        if self.dim == 1:
            return np.arctan(r * self.len_scale) * 2 / np.pi
        if self.dim == 2:
            return 1.0 - 1 / np.sqrt(1 + (r * self.len_scale) ** 2)
        if self.dim == 3:
            return (
                (
                    np.arctan(r * self.len_scale)
                    - r * self.len_scale / (1 + (r * self.len_scale) ** 2)
                )
                * 2
                / np.pi
            )
        return None

    def spectral_rad_ppf(self, u):
        r"""The ppf of the radial spectral density

        Note
        ----
        To sample the radii of the given spectral density we can calculate
        the PPF (Percent Point Function), to sample from a uniform distribution

        .. math::
           \mathrm{PPF}(u) = \mathrm{CDF}^{-1}(u)
        """
        if self.dim == 1:
            return np.tan(np.pi / 2 * u) / self.len_scale
        if self.dim == 2:
            return np.sqrt(1 / u ** 2 - 1.0) / self.len_scale
        return None

    def _has_ppf(self):
        """ppf for 3 dimensions is not analytical"""
        # since the ppf is not analytical for dim=3, we have to state that
        if self.dim == 3:
            return False
        return True

    def calc_integral_scale(self):
        """The integral scale of the exponential model is the length scale"""
        return self.len_scale


# Spherical Model #############################################################


class Spherical(CovModel):
    r"""The Spherical covariance model

    Note
    ----
    This model is given by the following normalized covariance function:

    .. math::
       \tilde{C}(r) =
       \begin{cases}
       1-\frac{3}{2}\cdot\frac{r}{\ell} +
       \frac{1}{2}\cdot\left(\frac{r}{\ell}\right)^{3}
       & r<\ell\\
       0 & r\geq\ell
       \end{cases}
    """

    def covariance_normed(self, r):
        r"""Spherical normalized covariance

        .. math::
           \tilde{C}(r) =
           \begin{cases}
           1-\frac{3}{2}\cdot\frac{r}{\ell} +
           \frac{1}{2}\cdot\left(\frac{r}{\ell}\right)^{3}
           & r<\ell\\
           0 & r\geq\ell
           \end{cases}
        """
        r = np.array(np.abs(r), dtype=float)
        res = np.zeros_like(r)
        res[r < self.len_scale] = (
            1.0
            - 3.0 / 2.0 * r[r < self.len_scale] / self.len_scale
            + 1.0 / 2.0 * (r[r < self.len_scale] / self.len_scale) ** 3
        )
        return res


class SphericalRescal(CovModel):
    r"""The rescaled Spherical covariance model

    Note
    ----
    This model is given by the following normalized covariance function:

    .. math::
       \tilde{C}(r) =
       \begin{cases}
       1-\frac{9}{16}\cdot\frac{r}{\ell} +
       \frac{27}{1024}\cdot\left(\frac{r}{\ell}\right)^{3}
       & r<\frac{8}{3}\ell\\
       0 & r\geq\frac{8}{3}\ell
       \end{cases}
    """

    def covariance_normed(self, r):
        r"""Rescaled Spherical normalized covariance

        .. math::
           \tilde{C}(r) =
           \begin{cases}
           1-\frac{9}{16}\cdot\frac{r}{\ell} +
           \frac{27}{1024}\cdot\left(\frac{r}{\ell}\right)^{3}
           & r<\frac{8}{3}\ell\\
           0 & r\geq\frac{8}{3}\ell
           \end{cases}
        """
        r = np.array(np.abs(r), dtype=float)
        res = np.zeros_like(r)
        res[r < 8 / 3 * self.len_scale] = (
            1.0
            - 9.0 / 16.0 * r[r < 8 / 3 * self.len_scale] / self.len_scale
            + 27.0
            / 1024.0
            * (r[r < 8 / 3 * self.len_scale] / self.len_scale) ** 3
        )
        return res

    def calc_integral_scale(self):
        """The integral scale of the spherical model is the length scale"""
        return self.len_scale


# Rational Model ##############################################################


class Rational(CovModel):
    r"""The rational quadratic covariance model

    Note
    ----
    This model is given by the following normalized covariance function:

    .. math::
       \tilde{C}(r) =
       \left(1 + \frac{1}{2\alpha} \cdot
       \left(\frac{r}{\ell}\right)^2\right)^{-\alpha}

    :math:`\alpha` is a shape parameter and should be > 0.5.
    """

    def default_opt_arg(self):
        return {"alpha": 1.0}

    def default_opt_arg_bounds(self):
        return {"alpha": [0.5, np.inf]}

    def covariance_normed(self, r):
        r"""Rational normalized covariance

        .. math::
           \tilde{C}(r) =
           \left(1 + \frac{1}{2\alpha} \cdot
           \left(\frac{r}{\ell}\right)^2\right)^{-\alpha}
        """
        r = np.array(np.abs(r), dtype=float)
        return np.power(
            1 + 0.5 / self.alpha * (r / self.len_scale) ** 2, -self.alpha
        )


# Stable Model ################################################################


class Stable(CovModel):
    r"""The stable covariance model

    Note
    ----
    This model is given by the following normalized covariance function:

    .. math::
       \tilde{C}(r) =
       \exp\left(- \left(\frac{r}{\ell}\right)^{\alpha}\right)

    :math:`\alpha` is a shape parameter with :math:`\alpha\in(0,2]`
    """

    def default_opt_arg(self):
        return {"alpha": 1.5}

    def default_opt_arg_bounds(self):
        return {"alpha": [0, 2, "oc"]}

    def check_opt_arg(self):
        if self.alpha < 0.3:
            warnings.warn(
                "TPLStable: parameter 'alpha' is < 0.3, "
                + "count with unstable results"
            )

    def covariance_normed(self, r):
        r"""Stable normalized covariance

        .. math::
           \tilde{C}(r) =
           \exp\left(- \left(\frac{r}{\ell}\right)^{\alpha}\right)
        """
        r = np.array(np.abs(r), dtype=float)
        return np.exp(-np.power(r / self.len_scale, self.alpha))


# Matérn Model ################################################################


class Matern(CovModel):
    r"""The Matérn covariance model

    Note
    ----
    This model is given by the following normalized covariance function:

    .. math::
       \tilde{C}(r) =
       \frac{2^{1-\nu}}{\Gamma\left(\nu\right)} \cdot
       \left(\sqrt{2\nu}\cdot\frac{r}{\ell}\right)^{\nu} \cdot
       \mathrm{K}_{\nu}\left(\sqrt{2\nu}\cdot\frac{r}{\ell}\right)

    Where :math:`\Gamma` is the gamma function and :math:`\mathrm{K}_{\nu}`
    is the modified Bessel function of the second kind.

    :math:`\nu` is a shape parameter and should be >= 0.5.
    """

    def default_opt_arg(self):
        return {"nu": 1.0}

    def default_opt_arg_bounds(self):
        return {"nu": [0.5, 60.0, "cc"]}

    def check_opt_arg(self):
        if self.nu > 50.0:
            warnings.warn(
                "Mat: parameter 'nu' is > 50, "
                + "calculations most likely get unstable here"
            )

    def covariance_normed(self, r):
        r"""Matérn normalized covariance#

        .. math::
           \tilde{C}(r) =
           \frac{2^{1-\nu}}{\Gamma\left(\nu\right)} \cdot
           \left(\sqrt{2\nu}\cdot\frac{r}{\ell}\right)^{\nu} \cdot
           \mathrm{K}_{\nu}\left(\sqrt{2\nu}\cdot\frac{r}{\ell}\right)
        """
        r = np.array(np.abs(r), dtype=float)
        r_gz = r[r > 0.0]
        res = np.ones_like(r)
        with np.errstate(over="ignore", invalid="ignore"):
            res[r > 0.0] = (
                np.power(2.0, 1.0 - self.nu)
                / sps.gamma(self.nu)
                * np.power(
                    np.sqrt(2.0 * self.nu) * r_gz / self.len_scale, self.nu
                )
                * sps.kv(
                    self.nu, np.sqrt(2.0 * self.nu) * r_gz / self.len_scale
                )
            )
        # if nu >> 1 we get errors for the farfield, there 0 is approached
        res[np.logical_not(np.isfinite(res))] = 0.0
        # covariance is positiv
        res = np.maximum(res, 0.0)
        return res


class MaternRescal(CovModel):
    r"""The rescaled Matérn covariance model

    Note
    ----
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
    """

    def default_opt_arg(self):
        return {"nu": 1.0}

    def default_opt_arg_bounds(self):
        return {"nu": [0.5, 60.0, "cc"]}

    def check_opt_arg(self):
        if self.nu > 50.0:
            warnings.warn(
                "Mat: parameter 'nu' is > 50, "
                + "calculations most likely get unstable here"
            )

    def covariance_normed(self, r):
        r"""Rescaled Matérn normalized covariance

        .. math::
           \tilde{C}(r) =
           \frac{2^{1-\nu}}{\Gamma\left(\nu\right)} \cdot
           \left(\frac{\pi}{B\left(\nu,\frac{1}{2}\right)} \cdot
           \frac{r}{\ell}\right)^{\nu} \cdot
           \mathrm{K}_{\nu}\left(\frac{\pi}{B\left(\nu,\frac{1}{2}\right)}
           \cdot\frac{r}{\ell}\right)
        """
        r = np.array(np.abs(r), dtype=float)
        r_gz = r[r > 0.0]
        res = np.ones_like(r)
        with np.errstate(over="ignore", invalid="ignore"):
            res[r > 0.0] = (
                np.power(2, 1.0 - self.nu)
                / sps.gamma(self.nu)
                * np.power(
                    np.pi / sps.beta(self.nu, 0.5) * r_gz / self.len_scale,
                    self.nu,
                )
                * sps.kv(
                    self.nu,
                    np.pi / sps.beta(self.nu, 0.5) * r_gz / self.len_scale,
                )
            )
        # if nu >> 1 we get errors for the farfield, there 0 is approached
        res[np.logical_not(np.isfinite(res))] = 0.0
        # covariance is positiv
        res = np.maximum(res, 0.0)
        return res


# Bounded linear Model ########################################################


class Linear(CovModel):
    r"""The bounded linear covariance model

    Note
    ----
    This model is given by the following normalized covariance function:

    .. math::
       \tilde{C}(r) =
       \begin{cases}
       1-\frac{r}{\ell}
       & r<\ell\\
       0 & r\geq\ell
       \end{cases}
    """

    def covariance_normed(self, r):
        r"""Linear normalized covariance

        .. math::
           \tilde{C}(r) =
           \begin{cases}
           1-\frac{r}{\ell}
           & r<\ell\\
           0 & r\geq\ell
           \end{cases}
        """
        r = np.array(np.abs(r), dtype=float)
        res = np.zeros_like(r)
        res[r < self.len_scale] = 1.0 - r[r < self.len_scale] / self.len_scale
        return res


# Truncated power law #########################################################


class TPLGaussian(CovModel):
    r"""Truncated-Power-Law with Gaussian modes

    Note
    ----
    This model is given by the following variogram function:

    .. math::
       \gamma_{\ell_u}(r) =
       \intop_0^{\ell_u} \gamma(r,\ell) \frac{\rm d \ell}{\ell}

    with gaussian modes on each scale:

    .. math::
       \gamma(r,\ell) &=
       \sigma^2(\ell)\cdot\left(1-
       \exp\left(- \frac{\pi}{4} \cdot \left(\frac{r}{\ell}\right)^2\right)
       \right)\\
       \sigma^2(\ell) &= C\cdot\ell^{2H}

    This results in:

    .. math::
       \gamma(r) &=
       \sigma^2_{\ell_u}\cdot\left(1-
       H \cdot E_{1+H}\left( \frac{\pi}{4} \cdot
       \left(\frac{r}{\ell_u}\right)^2\right)
       \right) \\
       \sigma^2_{\ell_u} &= \frac{C\cdot\ell_u^{2H}}{2H}

    The following Parameters occure:

        * :math:`C>0` : The scaling factor from the Power-Law.
          This parameter will be calculated internally by the given variance.
          You can access C directly by ``model.var_raw``
        * :math:`0<H<1` : The hurst coefficient (``model.hurst``)
        * :math:`\ell_u>0` : The upper length scale for truncation
          (``model.len_scale``)
    """

    def var_factor(self):
        """Factor for C (Power-Law scale) to result in the right variance"""
        return self.len_scale ** (2 * self.hurst) / (2 * self.hurst)

    def default_opt_arg(self):
        return {"hurst": 0.5}

    def default_opt_arg_bounds(self):
        return {"hurst": [0, 1, "oo"]}

    def covariance_normed(self, r):
        r"""Truncated-Power-Law with Gaussian modes - normalized covariance

        .. math::
           \tilde{C}(r) =
           H \cdot E_{1+H}\left( \frac{\pi}{4} \cdot
           \left(\frac{r}{\ell_u}\right)^2 \right)
       """
        r = np.array(np.abs(r), dtype=float)
        r[r < 1e-8] = 0
        res = np.ones_like(r)
        res[r > 0] = self.hurst * exp_int(
            1 + self.hurst, np.pi / 4 * (r[r > 0] / self.len_scale) ** 2
        )
        return res


class TPLExponential(CovModel):
    r"""Truncated-Power-Law with Exponential modes

    Note
    ----
    This model is given by the following variogram function:

    .. math::
       \gamma_{\ell_u}(r) =
       \intop_0^{\ell_u} \gamma(r,\ell) \frac{\rm d \ell}{\ell}

    with Exponential modes on each scale:

    .. math::
       \gamma(r,\ell) &=
       \sigma^2(\ell)\cdot\left(1-
       \exp\left(- \frac{r}{\ell} \right)
       \right)\\
       \sigma^2(\ell) &= C\cdot\ell^{2H}

    This results in:

    .. math::
       \gamma(r) &=
       \sigma^2_{\ell_u}\cdot\left(1-
       2H \cdot
       E_{1+2H} \left(\frac{r}{\ell_u}\right)
       \right) \\
       \sigma^2_{\ell_u} &= \frac{C\cdot\ell_u^{2H}}{2H}

    The following Parameters occure:

        * :math:`C>0` : The scaling factor from the Power-Law.
          This parameter will be calculated internally by the given variance.
          You can access C directly by ``model.var_raw``
        * :math:`0<H<1` : The hurst coefficient (``model.hurst``)
        * :math:`\ell_u>0` : The upper length scale for truncation
          (``model.len_scale``)
    """

    def var_factor(self):
        """Factor for C (Power-Law scale) to result in the right variance"""
        return self.len_scale ** (2 * self.hurst) / (2 * self.hurst)

    def default_opt_arg(self):
        return {"hurst": 0.5}

    def default_opt_arg_bounds(self):
        return {"hurst": [0, 1, "oo"]}

    def covariance_normed(self, r):
        r"""Truncated-Power-Law with Exponential modes - normalized covariance

        .. math::
           \tilde{C}(r) = 2H \cdot E_{1+2H} \left( \frac{r}{\ell_u} \right)
       """
        r = np.array(np.abs(r), dtype=float)
        r[r < 1e-8] = 0
        res = np.ones_like(r)
        res[r > 0] = (2 * self.hurst) * exp_int(
            1 + 2 * self.hurst, r[r > 0] / self.len_scale
        )
        return res


class TPLStable(CovModel):
    r"""Truncated-Power-Law with Stable modes

    Note
    ----
    This model is given by the following variogram function:

    .. math::
       \gamma_{\ell_u}(r) =
       \intop_0^{\ell_u} \gamma(r,\ell) \frac{\rm d \ell}{\ell}

    with stable modes on each scale:

    .. math::
       \gamma(r,\ell) &=
       \sigma^2(\ell)\cdot\left(1-
       \exp\left(- \left(\frac{r}{\ell}\right)^{\alpha}\right)
       \right)\\
       \sigma^2(\ell) &= C\cdot\ell^{2H}

    This results in:

    .. math::
       \gamma(r) &=
       \sigma^2_{\ell_u}\cdot\left(1-
       \frac{2H}{\alpha} \cdot
       E_{1+\frac{2H}{\alpha}}\left(\frac{r}{\ell_u}\right)
       \right) \\
       \sigma^2_{\ell_u} &= \frac{C\cdot\ell_u^{2H}}{2H}

    The following Parameters occure:

        * :math:`0<\alpha\leq 2` : The shape parameter of the Stable model.
        * :math:`C>0` : The scaling factor from the Power-Law.
          This parameter will be calculated internally by the given variance.
          You can access C directly by ``model.var_raw``
        * :math:`0<H<1` : The hurst coefficient (``model.hurst``)
        * :math:`\ell_u>0` : The upper length scale for truncation
          (``model.len_scale``)
    """

    def var_factor(self):
        """Factor for C (Power-Law scale) to result in the right variance"""
        return self.len_scale ** (2 * self.hurst) / (2 * self.hurst)

    def default_opt_arg(self):
        return {"hurst": 0.5, "alpha": 1.5}

    def default_opt_arg_bounds(self):
        return {"hurst": [0, 1, "oo"], "alpha": [0, 2, "oc"]}

    def check_opt_arg(self):
        if self.alpha < 0.3:
            warnings.warn(
                "TPLStable: parameter 'alpha' is < 0.3, "
                + "count with unstable results"
            )

    def covariance_normed(self, r):
        r"""Truncated-Power-Law with Stable modes - normalized covariance

        .. math::
           \tilde{C}(r) =
           \frac{2H}{\alpha} \cdot
           E_{1+\frac{2H}{\alpha}}
           \left(\left(\frac{r}{\ell_u}\right)^{\alpha} \right)
       """
        r = np.array(np.abs(r), dtype=float)
        r[r < 1e-8] = 0
        res = np.ones_like(r)
        res[r > 0] = (2 * self.hurst / self.alpha) * exp_int(
            1 + 2 * self.hurst / self.alpha,
            (r[r > 0] / self.len_scale) ** self.alpha,
        )
        return res
