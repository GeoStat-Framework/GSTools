# -*- coding: utf-8 -*-
"""
GStools subpackage providing different covariance models.

.. currentmodule:: gstools.covmodel.models

The following classes and functions are provided

.. autosummary::
   Gaussian
   Exponential
   Matern
   Stable
   Rational
   Linear
   Circular
   Spherical
   Intersection
"""
# pylint: disable=C0103, E1101, E1137

import warnings
import numpy as np
from scipy import special as sps
from gstools.covmodel import CovModel

__all__ = [
    "Gaussian",
    "Exponential",
    "Matern",
    "Stable",
    "Rational",
    "Linear",
    "Circular",
    "Spherical",
    "Intersection",
]


# Gaussian Model ##############################################################


class Gaussian(CovModel):
    r"""The Gaussian covariance model.

    Notes
    -----
    This model is given by the following variogram:

    .. math::
       \gamma(r)=
       \sigma^{2}
       \left(1-\exp\left(-\left(s\cdot\frac{r}{\ell}\right)^{2}\right)\right)+n

    Where the standard rescale factor is :math:`s=\frac{\sqrt{\pi}}{2}`.
    """

    def cor(self, h):
        """Gaussian normalized correlation function."""
        return np.exp(-(h ** 2))

    def default_rescale(self):
        """Gaussian rescaling factor to result in integral scale."""
        return np.sqrt(np.pi) / 2.0

    def spectral_density(self, k):  # noqa: D102
        k = np.array(k, dtype=np.double)
        return (self.len_rescaled / 2.0 / np.sqrt(np.pi)) ** self.dim * np.exp(
            -((k * self.len_rescaled / 2.0) ** 2)
        )

    def spectral_rad_cdf(self, r):
        """Gaussian radial spectral cdf."""
        r = np.array(r, dtype=np.double)
        if self.dim == 1:
            return sps.erf(r * self.len_rescaled / 2.0)
        if self.dim == 2:
            return 1.0 - np.exp(-((r * self.len_rescaled / 2.0) ** 2))
        if self.dim == 3:
            return sps.erf(
                r * self.len_rescaled / 2.0
            ) - r * self.len_rescaled / np.sqrt(np.pi) * np.exp(
                -((r * self.len_rescaled / 2.0) ** 2)
            )

    def spectral_rad_ppf(self, u):
        """Gaussian radial spectral ppf.

        Notes
        -----
        Not defined for 3D.
        """
        u = np.array(u, dtype=np.double)
        if self.dim == 1:
            return 2.0 / self.len_rescaled * sps.erfinv(u)
        if self.dim == 2:
            return 2.0 / self.len_rescaled * np.sqrt(-np.log(1.0 - u))

    def _has_cdf(self):
        return self.dim in [1, 2, 3]

    def _has_ppf(self):
        return self.dim in [1, 2]

    def calc_integral_scale(self):  # noqa: D102
        return self.len_rescaled * np.sqrt(np.pi) / 2.0


# Exponential Model ###########################################################


class Exponential(CovModel):
    r"""The Exponential covariance model.

    Notes
    -----
    This model is given by the following variogram:

    .. math::
       \gamma(r)=
       \sigma^{2}
       \left(1-\exp\left(-s\cdot\frac{r}{\ell}\right)\right)+n

    Where the standard rescale factor is :math:`s=1`.
    """

    def cor(self, h):
        """Exponential normalized correlation function."""
        return np.exp(-h)

    def spectral_density(self, k):  # noqa: D102
        k = np.array(k, dtype=np.double)
        return (
            self.len_rescaled ** self.dim
            * sps.gamma((self.dim + 1) / 2.0)
            / (np.pi * (1.0 + (k * self.len_rescaled) ** 2))
            ** ((self.dim + 1) / 2.0)
        )

    def spectral_rad_cdf(self, r):
        """Exponential radial spectral cdf."""
        r = np.array(r, dtype=np.double)
        if self.dim == 1:
            return np.arctan(r * self.len_rescaled) * 2.0 / np.pi
        if self.dim == 2:
            return 1.0 - 1.0 / np.sqrt(1.0 + (r * self.len_rescaled) ** 2)
        if self.dim == 3:
            return (
                (
                    np.arctan(r * self.len_rescaled)
                    - r
                    * self.len_rescaled
                    / (1.0 + (r * self.len_rescaled) ** 2)
                )
                * 2.0
                / np.pi
            )
        return None

    def spectral_rad_ppf(self, u):
        """Exponential radial spectral ppf.

        Notes
        -----
        Not defined for 3D.
        """
        u = np.array(u, dtype=np.double)
        if self.dim == 1:
            return np.tan(np.pi / 2 * u) / self.len_rescaled
        if self.dim == 2:
            u_power = np.divide(
                1,
                u ** 2,
                out=np.full_like(u, np.inf),
                where=np.logical_not(np.isclose(u, 0)),
            )
            return np.sqrt(u_power - 1.0) / self.len_rescaled
        return None

    def _has_cdf(self):
        return self.dim in [1, 2, 3]

    def _has_ppf(self):
        return self.dim in [1, 2]

    def calc_integral_scale(self):  # noqa: D102
        return self.len_rescaled


# Rational Model ##############################################################


class Rational(CovModel):
    r"""The rational quadratic covariance model.

    Notes
    -----
    This model is given by the following correlation function:

    .. math::
       \rho(r) =
       \left(1 + \frac{1}{2\alpha} \cdot
       \left(s\cdot\frac{r}{\ell}\right)^2\right)^{-\alpha}

    Where the standard rescale factor is :math:`s=1`.
    :math:`\alpha` is a shape parameter and should be > 0.5.

    Other Parameters
    ----------------
    alpha : :class:`float`, optional
        Shape parameter. Standard range: ``(0, inf)``
        Default: ``1.0``
    """

    def default_opt_arg(self):
        """Defaults for the optional arguments.

            * ``{"alpha": 1.0}``

        Returns
        -------
        :class:`dict`
            Defaults for optional arguments
        """
        return {"alpha": 1.0}

    def default_opt_arg_bounds(self):
        """Defaults for boundaries of the optional arguments.

            * ``{"alpha": [0.5, inf]}``

        Returns
        -------
        :class:`dict`
            Boundaries for optional arguments
        """
        return {"alpha": [0.5, np.inf]}

    def cor(self, h):
        """Rational normalized correlation function."""
        return np.power(1 + 0.5 / self.alpha * h ** 2, -self.alpha)

    def calc_integral_scale(self):  # noqa: D102
        return (
            self.len_rescaled
            * np.sqrt(np.pi * self.alpha * 0.5)
            * sps.gamma(self.alpha - 0.5)
            / sps.gamma(self.alpha)
        )


# Stable Model ################################################################


class Stable(CovModel):
    r"""The stable covariance model.

    Notes
    -----
    This model is given by the following correlation function:

    .. math::
       \rho(r) =
       \exp\left(- \left(s\cdot\frac{r}{\ell}\right)^{\alpha}\right)

    Where the standard rescale factor is :math:`s=1`.
    :math:`\alpha` is a shape parameter with :math:`\alpha\in(0,2]`

    Other Parameters
    ----------------
    alpha : :class:`float`, optional
        Shape parameter. Standard range: ``(0, 2]``
        Default: ``1.5``
    """

    def default_opt_arg(self):
        """Defaults for the optional arguments.

            * ``{"alpha": 1.5}``

        Returns
        -------
        :class:`dict`
            Defaults for optional arguments
        """
        return {"alpha": 1.5}

    def default_opt_arg_bounds(self):
        """Defaults for boundaries of the optional arguments.

            * ``{"alpha": [0, 2, "oc"]}``

        Returns
        -------
        :class:`dict`
            Boundaries for optional arguments
        """
        return {"alpha": [0, 2, "oc"]}

    def check_opt_arg(self):
        """Check the optional arguments.

        Warns
        -----
        alpha
            If alpha is < 0.3, the model tends to a nugget model and gets
            numerically unstable.
        """
        if self.alpha < 0.3:
            warnings.warn(
                "Stable: parameter 'alpha' is < 0.3, "
                + "count with unstable results"
            )

    def cor(self, h):
        r"""Stable normalized correlation function."""
        return np.exp(-np.power(h, self.alpha))

    def calc_integral_scale(self):  # noqa: D102
        return self.len_rescaled * sps.gamma(1.0 + 1.0 / self.alpha)


# Matérn Model ################################################################


class Matern(CovModel):
    r"""The Matérn covariance model.

    Notes
    -----
    This model is given by the following correlation function:

    .. math::
       \rho(r) =
       \frac{2^{1-\nu}}{\Gamma\left(\nu\right)} \cdot
       \left(\sqrt{\nu}\cdot s\cdot\frac{r}{\ell}\right)^{\nu} \cdot
       \mathrm{K}_{\nu}\left(\sqrt{\nu}\cdot s\cdot\frac{r}{\ell}\right)

    Where the standard rescale factor is :math:`s=1`.
    :math:`\Gamma` is the gamma function and :math:`\mathrm{K}_{\nu}`
    is the modified Bessel function of the second kind.

    :math:`\nu` is a shape parameter and should be >= 0.2.

    If :math:`\nu > 20`, a gaussian model is used, since it is the limit
    case:

    .. math::
       \rho(r) =
       \exp\left(-\left(\frac{r}{2\ell}\right)^2\right)

    Other Parameters
    ----------------
    nu : :class:`float`, optional
        Shape parameter. Standard range: ``[0.2, 30]``
        Default: ``1.0``
    """

    def default_opt_arg(self):
        """Defaults for the optional arguments.

            * ``{"nu": 1.0}``

        Returns
        -------
        :class:`dict`
            Defaults for optional arguments
        """
        return {"nu": 1.0}

    def default_opt_arg_bounds(self):
        """Defaults for boundaries of the optional arguments.

            * ``{"nu": [0.5, 30.0, "cc"]}``

        Returns
        -------
        :class:`dict`
            Boundaries for optional arguments
        """
        return {"nu": [0.2, 30.0, "cc"]}

    def cor(self, h):
        """Matérn normalized correlation function."""
        h = np.array(np.abs(h), dtype=np.double)
        # for nu > 20 we just use the gaussian model
        if self.nu > 20.0:
            return np.exp(-((h / 2.0) ** 2))
        # calculate by log-transformation to prevent numerical errors
        h_gz = h[h > 0.0]
        res = np.ones_like(h)
        res[h > 0.0] = np.exp(
            (1.0 - self.nu) * np.log(2)
            - sps.loggamma(self.nu)
            + self.nu * np.log(np.sqrt(self.nu) * h_gz)
        ) * sps.kv(self.nu, np.sqrt(self.nu) * h_gz)
        # if nu >> 1 we get errors for the farfield, there 0 is approached
        res[np.logical_not(np.isfinite(res))] = 0.0
        # covariance is positiv
        res = np.maximum(res, 0.0)
        return res

    def spectral_density(self, k):  # noqa: D102
        k = np.array(k, dtype=np.double)
        # for nu > 20 we just use an approximation of the gaussian model
        if self.nu > 20.0:
            return (
                (self.len_rescaled / np.sqrt(np.pi)) ** self.dim
                * np.exp(-((k * self.len_rescaled) ** 2))
                * (
                    1
                    + (
                        ((k * self.len_rescaled) ** 2 - self.dim / 2.0) ** 2
                        - self.dim / 2.0
                    )
                    / self.nu
                )
            )
        return (self.len_rescaled / np.sqrt(np.pi)) ** self.dim * np.exp(
            -(self.nu + self.dim / 2.0)
            * np.log(1.0 + (k * self.len_rescaled) ** 2 / self.nu)
            + sps.loggamma(self.nu + self.dim / 2.0)
            - sps.loggamma(self.nu)
            - self.dim * np.log(np.sqrt(self.nu))
        )

    def calc_integral_scale(self):  # noqa: D102
        return (
            self.len_rescaled
            * np.pi
            / np.sqrt(self.nu)
            / sps.beta(self.nu, 0.5)
        )


# Bounded linear Model ########################################################


class Linear(CovModel):
    r"""The bounded linear covariance model.

    This model is derived from the relative intersection area of
    two lines in 1D, where the middle points have a distance of :math:`r`
    and the line lengths are :math:`\ell`.

    Notes
    -----
    This model is given by the following correlation function:

    .. math::
       \rho(r) =
       \begin{cases}
       1-\frac{r}{\ell}
       & r<\ell\\
       0 & r\geq\ell
       \end{cases}
    """

    def correlation(self, r):
        r"""Linear correlation function.

        .. math::
           \rho(r) =
           \begin{cases}
           1-\frac{r}{\ell}
           & r<\ell\\
           0 & r\geq\ell
           \end{cases}
        """
        r = np.array(np.abs(r), dtype=np.double)
        res = np.zeros_like(r)
        r_ll = r < self.len_scale
        r_low = r[r_ll]
        res[r_ll] = 1.0 - r_low / self.len_scale
        return res


# Circular Model ##############################################################


class Circular(CovModel):
    r"""The circular covariance model.

    This model is derived as the relative intersection area of
    two discs in 2D, where the middle points have a distance of :math:`r`
    and the diameters are given by :math:`\ell`.

    Notes
    -----
    This model is given by the following correlation function:

    .. math::
       \rho(r) =
       \begin{cases}
       \frac{2}{\pi}\cdot\left(
       \cos^{-1}\left(\frac{r}{\ell}\right) -
       \frac{r}{\ell}\cdot\sqrt{1-\left(\frac{r}{\ell}\right)^{2}}
       \right)
       & r<\ell\\
       0 & r\geq\ell
       \end{cases}
    """

    def correlation(self, r):
        r"""Circular correlation function.

        .. math::
           \rho(r) =
           \begin{cases}
           \frac{2}{\pi}\cdot\left(
           \cos^{-1}\left(\frac{r}{\ell}\right) -
           \frac{r}{\ell}\cdot\sqrt{1-\left(\frac{r}{\ell}\right)^{2}}
           \right)
           & r<\ell\\
           0 & r\geq\ell
           \end{cases}

        """
        r = np.array(np.abs(r), dtype=np.double)
        res = np.zeros_like(r)
        r_ll = r < self.len_scale
        r_low = r[r_ll]
        res[r_ll] = (
            2
            / np.pi
            * (
                np.arccos(r_low / self.len_scale)
                - r_low
                / self.len_scale
                * np.sqrt(1 - (r_low / self.len_scale) ** 2)
            )
        )
        return res


# Spherical Model #############################################################


class Spherical(CovModel):
    r"""The Spherical covariance model.

    This model is derived from the relative intersection area of
    two spheres in 3D, where the middle points have a distance of :math:`r`
    and the diameters are given by :math:`\ell`.

    Notes
    -----
    This model is given by the following correlation function:

    .. math::
       \rho(r) =
       \begin{cases}
       1-\frac{3}{2}\cdot\frac{r}{\ell} +
       \frac{1}{2}\cdot\left(\frac{r}{\ell}\right)^{3}
       & r<\ell\\
       0 & r\geq\ell
       \end{cases}
    """

    def correlation(self, r):
        r"""Spherical correlation function.

        .. math::
           \rho(r) =
           \begin{cases}
           1-\frac{3}{2}\cdot\frac{r}{\ell} +
           \frac{1}{2}\cdot\left(\frac{r}{\ell}\right)^{3}
           & r<\ell\\
           0 & r\geq\ell
           \end{cases}
        """
        r = np.array(np.abs(r), dtype=np.double)
        res = np.zeros_like(r)
        r_ll = r < self.len_scale
        r_low = r[r_ll]
        res[r_ll] = (
            1.0
            - 3.0 / 2.0 * r_low / self.len_scale
            + 1.0 / 2.0 * (r_low / self.len_scale) ** 3
        )
        return res


class Intersection(CovModel):
    r"""The Intersection covariance model.

    This model is derived from the relative intersection area of
    two d-dimensional spheres,
    where the middle points have a distance of :math:`r`
    and the diameters are given by :math:`\ell`.

    In 1D this is the Linear model, in 2D this is the Circular model
    and in 3D this is the Spherical model.

    Notes
    -----
    This model is given by the following correlation functions.

    In 1D:

    .. math::
       \rho(r) =
       \begin{cases}
       1-\frac{r}{\ell}
       & r<\ell\\
       0 & r\geq\ell
       \end{cases}

    In 2D:

    .. math::
       \rho(r) =
       \begin{cases}
       \frac{2}{\pi}\cdot\left(
       \cos^{-1}\left(\frac{r}{\ell}\right) -
       \frac{r}{\ell}\cdot\sqrt{1-\left(\frac{r}{\ell}\right)^{2}}
       \right)
       & r<\ell\\
       0 & r\geq\ell
       \end{cases}

    In >=3D:

    .. math::
       \rho(r) =
       \begin{cases}
       1-\frac{3}{2}\cdot\frac{r}{\ell} +
       \frac{1}{2}\cdot\left(\frac{r}{\ell}\right)^{3}
       & r<\ell\\
       0 & r\geq\ell
       \end{cases}
    """

    def correlation(self, r):  # noqa: D102
        r"""
        Intersection correlation function.

        In 1D:

        .. math::
           \rho(r) =
           \begin{cases}
           1-\frac{r}{\ell}
           & r<\ell\\
           0 & r\geq\ell
           \end{cases}

        In 2D:

        .. math::
           \rho(r) =
           \begin{cases}
           \frac{2}{\pi}\cdot\left(
           \cos^{-1}\left(\frac{r}{\ell}\right) -
           \frac{r}{\ell}\cdot\sqrt{1-\left(\frac{r}{\ell}\right)^{2}}
           \right)
           & r<\ell\\
           0 & r\geq\ell
           \end{cases}

        In >=3D:

        .. math::
           \rho(r) =
           \begin{cases}
           1-\frac{3}{2}\cdot\frac{r}{\ell} +
           \frac{1}{2}\cdot\left(\frac{r}{\ell}\right)^{3}
           & r<\ell\\
           0 & r\geq\ell
           \end{cases}
        """
        r = np.array(np.abs(r), dtype=np.double)
        res = np.zeros_like(r)
        r_ll = r < self.len_scale
        r_low = r[r_ll]
        if self.dim == 1:
            res[r_ll] = 1.0 - r_low / self.len_scale
        elif self.dim == 2:
            res[r_ll] = (
                2
                / np.pi
                * (
                    np.arccos(r_low / self.len_scale)
                    - r_low
                    / self.len_scale
                    * np.sqrt(1 - (r_low / self.len_scale) ** 2)
                )
            )
        else:
            res[r_ll] = (
                1.0
                - 3.0 / 2.0 * r_low / self.len_scale
                + 1.0 / 2.0 * (r_low / self.len_scale) ** 3
            )
        return res

    def spectral_density(self, k):  # noqa: D102
        k = np.array(k, dtype=np.double)
        res = np.empty_like(k)
        kl = k * self.len_scale
        kl_gz = kl > 0
        # for k=0 we calculate the limit by hand
        if self.dim == 1:
            res[kl_gz] = (1.0 - np.cos(kl[kl_gz])) / (
                np.pi * k[kl_gz] * kl[kl_gz]
            )
            res[np.logical_not(kl_gz)] = self.len_scale / 2.0 / np.pi
        elif self.dim == 2:
            res[kl_gz] = sps.j1(kl[kl_gz] / 2.0) ** 2 / np.pi / k[kl_gz] ** 2
            res[np.logical_not(kl_gz)] = self.len_scale ** 2 / 16.0 / np.pi
        else:
            res[kl_gz] = -(
                12 * kl[kl_gz] * np.sin(kl[kl_gz])
                + (12 - 3 * kl[kl_gz] ** 2) * np.cos(kl[kl_gz])
                - 3 * kl[kl_gz] ** 2
                - 12
            ) / (2 * np.pi ** 2 * kl[kl_gz] ** 3 * k[kl_gz] ** 3)
            res[np.logical_not(kl_gz)] = (
                self.len_scale ** 3 / 48.0 / np.pi ** 2
            )
        return res
