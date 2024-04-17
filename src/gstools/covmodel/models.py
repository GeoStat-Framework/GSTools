"""
GStools subpackage providing different covariance models.

.. currentmodule:: gstools.covmodel.models

The following classes are provided

.. autosummary::
   Gaussian
   Exponential
   Matern
   Integral
   Stable
   Rational
   Cubic
   Linear
   Circular
   Spherical
   HyperSpherical
   SuperSpherical
   JBessel
"""

# pylint: disable=C0103, E1101, R0201
import warnings

import numpy as np
from scipy import special as sps

from gstools.covmodel.base import CovModel
from gstools.covmodel.tools import AttributeWarning
from gstools.tools.special import exp_int, inc_gamma_low

__all__ = [
    "Gaussian",
    "Exponential",
    "Matern",
    "Integral",
    "Stable",
    "Rational",
    "Cubic",
    "Linear",
    "Circular",
    "Spherical",
    "HyperSpherical",
    "SuperSpherical",
    "JBessel",
]


class Gaussian(CovModel):
    r"""The Gaussian covariance model.

    Notes
    -----
    This model is given by the following variogram [Webster2007]_:

    .. math::
       \gamma(r)=
       \sigma^{2}
       \left(1-\exp\left(-\left(s\cdot\frac{r}{\ell}\right)^{2}\right)\right)+n

    Where the standard rescale factor is :math:`s=\frac{\sqrt{\pi}}{2}`.

    References
    ----------
    .. [Webster2007] Webster, R. and Oliver, M. A.
           "Geostatistics for environmental scientists.",
           John Wiley & Sons. (2007)
    """

    def cor(self, h):
        """Gaussian normalized correlation function."""
        return np.exp(-(h**2))

    def default_rescale(self):
        """Gaussian rescaling factor to result in integral scale."""
        return np.sqrt(np.pi) / 2.0

    def spectral_density(self, k):  # noqa: D102
        k = np.asarray(k, dtype=np.double)
        return (self.len_rescaled / 2.0 / np.sqrt(np.pi)) ** self.dim * np.exp(
            -((k * self.len_rescaled / 2.0) ** 2)
        )

    def spectral_rad_cdf(self, r):
        """Gaussian radial spectral cdf."""
        r = np.asarray(r, dtype=np.double)
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
        return None  # pragma: no cover

    def spectral_rad_ppf(self, u):
        """Gaussian radial spectral ppf.

        Notes
        -----
        Not defined for 3D.
        """
        u = np.asarray(u, dtype=np.double)
        if self.dim == 1:
            return 2.0 / self.len_rescaled * sps.erfinv(u)
        if self.dim == 2:
            return 2.0 / self.len_rescaled * np.sqrt(-np.log(1.0 - u))
        return None  # pragma: no cover

    def _has_cdf(self):
        return self.dim in [1, 2, 3]

    def _has_ppf(self):
        return self.dim in [1, 2]

    def calc_integral_scale(self):  # noqa: D102
        return self.len_rescaled * np.sqrt(np.pi) / 2.0


class Exponential(CovModel):
    r"""The Exponential covariance model.

    Notes
    -----
    This model is given by the following variogram [Webster2007]_:

    .. math::
       \gamma(r)=
       \sigma^{2}
       \left(1-\exp\left(-s\cdot\frac{r}{\ell}\right)\right)+n

    Where the standard rescale factor is :math:`s=1`.

    References
    ----------
    .. [Webster2007] Webster, R. and Oliver, M. A.
           "Geostatistics for environmental scientists.",
           John Wiley & Sons. (2007)
    """

    def cor(self, h):
        """Exponential normalized correlation function."""
        return np.exp(-h)

    def spectral_density(self, k):  # noqa: D102
        k = np.asarray(k, dtype=np.double)
        return (
            self.len_rescaled**self.dim
            * sps.gamma((self.dim + 1) / 2.0)
            / (np.pi * (1.0 + (k * self.len_rescaled) ** 2))
            ** ((self.dim + 1) / 2.0)
        )

    def spectral_rad_cdf(self, r):
        """Exponential radial spectral cdf."""
        r = np.asarray(r, dtype=np.double)
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
        return None  # pragma: no cover

    def spectral_rad_ppf(self, u):
        """Exponential radial spectral ppf.

        Notes
        -----
        Not defined for 3D.
        """
        u = np.asarray(u, dtype=np.double)
        if self.dim == 1:
            return np.tan(np.pi / 2 * u) / self.len_rescaled
        if self.dim == 2:
            u_power = np.divide(
                1,
                u**2,
                out=np.full_like(u, np.inf),
                where=np.logical_not(np.isclose(u, 0)),
            )
            return np.sqrt(u_power - 1.0) / self.len_rescaled
        return None  # pragma: no cover

    def _has_cdf(self):
        return self.dim in [1, 2, 3]

    def _has_ppf(self):
        return self.dim in [1, 2]

    def calc_integral_scale(self):  # noqa: D102
        return self.len_rescaled


class Stable(CovModel):
    r"""The stable covariance model.

    Notes
    -----
    This model is given by the following correlation function
    [Wackernagel2003]_:

    .. math::
       \rho(r) =
       \exp\left(- \left(s\cdot\frac{r}{\ell}\right)^{\alpha}\right)

    Where the standard rescale factor is :math:`s=1`.
    :math:`\alpha` is a shape parameter with :math:`\alpha\in(0,2]`

    References
    ----------
    .. [Wackernagel2003] Wackernagel, H. "Multivariate geostatistics",
           Springer, Berlin, Heidelberg (2003)

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
                "count with unstable results",
                AttributeWarning,
            )

    def cor(self, h):
        r"""Stable normalized correlation function."""
        return np.exp(-np.power(h, self.alpha))

    def calc_integral_scale(self):  # noqa: D102
        return self.len_rescaled * sps.gamma(1.0 + 1.0 / self.alpha)


class Matern(CovModel):
    r"""The Matérn covariance model.

    Notes
    -----
    This model is given by the following correlation function [Rasmussen2003]_:

    .. math::
       \rho(r) =
       \frac{2^{1-\nu}}{\Gamma\left(\nu\right)} \cdot
       \left(\sqrt{\nu}\cdot s\cdot\frac{r}{\ell}\right)^{\nu} \cdot
       \mathrm{K}_{\nu}\left(\sqrt{\nu}\cdot s\cdot\frac{r}{\ell}\right)

    Where the standard rescale factor is :math:`s=1`.
    :math:`\Gamma` is the gamma function and :math:`\mathrm{K}_{\nu}`
    is the modified Bessel function of the second kind.

    :math:`\nu` is a shape parameter and should be >= 0.2.

    If :math:`\nu > 20`, a gaussian model is used, since it represents
    the limiting case:

    .. math::
       \rho(r) =
       \exp\left(-\left(s\cdot\frac{r}{2\ell}\right)^2\right)

    References
    ----------
    .. [Rasmussen2003] Rasmussen, C. E.,
           "Gaussian processes in machine learning." Summer school on
           machine learning. Springer, Berlin, Heidelberg, (2003)

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

            * ``{"nu": [0.2, 30.0, "cc"]}``

        Returns
        -------
        :class:`dict`
            Boundaries for optional arguments
        """
        return {"nu": [0.2, 30.0, "cc"]}

    def cor(self, h):
        """Matérn normalized correlation function."""
        h = np.asarray(np.abs(h), dtype=np.double)
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
        # covariance is positive
        res = np.maximum(res, 0.0)
        return res

    def spectral_density(self, k):  # noqa: D102
        k = np.asarray(k, dtype=np.double)
        x = (k * self.len_rescaled) ** 2
        # for nu > 20 we just use an approximation of the gaussian model
        if self.nu > 20.0:
            return (
                (self.len_rescaled / np.sqrt(np.pi)) ** self.dim
                * np.exp(-x)
                * (1 + 0.5 * x**2 / self.nu)
                * np.sqrt(1 + x / self.nu) ** (-self.dim)
            )
        return (self.len_rescaled / np.sqrt(np.pi)) ** self.dim * np.exp(
            -(self.nu + self.dim / 2.0) * np.log(1.0 + x / self.nu)
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


class Integral(CovModel):
    r"""The Exponential Integral covariance model.

    Notes
    -----
    This model is given by the following correlation function [Mueller2021]_:

    .. math::
       \rho(r) =
       \frac{\nu}{2}\cdot
       E_{1+\frac{\nu}{2}}\left( \left( s\cdot\frac{r}{\ell} \right)^2 \right)

    Where the standard rescale factor is :math:`s=1`.
    :math:`E_s(x)` is the exponential integral.

    :math:`\nu` is a shape parameter (1 by default).

    For :math:`\nu \to \infty`, a gaussian model is approached, since it represents
    the limiting case:

    .. math::
       \rho(r) =
       \exp\left(-\left(s\cdot\frac{r}{\ell}\right)^2\right)

    References
    ----------
    .. [Mueller2021] Müller, S., Heße, F., Attinger, S., and Zech, A.,
           "The extended generalized radial flow model and effective
           conductivity for truncated power law variograms",
           Adv. Water Resour., 156, 104027, (2021)

    Other Parameters
    ----------------
    nu : :class:`float`, optional
        Shape parameter. Standard range: ``(0.0, 50]``
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

            * ``{"nu": [0.0, 50.0, "oc"]}``

        Returns
        -------
        :class:`dict`
            Boundaries for optional arguments
        """
        return {"nu": [0.0, 50.0, "oc"]}

    def cor(self, h):
        """Exponential Integral normalized correlation function."""
        h = np.asarray(h, dtype=np.double)
        return 0.5 * self.nu * exp_int(1.0 + 0.5 * self.nu, h**2)

    def spectral_density(self, k):  # noqa: D102
        k = np.asarray(k, dtype=np.double)
        fac = (0.5 * self.len_rescaled / np.sqrt(np.pi)) ** self.dim
        lim = fac * self.nu / (self.nu + self.dim)
        # for nu > 50 we just use an approximation of the gaussian model
        if self.nu > 50.0:
            x = (k * self.len_rescaled / 2) ** 2
            return lim * np.exp(-x) * (1 + 2 * x / (self.nu + self.dim + 2))
        # separate calculation at origin
        s = (self.nu + self.dim) / 2
        res = np.empty_like(k)
        k_gz = np.logical_not(np.isclose(k, 0))
        x = (k[k_gz] * self.len_rescaled / 2) ** 2
        # limit at k=0 (inc_gamma_low(s, x) / x**s -> 1/s for x -> 0)
        res[np.logical_not(k_gz)] = lim
        res[k_gz] = 0.5 * self.nu * fac / x**s * inc_gamma_low(s, x)
        return res

    def calc_integral_scale(self):  # noqa: D102
        return (
            self.len_rescaled * self.nu * np.sqrt(np.pi) / (2 * self.nu + 2.0)
        )


class Rational(CovModel):
    r"""The rational quadratic covariance model.

    Notes
    -----
    This model is given by the following correlation function [Rasmussen2003]_:

    .. math::
       \rho(r) =
       \left(1 + \frac{1}{\alpha} \cdot
       \left(s\cdot\frac{r}{\ell}\right)^2\right)^{-\alpha}

    Where the standard rescale factor is :math:`s=1`.
    :math:`\alpha` is a shape parameter and should be > 0.5.

    For :math:`\alpha\to\infty` this model converges to the Gaussian model:

    .. math::
       \rho(r)=
       \exp\left(-\left(s\cdot\frac{r}{\ell}\right)^{2}\right)

    References
    ----------
    .. [Rasmussen2003] Rasmussen, C. E.,
           "Gaussian processes in machine learning." Summer school on
           machine learning. Springer, Berlin, Heidelberg, (2003)

    Other Parameters
    ----------------
    alpha : :class:`float`, optional
        Shape parameter. Standard range: ``[0.5, 50]``
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

            * ``{"alpha": [0.5, 50.0]}``

        Returns
        -------
        :class:`dict`
            Boundaries for optional arguments
        """
        return {"alpha": [0.5, 50.0]}

    def cor(self, h):
        """Rational normalized correlation function."""
        return np.power(1 + h**2 / self.alpha, -self.alpha)

    def calc_integral_scale(self):  # noqa: D102
        return (
            self.len_rescaled
            * np.sqrt(np.pi * self.alpha)
            * sps.gamma(self.alpha - 0.5)
            / sps.gamma(self.alpha)
            / 2.0
        )


class Cubic(CovModel):
    r"""The Cubic covariance model.

    A model with reverse curvature near the origin and a finite range of
    correlation.

    Notes
    -----
    This model is given by the following correlation function [Chiles2009]_:

    .. math::
       \rho(r) =
       \begin{cases}
       1- 7 \left(s\cdot\frac{r}{\ell}\right)^{2}
       + \frac{35}{4} \left(s\cdot\frac{r}{\ell}\right)^{3}
       - \frac{7}{2} \left(s\cdot\frac{r}{\ell}\right)^{5}
       + \frac{3}{4} \left(s\cdot\frac{r}{\ell}\right)^{7}
       & r<\frac{\ell}{s}\\
       0 & r\geq\frac{\ell}{s}
       \end{cases}

    Where the standard rescale factor is :math:`s=1`.

    References
    ----------
    .. [Chiles2009] Chiles, J. P., & Delfiner, P.,
           "Geostatistics: modeling spatial uncertainty" (Vol. 497),
           John Wiley & Sons. (2009)
    """

    def cor(self, h):
        """Spherical normalized correlation function."""
        h = np.minimum(np.abs(h, dtype=np.double), 1.0)
        return 1.0 - 7 * h**2 + 8.75 * h**3 - 3.5 * h**5 + 0.75 * h**7


class Linear(CovModel):
    r"""The bounded linear covariance model.

    This model is derived from the relative intersection area of
    two lines in 1D, where the middle points have a distance of :math:`r`
    and the line lengths are :math:`\ell`.

    Notes
    -----
    This model is given by the following correlation function [Webster2007]_:

    .. math::
       \rho(r) =
       \begin{cases}
       1-s\cdot\frac{r}{\ell} & r<\frac{\ell}{s}\\
       0 & r\geq\frac{\ell}{s}
       \end{cases}

    Where the standard rescale factor is :math:`s=1`.

    References
    ----------
    .. [Webster2007] Webster, R. and Oliver, M. A.
           "Geostatistics for environmental scientists.",
           John Wiley & Sons. (2007)
    """

    def cor(self, h):
        """Linear normalized correlation function."""
        return np.maximum(1 - np.abs(h, dtype=np.double), 0.0)

    def check_dim(self, dim):
        """Linear model is only valid in 1D."""
        return dim < 2


class Circular(CovModel):
    r"""The circular covariance model.

    This model is derived as the relative intersection area of
    two discs in 2D, where the middle points have a distance of :math:`r`
    and the diameters are given by :math:`\ell`.

    Notes
    -----
    This model is given by the following correlation function [Webster2007]_:

    .. math::
       \rho(r) =
       \begin{cases}
       \frac{2}{\pi}\cdot
       \left(
       \cos^{-1}\left(s\cdot\frac{r}{\ell}\right) -
       s\cdot\frac{r}{\ell}\cdot\sqrt{1-\left(s\cdot\frac{r}{\ell}\right)^{2}}
       \right)
       & r<\frac{\ell}{s}\\
       0 & r\geq\frac{\ell}{s}
       \end{cases}

    Where the standard rescale factor is :math:`s=1`.

    References
    ----------
    .. [Webster2007] Webster, R. and Oliver, M. A.
           "Geostatistics for environmental scientists.",
           John Wiley & Sons. (2007)
    """

    def cor(self, h):
        """Circular normalized correlation function."""
        h = np.asarray(np.abs(h), dtype=np.double)
        res = np.zeros_like(h)
        # arccos is instable around h=1
        h_l1 = h < 1.0
        h_low = h[h_l1]
        res[h_l1] = (
            2 / np.pi * (np.arccos(h_low) - h_low * np.sqrt(1 - h_low**2))
        )
        return res

    def check_dim(self, dim):
        """Circular model is only valid in 1D and 2D."""
        return dim < 3


class Spherical(CovModel):
    r"""The Spherical covariance model.

    This model is derived from the relative intersection area of
    two spheres in 3D, where the middle points have a distance of :math:`r`
    and the diameters are given by :math:`\ell`.

    Notes
    -----
    This model is given by the following correlation function [Webster2007]_:

    .. math::
       \rho(r) =
       \begin{cases}
       1-\frac{3}{2}\cdot s\cdot\frac{r}{\ell} +
       \frac{1}{2}\cdot\left(s\cdot\frac{r}{\ell}\right)^{3}
       & r<\frac{\ell}{s}\\
       0 & r\geq\frac{\ell}{s}
       \end{cases}

    Where the standard rescale factor is :math:`s=1`.

    References
    ----------
    .. [Webster2007] Webster, R. and Oliver, M. A.
           "Geostatistics for environmental scientists.",
           John Wiley & Sons. (2007)
    """

    def cor(self, h):
        """Spherical normalized correlation function."""
        h = np.minimum(np.abs(h, dtype=np.double), 1.0)
        return 1.0 - 1.5 * h + 0.5 * h**3

    def check_dim(self, dim):
        """Spherical model is only valid in 1D, 2D and 3D."""
        return dim < 4


class HyperSpherical(CovModel):
    r"""The Hyper-Spherical covariance model.

    This model is derived from the relative intersection area of
    two d-dimensional hyperspheres,
    where the middle points have a distance of :math:`r`
    and the diameters are given by :math:`\ell`.

    In 1D this is the Linear model, in 2D the Circular model
    and in 3D the Spherical model.

    Notes
    -----
    This model is given by the following correlation function [Matern1960]_:

    .. math::
       \rho(r) =
       \begin{cases}
       1-s\cdot\frac{r}{\ell}\cdot\frac{
       _{2}F_{1}\left(\frac{1}{2},-\frac{d-1}{2},\frac{3}{2},
       \left(s\cdot\frac{r}{\ell}\right)^{2}\right)}
       {_{2}F_{1}\left(\frac{1}{2},-\frac{d-1}{2},\frac{3}{2},1\right)}
       & r<\frac{\ell}{s}\\
       0 & r\geq\frac{\ell}{s}
       \end{cases}

    Where the standard rescale factor is :math:`s=1`.
    :math:`d` is the dimension.

    References
    ----------
    .. [Matern1960] Matern B., "Spatial Variation",
           Swedish National Institute for Forestry Research, (1960)
    """

    def cor(self, h):
        """Hyper-Spherical normalized correlation function."""
        h = np.asarray(h, dtype=np.double)
        res = np.zeros_like(h)
        h_l1 = h < 1
        nu = (self.dim - 1.0) / 2.0
        fac = 1.0 / sps.hyp2f1(0.5, -nu, 1.5, 1)
        res[h_l1] = 1 - h[h_l1] * fac * sps.hyp2f1(0.5, -nu, 1.5, h[h_l1] ** 2)
        return res

    def spectral_density(self, k):  # noqa: D102
        k = np.asarray(k, dtype=np.double)
        res = np.empty_like(k)
        kl = k * self.len_rescaled
        kl_gz = np.logical_not(np.isclose(k, 0))
        res[kl_gz] = sps.gamma(self.dim / 2 + 1) / np.sqrt(np.pi) ** self.dim
        res[kl_gz] *= sps.jv(self.dim / 2, kl[kl_gz] / 2) ** 2
        res[kl_gz] /= k[kl_gz] ** self.dim
        res[np.logical_not(kl_gz)] = (
            (self.len_rescaled / 4) ** self.dim
            / sps.gamma(self.dim / 2 + 1)
            / np.sqrt(np.pi) ** self.dim
        )
        return res


class SuperSpherical(CovModel):
    r"""The Super-Spherical covariance model.

    This model is derived from the relative intersection area of
    two d-dimensional hyperspheres,
    where the middle points have a distance of :math:`r`
    and the diameters are given by :math:`\ell`.
    It is than valid in all lower dimensions.
    By default it coincides with the Hyper-Spherical model.

    Notes
    -----
    This model is given by the following correlation function [Matern1960]_:

    .. math::
       \rho(r) =
       \begin{cases}
       1-s\cdot\frac{r}{\ell}\cdot\frac{
       _{2}F_{1}\left(\frac{1}{2},-\nu,\frac{3}{2},
       \left(s\cdot\frac{r}{\ell}\right)^{2}\right)}
       {_{2}F_{1}\left(\frac{1}{2},-\nu,\frac{3}{2},1\right)}
       & r<\frac{\ell}{s}\\
       0 & r\geq\frac{\ell}{s}
       \end{cases}

    Where the standard rescale factor is :math:`s=1`.
    :math:`\nu\geq\frac{d-1}{2}` is a shape parameter.

    References
    ----------
    .. [Matern1960] Matern B., "Spatial Variation",
           Swedish National Institute for Forestry Research, (1960)

    Other Parameters
    ----------------
    nu : :class:`float`, optional
        Shape parameter. Standard range: ``[(dim-1)/2, 50]``
        Default: ``(dim-1)/2``
    """

    def default_opt_arg(self):
        """Defaults for the optional arguments.

            * ``{"nu": (dim-1)/2}``

        Returns
        -------
        :class:`dict`
            Defaults for optional arguments
        """
        return {"nu": (self.dim - 1) / 2}

    def default_opt_arg_bounds(self):
        """Defaults for boundaries of the optional arguments.

            * ``{"nu": [(dim-1)/2, 50.0]}``

        Returns
        -------
        :class:`dict`
            Boundaries for optional arguments
        """
        return {"nu": [(self.dim - 1) / 2, 50.0]}

    def cor(self, h):
        """Super-Spherical normalized correlation function."""
        h = np.asarray(h, dtype=np.double)
        res = np.zeros_like(h)
        h_l1 = h < 1
        fac = 1.0 / sps.hyp2f1(0.5, -self.nu, 1.5, 1.0)
        res[h_l1] = 1.0 - h[h_l1] * fac * sps.hyp2f1(
            0.5, -self.nu, 1.5, h[h_l1] ** 2
        )
        return res


class JBessel(CovModel):
    r"""The J-Bessel hole model.

    This covariance model is a valid hole model, meaning it has areas
    of negative correlation but a valid spectral density.

    Notes
    -----
    This model is given by the following correlation function [Chiles2009]_:

    .. math::
       \rho(r) =
       \Gamma(\nu+1) \cdot
       \frac{\mathrm{J}_{\nu}\left(s\cdot\frac{r}{\ell}\right)}
       {\left(s\cdot\frac{r}{2\ell}\right)^{\nu}}

    Where the standard rescale factor is :math:`s=1`.
    :math:`\Gamma` is the gamma function and :math:`\mathrm{J}_{\nu}`
    is the Bessel functions of the first kind.
    :math:`\nu\geq\frac{d}{2}-1` is a shape parameter,
    which defaults to :math:`\nu=\frac{d}{2}`,
    since the spectrum of the model gets instable for
    :math:`\nu\to\frac{d}{2}-1`.

    For :math:`\nu=\frac{1}{2}` (valid in d=1,2,3)
    we get the so-called 'Wave' model:

    .. math::
       \rho(r) =
       \frac{\sin\left(s\cdot\frac{r}{\ell}\right)}{s\cdot\frac{r}{\ell}}

    References
    ----------
    .. [Chiles2009] Chiles, J. P., & Delfiner, P.,
           "Geostatistics: modeling spatial uncertainty" (Vol. 497),
           John Wiley & Sons. (2009)

    Other Parameters
    ----------------
    nu : :class:`float`, optional
        Shape parameter. Standard range: ``[dim/2 - 1, 50]``
        Default: ``dim/2``
    """

    def default_opt_arg(self):
        """Defaults for the optional arguments.

            * ``{"nu": dim/2}``

        Returns
        -------
        :class:`dict`
            Defaults for optional arguments
        """
        return {"nu": self.dim / 2}

    def default_opt_arg_bounds(self):
        """Defaults for boundaries of the optional arguments.

            * ``{"nu": [dim/2 - 1, 50.0]}``

        Returns
        -------
        :class:`dict`
            Boundaries for optional arguments
        """
        return {"nu": [self.dim / 2 - 1, 50.0]}

    def check_opt_arg(self):
        """Check the optional arguments.

        Warns
        -----
        nu
            If nu is close to dim/2 - 1, the model tends to get unstable.
        """
        if abs(self.nu - self.dim / 2 + 1) < 0.01:
            warnings.warn(
                "JBessel: parameter 'nu' is close to d/2-1, "
                "count with unstable results",
                AttributeWarning,
            )

    def cor(self, h):
        """J-Bessel correlation."""
        h = np.asarray(h, dtype=np.double)
        h_gz = np.logical_not(np.isclose(h, 0))
        hh = h[h_gz]
        res = np.ones_like(h)
        nu = self.nu
        res[h_gz] = sps.gamma(nu + 1) * sps.jv(nu, hh) / (hh / 2.0) ** nu
        return res

    def spectral_density(self, k):  # noqa: D102
        k = np.asarray(k, dtype=np.double)
        k_ll = k < 1.0 / self.len_rescaled
        kk = k[k_ll]
        res = np.zeros_like(k)
        # the model is degenerated for nu=d/2-1, so we tweak the spectral pdf
        # and cut of the divisor at nu-(d/2-1)=0.01 (gamma(0.01) about 100)
        res[k_ll] = (
            (self.len_rescaled / np.sqrt(np.pi)) ** self.dim
            * sps.gamma(self.nu + 1.0)
            / np.minimum(sps.gamma(self.nu - self.dim / 2 + 1), 100.0)
            * (1.0 - (kk * self.len_rescaled) ** 2) ** (self.nu - self.dim / 2)
        )
        return res
