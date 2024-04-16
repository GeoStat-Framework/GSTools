"""
GStools subpackage providing truncated power law covariance models.

.. currentmodule:: gstools.covmodel.tpl_models

The following classes and functions are provided

.. autosummary::
   TPLGaussian
   TPLExponential
   TPLStable
   TPLSimple
"""

# pylint: disable=C0103, E1101
import warnings

import numpy as np

from gstools.covmodel.base import CovModel
from gstools.covmodel.tools import AttributeWarning
from gstools.tools.special import (
    tpl_exp_spec_dens,
    tpl_gau_spec_dens,
    tplstable_cor,
)

__all__ = ["TPLGaussian", "TPLExponential", "TPLStable", "TPLSimple"]


class TPLCovModel(CovModel):
    """Truncated-Power-Law Covariance Model base class for super-position."""

    @property
    def len_up(self):
        """:class:`float`: Upper length scale truncation of the model.

        * ``len_up = len_low + len_scale``
        """
        return self.len_low + self.len_scale

    @property
    def len_up_rescaled(self):
        """:class:`float`: Upper length scale truncation rescaled.

        * ``len_up_rescaled = (len_low + len_scale) / rescale``
        """
        return self.len_up / self.rescale

    @property
    def len_low_rescaled(self):
        """:class:`float`: Lower length scale truncation rescaled.

        * ``len_low_rescaled = len_low / rescale``
        """
        return self.len_low / self.rescale

    def var_factor(self):
        """Factor for C (intensity of variation) to result in variance."""
        return (
            self.len_up_rescaled ** (2 * self.hurst)
            - self.len_low_rescaled ** (2 * self.hurst)
        ) / (2 * self.hurst)

    def cor(self, h):
        """TPL - normalized correlation function."""

    def correlation(self, r):
        """TPL - correlation function."""


# Truncated power law #########################################################


class TPLGaussian(TPLCovModel):
    r"""Truncated-Power-Law with Gaussian modes.

    Notes
    -----
    The truncated power law is given by a superposition of scale-dependent
    variograms [Federico1997]_:

    .. math::
       \gamma_{\ell_{\mathrm{low}},\ell_{\mathrm{up}}}(r) =
       \intop_{\ell_{\mathrm{low}}}^{\ell_{\mathrm{up}}}
       \gamma(r,\lambda) \frac{\rm d \lambda}{\lambda}

    with `Gaussian` modes on each scale:

    .. math::
       \gamma(r,\lambda) &=
       \sigma^2(\lambda)\cdot\left(1-
       \exp\left[- \left(\frac{r}{\lambda}\right)^{2}\right]
       \right)\\
       \sigma^2(\lambda) &= C\cdot\lambda^{2H}

    This results in:

    .. math::
       \gamma_{\ell_{\mathrm{low}},\ell_{\mathrm{up}}}(r) &=
       \sigma^2_{\ell_{\mathrm{low}},\ell_{\mathrm{up}}}\cdot\left(1-
       H \cdot
       \frac{\ell_{\mathrm{up}}^{2H} \cdot
       E_{1+H}
       \left[\left(\frac{r}{\ell_{\mathrm{up}}}\right)^{2}\right]
       - \ell_{\mathrm{low}}^{2H} \cdot
       E_{1+H}
       \left[\left(\frac{r}{\ell_{\mathrm{low}}}\right)^{2}\right]}
       {\ell_{\mathrm{up}}^{2H}-\ell_{\mathrm{low}}^{2H}}
       \right) \\
       \sigma^2_{\ell_{\mathrm{low}},\ell_{\mathrm{up}}} &=
       \frac{C\cdot\left(\ell_{\mathrm{up}}^{2H}
       -\ell_{\mathrm{low}}^{2H}\right)}{2H}

    The "length scale" of this model is equivalent by the integration range:

    .. math::
       \ell = \ell_{\mathrm{up}} -\ell_{\mathrm{low}}

    If you want to define an upper scale truncation, you should set ``len_low``
    and ``len_scale`` accordingly.

    The following Parameters occur:

        * :math:`C>0` :
          scaling factor from the Power-Law (intensity of variation)
          This parameter will be calculated internally by the given variance.
          You can access C directly by ``model.var_raw``
        * :math:`0<H<1` : hurst coefficient (``model.hurst``)
        * :math:`\ell_{\mathrm{low}}\geq 0` : lower length scale truncation
          of the model (``model.len_low``)
        * :math:`\ell_{\mathrm{up}}\geq 0` : upper length scale truncation
          of the model (``model.len_up``)

          This will be calculated internally by:

            * ``len_up = len_low + len_scale``

          That means, that the ``len_scale`` in this model actually represents
          the integration range for the truncated power law.
        * :math:`E_s(x)` is the exponential integral.

    References
    ----------
    .. [Federico1997] Di Federico, V. and Neuman, S. P.,
           "Scaling of random fields by means of truncated power variograms and
           associated spectra", Water Resources Research, 33, 1075–1085. (1997)

    Other Parameters
    ----------------
    hurst : :class:`float`, optional
        Hurst coefficient of the power law.
        Standard range: ``(0, 1)``.
        Default: ``0.5``
    len_low : :class:`float`, optional
        The lower length scale truncation of the model.
        Standard range: ``[0, inf]``.
        Default: ``0.0``
    """

    def default_opt_arg(self):
        """Defaults for the optional arguments.

            * ``{"hurst": 0.5, "len_low": 0.0}``

        Returns
        -------
        :class:`dict`
            Defaults for optional arguments
        """
        return {"hurst": 0.5, "len_low": 0.0}

    def default_opt_arg_bounds(self):
        """Defaults for boundaries of the optional arguments.

            * ``{"hurst": [0, 1, "oo"], "len_low": [0, inf, "cc"]}``

        Returns
        -------
        :class:`dict`
            Boundaries for optional arguments
        """
        return {"hurst": (0.1, 1, "oo"), "len_low": (0.0, np.inf, "co")}

    def cor(self, h):
        """TPL with Gaussian modes - normalized correlation function."""
        return tplstable_cor(h, 1.0, self.hurst, 2)

    def correlation(self, r):
        """TPL with Gaussian modes - correlation function."""
        # if lower limit is 0 we use the simplified version (faster)
        if np.isclose(self.len_low_rescaled, 0.0):
            return tplstable_cor(r, self.len_rescaled, self.hurst, 2)
        return (
            self.len_up_rescaled ** (2 * self.hurst)
            * tplstable_cor(r, self.len_up_rescaled, self.hurst, 2)
            - self.len_low_rescaled ** (2 * self.hurst)
            * tplstable_cor(r, self.len_low_rescaled, self.hurst, 2)
        ) / (
            self.len_up_rescaled ** (2 * self.hurst)
            - self.len_low_rescaled ** (2 * self.hurst)
        )

    def spectral_density(self, k):  # noqa: D102
        return tpl_gau_spec_dens(
            k, self.dim, self.len_rescaled, self.hurst, self.len_low_rescaled
        )


class TPLExponential(TPLCovModel):
    r"""Truncated-Power-Law with Exponential modes.

    Notes
    -----
    The truncated power law is given by a superposition of scale-dependent
    variograms [Federico1997]_:

    .. math::
       \gamma_{\ell_{\mathrm{low}},\ell_{\mathrm{up}}}(r) =
       \intop_{\ell_{\mathrm{low}}}^{\ell_{\mathrm{up}}}
       \gamma(r,\lambda) \frac{\rm d \lambda}{\lambda}

    with `Exponential` modes on each scale:

    .. math::
       \gamma(r,\lambda) &=
       \sigma^2(\lambda)\cdot\left(1-
       \exp\left[- \frac{r}{\lambda}\right]
       \right)\\
       \sigma^2(\lambda) &= C\cdot\lambda^{2H}

    This results in:

    .. math::
       \gamma_{\ell_{\mathrm{low}},\ell_{\mathrm{up}}}(r) &=
       \sigma^2_{\ell_{\mathrm{low}},\ell_{\mathrm{up}}}\cdot\left(1-
       2H \cdot
       \frac{\ell_{\mathrm{up}}^{2H} \cdot
       E_{1+2H}\left[\frac{r}{\ell_{\mathrm{up}}}\right]
       - \ell_{\mathrm{low}}^{2H} \cdot
       E_{1+2H}\left[\frac{r}{\ell_{\mathrm{low}}}\right]}
       {\ell_{\mathrm{up}}^{2H}-\ell_{\mathrm{low}}^{2H}}
       \right) \\
       \sigma^2_{\ell_{\mathrm{low}},\ell_{\mathrm{up}}} &=
       \frac{C\cdot\left(\ell_{\mathrm{up}}^{2H}
       -\ell_{\mathrm{low}}^{2H}\right)}{2H}

    The "length scale" of this model is equivalent by the integration range:

    .. math::
       \ell = \ell_{\mathrm{up}} -\ell_{\mathrm{low}}

    If you want to define an upper scale truncation, you should set ``len_low``
    and ``len_scale`` accordingly.

    The following Parameters occur:

        * :math:`C>0` :
          scaling factor from the Power-Law (intensity of variation)
          This parameter will be calculated internally by the given variance.
          You can access C directly by ``model.var_raw``
        * :math:`0<H<\frac{1}{2}` : hurst coefficient (``model.hurst``)
        * :math:`\ell_{\mathrm{low}}\geq 0` : lower length scale truncation
          of the model (``model.len_low``)
        * :math:`\ell_{\mathrm{up}}\geq 0` : upper length scale truncation
          of the model (``model.len_up``)

          This will be calculated internally by:

            * ``len_up = len_low + len_scale``

          That means, that the ``len_scale`` in this model actually represents
          the integration range for the truncated power law.
        * :math:`E_s(x)` is the exponential integral.

    References
    ----------
    .. [Federico1997] Di Federico, V. and Neuman, S. P.,
           "Scaling of random fields by means of truncated power variograms and
           associated spectra", Water Resources Research, 33, 1075–1085. (1997)

    Other Parameters
    ----------------
    hurst : :class:`float`, optional
        Hurst coefficient of the power law.
        Standard range: ``(0, 1)``.
        Default: ``0.5``
    len_low : :class:`float`, optional
        The lower length scale truncation of the model.
        Standard range: ``[0, inf]``.
        Default: ``0.0``
    """

    def default_opt_arg(self):
        """Defaults for the optional arguments.

            * ``{"hurst": 0.25, "len_low": 0.0}``

        Returns
        -------
        :class:`dict`
            Defaults for optional arguments
        """
        return {"hurst": 0.25, "len_low": 0.0}

    def default_opt_arg_bounds(self):
        """Defaults for boundaries of the optional arguments.

            * ``{"hurst": [0, 1, "oo"], "len_low": [0, inf, "cc"]}``

        Returns
        -------
        :class:`dict`
            Boundaries for optional arguments
        """
        return {"hurst": (0.1, 1, "oo"), "len_low": (0.0, np.inf, "co")}

    def cor(self, h):
        """TPL with Exponential modes - normalized correlation function."""
        return tplstable_cor(h, 1.0, self.hurst, 1)

    def correlation(self, r):
        """TPL with Exponential modes - correlation function."""
        # if lower limit is 0 we use the simplified version (faster)
        if np.isclose(self.len_low_rescaled, 0.0):
            return tplstable_cor(r, self.len_rescaled, self.hurst, 1)
        return (
            self.len_up_rescaled ** (2 * self.hurst)
            * tplstable_cor(r, self.len_up_rescaled, self.hurst, 1)
            - self.len_low_rescaled ** (2 * self.hurst)
            * tplstable_cor(r, self.len_low_rescaled, self.hurst, 1)
        ) / (
            self.len_up_rescaled ** (2 * self.hurst)
            - self.len_low_rescaled ** (2 * self.hurst)
        )

    def spectral_density(self, k):  # noqa: D102
        return tpl_exp_spec_dens(
            k, self.dim, self.len_rescaled, self.hurst, self.len_low_rescaled
        )


class TPLStable(TPLCovModel):
    r"""Truncated-Power-Law with Stable modes.

    Notes
    -----
    The truncated power law is given by a superposition of scale-dependent
    variograms:

    .. math::
       \gamma_{\ell_{\mathrm{low}},\ell_{\mathrm{up}}}(r) =
       \intop_{\ell_{\mathrm{low}}}^{\ell_{\mathrm{up}}}
       \gamma(r,\lambda) \frac{\rm d \lambda}{\lambda}

    with `Stable` modes on each scale:

    .. math::
       \gamma(r,\lambda) &=
       \sigma^2(\lambda)\cdot\left(1-
       \exp\left[- \left(\frac{r}{\lambda}\right)^{\alpha}\right]
       \right)\\
       \sigma^2(\lambda) &= C\cdot\lambda^{2H}

    This results in:

    .. math::
       \gamma_{\ell_{\mathrm{low}},\ell_{\mathrm{up}}}(r) &=
       \sigma^2_{\ell_{\mathrm{low}},\ell_{\mathrm{up}}}\cdot\left(1-
       \frac{2H}{\alpha} \cdot
       \frac{\ell_{\mathrm{up}}^{2H} \cdot
       E_{1+\frac{2H}{\alpha}}
       \left[\left(\frac{r}{\ell_{\mathrm{up}}}\right)^{\alpha}\right]
       - \ell_{\mathrm{low}}^{2H} \cdot
       E_{1+\frac{2H}{\alpha}}
       \left[\left(\frac{r}{\ell_{\mathrm{low}}}\right)^{\alpha}\right]}
       {\ell_{\mathrm{up}}^{2H}-\ell_{\mathrm{low}}^{2H}}
       \right) \\
       \sigma^2_{\ell_{\mathrm{low}},\ell_{\mathrm{up}}} &=
       \frac{C\cdot\left(\ell_{\mathrm{up}}^{2H}
       -\ell_{\mathrm{low}}^{2H}\right)}{2H}

    The "length scale" of this model is equivalent by the integration range:

    .. math::
       \ell = \ell_{\mathrm{up}} -\ell_{\mathrm{low}}

    If you want to define an upper scale truncation, you should set ``len_low``
    and ``len_scale`` accordingly.

    The following Parameters occur:

        * :math:`0<\alpha\leq 2` : The shape parameter of the Stable model.

            * :math:`\alpha=1` : Exponential modes
            * :math:`\alpha=2` : Gaussian modes

        * :math:`C>0` :
          scaling factor from the Power-Law (intensity of variation)
          This parameter will be calculated internally by the given variance.
          You can access C directly by ``model.var_raw``
        * :math:`0<H<\frac{\alpha}{2}` : hurst coefficient (``model.hurst``)
        * :math:`\ell_{\mathrm{low}}\geq 0` : lower length scale truncation
          of the model (``model.len_low``)
        * :math:`\ell_{\mathrm{up}}\geq 0` : upper length scale truncation
          of the model (``model.len_up``)

          This will be calculated internally by:

            * ``len_up = len_low + len_scale``

          That means, that the ``len_scale`` in this model actually represents
          the integration range for the truncated power law.
        * :math:`E_s(x)` is the exponential integral.

    Other Parameters
    ----------------
    hurst : :class:`float`, optional
        Hurst coefficient of the power law.
        Standard range: ``(0, 1)``.
        Default: ``0.5``
    alpha : :class:`float`, optional
        Shape parameter of the stable model.
        Standard range: ``(0, 2]``.
        Default: ``1.5``
    len_low : :class:`float`, optional
        The lower length scale truncation of the model.
        Standard range: ``[0, inf]``.
        Default: ``0.0``
    """

    def default_opt_arg(self):
        """Defaults for the optional arguments.

            * ``{"hurst": 0.5, "alpha": 1.5, "len_low": 0.0}``

        Returns
        -------
        :class:`dict`
            Defaults for optional arguments
        """
        return {"hurst": 0.5, "alpha": 1.5, "len_low": 0.0}

    def default_opt_arg_bounds(self):
        """Defaults for boundaries of the optional arguments.

            * ``{"hurst": [0, 1, "oo"],
              "alpha": [0, 2, "oc"],
              "len_low": [0, inf, "cc"]}``

        Returns
        -------
        :class:`dict`
            Boundaries for optional arguments
        """
        return {
            "hurst": (0.1, 1, "oo"),
            "alpha": (0, 2, "oc"),
            "len_low": (0, np.inf, "co"),
        }

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
                "TPLStable: parameter 'alpha' is < 0.3, "
                "count with unstable results",
                AttributeWarning,
            )

    def cor(self, h):
        """TPL with Stable modes - normalized correlation function."""
        return tplstable_cor(h, 1.0, self.hurst, self.alpha)

    def correlation(self, r):
        """TPL with Stable modes - correlation function."""
        # if lower limit is 0 we use the simplified version (faster)
        if np.isclose(self.len_low_rescaled, 0.0):
            return tplstable_cor(r, self.len_rescaled, self.hurst, self.alpha)
        return (
            self.len_up_rescaled ** (2 * self.hurst)
            * tplstable_cor(r, self.len_up_rescaled, self.hurst, self.alpha)
            - self.len_low_rescaled ** (2 * self.hurst)
            * tplstable_cor(r, self.len_low_rescaled, self.hurst, self.alpha)
        ) / (
            self.len_up_rescaled ** (2 * self.hurst)
            - self.len_low_rescaled ** (2 * self.hurst)
        )


class TPLSimple(CovModel):
    r"""The simply truncated power law model.

    This model describes a simple truncated power law
    with a finite length scale. In contrast to other models,
    this one is not derived from super-positioning modes.

    Notes
    -----
    This model is given by the following correlation function [Wendland1995]_:

    .. math::
       \rho(r) =
       \begin{cases}
       \left(1-s\cdot\frac{r}{\ell}\right)^{\nu} & r<\frac{\ell}{s}\\
       0 & r\geq\frac{\ell}{s}
       \end{cases}

    Where the standard rescale factor is :math:`s=1`.
    :math:`\nu\geq\frac{d+1}{2}` is a shape parameter,
    which defaults to :math:`\nu=\frac{d+1}{2}`,

    For :math:`\nu=1` (valid only in d=1)
    this coincides with the truncated linear model:

    .. math::
       \rho(r) =
       \begin{cases}
       1-s\cdot\frac{r}{\ell} & r<\frac{\ell}{s}\\
       0 & r\geq\frac{\ell}{s}
       \end{cases}

    References
    ----------
    .. [Wendland1995] Wendland, H.,
           "Piecewise polynomial, positive definite and compactly supported
           radial functions of minimal degree.",
           Advances in computational Mathematics 4.1, 389-396. (1995)

    Other Parameters
    ----------------
    nu : :class:`float`, optional
        Shape parameter. Standard range: ``[(dim+1)/2, 50]``
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
        return {"nu": (self.dim + 1) / 2}

    def default_opt_arg_bounds(self):
        """Defaults for boundaries of the optional arguments.

            * ``{"nu": [dim/2 - 1, 50.0]}``

        Returns
        -------
        :class:`dict`
            Boundaries for optional arguments
        """
        return {"nu": [(self.dim + 1) / 2, 50.0]}

    def cor(self, h):
        """TPL Simple - normalized correlation function."""
        return np.maximum(1 - np.abs(h, dtype=np.double), 0.0) ** self.nu
