# -*- coding: utf-8 -*-
"""
GStools subpackage providing truncated power law covariance models.

.. currentmodule:: gstools.covmodel.tpl_models

The following classes and functions are provided

.. autosummary::
   TPLGaussian
   TPLExponential
   TPLStable
"""
# pylint: disable=C0103, E1101

import warnings
import numpy as np
from gstools.covmodel.base import CovModel
from gstools.tools.special import (
    tplstable_cor,
    tpl_gau_spec_dens,
    tpl_exp_spec_dens,
)


__all__ = ["TPLGaussian", "TPLExponential", "TPLStable"]


# Truncated power law #########################################################


class TPLGaussian(CovModel):
    r"""Truncated-Power-Law with Gaussian modes.

    Notes
    -----
    The truncated power law is given by a superposition of scale-dependent
    variograms:

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

    The following Parameters occure:

        * :math:`C>0` : scaling factor from the Power-Law
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

    Other Parameters
    ----------------
    **opt_arg
        The following parameters are covered by these keyword arguments
    hurst : :class:`float`, optional
        Hurst coefficient of the power law.
        Standard range: ``(0, 1)``.
        Default: ``0.5``
    len_low : :class:`float`, optional
        The lower length scale truncation of the model.
        Standard range: ``[0, 1000]``.
        Default: ``0.0``
    """

    @property
    def len_up(self):
        """:class:`float`: Upper length scale truncation of the model.

        * ``len_up = len_low + len_scale``
        """
        return self.len_low + self.len_scale

    def var_factor(self):
        r"""Factor for C (Power-Law factor) to result in variance.

        This is used to result in the right variance, which is depending
        on the hurst coefficient and the length-scale extents

        .. math::
           \frac{\ell_{\mathrm{up}}^{2H} - \ell_{\mathrm{low}}^{2H}}{2H}

        Returns
        -------
        :class:`float`
            factor
        """
        return (
            self.len_up ** (2 * self.hurst) - self.len_low ** (2 * self.hurst)
        ) / (2 * self.hurst)

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

            * ``{"hurst": [0, 1, "oo"], "len_low": [0, 1000, "cc"]}``

        Returns
        -------
        :class:`dict`
            Boundaries for optional arguments
        """
        return {"hurst": (0.1, 1, "oo"), "len_low": (0.0, np.inf, "co")}

    def correlation(self, r):
        r"""Truncated-Power-Law with Gaussian modes - correlation function.

        If ``len_low=0`` we have a simple representation:

        .. math::
           \rho(r) =
           H \cdot
           E_{1+H}
           \left[
           \left(\frac{r}{\ell}\right)^{2}
           \right]

        The general case:

        .. math::
           \rho(r) =
           H \cdot
           \frac{\ell_{\mathrm{up}}^{2H} \cdot
           E_{1+H}
           \left[\left(\frac{r}{\ell_{\mathrm{up}}}\right)^{2}\right]
           - \ell_{\mathrm{low}}^{2H} \cdot
           E_{1+H}
           \left[\left(\frac{r}{\ell_{\mathrm{low}}}\right)^{2}\right]}
           {\ell_{\mathrm{up}}^{2H}-\ell_{\mathrm{low}}^{2H}}
        """
        # if lower limit is 0 we use the simplified version (faster)
        if np.isclose(self.len_low, 0.0):
            return tplstable_cor(r, self.len_scale, self.hurst, 2)
        return (
            self.len_up ** (2 * self.hurst)
            * tplstable_cor(r, self.len_up, self.hurst, 2)
            - self.len_low ** (2 * self.hurst)
            * tplstable_cor(r, self.len_low, self.hurst, 2)
        ) / (
            self.len_up ** (2 * self.hurst) - self.len_low ** (2 * self.hurst)
        )

    def spectral_density(self, k):  # noqa: D102
        return tpl_gau_spec_dens(
            k, self.dim, self.len_scale, self.hurst, self.len_low
        )


class TPLExponential(CovModel):
    r"""Truncated-Power-Law with Exponential modes.

    Notes
    -----
    The truncated power law is given by a superposition of scale-dependent
    variograms:

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

    The following Parameters occure:

        * :math:`C>0` : scaling factor from the Power-Law
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

    Other Parameters
    ----------------
    **opt_arg
        The following parameters are covered by these keyword arguments
    hurst : :class:`float`, optional
        Hurst coefficient of the power law.
        Standard range: ``(0, 1)``.
        Default: ``0.5``
    len_low : :class:`float`, optional
        The lower length scale truncation of the model.
        Standard range: ``[0, 1000]``.
        Default: ``0.0``
    """

    @property
    def len_up(self):
        """:class:`float`: Upper length scale truncation of the model.

        * ``len_up = len_low + len_scale``
        """
        return self.len_low + self.len_scale

    def var_factor(self):
        r"""Factor for C (Power-Law factor) to result in variance.

        This is used to result in the right variance, which is depending
        on the hurst coefficient and the length-scale extents

        .. math::
           \frac{\ell_{\mathrm{up}}^{2H} - \ell_{\mathrm{low}}^{2H}}{2H}

        Returns
        -------
        :class:`float`
            factor
        """
        return (
            self.len_up ** (2 * self.hurst) - self.len_low ** (2 * self.hurst)
        ) / (2 * self.hurst)

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

            * ``{"hurst": [0, 1, "oo"], "len_low": [0, 1000, "cc"]}``

        Returns
        -------
        :class:`dict`
            Boundaries for optional arguments
        """
        return {"hurst": (0.1, 1, "oo"), "len_low": (0.0, np.inf, "co")}

    def correlation(self, r):
        r"""Truncated-Power-Law with Exponential modes - correlation function.

        If ``len_low=0`` we have a simple representation:

        .. math::
           \rho(r) =
           H \cdot
           E_{1+H}
           \left[
           \frac{r}{\ell}
           \right]

        The general case:

        .. math::
           \rho(r) =
           2H \cdot
           \frac{\ell_{\mathrm{up}}^{2H} \cdot
           E_{1+2H}\left[\frac{r}{\ell_{\mathrm{up}}}\right]
           - \ell_{\mathrm{low}}^{2H} \cdot
           E_{1+2H}\left[\frac{r}{\ell_{\mathrm{low}}}\right]}
           {\ell_{\mathrm{up}}^{2H}-\ell_{\mathrm{low}}^{2H}}
        """
        # if lower limit is 0 we use the simplified version (faster)
        if np.isclose(self.len_low, 0.0):
            return tplstable_cor(r, self.len_scale, self.hurst, 1)
        return (
            self.len_up ** (2 * self.hurst)
            * tplstable_cor(r, self.len_up, self.hurst, 1)
            - self.len_low ** (2 * self.hurst)
            * tplstable_cor(r, self.len_low, self.hurst, 1)
        ) / (
            self.len_up ** (2 * self.hurst) - self.len_low ** (2 * self.hurst)
        )

    def spectral_density(self, k):  # noqa: D102
        return tpl_exp_spec_dens(
            k, self.dim, self.len_scale, self.hurst, self.len_low
        )


class TPLStable(CovModel):
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

    The following Parameters occure:

        * :math:`0<\alpha\leq 2` : The shape parameter of the Stable model.

            * :math:`\alpha=1` : Exponential modes
            * :math:`\alpha=2` : Gaussian modes

        * :math:`C>0` : scaling factor from the Power-Law
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
    **opt_arg
        The following parameters are covered by these keyword arguments
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
        Standard range: ``[0, 1000]``.
        Default: ``0.0``
    """

    @property
    def len_up(self):
        """:class:`float`: Upper length scale truncation of the model.

        * ``len_up = len_low + len_scale``
        """
        return self.len_low + self.len_scale

    def var_factor(self):
        r"""Factor for C (Power-Law factor) to result in variance.

        This is used to result in the right variance, which is depending
        on the hurst coefficient and the length-scale extents

        .. math::
           \frac{\ell_{\mathrm{up}}^{2H} - \ell_{\mathrm{low}}^{2H}}{2H}

        Returns
        -------
        :class:`float`
            factor
        """
        return (
            self.len_up ** (2 * self.hurst) - self.len_low ** (2 * self.hurst)
        ) / (2 * self.hurst)

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
              "len_low": [0, 1000, "cc"]}``

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
                + "count with unstable results"
            )

    def correlation(self, r):
        r"""Truncated-Power-Law with Stable modes - correlation function.

        If ``len_low=0`` we have a simple representation:

        .. math::
           \rho(r) =
           \frac{2H}{\alpha} \cdot
           E_{1+\frac{2H}{\alpha}}
           \left[
           \left(\frac{r}{\ell}\right)^{\alpha}
           \right]

        The general case:

        .. math::
           \rho(r) =
           \frac{2H}{\alpha} \cdot
           \frac{\ell_{\mathrm{up}}^{2H} \cdot
           E_{1+\frac{2H}{\alpha}}
           \left[\left(\frac{r}{\ell_{\mathrm{up}}}\right)^{\alpha}\right]
           - \ell_{\mathrm{low}}^{2H} \cdot
           E_{1+\frac{2H}{\alpha}}
           \left[\left(\frac{r}{\ell_{\mathrm{low}}}\right)^{\alpha}\right]}
           {\ell_{\mathrm{up}}^{2H}-\ell_{\mathrm{low}}^{2H}}
        """
        # if lower limit is 0 we use the simplified version (faster)
        if np.isclose(self.len_low, 0.0):
            return tplstable_cor(r, self.len_scale, self.hurst, self.alpha)
        return (
            self.len_up ** (2 * self.hurst)
            * tplstable_cor(r, self.len_up, self.hurst, self.alpha)
            - self.len_low ** (2 * self.hurst)
            * tplstable_cor(r, self.len_low, self.hurst, self.alpha)
        ) / (
            self.len_up ** (2 * self.hurst) - self.len_low ** (2 * self.hurst)
        )
