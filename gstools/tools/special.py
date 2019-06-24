# -*- coding: utf-8 -*-
"""
GStools subpackage providing special functions.

.. currentmodule:: gstools.tools.special

The following functions are provided

.. autosummary::
   inc_gamma
   exp_int
   inc_beta
   tplstable_cor
"""
# pylint: disable=C0103, E1101
from __future__ import print_function, division, absolute_import

import numpy as np
from scipy import special as sps

__all__ = ["inc_gamma", "exp_int", "inc_beta", "tplstable_cor"]


# special functions ###########################################################


def inc_gamma(s, x):
    r"""The (upper) incomplete gamma function.

    Given by: :math:`\Gamma(s,x) = \int_x^{\infty} t^{s-1}\,e^{-t}\,{\rm d}t`

    Parameters
    ----------
    s : :class:`float`
        exponent in the integral
    x : :class:`numpy.ndarray`
        input values
    """
    if np.isclose(s, 0):
        return sps.exp1(x)
    if np.isclose(s, np.around(s)) and s < -0.5:
        return x ** (s - 1) * sps.expn(int(1 - np.around(s)), x)
    if s < 0:
        return (inc_gamma(s + 1, x) - x ** s * np.exp(-x)) / s
    return sps.gamma(s) * sps.gammaincc(s, x)


def exp_int(s, x):
    r"""The exponential integral :math:`E_s(x)`.

    Given by: :math:`E_s(x) = \int_1^\infty \frac{e^{-xt}}{t^s}\,\mathrm dt`

    Parameters
    ----------
    s : :class:`float`
        exponent in the integral
    x : :class:`numpy.ndarray`
        input values
    """
    if np.isclose(s, 1):
        return sps.exp1(x)
    if np.isclose(s, np.around(s)) and s > -0.5:
        return sps.expn(int(np.around(s)), x)
    x = np.array(x, dtype=np.double)
    x[np.isclose(x, 0)] = 0  # hack to prevent numerical errors
    x[np.isclose(1.0 / x, 0)] = np.inf  # hack to prevent numerical errors
    res = np.full_like(x, 1.0 / (s - 1.0))  # limit at x=0
    res[x == np.inf] = 0  # limit at x=inf
    x_fin = np.logical_and(x > 0, x < np.inf)
    res[x_fin] = inc_gamma(1 - s, x[x_fin]) * x[x_fin] ** (s - 1)
    return res * 1  # this will create a float out of an array


def tplstable_cor(r, len_scale, hurst, alpha):
    r"""The correlation function of the TPLStable model.

    Given by

    .. math::
       \mathrm{cor}(r) =
       \frac{2H}{\alpha} \cdot
       E_{1+\frac{2H}{\alpha}}
       \left(\left(\frac{r}{\ell}\right)^{\alpha} \right)


    Parameters
    ----------
    r : :class:`numpy.ndarray`
        input values
    len_scale : :class:`float`
        length-scale of the model.
    hurst : :class:`float`
        Hurst coefficient of the power law.
    alpha : :class:`float`, optional
        Shape parameter of the stable model.
    """
    r = np.array(np.abs(r / len_scale), dtype=np.double)
    r[np.isclose(r, 0)] = 0  # hack to prevent numerical errors
    res = np.ones_like(r)
    res[r > 0] = (2 * hurst / alpha) * exp_int(
        1 + 2 * hurst / alpha, (r[r > 0]) ** alpha
    )
    return res


def inc_beta(a, b, x):
    r"""The incomplete Beta function.

    Given by: :math:`B(a,b;\,x) = \int_0^x t^{a-1}\,(1-t)^{b-1}\,dt`

    Parameters
    ----------
    a : :class:`float`
        first exponent in the integral
    b : :class:`float`
        second exponent in the integral
    x : :class:`numpy.ndarray`
        input values
    """
    return sps.betainc(a, b, x) * sps.beta(a, b)
