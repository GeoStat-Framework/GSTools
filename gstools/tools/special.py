# -*- coding: utf-8 -*-
"""
GStools subpackage providing special functions.

.. currentmodule:: gstools.tools.special

The following functions are provided

.. autosummary::
   inc_gamma
   exp_int
   inc_beta
   isclose
"""
# pylint: disable=C0103, E1101
from __future__ import print_function, division, absolute_import

try:
    # first added to Python in version 3.5 (numpy version not symmetric)
    from math import isclose
except ImportError:
    import cmath

    def isclose(a, b, rel_tol=1e-9, abs_tol=0.0):
        """Determine whether two floating point numbers are close in value.

        rel_tol : float
            maximum difference for being considered "close",
            relative to the magnitude of the input values
        abs_tol : float
            maximum difference for being considered "close",
            regardless of the magnitude of the input values

        Return True if a is close in value to b, and False otherwise.

        For the values to be considered close, the difference between them
        must be smaller than at least one of the tolerances.

        -inf, inf and NaN behave similarly to the IEEE 754 Standard. That is,
        NaN is not close to anything, even itself.
        inf and -inf are only close to themselves.
        """
        if rel_tol < 0.0 or abs_tol < 0.0:
            raise ValueError("error tolerances must be non-negative")

        if a == b:
            return True
        if cmath.isinf(a) or cmath.isinf(b):
            return False
        diff = abs(b - a)
        return ((diff <= abs(rel_tol * b)) or (diff <= abs(rel_tol * a))) or (
            diff <= abs_tol
        )


import numpy as np
from scipy import special as sps

__all__ = ["inc_gamma", "exp_int", "inc_beta", "isclose"]


# special functions ###########################################################


def inc_gamma(s, x):
    r"""The (upper) incomplete gamma function

    Given by: :math:`\Gamma(s,x) = \int_x^{\infty} t^{s-1}\,e^{-t}\,{\rm d}t`

    Parameters
    ----------
    s : :class:`float`
        exponent in the integral
    x : :class:`numpy.ndarray`
        input values
    """
    if isclose(s, 0):
        return sps.exp1(x)
    if isclose(s, np.around(s)) and s < 0:
        return x ** (s - 1) * sps.expn(int(1 - np.around(s)), x)
    if s < 0:
        return (inc_gamma(s + 1, x) - x ** s * np.exp(-x)) / s
    return sps.gamma(s) * sps.gammaincc(s, x)


def exp_int(s, x):
    r"""The exponential integral :math:`E_s(x)`

    Given by: :math:`E_s(x) = \int_1^\infty \frac{e^{-xt}}{t^s}\,\mathrm dt`

    Parameters
    ----------
    s : :class:`float`
        exponent in the integral
    x : :class:`numpy.ndarray`
        input values
    """
    #    print("s, x", s, x)
    if isclose(s, 1):
        return sps.exp1(x)
    if isclose(s, np.around(s)) and s > -1:
        return sps.expn(int(np.around(s)), x)
    return inc_gamma(1 - s, x) * x ** (s - 1)


def inc_beta(a, b, x):
    r"""The incomplete Beta function

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
