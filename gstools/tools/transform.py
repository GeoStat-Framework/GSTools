# -*- coding: utf-8 -*-
"""
GStools subpackage providing field transformations.

.. currentmodule:: gstools.tools.transform

The following functions are provided

.. autosummary::
   zinnharvey
   normal_force_moments
   normal_to_lognormal
   normal_to_uniform
   normal_to_arcsin
   normal_to_uquad
"""
# pylint: disable=C0103, E1101
from __future__ import print_function, division, absolute_import

import numpy as np
from scipy.special import erf, erfinv


__all__ = [
    "zinnharvey",
    "normal_force_moments",
    "normal_to_lognormal",
    "normal_to_uniform",
    "normal_to_arcsin",
    "normal_to_uquad",
]


def zinnharvey(field, conn="high", mean=None, var=None):
    """
    Zinn and Harvey transformation to connect low or high values

    Parameters
    ----------
    field : :class:`numpy.ndarray`
        Spatial Random Field with normal distributed values.
        As returned by SRF.
    conn : :class:`str`, optional
        Desired connectivity. Either "low" or "high".
        Default: "high"
    mean : :class:`float` or :any:`None`, optional
        Mean of the given field. If None is given, the mean will be calculated.
        Default: :any:`None`
    var : :class:`float` or :any:`None`, optional
        Variance of the given field.
        If None is given, the mean will be calculated.
        Default: :any:`None`

    Returns
    -------
        :class:`numpy.ndarray`
            Transformed field.
    """
    if mean is None:
        mean = np.mean(field)
    if var is None:
        var = np.var(field)
    field = np.abs((field - mean) / var)
    field = 2 * erf(field / np.sqrt(2)) - 1
    field = np.sqrt(2) * erfinv(field)
    if conn == "high":
        field = -field
    return field * var + mean


def normal_force_moments(field, mean=0, var=1):
    """
    Force moments of a normal distributed field.

    Parameters
    ----------
    field : :class:`numpy.ndarray`
        Spatial Random Field with normal distributed values.
        As returned by SRF.
    mean : :class:`float`, optional
        Desired mean of the field.
        Default: 0
    var : :class:`float` or :any:`None`, optional
        Desired variance of the field.
        Default: 1

    Returns
    -------
        :class:`numpy.ndarray`
            Transformed field.
    """
    var_in = np.var(field)
    mean_in = np.mean(field)
    rescale = np.sqrt(var / var_in)
    return rescale * (field - mean_in) + mean


def normal_to_lognormal(field):
    """
    Transform normal distribution to log-normal distribution

    Parameters
    ----------
    field : :class:`numpy.ndarray`
        Spatial Random Field with normal distributed values.
        As returned by SRF.

    Returns
    -------
        :class:`numpy.ndarray`
            Transformed field.
    """
    return np.exp(field)


def normal_to_uniform(field, mean=None, var=None):
    """
    Transform normal distribution to uniform distribution on [0, 1]

    Parameters
    ----------
    field : :class:`numpy.ndarray`
        Spatial Random Field with normal distributed values.
        As returned by SRF.
    mean : :class:`float` or :any:`None`, optional
        Mean of the given field. If None is given, the mean will be calculated.
        Default: :any:`None`
    var : :class:`float` or :any:`None`, optional
        Variance of the given field.
        If None is given, the mean will be calculated.
        Default: :any:`None`

    Returns
    -------
        :class:`numpy.ndarray`
            Transformed field.
    """
    if mean is None:
        mean = np.mean(field)
    if var is None:
        var = np.var(field)
    return 0.5 * (1 + erf((field - mean) / np.sqrt(2 * var)))


def normal_to_arcsin(field, mean=None, var=None, a=0, b=1):
    """
    Transform normal distribution to arcsin distribution

    See: https://en.wikipedia.org/wiki/Arcsine_distribution

    Parameters
    ----------
    field : :class:`numpy.ndarray`
        Spatial Random Field with normal distributed values.
        As returned by SRF.
    mean : :class:`float` or :any:`None`, optional
        Mean of the given field. If None is given, the mean will be calculated.
        Default: :any:`None`
    var : :class:`float` or :any:`None`, optional
        Variance of the given field.
        If None is given, the mean will be calculated.
        Default: :any:`None`
    a : :class:`float`, optional
        Parameter a of the arcsin distribution. Default: 0
    b : :class:`float`, optional
        Parameter b of the arcsin distribution. Default: 1

    Returns
    -------
        :class:`numpy.ndarray`
            Transformed field.
    """
    return _uniform_to_arcsin(normal_to_uniform(field, mean, var), a, b)


def normal_to_uquad(field, mean=None, var=None, a=0, b=1):
    """
    Transform normal distribution to U-quadratic distribution

    See: https://en.wikipedia.org/wiki/U-quadratic_distribution

    Parameters
    ----------
    field : :class:`numpy.ndarray`
        Spatial Random Field with normal distributed values.
        As returned by SRF.
    mean : :class:`float` or :any:`None`, optional
        Mean of the given field. If None is given, the mean will be calculated.
        Default: :any:`None`
    var : :class:`float` or :any:`None`, optional
        Variance of the given field.
        If None is given, the mean will be calculated.
        Default: :any:`None`
    a : :class:`float`, optional
        Parameter a of the U-quadratic distribution. Default: 0
    b : :class:`float`, optional
        Parameter b of the U-quadratic distribution. Default: 1

    Returns
    -------
        :class:`numpy.ndarray`
            Transformed field.
    """
    return _uniform_to_uquad(normal_to_uniform(field, mean, var), a, b)


def _uniform_to_arcsin(field, a=0, b=1):
    """
    PPF of your desired distribution

    The PPF is the inverse of the CDF and is used to sample a distribution
    from uniform distributed values on [0, 1]

    in this case: the arcsin distribution
    See: https://en.wikipedia.org/wiki/Arcsine_distribution
    """
    return (b - a) * np.sin(np.pi * 0.5 * field) ** 2 + a


def _uniform_to_uquad(field, a=0, b=1):
    """
    PPF of your desired distribution

    The PPF is the inverse of the CDF and is used to sample a distribution
    from uniform distributed values on [0, 1]

    in this case: the U-quadratic distribution
    See: https://en.wikipedia.org/wiki/U-quadratic_distribution
    """
    al = 12 / (b - a) ** 3
    be = (a + b) / 2
    ga = (a - b) ** 3 / 8
    y_raw = 3 * field / al + ga
    out = np.zeros_like(y_raw)
    out[y_raw > 0] = y_raw[y_raw > 0] ** (1 / 3)
    out[y_raw < 0] = -(-y_raw[y_raw < 0]) ** (1 / 3)
    return out + be
