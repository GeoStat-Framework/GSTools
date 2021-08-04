# -*- coding: utf-8 -*-
"""
GStools subpackage providing array transformations.

.. currentmodule:: gstools.transform.array

The following functions are provided

Transformations
^^^^^^^^^^^^^^^

.. autosummary::
   array_discrete
   array_boxcox
   array_zinnharvey
   array_force_moments
   array_to_lognormal
   array_to_uniform
   array_to_arcsin
   array_to_uquad
"""
# pylint: disable=C0103, C0123, R0911
from warnings import warn
import numpy as np
from scipy.special import erf, erfinv

__all__ = [
    "array_discrete",
    "array_boxcox",
    "array_zinnharvey",
    "array_force_moments",
    "array_to_lognormal",
    "array_to_uniform",
    "array_to_arcsin",
    "array_to_uquad",
]


def array_discrete(
    field, values, thresholds="arithmetic", mean=None, var=None
):
    """
    Discrete transformation.

    After this transformation, the field has only `len(values)` discrete
    values.

    Parameters
    ----------
    field : :class:`numpy.ndarray`
        Normal distributed values.
    values : :any:`numpy.ndarray`
        The discrete values the field will take
    thresholds : :class:`str` or :any:`numpy.ndarray`, optional
        the thresholds, where the value classes are separated
        possible values are:
        * "arithmetic": the mean of the 2 neighbouring values
        * "equal": devide the field into equal parts
        * an array of explicitly given thresholds
        Default: "arithmetic"
    mean : :class:`float`or :any:`None`
        Mean of the field for "equal" thresholds. Default: np.mean(field)
    var : :class:`float`or :any:`None`
        Variance of the field for "equal" thresholds. Default: np.var(field)

    Returns
    -------
    :class:`numpy.ndarray`
        Transformed field.
    """
    field = np.asarray(field)
    if thresholds == "arithmetic":
        # just in case, sort the values
        values = np.sort(values)
        thresholds = (values[1:] + values[:-1]) / 2
    elif thresholds == "equal":
        mean = np.mean(field) if mean is None else float(mean)
        var = np.var(field) if var is None else float(var)
        values = np.asarray(values)
        n = len(values)
        p = np.arange(1, n) / n  # n-1 equal subdivisions of [0, 1]
        rescale = np.sqrt(var * 2)
        # use quantile of the normal distribution to get equal ratios
        thresholds = mean + rescale * erfinv(2 * p - 1)
    else:
        if len(values) != len(thresholds) + 1:
            raise ValueError(
                "discrete transformation: len(values) != len(thresholds) + 1"
            )
        values = np.asarray(values)
        thresholds = np.asarray(thresholds)
    # check thresholds
    if not np.all(thresholds[:-1] < thresholds[1:]):
        raise ValueError(
            "discrete transformation: thresholds need to be ascending"
        )
    # use a separate result so the intermediate results are not affected
    result = np.empty_like(field)
    # handle edge cases
    result[field <= thresholds[0]] = values[0]
    result[field > thresholds[-1]] = values[-1]
    for i, value in enumerate(values[1:-1]):
        result[
            np.logical_and(thresholds[i] < field, field <= thresholds[i + 1])
        ] = value
    return result


def array_boxcox(field, lmbda=1, shift=0):
    """
    (Inverse) Box-Cox transformation to denormalize data.

    After this transformation, the again Box-Cox transformed field is normal
    distributed.

    See: https://en.wikipedia.org/wiki/Power_transform#Box%E2%80%93Cox_transformation

    Parameters
    ----------
    field : :class:`numpy.ndarray`
        Normal distributed values.
    lmbda : :class:`float`, optional
        The lambda parameter of the Box-Cox transformation.
        For ``lmbda=0`` one obtains the log-normal transformation.
        Default: ``1``
    shift : :class:`float`, optional
        The shift parameter from the two-parametric Box-Cox transformation.
        The field will be shifted by that value before transformation.
        Default: ``0``
    """
    field = np.asarray(field)
    result = field + shift
    if np.isclose(lmbda, 0):
        return array_to_lognormal(result)
    if np.min(result) < -1 / lmbda:
        warn("Box-Cox: Some values will be cut off!")
    return (np.maximum(lmbda * result + 1, 0)) ** (1 / lmbda)


def array_zinnharvey(field, conn="high", mean=None, var=None):
    """
    Zinn and Harvey transformation to connect low or high values.

    Parameters
    ----------
    field : :class:`numpy.ndarray`
        Normal distributed values.
    conn : :class:`str`, optional
        Desired connectivity. Either "low" or "high".
        Default: "high"
    mean : :class:`float` or :any:`None`, optional
        Mean of the given field. If None is given, the mean will be calculated.
        Default: :any:`None`
    var : :class:`float` or :any:`None`, optional
        Variance of the given field.
        If None is given, the variance will be calculated.
        Default: :any:`None`

    Returns
    -------
    :class:`numpy.ndarray`
        Transformed field.
    """
    field = np.asarray(field)
    mean = np.mean(field) if mean is None else float(mean)
    var = np.var(field) if var is None else float(var)
    result = np.abs((field - mean) / np.sqrt(var))
    result = np.sqrt(2) * erfinv(2 * erf(result / np.sqrt(2)) - 1)
    if conn == "high":
        result = -result
    return result * np.sqrt(var) + mean


def array_force_moments(field, mean=0, var=1):
    """
    Force moments of a normal distributed field.

    Parameters
    ----------
    field : :class:`numpy.ndarray`
        Normal distributed values.
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
    field = np.asarray(field)
    var_in = np.var(field)
    mean_in = np.mean(field)
    rescale = np.sqrt(var / var_in)
    return rescale * (field - mean_in) + mean


def array_to_lognormal(field):
    """
    Transform normal distribution to log-normal distribution.

    Parameters
    ----------
    field : :class:`numpy.ndarray`
        Normal distributed values.

    Returns
    -------
    :class:`numpy.ndarray`
        Transformed field.
    """
    return np.exp(field)


def array_to_uniform(field, mean=None, var=None):
    """
    Transform normal distribution to uniform distribution on [0, 1].

    Parameters
    ----------
    field : :class:`numpy.ndarray`
        Normal distributed values.
    mean : :class:`float` or :any:`None`, optional
        Mean of the given field. If None is given, the mean will be calculated.
        Default: :any:`None`
    var : :class:`float` or :any:`None`, optional
        Variance of the given field.
        If None is given, the variance will be calculated.
        Default: :any:`None`

    Returns
    -------
    :class:`numpy.ndarray`
        Transformed field.
    """
    field = np.asarray(field)
    mean = np.mean(field) if mean is None else float(mean)
    var = np.var(field) if var is None else float(var)
    return 0.5 * (1 + erf((field - mean) / np.sqrt(2 * var)))


def array_to_arcsin(field, mean=None, var=None, a=None, b=None):
    """
    Transform normal distribution to arcsin distribution.

    See: https://en.wikipedia.org/wiki/Arcsine_distribution

    Parameters
    ----------
    field : :class:`numpy.ndarray`
        Normal distributed values.
    mean : :class:`float` or :any:`None`, optional
        Mean of the given field. If None is given, the mean will be calculated.
        Default: :any:`None`
    var : :class:`float` or :any:`None`, optional
        Variance of the given field.
        If None is given, the mean will be calculated.
        Default: :any:`None`
    a : :class:`float`, optional
        Parameter a of the arcsin distribution (lower bound).
        Default: keep mean and variance
    b : :class:`float`, optional
        Parameter b of the arcsin distribution (upper bound).
        Default: keep mean and variance

    Returns
    -------
    :class:`numpy.ndarray`
        Transformed field.
    """
    field = np.asarray(field)
    mean = np.mean(field) if mean is None else float(mean)
    var = np.var(field) if var is None else float(var)
    a = mean - np.sqrt(2.0 * var) if a is None else float(a)
    b = mean + np.sqrt(2.0 * var) if b is None else float(b)
    return _uniform_to_arcsin(array_to_uniform(field, mean, var), a, b)


def array_to_uquad(field, mean=None, var=None, a=None, b=None):
    """
    Transform normal distribution to U-quadratic distribution.

    See: https://en.wikipedia.org/wiki/U-quadratic_distribution

    Parameters
    ----------
    field : :class:`numpy.ndarray`
        Normal distributed values.
    mean : :class:`float` or :any:`None`, optional
        Mean of the given field. If None is given, the mean will be calculated.
        Default: :any:`None`
    var : :class:`float` or :any:`None`, optional
        Variance of the given field.
        If None is given, the variance will be calculated.
        Default: :any:`None`
    a : :class:`float`, optional
        Parameter a of the U-quadratic distribution (lower bound).
        Default: keep mean and variance
    b : :class:`float`, optional
        Parameter b of the U-quadratic distribution (upper bound).
        Default: keep mean and variance

    Returns
    -------
    :class:`numpy.ndarray`
        Transformed field.
    """
    field = np.asarray(field)
    mean = np.mean(field) if mean is None else float(mean)
    var = np.var(field) if var is None else float(var)
    a = mean - np.sqrt(5.0 / 3.0 * var) if a is None else float(a)
    b = mean + np.sqrt(5.0 / 3.0 * var) if b is None else float(b)
    return _uniform_to_uquad(array_to_uniform(field, mean, var), a, b)


def _uniform_to_arcsin(field, a=0, b=1):
    """
    PPF of your desired distribution.

    The PPF is the inverse of the CDF and is used to sample a distribution
    from uniform distributed values on [0, 1]

    in this case: the arcsin distribution
    See: https://en.wikipedia.org/wiki/Arcsine_distribution
    """
    field = np.asarray(field)
    return (b - a) * np.sin(np.pi * 0.5 * field) ** 2 + a


def _uniform_to_uquad(field, a=0, b=1):
    """
    PPF of your desired distribution.

    The PPF is the inverse of the CDF and is used to sample a distribution
    from uniform distributed values on [0, 1]

    in this case: the U-quadratic distribution
    See: https://en.wikipedia.org/wiki/U-quadratic_distribution
    """
    field = np.asarray(field)
    al = 12 / (b - a) ** 3
    be = (a + b) / 2
    ga = (a - b) ** 3 / 8
    y_raw = 3 * field / al + ga
    result = np.zeros_like(y_raw)
    result[y_raw > 0] = y_raw[y_raw > 0] ** (1 / 3)
    result[y_raw < 0] = -((-y_raw[y_raw < 0]) ** (1 / 3))
    return result + be
