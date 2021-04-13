# -*- coding: utf-8 -*-
"""
GStools subpackage providing field transformations.

.. currentmodule:: gstools.transform.field

The following functions are provided

.. autosummary::
   binary
   discrete
   boxcox
   zinnharvey
   normal_force_moments
   normal_to_lognormal
   normal_to_uniform
   normal_to_arcsin
   normal_to_uquad
"""
# pylint: disable=C0103
from warnings import warn
import numpy as np
from scipy.special import erf, erfinv


__all__ = [
    "binary",
    "discrete",
    "boxcox",
    "zinnharvey",
    "normal_force_moments",
    "normal_to_lognormal",
    "normal_to_uniform",
    "normal_to_arcsin",
    "normal_to_uquad",
]


def binary(fld, divide=None, upper=None, lower=None):
    """
    Binary transformation.

    After this transformation, the field only has two values.

    Parameters
    ----------
    fld : :any:`Field`
        Spatial Random Field class containing a generated field.
        Field will be transformed inplace.
    divide : :class:`float`, optional
        The dividing value.
        Default: ``fld.mean``
    upper : :class:`float`, optional
        The resulting upper value of the field.
        Default: ``mean + sqrt(fld.model.sill)``
    lower : :class:`float`, optional
        The resulting lower value of the field.
        Default: ``mean - sqrt(fld.model.sill)``
    """
    if fld.field is None:
        print("binary: no field stored in SRF class.")
    else:
        divide = fld.mean if divide is None else divide
        upper = fld.mean + np.sqrt(fld.model.sill) if upper is None else upper
        lower = fld.mean - np.sqrt(fld.model.sill) if lower is None else lower
        discrete(fld, [lower, upper], thresholds=[divide])


def discrete(fld, values, thresholds="arithmetic"):
    """
    Discrete transformation.

    After this transformation, the field has only `len(values)` discrete
    values.

    Parameters
    ----------
    fld : :any:`Field`
        Spatial Random Field class containing a generated field.
        Field will be transformed inplace.
    values : :any:`numpy.ndarray`
        The discrete values the field will take
    thresholds : :class:`str` or :any:`numpy.ndarray`, optional
        the thresholds, where the value classes are separated
        possible values are:
        * "arithmetic": the mean of the 2 neighbouring values
        * "equal": devide the field into equal parts
        * an array of explicitly given thresholds
        Default: "arithmetic"
    """
    if fld.field is None:
        print("discrete: no field stored in SRF class.")
    else:
        if thresholds == "arithmetic":
            # just in case, sort the values
            values = np.sort(values)
            thresholds = (values[1:] + values[:-1]) / 2
        elif thresholds == "equal":
            values = np.array(values)
            n = len(values)
            p = np.arange(1, n) / n  # n-1 equal subdivisions of [0, 1]
            rescale = np.sqrt(fld.model.sill * 2)
            # use quantile of the normal distribution to get equal ratios
            thresholds = fld.mean + rescale * erfinv(2 * p - 1)
        else:
            if len(values) != len(thresholds) + 1:
                raise ValueError(
                    "discrete transformation: "
                    "len(values) != len(thresholds) + 1"
                )
            values = np.array(values)
            thresholds = np.array(thresholds)
        # check thresholds
        if not np.all(thresholds[:-1] < thresholds[1:]):
            raise ValueError(
                "discrete transformation: thresholds need to be ascending."
            )
        # use a separate result so the intermediate results are not affected
        result = np.empty_like(fld.field)
        # handle edge cases
        result[fld.field <= thresholds[0]] = values[0]
        result[fld.field > thresholds[-1]] = values[-1]
        for i, value in enumerate(values[1:-1]):
            result[
                np.logical_and(
                    thresholds[i] < fld.field, fld.field <= thresholds[i + 1]
                )
            ] = value
        # overwrite the field
        fld.field = result


def boxcox(fld, lmbda=1, shift=0):
    """
    (Inverse) Box-Cox transformation to denormalize data.

    After this transformation, the again Box-Cox transformed field is normal
    distributed.

    See: https://en.wikipedia.org/wiki/Power_transform#Box%E2%80%93Cox_transformation

    Parameters
    ----------
    fld : :any:`Field`
        Spatial Random Field class containing a generated field.
        Field will be transformed inplace.
    lmbda : :class:`float`, optional
        The lambda parameter of the Box-Cox transformation.
        For ``lmbda=0`` one obtains the log-normal transformation.
        Default: ``1``
    shift : :class:`float`, optional
        The shift parameter from the two-parametric Box-Cox transformation.
        The field will be shifted by that value before transformation.
        Default: ``0``
    """
    if fld.field is None:
        print("Box-Cox: no field stored in SRF class.")
    else:
        fld.mean += shift
        fld.field += shift
        if np.isclose(lmbda, 0):
            normal_to_lognormal(fld)
        if np.min(fld.field) < -1 / lmbda:
            warn("Box-Cox: Some values will be cut off!")
        fld.field = (np.maximum(lmbda * fld.field + 1, 0)) ** (1 / lmbda)


def zinnharvey(fld, conn="high"):
    """
    Zinn and Harvey transformation to connect low or high values.

    After this transformation, the field is still normal distributed.

    Parameters
    ----------
    fld : :any:`Field`
        Spatial Random Field class containing a generated field.
        Field will be transformed inplace.
    conn : :class:`str`, optional
        Desired connectivity. Either "low" or "high".
        Default: "high"
    """
    if fld.field is None:
        print("zinnharvey: no field stored in SRF class.")
    else:
        fld.field = _zinnharvey(fld.field, conn, fld.mean, fld.model.sill)


def normal_force_moments(fld):
    """
    Force moments of a normal distributed field.

    After this transformation, the field is still normal distributed.

    Parameters
    ----------
    fld : :any:`Field`
        Spatial Random Field class containing a generated field.
        Field will be transformed inplace.
    """
    if fld.field is None:
        print("normal_force_moments: no field stored in SRF class.")
    else:
        fld.field = _normal_force_moments(fld.field, fld.mean, fld.model.sill)


def normal_to_lognormal(fld):
    """
    Transform normal distribution to log-normal distribution.

    After this transformation, the field is log-normal distributed.

    Parameters
    ----------
    fld : :any:`Field`
        Spatial Random Field class containing a generated field.
        Field will be transformed inplace.
    """
    if fld.field is None:
        print("normal_to_lognormal: no field stored in SRF class.")
    else:
        fld.field = _normal_to_lognormal(fld.field)


def normal_to_uniform(fld):
    """
    Transform normal distribution to uniform distribution on [0, 1].

    After this transformation, the field is uniformly distributed on [0, 1].

    Parameters
    ----------
    fld : :any:`Field`
        Spatial Random Field class containing a generated field.
        Field will be transformed inplace.
    """
    if fld.field is None:
        print("normal_to_uniform: no field stored in SRF class.")
    else:
        fld.field = _normal_to_uniform(fld.field, fld.mean, fld.model.sill)


def normal_to_arcsin(fld, a=None, b=None):
    """
    Transform normal distribution to the bimodal arcsin distribution.

    See: https://en.wikipedia.org/wiki/Arcsine_distribution

    After this transformation, the field is arcsin-distributed on [a, b].

    Parameters
    ----------
    fld : :any:`Field`
        Spatial Random Field class containing a generated field.
        Field will be transformed inplace.
    a : :class:`float`, optional
        Parameter a of the arcsin distribution (lower bound).
        Default: keep mean and variance
    b : :class:`float`, optional
        Parameter b of the arcsin distribution (upper bound).
        Default: keep mean and variance
    """
    if fld.field is None:
        print("normal_to_arcsin: no field stored in SRF class.")
    else:
        a = fld.mean - np.sqrt(2.0 * fld.model.sill) if a is None else a
        b = fld.mean + np.sqrt(2.0 * fld.model.sill) if b is None else b
        fld.field = _normal_to_arcsin(
            fld.field, fld.mean, fld.model.sill, a, b
        )
        fld.mean = (a + b) / 2.0


def normal_to_uquad(fld, a=None, b=None):
    """
    Transform normal distribution to U-quadratic distribution.

    See: https://en.wikipedia.org/wiki/U-quadratic_distribution

    After this transformation, the field is U-quadratic-distributed on [a, b].

    Parameters
    ----------
    fld : :any:`Field`
        Spatial Random Field class containing a generated field.
        Field will be transformed inplace.
    a : :class:`float`, optional
        Parameter a of the U-quadratic distribution (lower bound).
        Default: keep mean and variance
    b : :class:`float`, optional
        Parameter b of the U-quadratic distribution (upper bound).
        Default: keep mean and variance
    """
    if fld.field is None:
        print("normal_to_uquad: no field stored in SRF class.")
    else:
        a = fld.mean - np.sqrt(5.0 / 3.0 * fld.model.sill) if a is None else a
        b = fld.mean + np.sqrt(5.0 / 3.0 * fld.model.sill) if b is None else b
        fld.field = _normal_to_uquad(fld.field, fld.mean, fld.model.sill, a, b)
        fld.mean = (a + b) / 2.0


# low level functions


def _zinnharvey(field, conn="high", mean=None, var=None):
    """
    Zinn and Harvey transformation to connect low or high values.

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
    field = np.abs((field - mean) / np.sqrt(var))
    field = 2 * erf(field / np.sqrt(2)) - 1
    field = np.sqrt(2) * erfinv(field)
    if conn == "high":
        field = -field
    return field * np.sqrt(var) + mean


def _normal_force_moments(field, mean=0, var=1):
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


def _normal_to_lognormal(field):
    """
    Transform normal distribution to log-normal distribution.

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


def _normal_to_uniform(field, mean=None, var=None):
    """
    Transform normal distribution to uniform distribution on [0, 1].

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


def _normal_to_arcsin(field, mean=None, var=None, a=0, b=1):
    """
    Transform normal distribution to arcsin distribution.

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
    return _uniform_to_arcsin(_normal_to_uniform(field, mean, var), a, b)


def _normal_to_uquad(field, mean=None, var=None, a=0, b=1):
    """
    Transform normal distribution to U-quadratic distribution.

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
    return _uniform_to_uquad(_normal_to_uniform(field, mean, var), a, b)


def _uniform_to_arcsin(field, a=0, b=1):
    """
    PPF of your desired distribution.

    The PPF is the inverse of the CDF and is used to sample a distribution
    from uniform distributed values on [0, 1]

    in this case: the arcsin distribution
    See: https://en.wikipedia.org/wiki/Arcsine_distribution
    """
    return (b - a) * np.sin(np.pi * 0.5 * field) ** 2 + a


def _uniform_to_uquad(field, a=0, b=1):
    """
    PPF of your desired distribution.

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
    out[y_raw < 0] = -((-y_raw[y_raw < 0]) ** (1 / 3))
    return out + be
