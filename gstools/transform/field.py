# -*- coding: utf-8 -*-
"""
GStools subpackage providing field transformations.

.. currentmodule:: gstools.transform.field

The following functions are provided

Wrapper
^^^^^^^

.. autosummary::
   apply

Transformations
^^^^^^^^^^^^^^^

.. autosummary::
   apply_function
   binary
   discrete
   boxcox
   zinnharvey
   normal_force_moments
   normal_to_lognormal
   normal_to_uniform
   normal_to_arcsin
   normal_to_uquad

Low-Level Routines
^^^^^^^^^^^^^^^^^^

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
from gstools.normalizer import (
    Normalizer,
    remove_trend_norm_mean,
    apply_mean_norm_trend,
)

__all__ = [
    "apply",
    "apply_function",
    "binary",
    "discrete",
    "boxcox",
    "zinnharvey",
    "normal_force_moments",
    "normal_to_lognormal",
    "normal_to_uniform",
    "normal_to_arcsin",
    "normal_to_uquad",
    "array_discrete",
    "array_boxcox",
    "array_zinnharvey",
    "array_force_moments",
    "array_to_lognormal",
    "array_to_uniform",
    "array_to_arcsin",
    "array_to_uquad",
]


def _pre_process(fld, data, keep_mean):
    return remove_trend_norm_mean(
        pos=fld.pos,
        field=data,
        mean=None if keep_mean else fld.mean,
        normalizer=fld.normalizer,
        trend=fld.trend,
        mesh_type=fld.mesh_type,
        value_type=fld.value_type,
        check_shape=False,
    )


def _post_process(fld, data, keep_mean):
    return apply_mean_norm_trend(
        pos=fld.pos,
        field=data,
        mean=None if keep_mean else fld.mean,
        normalizer=fld.normalizer,
        trend=fld.trend,
        mesh_type=fld.mesh_type,
        value_type=fld.value_type,
        check_shape=False,
    )


def _check_for_default_normal(fld):
    if not type(fld.normalizer) == Normalizer:
        raise ValueError(
            "transform: need a normal field but there is a normalizer defined"
        )
    if fld.trend is not None:
        raise ValueError(
            "transform: need a normal field but there is a trend defined"
        )
    if callable(fld.mean) or fld.mean is None:
        raise ValueError(
            "transform: need a normal field but mean is not constant"
        )


def apply(fld, method, field="field", store=True, process=False, **kwargs):
    """
    Apply field transformation.

    Parameters
    ----------
    fld : :any:`Field`
        Field class containing a generated field.
    method : :class:`str`
        Method to use. See :any:`transform` for available transformations.
    field : :class:`str`, optional
        Name of field to be transformed. The default is "field".
    store : :class:`str` or :class:`bool`, optional
        Whether to store field inplace (True/False) or with a specified name.
        The default is True.
    process : :class:`bool`, optional
        Whether to process in/out fields with trend, normalizer and mean
        of given Field instance. The default is False.
    **kwargs
        Keyword arguments forwarded to selected method.

    Raises
    ------
    ValueError
        When method is unknown.

    Returns
    -------
    :class:`numpy.ndarray`
        Transformed field.
    """
    kwargs["field"] = field
    kwargs["store"] = store
    kwargs["process"] = process
    method = str(method)  # ensure method is a string
    if method == "binary":
        return binary(fld, **kwargs)
    if method == "discrete":
        return discrete(fld, **kwargs)
    if method == "boxcox":
        return boxcox(fld, **kwargs)
    if method == "zinnharvey":
        return zinnharvey(fld, **kwargs)
    if method.endswith("force_moments"):
        return normal_force_moments(fld, **kwargs)
    if method.endswith("lognormal"):
        return normal_to_lognormal(fld, **kwargs)
    if method.endswith("uniform"):
        return normal_to_uniform(fld, **kwargs)
    if method.endswith("arcsin"):
        return normal_to_arcsin(fld, **kwargs)
    if method.endswith("uquad"):
        return normal_to_uquad(fld, **kwargs)
    if method.endswith("function"):
        return apply_function(fld, **kwargs)
    raise ValueError(f"transform.apply: unknown method '{method}'")


def apply_function(
    fld,
    function,
    field="field",
    store=True,
    process=False,
    keep_mean=False,
    **kwargs,
):
    """
    Apply function as field transformation.

    Parameters
    ----------
    fld : :any:`Field`
        Field class containing a generated field.
    function : :any:`callable`
        Function to use.
    field : :class:`str`, optional
        Name of field to be transformed. The default is "field".
    store : :class:`str` or :class:`bool`, optional
        Whether to store field inplace (True/False) or under a given name.
        The default is True.
    process : :class:`bool`, optional
        Whether to process in/out fields with trend, normalizer and mean
        of given Field instance. The default is False.
    keep_mean : :class:`bool`, optional
        Whether to keep the mean of the field if process=True.
        The default is False.
    **kwargs
        Keyword arguments forwarded to given function.

    Raises
    ------
    ValueError
        When function is not callable.

    Returns
    -------
    :class:`numpy.ndarray`
        Transformed field.
    """
    if not callable(function):
        raise ValueError("transform.apply_function: function not a 'callable'")
    data = fld[field]
    name, save = fld._get_store_config(store, default=field)
    if process:
        data = _pre_process(fld, data, keep_mean=keep_mean)
    data = function(data, **kwargs)
    if process:
        data = _post_process(fld, data, keep_mean=keep_mean)
    return fld.post_field(data, name=name, process=False, save=save)


def binary(
    fld,
    divide=None,
    upper=None,
    lower=None,
    field="field",
    store=True,
    process=False,
):
    """
    Binary transformation.

    After this transformation, the field only has two values.

    Parameters
    ----------
    fld : :any:`Field`
        Field class containing a generated field.
    divide : :class:`float`, optional
        The dividing value.
        Default: ``fld.mean``
    upper : :class:`float`, optional
        The resulting upper value of the field.
        Default: ``mean + sqrt(fld.model.sill)``
    lower : :class:`float`, optional
        The resulting lower value of the field.
        Default: ``mean - sqrt(fld.model.sill)``
    field : :class:`str`, optional
        Name of field to be transformed. The default is "field".
    store : :class:`str` or :class:`bool`, optional
        Whether to store field inplace (True/False) or under a given name.
        The default is True.
    process : :class:`bool`, optional
        Whether to process in/out fields with trend, normalizer and mean
        of given Field instance. The default is False.

    Returns
    -------
    :class:`numpy.ndarray`
        Transformed field.
    """
    if not process and divide is None:
        _check_for_default_normal(fld)
    if divide is None:
        mean = 0.0 if process else fld.mean
    divide = mean if divide is None else divide
    upper = mean + np.sqrt(fld.model.sill) if upper is None else upper
    lower = mean - np.sqrt(fld.model.sill) if lower is None else lower
    kw = dict(
        values=[lower, upper],
        thresholds=[divide],
    )
    return apply_function(
        fld=fld,
        function=array_discrete,
        field=field,
        store=store,
        process=process,
        keep_mean=False,
        **kw,
    )


def discrete(
    fld,
    values,
    thresholds="arithmetic",
    field="field",
    store=True,
    process=False,
):
    """
    Discrete transformation.

    After this transformation, the field has only `len(values)` discrete
    values.

    Parameters
    ----------
    fld : :any:`Field`
        Field class containing a generated field.
    values : :any:`numpy.ndarray`
        The discrete values the field will take
    thresholds : :class:`str` or :any:`numpy.ndarray`, optional
        the thresholds, where the value classes are separated
        possible values are:
        * "arithmetic": the mean of the 2 neighbouring values
        * "equal": devide the field into equal parts
        * an array of explicitly given thresholds
        Default: "arithmetic"
    field : :class:`str`, optional
        Name of field to be transformed. The default is "field".
    store : :class:`str` or :class:`bool`, optional
        Whether to store field inplace (True/False) or under a given name.
        The default is True.
    process : :class:`bool`, optional
        Whether to process in/out fields with trend, normalizer and mean
        of given Field instance. The default is False.

    Returns
    -------
    :class:`numpy.ndarray`
        Transformed field.
    """
    if not process and thresholds == "equal":
        _check_for_default_normal(fld)
    kw = dict(
        values=values,
        thresholds=thresholds,
        mean=0.0 if process else fld.mean,
        var=fld.model.sill,
    )
    return apply_function(
        fld=fld,
        function=array_discrete,
        field=field,
        store=store,
        process=process,
        keep_mean=False,
        **kw,
    )


def boxcox(fld, lmbda=1, shift=0, field="field", store=True, process=False):
    """
    (Inverse) Box-Cox transformation to denormalize data.

    After this transformation, the again Box-Cox transformed field is normal
    distributed.

    See: https://en.wikipedia.org/wiki/Power_transform#Box%E2%80%93Cox_transformation

    Parameters
    ----------
    fld : :any:`Field`
        Field class containing a generated field.
    lmbda : :class:`float`, optional
        The lambda parameter of the Box-Cox transformation.
        For ``lmbda=0`` one obtains the log-normal transformation.
        Default: ``1``
    shift : :class:`float`, optional
        The shift parameter from the two-parametric Box-Cox transformation.
        The field will be shifted by that value before transformation.
        Default: ``0``
    field : :class:`str`, optional
        Name of field to be transformed. The default is "field".
    store : :class:`str` or :class:`bool`, optional
        Whether to store field inplace (True/False) or under a given name.
        The default is True.
    process : :class:`bool`, optional
        Whether to process in/out fields with trend, normalizer and mean
        of given Field instance. The default is False.

    Returns
    -------
    :class:`numpy.ndarray`
        Transformed field.
    """
    kw = dict(lmbda=lmbda, shift=shift)
    return apply_function(
        fld=fld,
        function=array_boxcox,
        field=field,
        store=store,
        process=process,
        keep_mean=False,
        **kw,
    )


def zinnharvey(fld, conn="high", field="field", store=True, process=False):
    """
    Zinn and Harvey transformation to connect low or high values.

    After this transformation, the field is still normal distributed.

    Parameters
    ----------
    fld : :any:`Field`
        Field class containing a generated field.
    conn : :class:`str`, optional
        Desired connectivity. Either "low" or "high".
        Default: "high"
    field : :class:`str`, optional
        Name of field to be transformed. The default is "field".
    store : :class:`str` or :class:`bool`, optional
        Whether to store field inplace (True/False) or under a given name.
        The default is True.
    process : :class:`bool`, optional
        Whether to process in/out fields with trend, normalizer and mean
        of given Field instance. The default is False.

    Returns
    -------
    :class:`numpy.ndarray`
        Transformed field.
    """
    if not process:
        _check_for_default_normal(fld)
    kw = dict(conn=conn, mean=0.0 if process else fld.mean, var=fld.model.sill)
    return apply_function(
        fld=fld,
        function=array_zinnharvey,
        field=field,
        store=store,
        process=process,
        keep_mean=False,
        **kw,
    )


def normal_force_moments(fld, field="field", store=True, process=False):
    """
    Force moments of a normal distributed field.

    After this transformation, the field is still normal distributed.

    Parameters
    ----------
    fld : :any:`Field`
        Field class containing a generated field.
    field : :class:`str`, optional
        Name of field to be transformed. The default is "field".
    store : :class:`str` or :class:`bool`, optional
        Whether to store field inplace (True/False) or under a given name.
        The default is True.
    process : :class:`bool`, optional
        Whether to process in/out fields with trend, normalizer and mean
        of given Field instance. The default is False.

    Returns
    -------
    :class:`numpy.ndarray`
        Transformed field.
    """
    if not process:
        _check_for_default_normal(fld)
    kw = dict(mean=0.0 if process else fld.mean, var=fld.model.sill)
    return apply_function(
        fld=fld,
        function=array_force_moments,
        field=field,
        store=store,
        process=process,
        keep_mean=False,
        **kw,
    )


def normal_to_lognormal(fld, field="field", store=True, process=False):
    """
    Transform normal distribution to log-normal distribution.

    After this transformation, the field is log-normal distributed.

    Parameters
    ----------
    fld : :any:`Field`
        Field class containing a generated field.
    field : :class:`str`, optional
        Name of field to be transformed. The default is "field".
    store : :class:`str` or :class:`bool`, optional
        Whether to store field inplace (True/False) or under a given name.
        The default is True.
    process : :class:`bool`, optional
        Whether to process in/out fields with trend, normalizer and mean
        of given Field instance. The default is False.

    Returns
    -------
    :class:`numpy.ndarray`
        Transformed field.
    """
    return apply_function(
        fld=fld,
        function=array_to_lognormal,
        field=field,
        store=store,
        process=process,
        keep_mean=True,  # apply to normal field including mean
    )


def normal_to_uniform(fld, field="field", store=True, process=False):
    """
    Transform normal distribution to uniform distribution on [0, 1].

    After this transformation, the field is uniformly distributed on [0, 1].

    Parameters
    ----------
    fld : :any:`Field`
        Field class containing a generated field.
    """
    if not process:
        _check_for_default_normal(fld)
    kw = dict(mean=0.0 if process else fld.mean, var=fld.model.sill)
    return apply_function(
        fld=fld,
        function=array_to_uniform,
        field=field,
        store=store,
        process=process,
        keep_mean=False,
        **kw,
    )


def normal_to_arcsin(
    fld, a=None, b=None, field="field", store=True, process=False
):
    """
    Transform normal distribution to the bimodal arcsin distribution.

    See: https://en.wikipedia.org/wiki/Arcsine_distribution

    After this transformation, the field is arcsin-distributed on [a, b].

    Parameters
    ----------
    fld : :any:`Field`
        Field class containing a generated field.
    a : :class:`float`, optional
        Parameter a of the arcsin distribution (lower bound).
        Default: keep mean and variance
    b : :class:`float`, optional
        Parameter b of the arcsin distribution (upper bound).
        Default: keep mean and variance
    field : :class:`str`, optional
        Name of field to be transformed. The default is "field".
    store : :class:`str` or :class:`bool`, optional
        Whether to store field inplace (True/False) or under a given name.
        The default is True.
    process : :class:`bool`, optional
        Whether to process in/out fields with trend, normalizer and mean
        of given Field instance. The default is False.

    Returns
    -------
    :class:`numpy.ndarray`
        Transformed field.
    """
    if not process:
        _check_for_default_normal(fld)
    kw = dict(mean=0.0 if process else fld.mean, var=fld.model.sill, a=a, b=b)
    return apply_function(
        fld=fld,
        function=array_to_arcsin,
        field=field,
        store=store,
        process=process,
        keep_mean=False,
        **kw,
    )


def normal_to_uquad(
    fld, a=None, b=None, field="field", store=True, process=False
):
    """
    Transform normal distribution to U-quadratic distribution.

    See: https://en.wikipedia.org/wiki/U-quadratic_distribution

    After this transformation, the field is U-quadratic-distributed on [a, b].

    Parameters
    ----------
    fld : :any:`Field`
        Field class containing a generated field.
    a : :class:`float`, optional
        Parameter a of the U-quadratic distribution (lower bound).
        Default: keep mean and variance
    b : :class:`float`, optional
        Parameter b of the U-quadratic distribution (upper bound).
        Default: keep mean and variance
    field : :class:`str`, optional
        Name of field to be transformed. The default is "field".
    store : :class:`str` or :class:`bool`, optional
        Whether to store field inplace (True/False) or under a given name.
        The default is True.
    process : :class:`bool`, optional
        Whether to process in/out fields with trend, normalizer and mean
        of given Field instance. The default is False.

    Returns
    -------
    :class:`numpy.ndarray`
        Transformed field.
    """
    if not process:
        _check_for_default_normal(fld)
    kw = dict(mean=0.0 if process else fld.mean, var=fld.model.sill, a=a, b=b)
    return apply_function(
        fld=fld,
        function=array_to_uquad,
        field=field,
        store=store,
        process=process,
        keep_mean=False,
        **kw,
    )


# low level functions


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
        values = np.array(values)
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
        values = np.array(values)
        thresholds = np.array(thresholds)
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
        array_to_lognormal(result)
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
