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
"""
# pylint: disable=C0103, C0123, R0911
import numpy as np
from gstools.normalizer import (
    Normalizer,
    remove_trend_norm_mean,
    apply_mean_norm_trend,
)
from gstools.transform.array import (
    array_discrete,
    array_boxcox,
    array_zinnharvey,
    array_force_moments,
    array_to_lognormal,
    array_to_uniform,
    array_to_arcsin,
    array_to_uquad,
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
        Method to use.
        See :any:`gstools.transform` for available transformations.
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
    keep_mean=True,
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
        The default is True.
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
    name, save = fld.get_store_config(store, default=field)
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
    keep_mean=True,
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
    keep_mean : :class:`bool`, optional
        Whether to keep the mean of the field if process=True.
        The default is True.

    Returns
    -------
    :class:`numpy.ndarray`
        Transformed field.
    """
    if not process and divide is None:
        _check_for_default_normal(fld)
    if divide is None:
        mean = 0.0 if process and not keep_mean else fld.mean
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
        keep_mean=keep_mean,
        **kw,
    )


def discrete(
    fld,
    values,
    thresholds="arithmetic",
    field="field",
    store=True,
    process=False,
    keep_mean=True,
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
    keep_mean : :class:`bool`, optional
        Whether to keep the mean of the field if process=True.
        The default is True.

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
        mean=0.0 if process and not keep_mean else fld.mean,
        var=fld.model.sill,
    )
    return apply_function(
        fld=fld,
        function=array_discrete,
        field=field,
        store=store,
        process=process,
        keep_mean=keep_mean,
        **kw,
    )


def boxcox(
    fld,
    lmbda=1,
    shift=0,
    field="field",
    store=True,
    process=False,
    keep_mean=True,
):
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
    keep_mean : :class:`bool`, optional
        Whether to keep the mean of the field if process=True.
        The default is True.

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
        keep_mean=keep_mean,
        **kw,
    )


def zinnharvey(
    fld,
    conn="high",
    field="field",
    store=True,
    process=False,
    keep_mean=True,
):
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
    keep_mean : :class:`bool`, optional
        Whether to keep the mean of the field if process=True.
        The default is True.

    Returns
    -------
    :class:`numpy.ndarray`
        Transformed field.
    """
    if not process:
        _check_for_default_normal(fld)
    kw = dict(
        conn=conn,
        mean=0.0 if process and not keep_mean else fld.mean,
        var=fld.model.sill,
    )
    return apply_function(
        fld=fld,
        function=array_zinnharvey,
        field=field,
        store=store,
        process=process,
        keep_mean=keep_mean,
        **kw,
    )


def normal_force_moments(
    fld,
    field="field",
    store=True,
    process=False,
    keep_mean=True,
):
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
    keep_mean : :class:`bool`, optional
        Whether to keep the mean of the field if process=True.
        The default is True.

    Returns
    -------
    :class:`numpy.ndarray`
        Transformed field.
    """
    if not process:
        _check_for_default_normal(fld)
    kw = dict(
        mean=0.0 if process and not keep_mean else fld.mean, var=fld.model.sill
    )
    return apply_function(
        fld=fld,
        function=array_force_moments,
        field=field,
        store=store,
        process=process,
        keep_mean=keep_mean,
        **kw,
    )


def normal_to_lognormal(
    fld, field="field", store=True, process=False, keep_mean=True
):
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
    keep_mean : :class:`bool`, optional
        Whether to keep the mean of the field if process=True.
        The default is True.

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
        keep_mean=keep_mean,
    )


def normal_to_uniform(
    fld,
    field="field",
    store=True,
    process=False,
    keep_mean=True,
):
    """
    Transform normal distribution to uniform distribution on [0, 1].

    After this transformation, the field is uniformly distributed on [0, 1].

    Parameters
    ----------
    fld : :any:`Field`
        Field class containing a generated field.
    keep_mean : :class:`bool`, optional
        Whether to keep the mean of the field if process=True.
        The default is True.
    """
    if not process:
        _check_for_default_normal(fld)
    kw = dict(
        mean=0.0 if process and not keep_mean else fld.mean, var=fld.model.sill
    )
    return apply_function(
        fld=fld,
        function=array_to_uniform,
        field=field,
        store=store,
        process=process,
        keep_mean=keep_mean,
        **kw,
    )


def normal_to_arcsin(
    fld,
    a=None,
    b=None,
    field="field",
    store=True,
    process=False,
    keep_mean=True,
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
    keep_mean : :class:`bool`, optional
        Whether to keep the mean of the field if process=True.
        The default is True.

    Returns
    -------
    :class:`numpy.ndarray`
        Transformed field.
    """
    if not process:
        _check_for_default_normal(fld)
    kw = dict(
        mean=0.0 if process and not keep_mean else fld.mean,
        var=fld.model.sill,
        a=a,
        b=b,
    )
    return apply_function(
        fld=fld,
        function=array_to_arcsin,
        field=field,
        store=store,
        process=process,
        keep_mean=keep_mean,
        **kw,
    )


def normal_to_uquad(
    fld,
    a=None,
    b=None,
    field="field",
    store=True,
    process=False,
    keep_mean=True,
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
    keep_mean : :class:`bool`, optional
        Whether to keep the mean of the field if process=True.
        The default is True.

    Returns
    -------
    :class:`numpy.ndarray`
        Transformed field.
    """
    if not process:
        _check_for_default_normal(fld)
    kw = dict(
        mean=0.0 if process and not keep_mean else fld.mean,
        var=fld.model.sill,
        a=a,
        b=b,
    )
    return apply_function(
        fld=fld,
        function=array_to_uquad,
        field=field,
        store=store,
        process=process,
        keep_mean=keep_mean,
        **kw,
    )
