# -*- coding: utf-8 -*-
"""
GStools subpackage providing tools for Normalizers.

.. currentmodule:: gstools.normalizer.tools

The following classes and functions are provided

.. autosummary::
   apply_mean_norm_trend
   remove_trend_norm_mean
"""
import numpy as np

from gstools.normalizer.base import Normalizer
from gstools.tools.misc import eval_func
from gstools.tools.geometric import (
    format_struct_pos_shape,
    format_unstruct_pos_shape,
)

__all__ = ["apply_mean_norm_trend", "remove_trend_norm_mean"]


def _check_normalizer(normalizer):
    if isinstance(normalizer, type) and issubclass(normalizer, Normalizer):
        normalizer = normalizer()
    elif normalizer is None:
        normalizer = Normalizer()
    elif not isinstance(normalizer, Normalizer):
        raise ValueError("Check: 'normalizer' not of type 'Normalizer'.")
    return normalizer


def apply_mean_norm_trend(
    pos,
    field,
    mean=None,
    normalizer=None,
    trend=None,
    mesh_type="unstructured",
    value_type="scalar",
    check_shape=True,
    stacked=False,
):
    """
    Apply mean, de-normalization and trend to given field.

    Parameters
    ----------
    pos : :any:`iterable`
        Position tuple, containing main direction and transversal directions.
    field : :class:`numpy.ndarray` or :class:`list` of :class:`numpy.ndarray`
        The spatially distributed data.
        You can pass a list of fields, that will be used simultaneously.
        Then you need to set ``stacked=True``.
    mean : :any:`None` or :class:`float` or :any:`callable`, optional
        Mean of the field if wanted. Could also be a callable.
        The default is None.
    normalizer : :any:`None` or :any:`Normalizer`, optional
        Normalizer to be applied to the field.
        The default is None.
    trend : :any:`None` or :class:`float` or :any:`callable`, optional
        Trend of the denormalized fields. If no normalizer is applied,
        this behaves equal to 'mean'.
        The default is None.
    mesh_type : :class:`str`, optional
        'structured' / 'unstructured'
        Default: 'unstructured'
    value_type : :class:`str`, optional
        Value type of the field. Either "scalar" or "vector".
        The default is "scalar".
    check_shape : :class:`bool`, optional
        Wheather to check pos and field shapes. The default is True.
    stacked : :class:`bool`, optional
        Wheather the field is stacked or not. The default is False.

    Returns
    -------
    field : :class:`numpy.ndarray`
        The transformed field.
    """
    normalizer = _check_normalizer(normalizer)
    if check_shape:
        if mesh_type != "unstructured":
            pos, shape, dim = format_struct_pos_shape(
                pos, field.shape, check_stacked_shape=stacked
            )
        else:
            pos, shape, dim = format_unstruct_pos_shape(
                pos, field.shape, check_stacked_shape=stacked
            )
        field = np.asarray(field, dtype=np.double).reshape(shape)
    else:
        dim = len(pos)
    if not stacked:
        field = [field]
    field_cnt = len(field)
    for i in range(field_cnt):
        field[i] += eval_func(mean, pos, dim, mesh_type, value_type, True)
    field = normalizer.denormalize(field)
    for i in range(field_cnt):
        field[i] += eval_func(trend, pos, dim, mesh_type, value_type, True)
    return field if stacked else field[0]


def remove_trend_norm_mean(
    pos,
    field,
    mean=None,
    normalizer=None,
    trend=None,
    mesh_type="unstructured",
    value_type="scalar",
    check_shape=True,
    stacked=False,
    fit_normalizer=False,
):
    """
    Remove trend, de-normalization and mean from given field.

    Parameters
    ----------
    pos : :any:`iterable`
        Position tuple, containing main direction and transversal directions.
    field : :class:`numpy.ndarray` or :class:`list` of :class:`numpy.ndarray`
        The spatially distributed data.
        You can pass a list of fields, that will be used simultaneously.
        Then you need to set ``stacked=True``.
    mean : :any:`None` or :class:`float` or :any:`callable`, optional
        Mean of the field if wanted. Could also be a callable.
        The default is None.
    normalizer : :any:`None` or :any:`Normalizer`, optional
        Normalizer to be applied to the field.
        The default is None.
    trend : :any:`None` or :class:`float` or :any:`callable`, optional
        Trend of the denormalized fields. If no normalizer is applied,
        this behaves equal to 'mean'.
        The default is None.
    mesh_type : :class:`str`, optional
        'structured' / 'unstructured'
        Default: 'unstructured'
    value_type : :class:`str`, optional
        Value type of the field. Either "scalar" or "vector".
        The default is "scalar".
    check_shape : :class:`bool`, optional
        Wheather to check pos and field shapes. The default is True.
    stacked : :class:`bool`, optional
        Wheather the field is stacked or not. The default is False.
    fit_normalizer : :class:`bool`, optional
        Wheater to fit the data-normalizer to the given (detrended) field.
        Default: False

    Returns
    -------
    field : :class:`numpy.ndarray`
        The cleaned field.
    normalizer : :any:`Normalizer`, optional
        The fitted normalizer for the given data.
        Only provided if `fit_normalizer` is True.
    """
    normalizer = _check_normalizer(normalizer)
    if check_shape:
        if mesh_type != "unstructured":
            pos, shape, dim = format_struct_pos_shape(
                pos, field.shape, check_stacked_shape=stacked
            )
        else:
            pos, shape, dim = format_unstruct_pos_shape(
                pos, field.shape, check_stacked_shape=stacked
            )
        field = np.asarray(field, dtype=np.double).reshape(shape)
    else:
        dim = len(pos)
    if not stacked:
        field = [field]
    field_cnt = len(field)
    for i in range(field_cnt):
        field[i] -= eval_func(trend, pos, dim, mesh_type, value_type, True)
    if fit_normalizer:
        normalizer.fit(field)
    field = normalizer.normalize(field)
    for i in range(field_cnt):
        field[i] -= eval_func(mean, pos, dim, mesh_type, value_type, True)
    out = field if stacked else field[0]
    return (out, normalizer) if fit_normalizer else out
