# -*- coding: utf-8 -*-
"""
GStools subpackage providing tools for Kriging.

.. currentmodule:: gstools.krige.tools

The following classes and functions are provided

.. autosummary::
   set_condition
   get_drift_functions
   no_trend
   eval_func
"""
# pylint: disable=C0103
from itertools import combinations_with_replacement
import numpy as np
from gstools.tools.geometric import pos2xyz, xyz2pos
from gstools.field.tools import (
    reshape_axis_from_struct_to_unstruct,
    reshape_field_from_unstruct_to_struct,
)

__all__ = ["no_trend", "eval_func", "set_condition", "get_drift_functions"]


def no_trend(*args, **kwargs):
    """
    Zero trend dummy function.

    Parameters
    ----------
    *args : any
        Ignored arguments.
    **kwargs : any
        Ignored keyword arguments.

    Returns
    -------
    float
        A zero trend given as single float.

    """
    return 0.0


def eval_func(func, pos, mesh_type="structured"):
    """
    Evaluate a function on a mesh.

    Parameters
    ----------
    func : :any:`callable`
        The function to be called. Should have the signiture f(x, [y, z])
    pos : :class:`list`
        the position tuple, containing main direction and transversal
        directions (x, [y, z])
    mesh_type : :class:`str`, optional
        'structured' / 'unstructured'

    Returns
    -------
    :class:`numpy.ndarray`
        Function values at the given points.
    """
    x, y, z, dim = pos2xyz(pos, calc_dim=True)
    if mesh_type == "structured":
        x, y, z, axis_lens = reshape_axis_from_struct_to_unstruct(dim, x, y, z)
    res = func(*[x, y, z][:dim])
    if mesh_type == "structured":
        res = reshape_field_from_unstruct_to_struct(dim, res, axis_lens)
    return res


def set_condition(cond_pos, cond_val, max_dim=3):
    """
    Set the conditions for kriging.

    Parameters
    ----------
    cond_pos : :class:`list`
        the position tuple of the conditions (x, [y, z])
    cond_val : :class:`numpy.ndarray`
        the values of the conditions
    max_dim : :class:`int`, optional
        Cut of information above the given dimension. Default: 3

    Raises
    ------
    ValueError
        If the given data does not match the given dimension.

    Returns
    -------
    cond_pos : :class:`list`
        the error checked cond_pos
    cond_val : :class:`numpy.ndarray`
        the error checked cond_val
    """
    # convert the input for right shapes and dimension checks
    c_x, c_y, c_z = pos2xyz(cond_pos, dtype=np.double, max_dim=max_dim)
    cond_pos = xyz2pos(c_x, c_y, c_z)
    if len(cond_pos) != max_dim:
        raise ValueError(
            "Please check your 'cond_pos' parameters. "
            + "The dimension does not match with the given one."
        )
    cond_val = np.array(cond_val, dtype=np.double).reshape(-1)
    if not all([len(cond_pos[i]) == len(cond_val) for i in range(max_dim)]):
        raise ValueError(
            "Please check your 'cond_pos' and 'cond_val' parameters. "
            + "The shapes do not match."
        )
    return cond_pos, cond_val


def get_drift_functions(dim, drift_type):
    """
    Get functions for a given drift type in universal kriging.

    Parameters
    ----------
    dim : :class:`int`
        Given dimension.
    drift_type : :class:`str` or :class:`int`
        Drift type: 'linear' or 'quadratic' or an integer for the polynomial
        order of the drift type. (linear equals 1, quadratic equals 2 ...)

    Returns
    -------
    :class:`list` of :any:`callable`
        List of drift functions.
    """
    if drift_type in ["lin", "linear"]:
        drift_type = 1
    elif drift_type in ["quad", "quadratic"]:
        drift_type = 2
    else:
        drift_type = int(drift_type)
    drift_functions = []
    for d in range(drift_type):
        selects = combinations_with_replacement(range(dim), d + 1)
        for select in selects:
            drift_functions.append(_f_factory(select))
    return drift_functions


def _f_factory(select):
    def f(*pos):
        res = 1.0
        for i in select:
            res *= np.asarray(pos[i])
        return res

    return f
