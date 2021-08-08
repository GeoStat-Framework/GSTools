# -*- coding: utf-8 -*-
"""
GStools subpackage providing tools for Kriging.

.. currentmodule:: gstools.krige.tools

The following classes and functions are provided

.. autosummary::
   set_condition
   get_drift_functions
"""
# pylint: disable=C0103
from itertools import combinations_with_replacement
import numpy as np


__all__ = ["set_condition", "get_drift_functions"]


def set_condition(cond_pos, cond_val, dim):
    """
    Set the conditions for kriging.

    Parameters
    ----------
    cond_pos : :class:`list`
        the position tuple of the conditions (x, [y, z])
    cond_val : :class:`numpy.ndarray`
        the values of the conditions (nan values will be ignored)
    dim : :class:`int`, optional
        Spatial dimension

    Raises
    ------
    ValueError
        If the given data does not match the given dimension.

    Returns
    -------
    cond_pos : :class:`list`
        the error checked cond_pos with all finite values
    cond_val : :class:`numpy.ndarray`
        the error checked cond_val for all finite cond_pos values
    """
    # convert the input for right shapes and dimension checks
    cond_val = np.asarray(cond_val, dtype=np.double).reshape(-1)
    cond_pos = np.asarray(cond_pos, dtype=np.double).reshape(dim, -1)
    if len(cond_pos[0]) != len(cond_val):
        raise ValueError(
            "Please check your 'cond_pos' and 'cond_val' parameters. "
            "The shapes do not match."
        )
    mask = np.isfinite(cond_val)
    return cond_pos[:, mask], cond_val[mask]


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
