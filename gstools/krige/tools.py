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

import numpy as np
from gstools.tools.geometric import pos2xyz, xyz2pos

__all__ = ["set_condition", "get_drift_functions"]


def set_condition(cond_pos, cond_val, max_dim=3):
    """Set the conditions for kriging.

    Parameters
    ----------
    cond_pos : :class:`list`
        the position tuple of the conditions (x, [y, z])
    cond_val : :class:`numpy.ndarray`
        the values of the conditions
    max_dim : :class:`int`, optional
        Cut of information above the given dimension. Default: 3

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
    cond_val = np.array(cond_val, dtype=np.double).reshape(-1)
    if not all([len(cond_pos[i]) == len(cond_val) for i in range(max_dim)]):
        raise ValueError(
            "Please check your 'cond_pos' and 'cond_val' parameters. "
            + "The shapes do not match."
        )
    return cond_pos, cond_val


def get_drift_functions(dim, drift_type):
    """
    Get functions for a given drift type.

    Parameters
    ----------
    dim : :class:`int`
        Given dimension.
    drift_type : :class:`str`
        Drift type: 'ordinary', 'linear' or 'quadratic'.

    Raises
    ------
    ValueError
        If given drift_type is unknown.

    Returns
    -------
    :class:`list` of :any:`callable`
        List of drift functions.
    """
    ord_drift = [_f_1]
    lin_drift = [_f_x, _f_y, _f_z][:dim]
    qu1_drift = [_f_xx, _f_yy, _f_zz][:dim]
    qu2_drift = [] if dim < 2 else [_f_xy]
    qu3_drift = [] if dim < 3 else [_f_yz, _f_xz]
    if drift_type in ["const", "ord", "constant", "ordinary"]:
        return ord_drift
    if drift_type in ["lin", "linear"]:
        return ord_drift + lin_drift
    if drift_type in ["quad", "quadratic"]:
        return ord_drift + lin_drift + qu1_drift + qu2_drift + qu3_drift
    raise ValueError("Drift: unknown drift given: '{}'".format(drift_type))


def _f_1(*pos):
    return np.ones_like(pos[0])


def _f_x(*pos):
    return pos[0]


def _f_y(*pos):
    return pos[1]


def _f_z(*pos):
    return pos[2]


def _f_xx(*pos):
    return pos[0] ** 2


def _f_yy(*pos):
    return pos[1] ** 2


def _f_zz(*pos):
    return pos[2] ** 2


def _f_xy(*pos):
    return pos[0] * pos[1]


def _f_xz(*pos):
    return pos[0] * pos[2]


def _f_yz(*pos):
    return pos[1] * pos[2]
