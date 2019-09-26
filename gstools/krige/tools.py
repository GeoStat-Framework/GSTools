# -*- coding: utf-8 -*-
"""
GStools subpackage providing tools for Kriging.

.. currentmodule:: gstools.krige.tools

The following classes and functions are provided

.. autosummary::
   set_condition
"""
# pylint: disable=C0103
from __future__ import print_function, division, absolute_import

import numpy as np
from gstools.tools.geometric import pos2xyz, xyz2pos

__all__ = ["set_condition"]


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
    error = False
    if max_dim > 1:
        if not all(
            [len(cond_pos[i]) == len(cond_val) for i in range(max_dim)]
        ):
            error = True
    else:
        if len(cond_pos) != len(cond_val):
            error = True
    if error:
        raise ValueError(
            "Please check your 'cond_pos' and 'cond_val' parameters. "
            + "The shapes do not match."
        )
    cond_val = np.array(cond_val, dtype=np.double).reshape(-1)
    return cond_pos, cond_val
