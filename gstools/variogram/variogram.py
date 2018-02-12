#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A collection of tools for estimating and fitting variograms.
"""
from __future__ import division, absolute_import, print_function

import numpy as np

from gstools.field import RNG
from gstools.variogram.estimator import (unstructured, structured_3d,
                                         structured_2d, structured_1d,
                                         ma_structured_3d, ma_structured_2d,
                                         ma_structured_1d)


def estimate_unstructured(field, bins, x, y=None, z=None):
    """Estimates the variogram of the unstructured input data.

    The algorithm calculates following equation:

    .. math::
        \\gamma(r_k) = \\frac{1}{2 N} \\sum_{i=1}^N (z(\\mathbf x_i) -
        z(\\mathbf x_i'))^2, \\; \mathrm{ with}

        r_k \\leq \\| \\mathbf x_i - \\mathbf x_i' \\| < r_{k+1}


    Args:
        f (ndarray): the spatially distributed data
        bins (ndarray): the bins on which the variogram will be calculated
        x (ndarray): first components of position vectors
        y (ndarray, opt.): analog to x
        z (ndarray, opt.): analog to x
    Returns:
        the estimated variogram
    """
    return unstructured(field, bins, x, y, z)

def estimate_structured(pos, field, direction='x'):
    """Estimates the variogram of the input data on a regular grid.

    The axis of the given direction is used for the bins.
    The algorithm calculates following equation:

    .. math::
        \\gamma(r_k) = \\frac{1}{2 N} \\sum_{i=1}^N (z(\\mathbf x_i) -
        z(\\mathbf x_i'))^2, \\; \mathrm{ with}

        r_k \\leq \\| \\mathbf x_i - \\mathbf x_i' \\| < r_{k+1}

    Args:
        pos (tuple): a tuple of ndarrays containing the axes
        field (ndarray): the spatially distributed data
        direction (string): the axis over which the variogram
                            will be estimated (x, y, z)
    Returns:
        the estimated variogram along the given direction.
    """
    shape = field.shape

    if direction == 'x':
        pass
    elif direction == 'y':
        field = field.swapaxes(0, 1)
    elif direction == 'z':
        field = field.swapaxes(0, 2)
    else:
        raise ValueError('Unknown direction {0}'.format(direction))

    try:
        mask = np.array(field.mask, dtype=np.int32)
    except AttributeError:
        mask = None

    if len(shape) == 3:
        if mask is None:
            gamma = structured_3d(pos[0], pos[1], pos[2], field)
        else:
            gamma = ma_structured_3d(pos[0], pos[1], pos[2], field, mask)
    elif len(shape) == 2:
        if mask is None:
            gamma = structured_2d(pos[0], pos[1], field)
        else:
            gamma = ma_structured_2d(pos[0], pos[1], field, mask)
    else:
        if mask is None:
            gamma = structured_1d(pos, field)
        else:
            gamma = ma_structured_1d(pos, field, mask)
    return gamma
