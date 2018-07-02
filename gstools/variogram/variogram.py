# -*- coding: utf-8 -*-
"""
GStools subpackage providing tools for estimating and fitting variograms.

.. currentmodule:: gstools.variogram.variogram

The following functions are provided

.. autosummary::
   estimate_unstructured
   estimate_structured
"""
from __future__ import division, absolute_import, print_function

import numpy as np
from gstools.variogram.estimator import (
    unstructured,
    structured_3d,
    structured_2d,
    structured_1d,
    ma_structured_3d,
    ma_structured_2d,
    ma_structured_1d,
)

__all__ = [
    "estimate_unstructured",
    "estimate_structured",
]


def estimate_unstructured(
        field,
        bin_edges,
        x,
        y=None,
        z=None,
        sampling_size=None,
):
    r"""
    Estimates the variogram of the unstructured input data.

    The algorithm calculates following equation:

    .. math::
       \gamma(r_k) = \frac{1}{2 N} \sum_{i=1}^N (z(\mathbf x_i) -
       z(\mathbf x_i'))^2, \; \mathrm{ with}

       r_k \leq \| \mathbf x_i - \mathbf x_i' \| < r_{k+1}

    Parameters
    ----------
        field : :class:`numpy.ndarray`
            the spatially distributed data
        bin_edges : :class:`numpy.ndarray`
            the bins on which the variogram will be calculated
        x : :class:`numpy.ndarray`
            first components of position vectors
        y : :class:`numpy.ndarray`, optional
            analog to x
        z : :class:`numpy.ndarray`, optional
            analog to x
        sampling_size : :class:`int`
            for large input data, this method can take a long
            time to compute the variogram, therefore this argument specifies
            the number of data points to sample randomly
    Returns
    -------
        :class:`tuple` of :class:`numpy.ndarray`
            the estimated variogram and the bin centers
    """

    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.

    if sampling_size is not None and sampling_size < len(field):
        sampled_idx = np.random.choice(np.arange(len(field)), sampling_size,
                                       replace=False)
        field = field[sampled_idx]
        x = x[sampled_idx]
        if y is not None:
            y = y[sampled_idx]
        if z is not None:
            z = z[sampled_idx]

    return unstructured(field, bin_edges, x, y, z), bin_centres


def estimate_structured(
        pos,
        field,
        direction='x',
):
    r"""Estimates the variogram of the input data on a regular grid.

    The axis of the given direction is used for the bins.
    The algorithm calculates following equation:

    .. math::
       \gamma(r_k) = \frac{1}{2 N} \sum_{i=1}^N (z(\mathbf x_i) -
       z(\mathbf x_i'))^2, \; \mathrm{ with}

       r_k \leq \| \mathbf x_i - \mathbf x_i' \| < r_{k+1}

    Parameters
    ----------
        pos : :class:`tuple`
            a tuple of :class:`numpy.ndarray` containing the axes
        field : :class:`numpy.ndarray`
            the spatially distributed data
        direction : :class:`str`
            the axis over which the variogram will be estimated (x, y, z)

    Returns
    -------
        :class:`numpy.ndarray`
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
