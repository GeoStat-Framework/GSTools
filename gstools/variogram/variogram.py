# -*- coding: utf-8 -*-
"""
GStools subpackage providing tools for estimating and fitting variograms.

.. currentmodule:: gstools.variogram.variogram

The following functions are provided

.. autosummary::
   vario_estimate_unstructured
   vario_estimate_structured
"""
# pylint: disable=C0103
from __future__ import division, absolute_import, print_function

import numpy as np

from gstools.tools.geometric import pos2xyz
from gstools.variogram.estimator import (
    unstructured,
    structured_3d,
    structured_2d,
    structured_1d,
    ma_structured_3d,
    ma_structured_2d,
    ma_structured_1d,
)

__all__ = ["vario_estimate_unstructured", "vario_estimate_structured"]


def vario_estimate_unstructured(
    pos, field, bin_edges, sampling_size=None, sampling_seed=None
):
    r"""
    Estimates the variogram on a unstructured grid.

    The algorithm calculates following equation:

    .. math::
       \gamma(r_k) = \frac{1}{2 N} \sum_{i=1}^N (z(\mathbf x_i) -
       z(\mathbf x_i'))^2, \; \mathrm{ with}

       r_k \leq \| \mathbf x_i - \mathbf x_i' \| < r_{k+1}

    Notes
    -----
    Internally uses double precision and also returns doubles.

    Parameters
    ----------
    pos : :class:`list`
        the position tuple, containing main direction and transversal
        directions
    field : :class:`numpy.ndarray`
        the spatially distributed data
    bin_edges : :class:`numpy.ndarray`
        the bins on which the variogram will be calculated
    sampling_size : :class:`int` or :any:`None`, optional
        for large input data, this method can take a long
        time to compute the variogram, therefore this argument specifies
        the number of data points to sample randomly
        Default: :any:`None`
    sampling_seed : :class:`int` or :any:`None`, optional
        seed for samples if sampling_size is given.
        Default: :any:`None`

    Returns
    -------
    :class:`tuple` of :class:`numpy.ndarray`
        the estimated variogram and the bin centers
    """
    # TODO check_mesh
    field = np.array(field, ndmin=1, dtype=np.double)
    bin_edges = np.array(bin_edges, ndmin=1, dtype=np.double)
    x, y, z, dim = pos2xyz(pos, calc_dim=True, dtype=np.double)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    if sampling_size is not None and sampling_size < len(field):
        sampled_idx = np.random.RandomState(sampling_seed).choice(
            np.arange(len(field)), sampling_size, replace=False
        )
        field = field[sampled_idx]
        x = x[sampled_idx]
        if dim > 1:
            y = y[sampled_idx]
        if dim > 2:
            z = z[sampled_idx]

    return bin_centres, unstructured(field, bin_edges, x, y, z)


def vario_estimate_structured(field, direction="x"):
    r"""Estimates the variogram on a regular grid.

    The indices of the given direction are used for the bins.
    The algorithm calculates following equation:

    .. math::
       \gamma(r_k) = \frac{1}{2 N} \sum_{i=1}^N (z(\mathbf x_i) -
       z(\mathbf x_i'))^2, \; \mathrm{ with}

       r_k \leq \| \mathbf x_i - \mathbf x_i' \| < r_{k+1}

    Warnings
    --------
    It is assumed that the field is defined on an equidistant Cartesian grid.

    Notes
    -----
    Internally uses double precision and also returns doubles.

    Parameters
    ----------
    field : :class:`numpy.ndarray`
        the spatially distributed data
    direction : :class:`str`
        the axis over which the variogram will be estimated (x, y, z)

    Returns
    -------
    :class:`numpy.ndarray`
        the estimated variogram along the given direction.
    """
    try:
        mask = np.array(field.mask, dtype=np.int32)
        field = np.ma.array(field, ndmin=1, dtype=np.double)
        masked = True
    except AttributeError:
        mask = None
        field = np.array(field, ndmin=1, dtype=np.double)
        masked = False
    shape = field.shape

    if direction == "x":
        axis_to_swap = 0
    elif direction == "y":
        axis_to_swap = 1
    elif direction == "z":
        axis_to_swap = 2
    else:
        raise ValueError("Unknown direction {0}".format(direction))

    field = field.swapaxes(0, axis_to_swap)
    if masked:
        mask = mask.swapaxes(0, axis_to_swap)

    if len(shape) == 3:
        if mask is None:
            gamma = structured_3d(field)
        else:
            gamma = ma_structured_3d(field, mask)
    elif len(shape) == 2:
        if mask is None:
            gamma = structured_2d(field)
        else:
            gamma = ma_structured_2d(field, mask)
    else:
        if mask is None:
            gamma = structured_1d(np.array(field, ndmin=1, dtype=np.double))
        else:
            gamma = ma_structured_1d(field, mask)
    return gamma
