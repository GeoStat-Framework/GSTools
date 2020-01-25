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

import numpy as np

from gstools.tools.geometric import pos2xyz
from gstools.variogram.estimator import unstructured, structured, ma_structured

__all__ = ["vario_estimate_unstructured", "vario_estimate_structured"]


def _set_estimator(estimator):
    """Translate the verbose Python estimator identifier to single char"""
    if estimator.lower() == "matheron":
        cython_estimator = "m"
    elif estimator.lower() == "cressie":
        cython_estimator = "c"
    else:
        raise ValueError(
            "Unknown variogram estimator function " + str(estimator)
        )
    return cython_estimator


def vario_estimate_unstructured(
    pos,
    field,
    bin_edges,
    sampling_size=None,
    sampling_seed=None,
    estimator="matheron",
):
    r"""
    Estimates the variogram on a unstructured grid.

    The algorithm calculates following equation:

    .. math::
       \gamma(r_k) = \frac{1}{2 N(r_k)} \sum_{i=1}^{N(r_k)} (z(\mathbf x_i) -
       z(\mathbf x_i'))^2 \; ,

    with :math:`r_k \leq \| \mathbf x_i - \mathbf x_i' \| < r_{k+1}` being the bins.

    Or if the estimator "cressie" was chosen:

    .. math::
       \gamma(r_k) = \frac{\left(\frac{1}{N(r_k)} \sum_{i=1}^{N(r_k)}
       \left|z(\mathbf x_i) - z(\mathbf x_i')\right|^{0.5}\right)^4}
       {0.457 + 0.494 / N(r_k) + 0.045 / N^2(r_k)} \; ,

    with :math:`r_k \leq \| \mathbf x_i - \mathbf x_i' \| < r_{k+1}` being the bins.
    The Cressie estimator is more robust to outliers.

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
    estimator : :class:`str`, optional
        the estimator function, possible choices:

            * "matheron": the standard method of moments of Matheron
            * "cressie": an estimator more robust to outliers

        Default: "matheron"

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

    cython_estimator = _set_estimator(estimator)

    return (
        bin_centres,
        unstructured(
            field, bin_edges, x, y, z, estimator_type=cython_estimator
        ),
    )


def vario_estimate_structured(field, direction="x", estimator="matheron"):
    r"""Estimates the variogram on a regular grid.

    The indices of the given direction are used for the bins.
    The algorithm calculates following equation:

    .. math::
       \gamma(r_k) = \frac{1}{2 N(r_k)} \sum_{i=1}^{N(r_k)} (z(\mathbf x_i) -
       z(\mathbf x_i'))^2 \; ,

    with :math:`r_k \leq \| \mathbf x_i - \mathbf x_i' \| < r_{k+1}` being the bins.

    Or if the estimator "cressie" was chosen:

    .. math::
       \gamma(r_k) = \frac{\left(\frac{1}{N(r_k)} \sum_{i=1}^{N(r_k)}
       \left|z(\mathbf x_i) - z(\mathbf x_i')\right|^{0.5}\right)^4}
       {0.457 + 0.494 / N(r_k) + 0.045 / N^2(r_k)} \; ,

    with :math:`r_k \leq \| \mathbf x_i - \mathbf x_i' \| < r_{k+1}` being the bins.
    The Cressie estimator is more robust to outliers.

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
    estimator : :class:`str`, optional
        the estimator function, possible choices:

            * "mathoron": the standard method of moments of Matheron
            * "cressie": an estimator more robust to outliers

        Default: "matheron"

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

    cython_estimator = _set_estimator(estimator)

    # fill up the field with empty dimensions up to a number of 3
    for i in range(3 - len(field.shape)):
        field = field[..., np.newaxis]
    if masked:
        for i in range(3 - len(mask.shape)):
            mask = mask[..., np.newaxis]

    if mask is None:
        gamma = structured(field, cython_estimator)
    else:
        gamma = ma_structured(field, mask, cython_estimator)
    return gamma
