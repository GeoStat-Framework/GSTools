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

from gstools.tools.geometric import pos2xyz, xyz2pos, ang2dir
from gstools.variogram.estimator import (
    unstructured,
    structured,
    ma_structured,
    directional,
)

__all__ = ["vario_estimate_unstructured", "vario_estimate_structured"]


def _set_estimator(estimator):
    """Translate the verbose Python estimator identifier to single char."""
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
    direction=None,
    angles=None,
    angles_tol=np.pi / 8,
    bandwith=None,
    sampling_size=None,
    sampling_seed=None,
    estimator="matheron",
    return_counts=False,
):
    r"""
    Estimates the variogram on a unstructured grid.

    The algorithm calculates following equation:

    .. math::
       \gamma(r_k) = \frac{1}{2 N(r_k)} \sum_{i=1}^{N(r_k)} (z(\mathbf x_i) -
       z(\mathbf x_i'))^2 \; ,

    with :math:`r_k \leq \| \mathbf x_i - \mathbf x_i' \| < r_{k+1}`
    being the bins.

    Or if the estimator "cressie" was chosen:

    .. math::
       \gamma(r_k) = \frac{\frac{1}{2}\left(\frac{1}{N(r_k)}\sum_{i=1}^{N(r_k)}
       \left|z(\mathbf x_i) - z(\mathbf x_i')\right|^{0.5}\right)^4}
       {0.457 + 0.494 / N(r_k) + 0.045 / N^2(r_k)} \; ,

    with :math:`r_k \leq \| \mathbf x_i - \mathbf x_i' \| < r_{k+1}`
    being the bins.
    The Cressie estimator is more robust to outliers.

    Parameters
    ----------
    pos : :class:`list`
        the position tuple, containing main direction and transversal
        directions
    field : :class:`numpy.ndarray`
        the spatially distributed data
    bin_edges : :class:`numpy.ndarray`
        the bins on which the variogram will be calculated
    direction : :class:`list` of :class:`numpy.ndarray`, optional
        directions to evaluate a directional variogram.
        Anglular tolerance is given by `angles_tol`.
        Bandwith to cut off how wide the search for point pairs should be
        is given by `bandwith`.
        You can provide multiple directions at once to get one variogram
        for each direction.
        For a single direction you can also use the `angles` parameter,
        to provide the direction by its spherical coordianates.
        Default: :any:`None`
    angles : :class:`numpy.ndarray`, optional
        the angles of the main axis to calculate the variogram for in radians
        angle definitions from ISO standard 80000-2:2009
        for 1d this parameter will have no effect at all
        for 2d supply one angle which is azimuth φ (ccw from +x in xy plane)
        for 3d supply two angles which are azimuth φ (ccw from +x in xy plane)
        and inclination θ (cw from +z).
        Can be used instead of direction for a single main axis.
        Default: :any:`None`
    angles_tol : class:`float`, optional
        the tolerance around the variogram angle to count a point as being
        within this direction from another point (the angular tolerance around
        the directional vector given by angles)
        Default: `np.pi/8` = 22.5°
    bandwith : class:`float`, optional
        Bandwith to cut off the angular tolerance for directional variograms.
        If None is given, only the `angles_tol` parameter will control the
        point selection.
        Default: :any:`None`
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
    return_counts: class:`bool`, optional
        if set to true, this function will also return the number of data
        points found at each lag distance as a third return value
        Default: False

    Returns
    -------
    :class:`tuple` of :class:`numpy.ndarray`
        1. the bin centers
        2. the estimated variogram values at bin centers
        3. (optional) the number of points found at each bin center
           (see argument return_counts)

    Notes
    -----
    Internally uses double precision and also returns doubles.
    """
    # TODO check_mesh
    field = np.array(field, ndmin=1, dtype=np.double)
    bin_edges = np.array(bin_edges, ndmin=1, dtype=np.double)
    x, y, z, dim = pos2xyz(pos, calc_dim=True, dtype=np.double)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    # initialize number of directions
    dir_no = 0
    if direction is not None and dim > 1:
        direction = np.array(direction, ndmin=2, dtype=np.double)
        if len(direction.shape) > 2:
            raise ValueError("Can't interpret directions {}".format(direction))
        if direction.shape[1] != dim:
            raise ValueError("Can't interpret directions {}".format(direction))
        dir_no = direction.shape[0]
    # convert given angles to direction vector
    if angles is not None and direction is None and dim > 1:
        direction = ang2dir(angles=angles, dtype=np.double, dim=dim)
        dir_no = direction.shape[0]
    # prepare directional variogram
    if dir_no > 0:
        norms = np.linalg.norm(direction, axis=1)
        if np.any(np.isclose(norms, 0)):
            raise ValueError("Zero length direction {}".format(direction))
        # only unit-vectors for directions
        direction = np.divide(direction, norms[:, np.newaxis])
        # negative bandwith to turn it off
        bandwith = float(bandwith) if bandwith is not None else -1.0
        angles_tol = float(angles_tol)
    # prepare positions
    pos = np.array(xyz2pos(x, y, z, dtype=np.double, max_dim=dim))
    # prepare sampled variogram
    if sampling_size is not None and sampling_size < len(field):
        sampled_idx = np.random.RandomState(sampling_seed).choice(
            np.arange(len(field)), sampling_size, replace=False
        )
        field = field[sampled_idx]
        pos = pos[:, sampled_idx]
    # select variogram estimator
    cython_estimator = _set_estimator(estimator)
    # run
    if dir_no == 0:
        estimates, counts = unstructured(
            dim,
            field,
            bin_edges,
            pos,
            estimator_type=cython_estimator,
        )
    else:
        estimates, counts = directional(
            dim,
            field,
            bin_edges,
            pos,
            direction,
            angles_tol,
            bandwith,
            estimator_type=cython_estimator,
        )
        if dir_no == 1:
            estimates, counts = estimates[0], counts[0]
    if return_counts:
        return bin_centres, estimates, counts
    return bin_centres, estimates


def vario_estimate_structured(field, direction="x", estimator="matheron"):
    r"""Estimates the variogram on a regular grid.

    The indices of the given direction are used for the bins.
    The algorithm calculates following equation:

    .. math::
       \gamma(r_k) = \frac{1}{2 N(r_k)} \sum_{i=1}^{N(r_k)} (z(\mathbf x_i) -
       z(\mathbf x_i'))^2 \; ,

    with :math:`r_k \leq \| \mathbf x_i - \mathbf x_i' \| < r_{k+1}`
    being the bins.

    Or if the estimator "cressie" was chosen:

    .. math::
       \gamma(r_k) = \frac{\frac{1}{2}\left(\frac{1}{N(r_k)}\sum_{i=1}^{N(r_k)}
       \left|z(\mathbf x_i) - z(\mathbf x_i')\right|^{0.5}\right)^4}
       {0.457 + 0.494 / N(r_k) + 0.045 / N^2(r_k)} \; ,

    with :math:`r_k \leq \| \mathbf x_i - \mathbf x_i' \| < r_{k+1}`
    being the bins.
    The Cressie estimator is more robust to outliers.

    Parameters
    ----------
    field : :class:`numpy.ndarray`
        the spatially distributed data
    direction : :class:`str`
        the axis over which the variogram will be estimated (x, y, z)
    estimator : :class:`str`, optional
        the estimator function, possible choices:

            * "matheron": the standard method of moments of Matheron
            * "cressie": an estimator more robust to outliers

        Default: "matheron"

    Returns
    -------
    :class:`numpy.ndarray`
        the estimated variogram along the given direction.

    Warnings
    --------
    It is assumed that the field is defined on an equidistant Cartesian grid.

    Notes
    -----
    Internally uses double precision and also returns doubles.
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
    for __ in range(3 - len(field.shape)):
        field = field[..., np.newaxis]
    if masked:
        for __ in range(3 - len(mask.shape)):
            mask = mask[..., np.newaxis]

    if mask is None:
        gamma = structured(field, cython_estimator)
    else:
        gamma = ma_structured(field, mask, cython_estimator)
    return gamma
