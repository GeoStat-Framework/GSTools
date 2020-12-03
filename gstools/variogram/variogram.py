# -*- coding: utf-8 -*-
"""
GStools subpackage providing tools for estimating and fitting variograms.

.. currentmodule:: gstools.variogram.variogram

The following functions are provided

.. autosummary::
   vario_estimate
   vario_estimate_axis
"""
# pylint: disable=C0103

import numpy as np

from gstools.tools.geometric import (
    gen_mesh,
    format_struct_pos_shape,
    format_unstruct_pos_shape,
    ang2dir,
)
from gstools.variogram.estimator import (
    unstructured,
    structured,
    ma_structured,
    directional,
)

__all__ = [
    "vario_estimate",
    "vario_estimate_axis",
    "vario_estimate_unstructured",
    "vario_estimate_structured",
]


AXIS = ["x", "y", "z"]
AXIS_DIR = {"x": 0, "y": 1, "z": 2}


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


def _separate_dirs_test(direction, angles_tol):
    """Check if given directions are separated."""
    if direction is None or direction.shape[0] < 2:
        return True
    separate_dirs = True
    for i in range(direction.shape[0] - 1):
        for j in range(i + 1, direction.shape[0]):
            s_prod = np.minimum(np.abs(np.dot(direction[i], direction[j])), 1)
            separate_dirs &= np.arccos(s_prod) >= 2 * angles_tol
    return separate_dirs


def vario_estimate(
    pos,
    field,
    bin_edges,
    sampling_size=None,
    sampling_seed=None,
    estimator="matheron",
    direction=None,
    angles=None,
    angles_tol=np.pi / 8,
    bandwidth=None,
    no_data=np.nan,
    mask=np.ma.nomask,
    mesh_type="unstructured",
    return_counts=False,
):
    r"""
    Estimates the empirical variogram.

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

    By provding `direction` vector[s] or angles, a directional variogram
    can be calculated. If multiple directions are given, a set of variograms
    will be returned.
    Directional bining is controled by a given angle tolerance (`angles_tol`)
    and an optional `bandwidth`, that truncates the width of the search band
    around the given direction[s].

    To reduce the calcuation time, `sampling_size` could be passed to sample
    down the number of field points.

    Parameters
    ----------
    pos : :class:`list`
        the position tuple, containing either the point coordinates (x, y, ...)
        or the axes descriptions (for mesh_type='structured')
    field : :class:`numpy.ndarray` or :class:`list` of :class:`numpy.ndarray`
        The spatially distributed data.
        Can also be of type :class:`numpy.ma.MaskedArray` to use masked values.
        You can pass a list of fields, that will be used simultaneously.
        This could be helpful, when there are multiple realizations at the
        same points, with the same statistical properties.
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
    direction : :class:`list` of :class:`numpy.ndarray`, optional
        directions to evaluate a directional variogram.
        Anglular tolerance is given by `angles_tol`.
        bandwidth to cut off how wide the search for point pairs should be
        is given by `bandwidth`.
        You can provide multiple directions at once to get one variogram
        for each direction.
        For a single direction you can also use the `angles` parameter,
        to provide the direction by its spherical coordianates.
        Default: :any:`None`
    angles : :class:`numpy.ndarray`, optional
        the angles of the main axis to calculate the variogram for in radians
        angle definitions from ISO standard 80000-2:2009
        for 1d this parameter will have no effect at all
        for 2d supply one angle which is
        azimuth :math:`\varphi` (ccw from +x in xy plane)
        for 3d supply two angles which are
        azimuth :math:`\varphi` (ccw from +x in xy plane)
        and inclination :math:`\theta` (cw from +z).
        Can be used instead of direction.
        Default: :any:`None`
    angles_tol : class:`float`, optional
        the tolerance around the variogram angle to count a point as being
        within this direction from another point (the angular tolerance around
        the directional vector given by angles)
        Default: `np.pi/8` = 22.5°
    bandwidth : class:`float`, optional
        bandwidth to cut off the angular tolerance for directional variograms.
        If None is given, only the `angles_tol` parameter will control the
        point selection.
        Default: :any:`None`
    no_data : :class:`float`, optional
        Value to identify missing data in the given field.
        Default: `numpy.nan`
    mask : :class:`numpy.ndarray` of :class:`bool`, optional
        Mask to deselect data in the given field.
        Default: :any:`numpy.ma.nomask`
    mesh_type : :class:`str`, optional
        'structured' / 'unstructured', indicates whether the pos tuple
        describes the axis or the point coordinates.
        Default: `'unstructured'`
    return_counts: :class:`bool`, optional
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
    bin_edges = np.array(bin_edges, ndmin=1, dtype=np.double)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    # allow multiple fields at same positions (ndmin=2: first axis -> field ID)
    # need to convert to ma.array, since list of ma.array is not recognised
    field = np.ma.array(field, ndmin=2, dtype=np.double)
    masked = np.ma.is_masked(field) or np.any(mask)
    # catch special case if everything is masked
    if masked and np.all(mask):
        estimates = np.zeros_like(bin_centres)
        if return_counts:
            return bin_centres, estimates, np.zeros_like(estimates, dtype=int)
        return bin_centres, estimates
    if not masked:
        field = field.filled()
    # check mesh shape
    if mesh_type != "unstructured":
        pos, __, dim = format_struct_pos_shape(
            pos, field.shape, check_stacked_shape=True
        )
        pos = gen_mesh(pos)
    else:
        pos, __, dim = format_unstruct_pos_shape(
            pos, field.shape, check_stacked_shape=True
        )
    # prepare the field
    pnt_cnt = len(pos[0])
    field = field.reshape((-1, pnt_cnt))
    # apply mask if wanted
    if masked:
        # if fields have different masks, take the minimal common mask
        # given mask will be applied in addition
        # selected region is the inverted masked (unmasked values)
        if np.size(mask) > 1:  # not only np.ma.nomask
            select = np.invert(
                np.logical_or(
                    np.reshape(mask, pnt_cnt), np.all(field.mask, axis=0)
                )
            )
        else:
            select = np.invert(np.all(field.mask, axis=0))
        pos = pos[:, select]
        field.fill_value = np.nan  # use no-data val. for remaining masked vals
        field = field[:, select].filled()  # convert to ndarray
        select = mask = None  # free space
    # set no_data values
    if not np.isnan(no_data):
        field[np.isclose(field, float(no_data))] = np.nan
    # set directions
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
        # negative bandwidth to turn it off
        bandwidth = float(bandwidth) if bandwidth is not None else -1.0
        angles_tol = float(angles_tol)
    # prepare sampled variogram
    if sampling_size is not None and sampling_size < pnt_cnt:
        sampled_idx = np.random.RandomState(sampling_seed).choice(
            np.arange(pnt_cnt), sampling_size, replace=False
        )
        field = field[:, sampled_idx]
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
            bandwidth,
            separate_dirs=_separate_dirs_test(direction, angles_tol),
            estimator_type=cython_estimator,
        )
        if dir_no == 1:
            estimates, counts = estimates[0], counts[0]
    if return_counts:
        return bin_centres, estimates, counts
    return bin_centres, estimates


def vario_estimate_axis(
    field, direction="x", estimator="matheron", no_data=np.nan
):
    r"""Estimates the variogram along array axis.

    The indices of the given direction are used for the bins.
    Uniform spacings along the given axis are assumed.

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
    field : :class:`numpy.ndarray` or :class:`numpy.ma.MaskedArray`
        the spatially distributed data (can be masked)
    direction : :class:`str` or :class:`int`
        the axis over which the variogram will be estimated (x, y, z)
        or (0, 1, 2, ...)
    estimator : :class:`str`, optional
        the estimator function, possible choices:

            * "matheron": the standard method of moments of Matheron
            * "cressie": an estimator more robust to outliers

        Default: "matheron"

    no_data : :class:`float`, optional
        Value to identify missing data in the given field.
        Default: `numpy.nan`

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
    missing_mask = (
        np.isnan(field) if np.isnan(no_data) else np.isclose(field, no_data)
    )
    missing = np.any(missing_mask)
    masked = np.ma.is_masked(field) or missing
    if masked:
        field = np.ma.array(field, ndmin=1, dtype=np.double)
        if missing:
            field.mask = np.logical_or(field.mask, missing_mask)
        mask = np.array(np.ma.getmaskarray(field), dtype=np.int32)
    else:
        field = np.array(field, ndmin=1, dtype=np.double)
        missing_mask = None  # free space

    axis_to_swap = AXIS_DIR[direction] if direction in AXIS else int(direction)
    # desired axis first, convert to 2D array afterwards
    field = field.swapaxes(0, axis_to_swap)
    field = field.reshape((field.shape[0], -1))
    if masked:
        mask = mask.swapaxes(0, axis_to_swap)
        mask = mask.reshape((mask.shape[0], -1))

    cython_estimator = _set_estimator(estimator)

    if masked:
        return ma_structured(field, mask, cython_estimator)
    return structured(field, cython_estimator)


# for backward compatibility
vario_estimate_unstructured = vario_estimate
vario_estimate_structured = vario_estimate_axis
