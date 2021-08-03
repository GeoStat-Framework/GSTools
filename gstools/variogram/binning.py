# -*- coding: utf-8 -*-
"""
GStools subpackage providing binning routines.

.. currentmodule:: gstools.variogram.binning

The following functions are provided

.. autosummary::
   standard_bins
"""
import numpy as np

from gstools.tools.geometric import (
    generate_grid,
    format_struct_pos_dim,
    latlon2pos,
    chordal_to_great_circle,
)

__all__ = ["standard_bins"]


def _sturges(pnt_cnt):
    return int(np.ceil(2 * np.log2(pnt_cnt) + 1))


def standard_bins(
    pos=None,
    dim=2,
    latlon=False,
    mesh_type="unstructured",
    bin_no=None,
    max_dist=None,
):
    r"""
    Get standard binning.

    Parameters
    ----------
    pos : :class:`list`, optional
        the position tuple, containing either the point coordinates (x, y, ...)
        or the axes descriptions (for mesh_type='structured')
    dim : :class:`int`, optional
        Field dimension.
    latlon : :class:`bool`, optional
        Whether the data is representing 2D fields on earths surface described
        by latitude and longitude. When using this, the estimator will
        use great-circle distance for variogram estimation.
        Note, that only an isotropic variogram can be estimated and a
        ValueError will be raised, if a direction was specified.
        Bin edges need to be given in radians in this case.
        Default: False
    mesh_type : :class:`str`, optional
        'structured' / 'unstructured', indicates whether the pos tuple
        describes the axis or the point coordinates.
        Default: `'unstructured'`
    bin_no: :class:`int`, optional
        number of bins to create. If None is given, will be determined by
        Sturges' rule from the number of points.
        Default: None
    max_dist: :class:`float`, optional
        Cut of length for the bins. If None is given, it will be set to one
        third of the box-diameter from the given points.
        Default: None

    Returns
    -------
    :class:`numpy.ndarray`
        The generated bin edges.

    Notes
    -----
    Internally uses double precision and also returns doubles.
    """
    dim = 2 if latlon else int(dim)
    if bin_no is None or max_dist is None:
        if pos is None:
            raise ValueError("standard_bins: no pos tuple given.")
        if mesh_type != "unstructured":
            pos = generate_grid(format_struct_pos_dim(pos, dim)[0])
        else:
            pos = np.asarray(pos, dtype=np.double).reshape(dim, -1)
        pos = latlon2pos(pos) if latlon else pos
        pnt_cnt = len(pos[0])
        box = []
        for axis in pos:
            box.append([np.min(axis), np.max(axis)])
        box = np.asarray(box)
        diam = np.linalg.norm(box[:, 0] - box[:, 1])
        # convert diameter to great-circle distance if using latlon
        diam = chordal_to_great_circle(diam) if latlon else diam
        bin_no = _sturges(pnt_cnt) if bin_no is None else int(bin_no)
        max_dist = diam / 3 if max_dist is None else float(max_dist)
    return np.linspace(0, max_dist, num=bin_no + 1, dtype=np.double)
