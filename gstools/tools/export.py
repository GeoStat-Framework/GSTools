# -*- coding: utf-8 -*-
"""
GStools subpackage providing export routines.

.. currentmodule:: gstools.tools.export

The following functions are provided

.. autosummary::
   vtk_export_structured
   vtk_export_unstructured
   vtk_export
"""
# pylint: disable=C0103, E1101
from __future__ import print_function, division, absolute_import

import numpy as np
from pyevtk.hl import gridToVTK, pointsToVTK
from gstools.tools.geometric import pos2xyz

__all__ = ["vtk_export_structured", "vtk_export_unstructured", "vtk_export"]


# export routines #############################################################


def vtk_export_structured(filename, pos, fields):  # pragma: no cover
    """Export a field to vtk structured rectilinear grid file.

    Parameters
    ----------
    filename : :class:`str`
        Filename of the file to be saved, including the path. Note that an
        ending (.vtr) will be added to the name.
    pos : :class:`list`
        the position tuple, containing main direction and transversal
        directions
    fields : :class:`dict` or :class:`numpy.ndarray`
        Structured fields to be saved.
        Either a single numpy array as returned by SRF,
        or a dictionary of fields with theirs names as keys.
    """
    if not isinstance(fields, dict):
        fields = {"field": fields}
    x, y, z = pos2xyz(pos)
    if y is None:
        y = np.array([0])
    if z is None:
        z = np.array([0])
    # need fortran order in VTK
    for field in fields:
        fields[field] = fields[field].reshape(-1, order="F")
        if len(fields[field]) != len(x) * len(y) * len(z):
            raise ValueError(
                "gstools.vtk_export_structured: "
                + "field shape doesn't match the given mesh"
            )
    gridToVTK(filename, x, y, z, pointData=fields)


def vtk_export_unstructured(filename, pos, fields):  # pragma: no cover
    """Export a field to vtk structured rectilinear grid file.

    Parameters
    ----------
    filename : :class:`str`
        Filename of the file to be saved, including the path. Note that an
        ending (.vtu) will be added to the name.
    pos : :class:`list`
        the position tuple, containing main direction and transversal
        directions
    fields : :class:`dict` or :class:`numpy.ndarray`
        Unstructured fields to be saved.
        Either a single numpy array as returned by SRF,
        or a dictionary of fields with theirs names as keys.
    """
    if not isinstance(fields, dict):
        fields = {"field": fields}
    x, y, z = pos2xyz(pos)
    if y is None:
        y = np.zeros_like(x)
    if z is None:
        z = np.zeros_like(x)
    for field in fields:
        fields[field] = fields[field].reshape(-1)
        if (
            len(fields[field]) != len(x)
            or len(fields[field]) != len(y)
            or len(fields[field]) != len(z)
        ):
            raise ValueError(
                "gstools.vtk_export_unstructured: "
                + "field shape doesn't match the given mesh"
            )
    pointsToVTK(filename, x, y, z, data=fields)


def vtk_export(
    filename, pos, fields, mesh_type="unstructured"
):  # pragma: no cover
    """Export a field to vtk.

    Parameters
    ----------
    filename : :class:`str`
        Filename of the file to be saved, including the path. Note that an
        ending (.vtr or .vtu) will be added to the name.
    pos : :class:`list`
        the position tuple, containing main direction and transversal
        directions
    fields : :class:`dict` or :class:`numpy.ndarray`
        [Un]structured fields to be saved.
        Either a single numpy array as returned by SRF,
        or a dictionary of fields with theirs names as keys.
    mesh_type : :class:`str`, optional
        'structured' / 'unstructured'. Default: structured
    """
    if mesh_type == "structured":
        vtk_export_structured(filename=filename, pos=pos, fields=fields)
    else:
        vtk_export_unstructured(filename=filename, pos=pos, fields=fields)
