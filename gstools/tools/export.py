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


def vtk_export_structured(filename, pos, field, fieldname="field"):
    """Export a field to vtk structured rectilinear grid file

    Parameters
    ----------
    filename : :class:`str`
        Filename of the file to be saved, including the path. Note that an
        ending (\*.vtr or \*.vtu) will be added to the name.
    pos : :class:`list`
        the position tuple, containing main direction and transversal
        directions
    field : :class:`numpy.ndarray`
        Structured field to be saved. As returned by SRF.
    fieldname : :class:`str`, optional
        Name of the field in the VTK file. Default: "field"
    """
    x, y, z = pos2xyz(pos)
    if y is None:
        y = np.array([0])
    if z is None:
        z = np.array([0])
    # need fortran order in VTK
    field = field.reshape(-1, order="F")
    if len(field) != len(x) * len(y) * len(z):
        raise ValueError(
            "gstools.vtk_export_structured: "
            + "field shape doesn't match the given mesh"
        )
    gridToVTK(filename, x, y, z, pointData={fieldname: field})


def vtk_export_unstructured(filename, pos, field, fieldname="field"):
    """Export a field to vtk structured rectilinear grid file

    Parameters
    ----------
    filename : :class:`str`
        Filename of the file to be saved, including the path. Note that an
        ending (\*.vtr or \*.vtu) will be added to the name.
    pos : :class:`list`
        the position tuple, containing main direction and transversal
        directions
    field : :class:`numpy.ndarray`
        Unstructured field to be saved. As returned by SRF.
    fieldname : :class:`str`, optional
        Name of the field in the VTK file. Default: "field"
    """
    x, y, z = pos2xyz(pos)
    if y is None:
        y = np.zeros_like(x)
    if z is None:
        z = np.zeros_like(x)
    field = np.array(field).reshape(-1)
    if len(field) != len(x) or len(field) != len(y) or len(field) != len(z):
        raise ValueError(
            "gstools.vtk_export_unstructured: "
            + "field shape doesn't match the given mesh"
        )
    pointsToVTK(filename, x, y, z, data={fieldname: field})


def vtk_export(filename, pos, field, fieldname="field", mesh_type="unstructured"):
    """Export a field to vtk

    Parameters
    ----------
    filename : :class:`str`
        Filename of the file to be saved, including the path. Note that an
        ending (\*.vtr or \*.vtu) will be added to the name.
    pos : :class:`list`
        the position tuple, containing main direction and transversal
        directions
    field : :class:`numpy.ndarray`
        Unstructured field to be saved. As returned by SRF.
    fieldname : :class:`str`, optional
        Name of the field in the VTK file. Default: "field"
    mesh_type : :class:`str`, optional
        'structured' / 'unstructured'. Default: structured
    """
    if mesh_type == "structured":
        vtk_export_structured(
            filename=filename, pos=pos, field=field, fieldname=fieldname
        )
    else:
        vtk_export_unstructured(
            filename=filename, pos=pos, field=field, fieldname=fieldname
        )
