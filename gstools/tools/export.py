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

__all__ = ["vtk_export_structured", "vtk_export_unstructured", "vtk_export"]


# export routines #############################################################


def vtk_export_structured(path, field, x, y=None, z=None, fieldname="field"):
    """Export a field to vtk structured rectilinear grid file

    Parameters
    ----------
    path : :class:`str`
        Path to the file to be saved. Note that a ".vtr" will be added to the
        name.
    field : :class:`numpy.ndarray`
        Structured field to be saved. As returned by SRF.
    x : :class:`numpy.ndarray`
        grid axis in x-direction
    y : :class:`numpy.ndarray`, optional
        analog to x
    z : :class:`numpy.ndarray`, optional
        analog to x
    fieldname : :class:`str`, optional
        Name of the field in the VTK file. Default: "field"
    """
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
    gridToVTK(path, x, y, z, pointData={fieldname: field})


def vtk_export_unstructured(path, field, x, y=None, z=None, fieldname="field"):
    """Export a field to vtk structured rectilinear grid file

    Parameters
    ----------
    path : :class:`str`
        Path to the file to be saved. Note that a ".vtr" will be added to the
        name.
    field : :class:`numpy.ndarray`
        Unstructured field to be saved. As returned by SRF.
    x : :class:`numpy.ndarray`
        first components of position vectors
    y : :class:`numpy.ndarray`, optional
        analog to x
    z : :class:`numpy.ndarray`, optional
        analog to x
    fieldname : :class:`str`, optional
        Name of the field in the VTK file. Default: "field"
    """
    if y is None:
        y = np.zeros_like(x)
    if z is None:
        z = np.zeros_like(x)
    if len(field) != len(x) or len(field) != len(y) or len(field) != len(z):
        raise ValueError(
            "gstools.vtk_export_unstructured: "
            + "field shape doesn't match the given mesh"
        )
    pointsToVTK(path, x, y, z, data={fieldname: field})


def vtk_export(
    path, field, x, y=None, z=None, fieldname="field", mesh_type="unstructured"
):
    """Export a field to vtk

    Parameters
    ----------
    path : :class:`str`
        Path to the file to be saved. Note that a ".vtr" will be added to the
        name.
    field : :class:`numpy.ndarray`
        Unstructured field to be saved. As returned by SRF.
    x : :class:`numpy.ndarray`
        first components of position vectors
    y : :class:`numpy.ndarray`, optional
        analog to x
    z : :class:`numpy.ndarray`, optional
        analog to x
    fieldname : :class:`str`, optional
        Name of the field in the VTK file. Default: "field"
    mesh_type : :class:`str`
        'structured' / 'unstructured'
    """
    if mesh_type == "structured":
        vtk_export_structured(
            path=path, field=field, x=x, y=y, z=z, fieldname=fieldname
        )
    else:
        vtk_export_unstructured(
            path=path, field=field, x=x, y=y, z=z, fieldname=fieldname
        )
