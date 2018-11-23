# -*- coding: utf-8 -*-
"""
GStools subpackage providing tools for the spatial random field.

.. currentmodule:: gstools.field.tools

The following classes and functions are provided

.. autosummary::
   r3d_x
   r3d_y
   r3d_z
   reshape_input
   reshape_input_axis_from_unstruct
   reshape_input_axis_from_struct
   check_mesh
   make_isotropic
   unrotate_mesh
   reshape_axis_from_struct_to_unstruct
   reshape_field_from_unstruct_to_struct
   vtk_export_structured
   vtk_export_unstructured
"""
from __future__ import print_function, division, absolute_import

import numpy as np
from pyevtk.hl import gridToVTK, pointsToVTK

__all__ = [
    "r3d_x",
    "r3d_y",
    "r3d_z",
    "reshape_input",
    "reshape_input_axis_from_unstruct",
    "reshape_input_axis_from_struct",
    "check_mesh",
    "make_isotropic",
    "unrotate_mesh",
    "reshape_axis_from_struct_to_unstruct",
    "reshape_field_from_unstruct_to_struct",
    "vtk_export_structured",
    "vtk_export_unstructured",
]


# Geometric functions #########################################################


def r3d_x(theta):
    """Rotation matrix about x axis.

    Parameters
    ----------
    theta : :class:`float`
        Rotation angle

    Returns
    -------
        :class:`numpy.ndarray`
            Rotation matrix.
    """
    sin = np.sin(theta)
    cos = np.cos(theta)
    return np.array(((1.0, +0.0, +0.0), (0.0, cos, -sin), (0.0, sin, cos)))


def r3d_y(theta):
    """Rotation matrix about y axis.

    Parameters
    ----------
    theta : :class:`float`
        Rotation angle

    Returns
    -------
        :class:`numpy.ndarray`
            Rotation matrix.
    """
    sin = np.sin(theta)
    cos = np.cos(theta)
    return np.array(((+cos, 0.0, sin), (+0.0, 1.0, +0.0), (-sin, 0.0, cos)))


def r3d_z(theta):
    """Rotation matrix about z axis.

    Parameters
    ----------
    theta : :class:`float`
        Rotation angle

    Returns
    -------
        :class:`numpy.ndarray`
            Rotation matrix.
    """
    sin = np.sin(theta)
    cos = np.cos(theta)
    return np.array(((cos, -sin, 0.0), (sin, +cos, 0.0), (+0.0, +0.0, 1.0)))


def reshape_input(x, y=None, z=None, mesh_type="unstructured"):
    """Reshape given axes, depending on the mesh type."""
    if mesh_type == "unstructured":
        x, y, z = reshape_input_axis_from_unstruct(x, y, z)
    elif mesh_type == "structured":
        x, y, z = reshape_input_axis_from_struct(x, y, z)
    return x, y, z


def reshape_input_axis_from_unstruct(x, y=None, z=None):
    """Reshape given axes for vectorisation on unstructured grid."""
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    x = np.reshape(x, (len(x), 1))
    y = np.reshape(y, (len(y), 1))
    z = np.reshape(z, (len(z), 1))
    return (x, y, z)


def reshape_input_axis_from_struct(x, y=None, z=None):
    """Reshape given axes for vectorisation on unstructured grid."""
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    x = np.reshape(x, (len(x), 1, 1, 1))
    y = np.reshape(y, (1, len(y), 1, 1))
    z = np.reshape(z, (1, 1, len(z), 1))
    return (x, y, z)


# SRF helpers #################################################################


def check_mesh(dim, x, y, z, mesh_type):
    """Do a basic check of the shapes of the input arrays."""
    if dim >= 2:
        if y is None:
            raise ValueError(
                "The y-component is missing for " "{0} dimensions".format(dim)
            )
    if dim == 3:
        if z is None:
            raise ValueError(
                "The z-component is missing for " "{0} dimensions".format(dim)
            )
    if mesh_type == "unstructured":
        if dim >= 2:
            try:
                if len(x) != len(y):
                    raise ValueError(
                        "len(x) = {0} != len(y) = {1} "
                        "for unstructured grids".format(len(x), len(y))
                    )
            except TypeError:
                pass
            if dim == 3:
                try:
                    if len(x) != len(z):
                        raise ValueError(
                            "len(x) = {0} != len(z) = {1} "
                            "for unstructured grids".format(len(x), len(z))
                        )
                except TypeError:
                    pass
    elif mesh_type == "structured":
        pass
    else:
        raise ValueError("Unknown mesh type {0}".format(mesh_type))


def make_isotropic(dim, anis, y, z):
    """Stretch given axes in order to implement anisotropy."""
    if dim == 1:
        return y, z
    if dim == 2:
        return y / anis[0], z
    if dim == 3:
        return y / anis[0], z / anis[1]
    return None


def unrotate_mesh(dim, angles, x, y, z):
    """Rotate axes in order to implement rotation.

    for 3d: yaw, pitch, and roll angles are alpha, beta, and gamma,
    of intrinsic rotation rotation whose Tait-Bryan angles are
    alpha, beta, gamma about axes x, y, z.
    """
    if dim == 1:
        return x, y, z
    if dim == 2:
        # extract 2d rotation matrix
        rot_mat = r3d_z(angles[0])[0:2, 0:2]
        pos_tuple = np.vstack((x, y))
        pos_tuple = np.vsplit(np.dot(rot_mat, pos_tuple), 2)
        x = np.squeeze(pos_tuple[0])
        y = np.squeeze(pos_tuple[1])
        return x, y, z
    if dim == 3:
        alpha = angles[0]
        beta = angles[1]
        gamma = angles[2]
        rot_mat = np.dot(np.dot(r3d_z(alpha), r3d_y(beta)), r3d_x(gamma))
        pos_tuple = np.vstack((x, y, z))
        pos_tuple = np.vsplit(np.dot(rot_mat, pos_tuple), 3)
        x = np.squeeze(pos_tuple[0])
        y = np.squeeze(pos_tuple[1])
        z = np.squeeze(pos_tuple[2])
        return x, y, z
    return None


def reshape_axis_from_struct_to_unstruct(dim, x, y=None, z=None):
    """Reshape given axes from struct to unstruct for rotation."""
    if dim == 1:
        return x, y, z, (len(x),)
    if dim == 2:
        x_u, y_u = np.meshgrid(x, y, indexing="ij")
        len_unstruct = len(x) * len(y)
        x_u = np.reshape(x_u, len_unstruct)
        y_u = np.reshape(y_u, len_unstruct)
        return x_u, y_u, z, (len(x), len(y))
    if dim == 3:
        x_u, y_u, z_u = np.meshgrid(x, y, z, indexing="ij")
        len_unstruct = len(x) * len(y) * len(z)
        x_u = np.reshape(x_u, len_unstruct)
        y_u = np.reshape(y_u, len_unstruct)
        z_u = np.reshape(z_u, len_unstruct)
        return x_u, y_u, z_u, (len(x), len(y), len(z))
    return None


def reshape_field_from_unstruct_to_struct(dim, field, axis_lens):
    """Reshape the rotated field back to struct."""
    if dim == 1:
        return field
    if dim == 2:
        field = np.reshape(field, axis_lens)
        return field
    if dim == 3:
        field = np.reshape(field, axis_lens)
        return field
    return None


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
