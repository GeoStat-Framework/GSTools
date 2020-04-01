# -*- coding: utf-8 -*-
"""
GStools subpackage providing tools for the spatial random field.

.. currentmodule:: gstools.field.tools

The following classes and functions are provided

.. autosummary::
   reshape_input
   reshape_input_axis_from_unstruct
   reshape_input_axis_from_struct
   check_mesh
   make_isotropic
   make_anisotropic
   unrotate_mesh
   rotate_mesh
   reshape_axis_from_struct_to_unstruct
   reshape_field_from_unstruct_to_struct
"""
# pylint: disable=C0103

import numpy as np
from gstools.tools.geometric import r3d_x, r3d_y, r3d_z

__all__ = [
    "reshape_input",
    "reshape_input_axis_from_unstruct",
    "reshape_input_axis_from_struct",
    "check_mesh",
    "make_isotropic",
    "make_anisotropic",
    "unrotate_mesh",
    "rotate_mesh",
    "reshape_axis_from_struct_to_unstruct",
    "reshape_field_from_unstruct_to_struct",
]


# Geometric functions #########################################################


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


def make_anisotropic(dim, anis, y, z):
    """Re-stretch given axes."""
    if dim == 1:
        return y, z
    if dim == 2:
        return y * anis[0], z
    if dim == 3:
        return y * anis[0], z * anis[1]
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
        rot_mat = r3d_z(-angles[0])[0:2, 0:2]
        pos_tuple = np.vstack((x, y))
        pos_tuple = np.vsplit(np.dot(rot_mat, pos_tuple), 2)
        x = pos_tuple[0].reshape(np.shape(x))
        y = pos_tuple[1].reshape(np.shape(y))
        return x, y, z
    if dim == 3:
        alpha = -angles[0]
        beta = -angles[1]
        gamma = -angles[2]
        rot_mat = np.dot(np.dot(r3d_z(alpha), r3d_y(beta)), r3d_x(gamma))
        pos_tuple = np.vstack((x, y, z))
        pos_tuple = np.vsplit(np.dot(rot_mat, pos_tuple), 3)
        x = pos_tuple[0].reshape(np.shape(x))
        y = pos_tuple[1].reshape(np.shape(y))
        z = pos_tuple[2].reshape(np.shape(z))
        return x, y, z
    return None


def rotate_mesh(dim, angles, x, y, z):
    """Rotate axes.

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
        x = pos_tuple[0].reshape(np.shape(x))
        y = pos_tuple[1].reshape(np.shape(y))
        return x, y, z
    if dim == 3:
        alpha = angles[0]
        beta = angles[1]
        gamma = angles[2]
        rot_mat = np.dot(np.dot(r3d_x(gamma), r3d_y(beta)), r3d_z(alpha))
        pos_tuple = np.vstack((x, y, z))
        pos_tuple = np.vsplit(np.dot(rot_mat, pos_tuple), 3)
        x = pos_tuple[0].reshape(np.shape(x))
        y = pos_tuple[1].reshape(np.shape(y))
        z = pos_tuple[2].reshape(np.shape(z))
        return x, y, z
    return None


def reshape_axis_from_struct_to_unstruct(
    dim, x, y=None, z=None, indexing="ij"
):
    """Reshape given axes from struct to unstruct for rotation."""
    if dim == 1:
        return x, y, z, (len(x),)
    if dim == 2:
        x_u, y_u = np.meshgrid(x, y, indexing=indexing)
        len_unstruct = len(x) * len(y)
        x_u = np.reshape(x_u, len_unstruct)
        y_u = np.reshape(y_u, len_unstruct)
        return x_u, y_u, z, (len(x), len(y))
    if dim == 3:
        x_u, y_u, z_u = np.meshgrid(x, y, z, indexing=indexing)
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


def _get_select(direction):
    select = []
    if not (0 < len(direction) < 4):
        raise ValueError(
            "Field.mesh: need 1 to 3 direction(s), got '{}'".format(direction)
        )
    for axis in direction:
        if axis == "x":
            if 0 in select:
                raise ValueError(
                    "Field.mesh: got duplicate directions {}".format(direction)
                )
            select.append(0)
        elif axis == "y":
            if 1 in select:
                raise ValueError(
                    "Field.mesh: got duplicate directions {}".format(direction)
                )
            select.append(1)
        elif axis == "z":
            if 2 in select:
                raise ValueError(
                    "Field.mesh: got duplicate directions {}".format(direction)
                )
            select.append(2)
        else:
            raise ValueError(
                "Field.mesh: got unknown direction {}".format(axis)
            )
    return select
