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
"""
from __future__ import print_function, division, absolute_import
import numpy as np


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
