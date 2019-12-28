# -*- coding: utf-8 -*-
"""
GStools subpackage providing geometric tools.

.. currentmodule:: gstools.tools.geometric

The following functions are provided

.. autosummary::
   r3d_x
   r3d_y
   r3d_z
   pos2xyz
   xyz2pos
"""
# pylint: disable=C0103

import numpy as np

__all__ = ["r3d_x", "r3d_y", "r3d_z", "pos2xyz", "xyz2pos"]


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


# conversion ##################################################################


def pos2xyz(pos, dtype=None, calc_dim=False, max_dim=3):
    """Convert postional arguments to x, y, z.

    Parameters
    ----------
    pos : :any:`iterable`
        the position tuple, containing main direction and transversal
        directions
    dtype : data-type, optional
        The desired data-type for the array.
        If not given, then the type will be determined as the minimum type
        required to hold the objects in the sequence. Default: None
    calc_dim : :class:`bool`, optional
        State if the dimension should be returned. Default: False
    max_dim : :class:`int`, optional
        Cut of information above the given dimension. Default: 3

    Returns
    -------
    x : :class:`numpy.ndarray`
        first components of position vectors
    y : :class:`numpy.ndarray` or None
        analog to x
    z : :class:`numpy.ndarray` or None
        analog to x
    dim : :class:`int`, optional
        dimension (only if calc_dim is True)

    Notes
    -----
    If len(pos) > 3, everything after pos[2] will be ignored.
    """
    if max_dim == 1:  # sanity check
        pos = np.array(pos, ndmin=2)
    x = np.array(pos[0], dtype=dtype).reshape(-1)
    dim = 1
    y = z = None
    if len(pos) > 1 and max_dim > 1:
        dim = 2
        y = np.array(pos[1], dtype=dtype).reshape(-1)
    if len(pos) > 2 and max_dim > 2:
        dim = 3
        z = np.array(pos[2], dtype=dtype).reshape(-1)
    if calc_dim:
        return x, y, z, dim
    return x, y, z


def xyz2pos(x, y=None, z=None, dtype=None, max_dim=3):
    """Convert x, y, z to postional arguments.

    Parameters
    ----------
    x : :class:`numpy.ndarray`
        grid axis in x-direction if structured, or first components of
        position vectors if unstructured
    y : :class:`numpy.ndarray`, optional
        analog to x
    z : :class:`numpy.ndarray`, optional
        analog to x
    dtype : data-type, optional
        The desired data-type for the array.
        If not given, then the type will be determined as the minimum type
        required to hold the objects in the sequence. Default: None
    max_dim : :class:`int`, optional
        Cut of information above the given dimension. Default: 3

    Returns
    -------
    pos : :class:`tuple` of :class:`numpy.ndarray`
        the position tuple
    """
    if y is None and z is not None:
        raise ValueError("gstools.tools.xyz2pos: if z is given, y is needed!")
    pos = []
    pos.append(np.array(x, dtype=dtype).reshape(-1))
    if y is not None and max_dim > 1:
        pos.append(np.array(y, dtype=dtype).reshape(-1))
    if z is not None and max_dim > 2:
        pos.append(np.array(z, dtype=dtype).reshape(-1))
    return tuple(pos)
