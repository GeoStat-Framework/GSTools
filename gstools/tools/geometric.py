# -*- coding: utf-8 -*-
"""
GStools subpackage providing geometric tools.

.. currentmodule:: gstools.tools.geometric

The following functions are provided

.. autosummary::
   r3d_x
   r3d_y
   r3d_z
"""
from __future__ import print_function, division, absolute_import

import numpy as np

__all__ = ["r3d_x", "r3d_y", "r3d_z"]


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


def pos2xyz(pos, dtype=None, calc_dim=False):
    """Convert postional arguments to x, y, z

    Parameters
    ----------
        pos : :any:`iterable`
            the position tuple, containing main direction and transversal
            directions
        dtype : data-type, optional
            The desired data-type for the array.
            If not given, then the type will be determined as the minimum type
            required to hold the objects in the sequence. Default: None
        calc_dim : :class:`bool`
            State if the dimension should be returned

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

    Note
    ----
    If len(pos) > 3, everything after pos[2] will be ignored.
    """
    x = np.array(pos[0], dtype=dtype).reshape(-1)
    dim = 1
    y = z = None
    if len(pos) > 1:
        dim = 2
        y = np.array(pos[1], dtype=dtype).reshape(-1)
    if len(pos) > 2:
        dim = 3
        z = np.array(pos[2], dtype=dtype).reshape(-1)
    if calc_dim:
        return x, y, z, dim
    return x, y, z


def xyz2pos(x, y=None, z=None, dtype=None):
    """Convert postional arguments to x, y, z

    Parameters
    ----------
        pos : :any:`iterable`
            the position tuple, containing main direction and transversal
            directions
        x : :class:`numpy.ndarray`
            grid axis in x-direction if structured, or first components of
            position vectors if unstructured
        y : :class:`numpy.ndarray`, optional
            analog to x
        z : :class:`numpy.ndarray`, optional
            analog to x
        seed : :class:`int`, optional
            seed for RNG for reseting. Default: keep seed from generator
        force_moments : :class:`bool`
            Force the generator to exactly match mean and variance.
            Default: ``False``
        point_volumes : :class:`float` or :class:`numpy.ndarray`
            If your evaluation points for the field are coming from a mesh,
            they are probably representing a certain element volume.
            This volumes can be passed by `point_volumes` to apply the
            given variance upscaling. If `point_volumes` is ``0`` nothing
            is changed. Default: ``0``
        mesh_type : :class:`str`
            'structured' / 'unstructured'
    Returns
    -------
        field : :class:`numpy.ndarray`
            the SRF
    """
    if y is None and z is not None:
        raise ValueError("gstools.tools.xyz2pos: if z is given, y is needed!")
    pos = []
    pos.append(np.array(x, dtype=dtype).reshape(-1))
    if y is not None:
        pos.append(np.array(y, dtype=dtype).reshape(-1))
    if z is not None:
        pos.append(np.array(z, dtype=dtype).reshape(-1))
    return tuple(pos)
