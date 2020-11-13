# -*- coding: utf-8 -*-
"""
GStools subpackage providing geometric tools.

.. currentmodule:: gstools.tools.geometric

The following functions are provided

.. autosummary::
   set_angles
   set_anis
   no_of_angles
   rotation_planes
   givens_rotation
   matrix_rotate
   matrix_derotate
   matrix_isotropify
   matrix_anisotropify
   matrix_isometrize
   matrix_anisometrize
   pos2xyz
   xyz2pos
   ang2dir
"""
# pylint: disable=C0103

import numpy as np

__all__ = [
    "set_angles",
    "set_anis",
    "no_of_angles",
    "rotation_planes",
    "givens_rotation",
    "matrix_rotate",
    "matrix_derotate",
    "matrix_isotropify",
    "matrix_anisotropify",
    "matrix_isometrize",
    "matrix_anisometrize",
    "pos2xyz",
    "xyz2pos",
    "ang2dir",
]


# Geometric functions #########################################################


def set_angles(dim, angles):
    """Set the angles for the given dimension.

    Parameters
    ----------
    dim : :class:`int`
        spatial dimension
    angles : :class:`float` or :class:`list`
        the angles of the SRF

    Returns
    -------
    angles : :class:`float`
        the angles fitting to the dimension

    Notes
    -----
        If too few angles are given, they are filled up with `0`.
    """
    out_angles = np.atleast_1d(angles)[: no_of_angles(dim)]
    # fill up the rotation angle array with zeros
    out_angles = np.pad(
        out_angles,
        (0, no_of_angles(dim) - len(out_angles)),
        "constant",
        constant_values=0.0,
    )
    return out_angles


def set_anis(dim, anis):
    """Set the anisotropy ratios for the given dimension.

    Parameters
    ----------
    dim : :class:`int`
        spatial dimension
    anis : :class:`list` of :class:`float`
        the anisotropy of length scales along the transversal directions

    Returns
    -------
    anis : :class:`list` of :class:`float`
        the anisotropy of length scales fitting the dimensions

    Notes
    -----
        If too few anisotrpy ratios are given, they are filled up with `1`.
    """
    out_anis = np.atleast_1d(anis)[: dim - 1]
    if len(out_anis) < dim - 1:
        # fill up the anisotropies with ones, such that len()==dim-1
        out_anis = np.pad(
            out_anis,
            (dim - len(out_anis) - 1, 0),
            "constant",
            constant_values=1.0,
        )
    return out_anis


def no_of_angles(dim):
    """Calculate number of rotation angles depending on the dimension.

    Parameters
    ----------
    dim : :class:`int`
        spatial dimension

    Returns
    -------
    :class:`int`
        Number of angles.
    """
    return (dim * (dim - 1)) // 2


def rotation_planes(dim):
    """Get all 2D sub-planes for rotation.

    Parameters
    ----------
    dim : :class:`int`
        spatial dimension

    Returns
    -------
    :class:`list` of :class:`tuple` of :class:`int`
        All 2D sub-planes for rotation.
    """
    return [(i, j) for j in range(1, dim) for i in range(j)]


def givens_rotation(dim, plane, angle):
    """Givens rotation matrix in arbitrary dimensions.

    Parameters
    ----------
    dim : :class:`int`
        spatial dimension
    plane : :class:`list` of :class:`int`
        the plane to rotate in, given by the indices of the two defining axes.
        For example the xy plane is defined by `(0,1)`
    angle : :class:`float` or :class:`list`
        the rotation angle in the given plane

    Returns
    -------
    :class:`numpy.ndarray`
        Rotation matrix.
    """
    result = np.eye(dim, dtype=np.double)
    result[plane[0], plane[0]] = np.cos(angle)
    result[plane[1], plane[1]] = np.cos(angle)
    result[plane[0], plane[1]] = -np.sin(angle)
    result[plane[1], plane[0]] = np.sin(angle)
    return result


def matrix_rotate(dim, angles):
    """Create a matrix to rotate points to the target coordinate-system.

    Parameters
    ----------
    dim : :class:`int`
        spatial dimension
    angles : :class:`float` or :class:`list`
        the rotation angles of the target coordinate-system

    Returns
    -------
    :class:`numpy.ndarray`
        Rotation matrix.
    """
    angles = set_angles(dim, angles)
    planes = rotation_planes(dim)
    result = np.eye(dim, dtype=np.double)
    for i, (angle, plane) in enumerate(zip(angles, planes)):
        # angles have alternating signs to match tait bryan
        result = np.matmul(
            givens_rotation(dim, plane, (-1) ** i * angle), result
        )
    return result


def matrix_derotate(dim, angles):
    """Create a matrix to derotate points to the initial coordinate-system.

    Parameters
    ----------
    dim : :class:`int`
        spatial dimension
    angles : :class:`float` or :class:`list`
        the rotation angles of the target coordinate-system

    Returns
    -------
        :class:`numpy.ndarray`
            Rotation matrix.
    """
    angles = -set_angles(dim, angles)
    planes = rotation_planes(dim)
    result = np.eye(dim, dtype=np.double)
    for i, (angle, plane) in enumerate(zip(angles, planes)):
        # angles have alternating signs to match tait bryan
        result = np.matmul(
            result, givens_rotation(dim, plane, (-1) ** i * angle)
        )
    return result


def matrix_isotropify(dim, anis):
    """Create a stretching matrix to make things isotrope.

    Parameters
    ----------
    dim : :class:`int`
        spatial dimension
    anis : :class:`list` of :class:`float`
        the anisotropy of length scales along the transversal directions

    Returns
    -------
        :class:`numpy.ndarray`
            Stretching matrix.
    """
    anis = set_anis(dim, anis)
    return np.diag(np.concatenate(([1.0], 1.0 / anis)))


def matrix_anisotropify(dim, anis):
    """Create a stretching matrix to make things anisotrope.

    Parameters
    ----------
    dim : :class:`int`
        spatial dimension
    anis : :class:`list` of :class:`float`
        the anisotropy of length scales along the transversal directions

    Returns
    -------
        :class:`numpy.ndarray`
            Stretching matrix.
    """
    anis = set_anis(dim, anis)
    return np.diag(np.concatenate(([1.0], anis)))


def matrix_isometrize(dim, angles, anis):
    """Create a matrix to derotate points and make them isotrope.

    Parameters
    ----------
    dim : :class:`int`
        spatial dimension
    angles : :class:`float` or :class:`list`
        the rotation angles of the target coordinate-system
    anis : :class:`list` of :class:`float`
        the anisotropy of length scales along the transversal directions

    Returns
    -------
        :class:`numpy.ndarray`
            Transformation matrix.
    """
    return np.matmul(
        matrix_isotropify(dim, anis), matrix_derotate(dim, angles)
    )


def matrix_anisometrize(dim, angles, anis):
    """Create a matrix to rotate points and make them anisotrope.

    Parameters
    ----------
    dim : :class:`int`
        spatial dimension
    angles : :class:`float` or :class:`list`
        the rotation angles of the target coordinate-system
    anis : :class:`list` of :class:`float`
        the anisotropy of length scales along the transversal directions

    Returns
    -------
        :class:`numpy.ndarray`
            Transformation matrix.
    """
    return np.matmul(
        matrix_rotate(dim, angles), matrix_anisotropify(dim, anis)
    )


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


def ang2dir(angles, dtype=np.double, dim=None):
    """Convert n-D spherical coordinates to Euclidean direction vectors.

    Parameters
    ----------
    angles : :class:`list` of :class:`numpy.ndarray`
        spherical coordinates given as angles.
    dtype : data-type, optional
        The desired data-type for the array.
        If not given, then the type will be determined as the minimum type
        required to hold the objects in the sequence. Default: None
    dim : :class:`int`, optional
        Cut of information above the given dimension.
        Otherwise, dimension is determined by number of angles
        Default: None

    Returns
    -------
    :class:`numpy.ndarray`
        the array of direction vectors
    """
    pre_dim = np.asanyarray(angles).ndim
    angles = np.array(angles, ndmin=2, dtype=dtype)
    if len(angles.shape) > 2:
        raise ValueError("Can't interpret angles array {}".format(angles))
    dim = angles.shape[1] + 1 if dim is None else dim
    if dim == 2 and angles.shape[0] == 1 and pre_dim < 2:
        # fix for 2D where only one angle per direction is given
        angles = angles.T  # can't be interpreted if dim=None is given
    if dim != angles.shape[1] + 1 or dim == 1:
        raise ValueError("Wrong dim. ({}) for angles {}".format(dim, angles))
    vec = np.empty((angles.shape[0], dim), dtype=dtype)
    vec[:, 0] = np.prod(np.sin(angles), axis=1)
    for i in range(1, dim):
        vec[:, i] = np.prod(np.sin(angles[:, i:]), axis=1)  # empty prod = 1
        vec[:, i] *= np.cos(angles[:, (i - 1)])
    if dim in [2, 3]:
        vec[:, [0, 1]] = vec[:, [1, 0]]  # to match convention in 2D and 3D
    return vec
