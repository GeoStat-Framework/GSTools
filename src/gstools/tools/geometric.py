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
   rotated_main_axes
   generate_grid
   generate_st_grid
   format_struct_pos_dim
   format_struct_pos_shape
   format_unstruct_pos_shape
   ang2dir
   latlon2pos
   pos2latlon
   chordal_to_great_circle
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
    "rotated_main_axes",
    "generate_grid",
    "generate_st_grid",
    "format_struct_pos_dim",
    "format_struct_pos_shape",
    "format_unstruct_pos_shape",
    "ang2dir",
    "latlon2pos",
    "pos2latlon",
    "chordal_to_great_circle",
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
    out_angles = np.asarray(angles, dtype=np.double)
    out_angles = np.atleast_1d(out_angles)[: no_of_angles(dim)]
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
        If too few anisotropy ratios are given, they are filled up with `1`.
    """
    out_anis = np.asarray(anis, dtype=np.double)
    out_anis = np.atleast_1d(out_anis)[: dim - 1]
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
        # angles have alternating signs to match tait-bryan
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
    # derotating by taking negative angles
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


def rotated_main_axes(dim, angles):
    """Create list of the main axis defined by the given system rotations.

    Parameters
    ----------
    dim : :class:`int`
        spatial dimension
    angles : :class:`float` or :class:`list`
        the rotation angles of the target coordinate-system

    Returns
    -------
    :class:`numpy.ndarray`
        Main axes of the target coordinate-system.
    """
    return matrix_rotate(dim, angles).T


# grid routines ###############################################################


def generate_grid(pos):
    """
    Generate grid from a structured position tuple.

    Parameters
    ----------
    pos : :class:`tuple` of :class:`numpy.ndarray`
        The structured position tuple.

    Returns
    -------
    :class:`numpy.ndarray`
        Unstructured position tuple.
    """
    return np.asarray(
        np.meshgrid(*pos, indexing="ij"), dtype=np.double
    ).reshape((len(pos), -1))


def generate_st_grid(pos, time, mesh_type="unstructured"):
    """
    Generate spatio-temporal grid from a position tuple and time array.

    Parameters
    ----------
    pos : :class:`tuple` of :class:`numpy.ndarray`
        The (un-)structured position tuple.
    time : :any:`iterable`
        The time array.
    mesh_type : :class:`str`, optional
        'structured' / 'unstructured'
        Default: `"unstructured"`

    Returns
    -------
    :class:`numpy.ndarray`
        Unstructured spatio-temporal point tuple.

    Notes
    -----
        Time dimension will be the last one.
    """
    time = np.asarray(time, dtype=np.double).reshape(-1)
    if mesh_type != "unstructured":
        pos = generate_grid(pos)
    else:
        pos = np.array(pos, dtype=np.double, ndmin=2, copy=False)
    out = [np.repeat(p.reshape(-1), np.size(time)) for p in pos]
    out.append(np.tile(time, np.size(pos[0])))
    return np.asarray(out, dtype=np.double)


# conversion ##################################################################


def format_struct_pos_dim(pos, dim):
    """
    Format a structured position tuple with given dimension.

    Parameters
    ----------
    pos : :any:`iterable`
        Position tuple, containing main direction and transversal directions.
    dim : :class:`int`
        Spatial dimension.

    Raises
    ------
    ValueError
        When position tuple doesn't match the given dimension.

    Returns
    -------
    pos : :class:`tuple` of :class:`numpy.ndarray`
        The formatted structured position tuple.
    shape : :class:`tuple`
        Shape of the resulting field.
    """
    if dim == 1:
        pos = (np.asarray(pos, dtype=np.double).reshape(-1),)
    elif len(pos) != dim:
        raise ValueError("Formatting: position tuple doesn't match dimension.")
    else:
        pos = tuple(np.asarray(p, dtype=np.double).reshape(-1) for p in pos)
    shape = tuple(len(p) for p in pos)
    return pos, shape


def format_struct_pos_shape(pos, shape, check_stacked_shape=False):
    """
    Format a structured position tuple with given shape.

    Shape could be stacked, when multiple fields are given.

    Parameters
    ----------
    pos : :any:`iterable`
        Position tuple, containing main direction and transversal directions.
    shape : :class:`tuple`
        Shape of the input field.
    check_stacked_shape : :class:`bool`, optional
        Whether to check if given shape comes from stacked fields.
        Default: False.

    Raises
    ------
    ValueError
        When position tuple doesn't match the given dimension.

    Returns
    -------
    pos : :class:`tuple` of :class:`numpy.ndarray`
        The formatted structured position tuple.
    shape : :class:`tuple`
        Shape of the resulting field.
    dim : :class:`int`
        Spatial dimension.
    """
    # some help from the given shape
    shape_size = np.prod(shape)
    stacked_shape_size = np.prod(shape[1:])
    wrong_shape = False
    # now we try to be smart
    try:
        # if this works we have either:
        # - a 1D array
        # - nD array where all axes have same length (corner case)
        check_pos = np.array(pos, dtype=np.double, ndmin=2)
    except ValueError:
        # if it doesn't work, we have a tuple of differently sized axes (easy)
        dim = len(pos)
        pos, pos_shape = format_struct_pos_dim(pos, dim)
        # determine if we have a stacked field if wanted
        if check_stacked_shape and stacked_shape_size == np.prod(pos_shape):
            shape = (shape[0],) + pos_shape
        # check if we have a single field with matching size
        elif shape_size == np.prod(pos_shape):
            shape = (1,) + pos_shape if check_stacked_shape else pos_shape
        # if nothing works, we raise an error
        else:
            wrong_shape = True
    else:
        struct_size = np.prod([p.size for p in check_pos])
        # case: 1D unstacked
        if check_pos.size == shape_size:
            dim = 1
            pos, pos_shape = format_struct_pos_dim(check_pos, dim)
            shape = (1,) + pos_shape if check_stacked_shape else pos_shape
        # case: 1D and stacked
        elif check_pos.size == stacked_shape_size:
            dim = 1
            pos, pos_shape = format_struct_pos_dim(check_pos, dim)
            cnt = shape[0]
            shape = (cnt,) + pos_shape
            wrong_shape = not check_stacked_shape
        # case: nD unstacked
        elif struct_size == shape_size:
            dim = len(check_pos)
            pos, pos_shape = format_struct_pos_dim(pos, dim)
            shape = (1,) + pos_shape if check_stacked_shape else pos_shape
        # case: nD and stacked
        elif struct_size == stacked_shape_size:
            dim = len(check_pos)
            pos, pos_shape = format_struct_pos_dim(pos, dim)
            cnt = shape[0]
            shape = (cnt,) + pos_shape
            wrong_shape = not check_stacked_shape
        # if nothing works, we raise an error
        else:
            wrong_shape = True

    # if shape was wrong at one point we raise an error
    if wrong_shape:
        raise ValueError("Formatting: position tuple doesn't match dimension.")

    return pos, shape, dim


def format_unstruct_pos_shape(pos, shape, check_stacked_shape=False):
    """
    Format an unstructured position tuple with given shape.

    Shape could be stacked, when multiple fields were given.

    Parameters
    ----------
    pos : :any:`iterable`
        Position tuple, containing point coordinates.
    shape : :class:`tuple`
        Shape of the input field.
    check_stacked_shape : :class:`bool`, optional
        Whether to check if given shape comes from stacked fields.
        Default: False.

    Raises
    ------
    ValueError
        When position tuple doesn't match the given dimension.

    Returns
    -------
    pos : :class:`tuple` of :class:`numpy.ndarray`
        The formatted structured position tuple.
    shape : :class:`tuple`
        Shape of the resulting field.
    dim : :class:`int`
        Spatial dimension.
    """
    # some help from the given shape
    shape_size = np.prod(shape)
    stacked_shape_size = np.prod(shape[1:])
    wrong_shape = False
    # now we try to be smart
    pre_len = len(np.atleast_1d(pos))
    # care about 1D: pos can be given as 1D array here -> convert to 2D array
    pos = np.array(pos, dtype=np.double, ndmin=2, copy=False)
    post_len = len(pos)
    # first array dimension should be spatial dimension (1D is special case)
    dim = post_len if pre_len == post_len else 1
    pnt_cnt = pos[0].size
    # case: 1D unstacked
    if dim == 1 and pos.size == shape_size:
        shape = (1, pos.size) if check_stacked_shape else (pos.size,)
    # case: 1D and stacked
    elif dim == 1 and pos.size == stacked_shape_size:
        shape = (shape[0], pos.size)
        wrong_shape = not check_stacked_shape
    # case: nD unstacked
    elif pnt_cnt == shape_size:
        shape = (1, pnt_cnt) if check_stacked_shape else pnt_cnt
    # case: nD and stacked
    elif pnt_cnt == stacked_shape_size:
        shape = (shape[0], pnt_cnt)
        wrong_shape = not check_stacked_shape
    # if nothing works, we raise an error
    else:
        wrong_shape = True

    # if shape was wrong at one point we raise an error
    if wrong_shape:
        raise ValueError("Formatting: position tuple doesn't match dimension.")

    pos = pos.reshape((dim, -1))

    return pos, shape, dim


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
    angles = np.array(angles, ndmin=2, dtype=dtype, copy=False)
    if len(angles.shape) > 2:
        raise ValueError(f"Can't interpret angles array {angles}")
    dim = angles.shape[1] + 1 if dim is None else dim
    if dim == 2 and angles.shape[0] == 1 and pre_dim < 2:
        # fix for 2D where only one angle per direction is given
        angles = angles.T  # can't be interpreted if dim=None is given
    if dim != angles.shape[1] + 1 or dim == 1:
        raise ValueError(f"Wrong dim. ({dim}) for angles {angles}")
    vec = np.empty((angles.shape[0], dim), dtype=dtype)
    vec[:, 0] = np.prod(np.sin(angles), axis=1)
    for i in range(1, dim):
        vec[:, i] = np.prod(np.sin(angles[:, i:]), axis=1)  # empty prod = 1
        vec[:, i] *= np.cos(angles[:, (i - 1)])
    if dim in [2, 3]:
        vec[:, [0, 1]] = vec[:, [1, 0]]  # to match convention in 2D and 3D
    return vec


def latlon2pos(latlon, radius=1.0, dtype=np.double):
    """Convert lat-lon geo coordinates to 3D position tuple.

    Parameters
    ----------
    latlon : :class:`list` of :class:`numpy.ndarray`
        latitude and longitude given in degrees.
    radius : :class:`float`, optional
        Earth radius. Default: `1.0`
    dtype : data-type, optional
        The desired data-type for the array.
        If not given, then the type will be determined as the minimum type
        required to hold the objects in the sequence. Default: None

    Returns
    -------
    :class:`numpy.ndarray`
        the 3D position array
    """
    latlon = np.asarray(latlon, dtype=dtype).reshape((2, -1))
    lat, lon = np.deg2rad(latlon)
    return np.array(
        (
            radius * np.cos(lat) * np.cos(lon),
            radius * np.cos(lat) * np.sin(lon),
            radius * np.sin(lat) * np.ones_like(lon),
        ),
        dtype=dtype,
    )


def pos2latlon(pos, radius=1.0, dtype=np.double):
    """Convert 3D position tuple from sphere to lat-lon geo coordinates.

    Parameters
    ----------
    pos : :class:`list` of :class:`numpy.ndarray`
        The position tuple containing points on a unit-sphere.
    radius : :class:`float`, optional
        Earth radius. Default: `1.0`
    dtype : data-type, optional
        The desired data-type for the array.
        If not given, then the type will be determined as the minimum type
        required to hold the objects in the sequence. Default: None

    Returns
    -------
    :class:`numpy.ndarray`
        the 3D position array
    """
    pos = np.asarray(pos, dtype=dtype).reshape((3, -1))
    # prevent numerical errors in arcsin
    lat = np.arcsin(np.maximum(np.minimum(pos[2] / radius, 1.0), -1.0))
    lon = np.arctan2(pos[1], pos[0])
    return np.rad2deg((lat, lon), dtype=dtype)


def chordal_to_great_circle(dist):
    """
    Calculate great circle distance corresponding to given chordal distance.

    Parameters
    ----------
    dist : array_like
        Chordal distance of two points on the unit-sphere.

    Returns
    -------
    :class:`numpy.ndarray`
        Great circle distance corresponding to given chordal distance.

    Notes
    -----
    If given values are not in [0, 1], they will be truncated.
    """
    return 2 * np.arcsin(np.maximum(np.minimum(np.divide(dist, 2), 1), 0))
