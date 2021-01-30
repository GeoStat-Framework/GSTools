# -*- coding: utf-8 -*-
"""
GStools subpackage providing miscellaneous tools.

.. currentmodule:: gstools.tools.misc

The following functions are provided

.. autosummary::
   list_format
   eval_func
"""
import numpy as np
from gstools.tools.geometric import format_struct_pos_dim, gen_mesh


__all__ = ["list_format", "eval_func"]


def list_format(lst, prec):
    """Format a list of floats."""
    return "[{}]".format(
        ", ".join("{x:.{p}}".format(x=x, p=prec) for x in lst)
    )


def eval_func(
    func_val,
    pos,
    dim,
    mesh_type="unstructured",
    value_type="scalar",
    broadcast=False,
):
    """
    Evaluate a function on a mesh.

    Parameters
    ----------
    func_val : :any:`callable` or :class:`float` or :any:`None`
        Function to be called or single value to be filled.
        Should have the signiture f(x, [y, z, ...]) in case of callable.
        In case of a float, the field will be filled with a single value and
        in case of None, this value will be set to 0.
    pos : :class:`list`
        The position tuple, containing main direction and transversal
        directions (x, [y, z, ...]).
    dim : :class:`int`
        The spatial dimension.
    mesh_type : :class:`str`, optional
        'structured' / 'unstructured'
        Default: 'unstructured'
    value_type : :class:`str`, optional
        Value type of the field. Either "scalar" or "vector".
        The default is "scalar".
    broadcast : :class:`bool`, optional
        Whether to return a single value, if a single value was given.
        Default: False

    Returns
    -------
    :class:`numpy.ndarray`
        Function values at the given points.
    """
    # care about scalar inputs
    if broadcast and not callable(func_val):
        return 0.0 if func_val is None else float(func_val)
    if not callable(func_val):
        func_val = _func_from_scalar(func_val)
    # care about mesh and function call
    if mesh_type != "unstructured":
        pos, shape = format_struct_pos_dim(pos, dim)
        pos = gen_mesh(pos)
    else:
        pos = np.array(pos, dtype=np.double).reshape(dim, -1)
        shape = np.shape(pos[0])
    # prepend dimension if we have a vector field
    if value_type == "vector":
        shape = (dim,) + shape
    return np.reshape(func_val(*pos), shape)


def _func_from_scalar(value, value_type="scalar"):
    value = 0.0 if value is None else float(value)

    def _f(*pos):
        if value_type == "vector":
            return np.full_like(pos, value, dtype=np.double)
        # scalar field has same shape like a single axis
        return np.full_like(pos[0], value, dtype=np.double)

    return _f
