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
from gstools.tools.geometric import format_struct_pos_dim, generate_grid


__all__ = ["list_format", "eval_func"]


def list_format(lst, prec):  # pragma: no cover
    """Format a list of floats."""
    return f"[{', '.join(f'{float(x):.{prec}}' for x in lst)}]"


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
    func_val = 0 if func_val is None else func_val
    if broadcast and not callable(func_val) and np.size(func_val) == 1:
        return np.array(func_val, dtype=np.double).item()
    if not callable(func_val):
        func_val = _func_from_single_val(func_val, dim, value_type=value_type)
    # care about mesh and function call
    if mesh_type != "unstructured":
        pos, shape = format_struct_pos_dim(pos, dim)
        pos = generate_grid(pos)
    else:
        pos = np.array(pos, dtype=np.double).reshape(dim, -1)
        shape = np.shape(pos[0])
    # prepend dimension if we have a vector field
    if value_type == "vector":
        shape = (dim,) + shape
    return np.reshape(func_val(*pos), shape)


def _func_from_single_val(value, dim=None, value_type="scalar"):
    # care about broadcasting vector values for each dim
    v_d = dim if value_type == "vector" else 1  # value dim
    if v_d is None:  # pragma: no cover
        raise ValueError("_func_from_single_val: dim needed for vector value.")
    value = np.array(value, dtype=np.double).ravel()[:v_d]
    # fill up vector valued output to dimension with last value
    value = np.pad(
        value, (0, v_d - len(value)), "constant", constant_values=value[-1]
    )

    def _f(*pos):
        # zip uses shortest len of iterables given (correct for scalar value)
        return np.concatenate(
            [
                np.full_like(p, val, dtype=np.double)
                for p, val in zip(pos, value)
            ]
        )

    return _f
