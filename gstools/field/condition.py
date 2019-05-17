# -*- coding: utf-8 -*-
"""
GStools subpackage providing routines for conditioned random fields.

.. currentmodule:: gstools.field.condition

The following functions are provided

.. autosummary::
   condition_ok
"""
import numpy as np
from scipy import interpolate as inter


def condition_ok(
    pos, field, cond_pos, cond_val, model, mesh_type="unstructured"
):
    """Condition a given spatial random field with ordinary kriging.

    Parameters
    ----------
    pos : :class:`list`
        the position tuple, containing main direction and transversal
        directions
    field : :class:`numpy.ndarray`
        A generated spatial random field.
    cond_pos : :class:`list`
        the position tuple of the conditions
    cond_pos : :class:`numpy.ndarray`
        the values of the conditions
    model : :any:`CovModel`
        Covariance Model to use for the field.
    mesh_type : :class:`str`
        'structured' / 'unstructured'

    Returns
    -------
    cond_field : :class:`numpy.ndarray`
        the conditioned field
    krige_field : :class:`numpy.ndarray`
        the kriged field
    err_field : :class:`numpy.ndarray`
        the error field to set the given random field to zero at the conditions
    krige_var : :class:`numpy.ndarray`
        the variance of the kriged field
    """
    kwargs = {
        "pos": pos,
        "field": field,
        "cond_pos": cond_pos,
        "cond_val": cond_val,
        "model": model,
        "mesh_type": mesh_type,
    }
    if model.dim == 1:
        return _cond_ok_1d(**kwargs)
    elif model.dim == 2:
        return _cond_ok_2d(**kwargs)
    else:
        return _cond_ok_3d(**kwargs)


def _cond_ok_1d(
    pos, field, cond_pos, cond_val, model, mesh_type="unstructured"
):
    """Condition a given 1D spatial random field with ordinary kriging."""
    try:
        from pykrige.ok import OrdinaryKriging
    except ImportError as err:
        raise ImportError(
            "GSTools: To generate conditioned fields, the "
            + "pykrige module must be correctly installed: "
            + str(err)
        )

    if model.dim != 1:
        raise ValueError("Dimension 1 needed for 1D kriging!")

    krige_ok = OrdinaryKriging(
        cond_pos[0],
        np.zeros_like(cond_pos[0]),
        z=cond_val,
        **model.pykrige_kwargs
    )
    krige_field, krige_var = krige_ok.execute("grid", pos[0], np.array([0.0]))
    krige_field = np.reshape(krige_field, -1)

    err_data = np.interp(
        cond_pos[0], np.reshape(pos[0], -1), np.reshape(field, -1)
    )
    err_ok = OrdinaryKriging(
        cond_pos[0],
        np.zeros_like(cond_pos[0]),
        z=err_data,
        **model.pykrige_kwargs
    )
    err_field, err_var = err_ok.execute("grid", pos[0], np.array([0.0]))
    err_field = np.reshape(err_field, -1)

    cond_field = field + krige_field - err_field
    return cond_field, krige_field, err_field, krige_var


def _cond_ok_2d(
    pos, field, cond_pos, cond_val, model, mesh_type="unstructured"
):
    """Condition a given 2D spatial random field with ordinary kriging."""
    try:
        from pykrige.ok import OrdinaryKriging
    except ImportError as err:
        raise ImportError(
            "GSTools: To generate conditioned fields, the "
            + "pykrige module must be correctly installed: "
            + str(err)
        )

    if model.dim != 2:
        raise ValueError("Dimension 2 needed for 2D kriging!")

    pykrige_style = "points" if mesh_type == "unstructured" else "grid"

    krige_ok = OrdinaryKriging(*cond_pos, z=cond_val, **model.pykrige_kwargs)
    krige_field, krige_var = krige_ok.execute(pykrige_style, *pos)
    krige_field = krige_field.T

    if mesh_type != "unstructured":
        err_data = inter.interpn(
            pos,
            field,
            np.array(cond_pos).T,
            bounds_error=False,
            fill_value=0.0,
        )
    else:
        grid_x = np.reshape(pos[0], -1)
        grid_y = np.reshape(pos[1], -1)
        grid_v = np.reshape(field, -1)
        err_data = inter.griddata(
            (grid_x, grid_y), grid_v, cond_pos, fill_value=0.0
        )

    err_ok = OrdinaryKriging(*cond_pos, z=err_data, **model.pykrige_kwargs)
    err_field, err_var = err_ok.execute(pykrige_style, *pos)
    err_field = err_field.T
    cond_field = field + krige_field - err_field
    return cond_field, krige_field, err_field, krige_var


def _cond_ok_3d(
    pos, field, cond_pos, cond_val, model, mesh_type="unstructured"
):
    """Condition a given 3D spatial random field with ordinary kriging."""
    try:
        from pykrige.ok3d import OrdinaryKriging3D
    except ImportError as err:
        raise ImportError(
            "GSTools: To generate conditioned fields, the "
            + "pykrige module must be correctly installed: "
            + str(err)
        )

    if model.dim != 3:
        raise ValueError("Dimension 3 needed for 3D kriging!")

    pykrige_style = "points" if mesh_type == "unstructured" else "grid"

    krige_ok = OrdinaryKriging3D(
        *cond_pos, val=cond_val, **model.pykrige_kwargs
    )
    krige_field, krige_var = krige_ok.execute(pykrige_style, *pos)
    krige_field = krige_field.T

    if mesh_type != "unstructured":
        err_data = inter.interpn(
            pos,
            field,
            np.array(cond_pos).T,
            bounds_error=False,
            fill_value=0.0,
        )
    else:
        grid_x = np.reshape(pos[0], -1)
        grid_y = np.reshape(pos[1], -1)
        grid_z = np.reshape(pos[2], -1)
        grid_v = np.reshape(field, -1)
        err_data = inter.griddata(
            (grid_x, grid_y, grid_z), grid_v, cond_pos, fill_value=0.0
        )

    err_ok = OrdinaryKriging3D(*cond_pos, val=err_data, **model.pykrige_kwargs)
    err_field, err_var = err_ok.execute(pykrige_style, *pos)
    err_field = err_field.T
    cond_field = field + krige_field - err_field
    return cond_field, krige_field, err_field, krige_var
