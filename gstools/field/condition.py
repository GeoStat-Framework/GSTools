# -*- coding: utf-8 -*-
"""
GStools subpackage providing routines for conditioned random fields.

.. currentmodule:: gstools.field.condition

The following functions are provided

.. autosummary::
   ordinary
   simple
"""
import numpy as np
from scipy import interpolate as inter
from gstools.krige import Ordinary, Simple


def ordinary(pos, srf, mesh_type="unstructured"):
    """Condition a given spatial random field with ordinary kriging.

    Parameters
    ----------
    pos : :class:`list`
        the position tuple, containing main direction and transversal
        directions
    srf : :any:`SRF`
        The spatial random field class containing all information
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
    krige_ok = Ordinary(
        model=srf.model, cond_pos=srf.cond_pos, cond_val=srf.cond_val
    )
    krige_field, krige_var = krige_ok(pos, mesh_type)

    if mesh_type != "unstructured":
        err_data = inter.interpn(
            pos,
            srf.raw_field,
            np.array(srf.cond_pos).T,
            bounds_error=False,
            fill_value=0.0,
        )
    else:
        err_data = inter.griddata(
            pos, np.reshape(srf.raw_field, -1), srf.cond_pos, fill_value=0.0
        )

    err_ok = Ordinary(
        model=srf.model, cond_pos=srf.cond_pos, cond_val=err_data
    )
    err_field, __ = err_ok(pos, mesh_type)
    cond_field = srf.raw_field + krige_field - err_field
    return cond_field, krige_field, err_field, krige_var


def simple(pos, srf, mesh_type="unstructured"):
    """Condition a given spatial random field with simple kriging.

    Parameters
    ----------
    pos : :class:`list`
        the position tuple, containing main direction and transversal
        directions
    srf : :any:`SRF`
        The spatial random field class containing all information
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
    krige_sk = Simple(
        model=srf.model,
        mean=srf.mean,
        cond_pos=srf.cond_pos,
        cond_val=srf.cond_val,
    )
    krige_field, krige_var = krige_sk(pos, mesh_type)

    if mesh_type != "unstructured":
        err_data = inter.interpn(
            pos,
            srf.raw_field + srf.mean,
            np.array(srf.cond_pos).T,
            bounds_error=False,
            fill_value=srf.mean,
        )
    else:
        err_data = inter.griddata(
            pos,
            np.reshape(srf.raw_field + srf.mean, -1),
            srf.cond_pos,
            fill_value=srf.mean,
        )

    err_ok = Simple(
        model=srf.model,
        mean=srf.mean,
        cond_pos=srf.cond_pos,
        cond_val=err_data,
    )
    err_field, __ = err_ok(pos, mesh_type)
    cond_field = srf.raw_field + krige_field - err_field + srf.mean
    return cond_field, krige_field, err_field, krige_var
