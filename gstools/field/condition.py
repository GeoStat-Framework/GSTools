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

from gstools.field.tools import make_isotropic, unrotate_mesh
from gstools.tools.geometric import pos2xyz
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

    # evaluate the field at the conditional points
    x, y, z = pos2xyz(srf.cond_pos)
    if srf.do_rotation:
        x, y, z = unrotate_mesh(srf.model.dim, srf.model.angles, x, y, z)
    y, z = make_isotropic(srf.model.dim, srf.model.anis, y, z)
    err_data = srf.generator.__call__(x, y, z, "unstructured")

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

    # evaluate the field at the conditional points
    x, y, z = pos2xyz(srf.cond_pos)
    if srf.do_rotation:
        x, y, z = unrotate_mesh(srf.model.dim, srf.model.angles, x, y, z)
    y, z = make_isotropic(srf.model.dim, srf.model.anis, y, z)
    err_data = srf.generator.__call__(x, y, z, "unstructured") + srf.mean

    err_ok = Simple(
        model=srf.model,
        mean=srf.mean,
        cond_pos=srf.cond_pos,
        cond_val=err_data,
    )
    err_field, __ = err_ok(pos, mesh_type)
    cond_field = srf.raw_field + krige_field - err_field + srf.mean
    return cond_field, krige_field, err_field, krige_var
