# -*- coding: utf-8 -*-
"""
GStools subpackage providing routines for conditioned random fields.

.. currentmodule:: gstools.field.condition

The following functions are provided

.. autosummary::
   ordinary
   simple
"""
# pylint: disable=C0103
from gstools.field.tools import make_isotropic, unrotate_mesh
from gstools.tools.geometric import pos2xyz
from gstools.krige import Ordinary, Simple


def ordinary(srf):
    """Condition a given spatial random field with ordinary kriging.

    Parameters
    ----------
    srf : :any:`SRF`
        The spatial random field class containing all information

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
    if srf._value_type != "scalar":
        raise ValueError("Conditioned SRF: only scalar fields allowed.")
    krige_ok = Ordinary(
        model=srf.model, cond_pos=srf.cond_pos, cond_val=srf.cond_val
    )
    krige_field, krige_var = krige_ok(srf.pos, srf.mesh_type)

    # evaluate the field at the conditional points
    x, y, z = pos2xyz(srf.cond_pos, max_dim=srf.model.dim)
    if srf.model.do_rotation:
        x, y, z = unrotate_mesh(srf.model.dim, srf.model.angles, x, y, z)
    y, z = make_isotropic(srf.model.dim, srf.model.anis, y, z)
    err_data = srf.generator.__call__(x, y, z, "unstructured")

    err_ok = Ordinary(
        model=srf.model, cond_pos=srf.cond_pos, cond_val=err_data
    )
    err_field, __ = err_ok(srf.pos, srf.mesh_type)
    cond_field = srf.raw_field + krige_field - err_field
    info = {"mean": krige_ok.mean}
    return cond_field, krige_field, err_field, krige_var, info


def simple(srf):
    """Condition a given spatial random field with simple kriging.

    Parameters
    ----------
    srf : :any:`SRF`
        The spatial random field class containing all information

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
    if srf._value_type != "scalar":
        raise ValueError("Conditioned SRF: only scalar fields allowed.")
    krige_sk = Simple(
        model=srf.model,
        mean=srf.mean,
        cond_pos=srf.cond_pos,
        cond_val=srf.cond_val,
    )
    krige_field, krige_var = krige_sk(srf.pos, srf.mesh_type)

    # evaluate the field at the conditional points
    x, y, z = pos2xyz(srf.cond_pos, max_dim=srf.model.dim)
    if srf.model.do_rotation:
        x, y, z = unrotate_mesh(srf.model.dim, srf.model.angles, x, y, z)
    y, z = make_isotropic(srf.model.dim, srf.model.anis, y, z)
    err_data = srf.generator.__call__(x, y, z, "unstructured") + srf.mean

    err_sk = Simple(
        model=srf.model,
        mean=srf.mean,
        cond_pos=srf.cond_pos,
        cond_val=err_data,
    )
    err_field, __ = err_sk(srf.pos, srf.mesh_type)
    cond_field = srf.raw_field + krige_field - err_field + srf.mean
    info = {}
    return cond_field, krige_field, err_field, krige_var, info
