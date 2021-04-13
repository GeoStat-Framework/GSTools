# -*- coding: utf-8 -*-
"""
GStools subpackage providing upscaling routines for the spatial random field.

.. currentmodule:: gstools.field.upscaling

The following functions are provided

.. autosummary::
   var_coarse_graining
   var_no_scaling
"""
# pylint: disable=W0613
import warnings
import numpy as np

__all__ = ["var_coarse_graining", "var_no_scaling"]


# scaling routines ############################################################


def var_coarse_graining(model, point_volumes=0.0):
    r"""Coarse Graning procedure to upscale the variance for uniform flow.

    Parameters
    ----------
    model : :any:`CovModel`
        Covariance Model used for the field.
    point_volumes : :class:`float` or :class:`numpy.ndarray`
        Volumes of the elements at the given points. Default: ``0``

    Returns
    -------
    scaled_var : :class:`float` or :class:`numpy.ndarray`
        The upscaled variance

    Notes
    -----
    This procedure was presented in [Attinger03]_. It applies the
    upscaling procedure 'Coarse Graining' to the Groundwater flow equation
    under uniform flow on a lognormal distributed conductivity field following
    a gaussian covariance function. A filter over a cube with a given
    edge-length :math:`\lambda` is applied and an upscaled conductivity field
    is obtained.
    The upscaled field is again following a gaussian covariance function with
    scale dependent variance and length-scale:

    .. math::
       \lambda &= V^{\frac{1}{d}} \\
       \sigma^2\left(\lambda\right) &=
       \sigma^2\cdot\left(
       \frac{\ell^2}{\ell^2+\left(\frac{\lambda}{2}\right)^2}
       \right)^{\frac{d}{2}} \\
       \ell\left(\lambda\right) &=
       \left(\ell^2+\left(\frac{\lambda}{2}\right)^2\right)^{\frac{1}{2}}

    Therby :math:`\lambda` will be calculated from the given
    ``point_volumes`` :math:`V` by assuming a cube with the given volume.

    The upscaled length scale will be ignored by this routine.

    References
    ----------
    .. [Attinger03] Attinger, S. 2003,
       ''Generalized coarse graining procedures for flow in porous media'',
       Computational Geosciences, 7(4), 253–273.
    """
    if not np.isclose(model.nugget, 0):
        warnings.warn(
            "var_coarse_graining: non-zero nugget will violate upscaling!"
        )
    # interprete volume as a hypercube and calculate the edge length
    edge = point_volumes ** (1.0 / model.dim)
    var_factor = (
        model.len_scale ** 2 / (model.len_scale ** 2 + edge ** 2 / 4)
    ) ** (model.dim / 2.0)

    return model.sill * var_factor


def var_no_scaling(model, *args, **kwargs):
    r"""Dummy function to bypass scaling.

    Parameters
    ----------
    model : :any:`CovModel`
        Covariance Model used for the field.

    Returns
    -------
    var : :class:`float`
        The model variance.
    """
    return model.sill
