# -*- coding: utf-8 -*-
"""
GStools subpackage providing a class for standard spatial random fields.

.. currentmodule:: gstools.field.srf

The following classes are provided

.. autosummary::
   SRF
"""
# pylint: disable=C0103

import numpy as np
from gstools.field.generator import RandMeth, IncomprRandMeth
from gstools.field.base import Field
from gstools.field.upscaling import var_coarse_graining, var_no_scaling

__all__ = ["SRF"]

GENERATOR = {
    "RandMeth": RandMeth,
    "IncomprRandMeth": IncomprRandMeth,
    "VectorField": IncomprRandMeth,
    "VelocityField": IncomprRandMeth,
}
"""dict: Standard generators for spatial random fields."""

UPSCALING = {
    "coarse_graining": var_coarse_graining,
    "no_scaling": var_no_scaling,
}
"""dict: Upscaling routines for spatial random fields."""


class SRF(Field):
    """A class to generate spatial random fields (SRF).

    Parameters
    ----------
    model : :any:`CovModel`
        Covariance Model of the spatial random field.
    mean : :class:`float`, optional
        mean value of the SRF
    upscaling : :class:`str`, optional
        Method to be used for upscaling the variance at each point
        depending on the related element volume.
        See the ``point_volumes`` keyword in the :any:`SRF.__call__` routine.
        At the moment, the following upscaling methods are provided:

            * "no_scaling" : No upscaling is applied to the variance.
              See: :any:`var_no_scaling`
            * "coarse_graining" : A volume depended variance is
              calculated by the upscaling technique coarse graining.
              See: :any:`var_coarse_graining`

        Default: "no_scaling"
    generator : :class:`str`, optional
        Name of the field generator to be used.
        At the moment, the following generators are provided:

            * "RandMeth" : The Randomization Method.
              See: :any:`RandMeth`
            * "IncomprRandMeth" : The incompressible Randomization Method.
              This is the original algorithm proposed by Kraichnan 1970
              See: :any:`IncomprRandMeth`
            * "VectorField" : an alias for "IncomprRandMeth"
            * "VelocityField" : an alias for "IncomprRandMeth"

        Default: "RandMeth"
    **generator_kwargs
        Keyword arguments that are forwarded to the generator in use.
        Have a look at the provided generators for further information.
    """

    def __init__(
        self,
        model,
        mean=0.0,
        upscaling="no_scaling",
        generator="RandMeth",
        **generator_kwargs
    ):
        super().__init__(model)
        # initialize private attributes
        self._generator = None
        self._upscaling = None
        self._upscaling_func = None
        self._mean = None
        # initialize attributes
        self.mean = mean
        self.upscaling = upscaling
        self.set_generator(generator, **generator_kwargs)

    def __call__(
        self, pos, seed=np.nan, point_volumes=0.0, mesh_type="unstructured"
    ):
        """Generate the spatial random field.

        The field is saved as `self.field` and is also returned.

        Parameters
        ----------
        pos : :class:`list`
            the position tuple, containing main direction and transversal
            directions
        seed : :class:`int`, optional
            seed for RNG for reseting. Default: keep seed from generator
        point_volumes : :class:`float` or :class:`numpy.ndarray`
            If your evaluation points for the field are coming from a mesh,
            they are probably representing a certain element volume.
            This volume can be passed by `point_volumes` to apply the
            given variance upscaling. If `point_volumes` is ``0`` nothing
            is changed. Default: ``0``
        mesh_type : :class:`str`
            'structured' / 'unstructured'

        Returns
        -------
        field : :class:`numpy.ndarray`
            the SRF
        """
        # update the model/seed in the generator if any changes were made
        self.generator.update(self.model, seed)
        # get isometrized positions and the resulting field-shape
        iso_pos, shape = self.pre_pos(pos, mesh_type)
        # generate the field
        self.field = np.reshape(self.generator(iso_pos), shape)
        # upscaled variance
        if not np.isscalar(point_volumes) or not np.isclose(point_volumes, 0):
            scaled_var = self.upscaling_func(self.model, point_volumes)
            if np.size(scaled_var) > 1:
                scaled_var = np.reshape(scaled_var, shape)
            self.field *= np.sqrt(scaled_var / self.model.sill)
        self.field += self.mean
        return self.field

    def upscaling_func(self, *args, **kwargs):
        """Upscaling method applied to the field variance."""
        return self._upscaling_func(*args, **kwargs)

    def set_generator(self, generator, **generator_kwargs):
        """Set the generator for the field.

        Parameters
        ----------
        generator : :class:`str`, optional
            Name of the generator to use for field generation.
            Default: "RandMeth"
        **generator_kwargs
            keyword arguments that are forwarded to the generator in use.
        """
        if generator in GENERATOR:
            gen = GENERATOR[generator]
            self._generator = gen(self.model, **generator_kwargs)
            self.value_type = self._generator.value_type
        else:
            raise ValueError("gstools.SRF: Unknown generator: " + generator)

    @property
    def generator(self):
        """:any:`callable`: The generator of the field.

        Default: :any:`RandMeth`
        """
        return self._generator

    @property
    def upscaling(self):  # pragma: no cover
        """:class:`str`: Name of the upscaling method.

        See the ``point_volumes`` keyword in the :any:`SRF.__call__` routine.
        Default: "no_scaling"
        """
        return self._upscaling

    @upscaling.setter
    def upscaling(self, upscaling):
        if upscaling in UPSCALING:
            self._upscaling = upscaling
            self._upscaling_func = UPSCALING[upscaling]
        else:
            raise ValueError(
                "gstools.SRF: Unknown upscaling method: " + upscaling
            )

    @property
    def mean(self):
        """:class:`float`: The mean of the field."""
        return self._mean

    @mean.setter
    def mean(self, mean):
        self._mean = float(mean)

    def __repr__(self):
        """Return String representation."""
        return "SRF(model={0}, mean={1:.{p}}, generator={2}".format(
            self.model, self.mean, self.generator, p=self.model._prec
        )
