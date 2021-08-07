# -*- coding: utf-8 -*-
"""
GStools subpackage providing a class for standard spatial random fields.

.. currentmodule:: gstools.field.srf

The following classes are provided

.. autosummary::
   SRF
"""
# pylint: disable=C0103, W0221, E1102
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
    mean : :class:`float` or :any:`callable`, optional
        Mean of the SRF (in normal form). Could also be a callable.
        The default is 0.0.
    normalizer : :any:`None` or :any:`Normalizer`, optional
        Normalizer to be applied to the SRF to transform the field values.
        The default is None.
    trend : :any:`None` or :class:`float` or :any:`callable`, optional
        Trend of the SRF (in transformed form).
        If no normalizer is applied, this behaves equal to 'mean'.
        The default is None.
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
        normalizer=None,
        trend=None,
        upscaling="no_scaling",
        generator="RandMeth",
        **generator_kwargs,
    ):
        super().__init__(model, mean=mean, normalizer=normalizer, trend=trend)
        # initialize private attributes
        self._generator = None
        self._upscaling = None
        self._upscaling_func = None
        # initialize attributes
        self.upscaling = upscaling
        self.set_generator(generator, **generator_kwargs)

    def __call__(
        self,
        pos=None,
        seed=np.nan,
        point_volumes=0.0,
        mesh_type="unstructured",
        post_process=True,
        store=True,
    ):
        """Generate the spatial random field.

        The field is saved as `self.field` and is also returned.

        Parameters
        ----------
        pos : :class:`list`, optional
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
        post_process : :class:`bool`, optional
            Whether to apply mean, normalizer and trend to the field.
            Default: `True`
        store : :class:`str` or :class:`bool`, optional
            Whether to store field (True/False) with default name
            or with specified name.
            The default is :any:`True` for default name "field".

        Returns
        -------
        field : :class:`numpy.ndarray`
            the SRF
        """
        name, save = self.get_store_config(store)
        # update the model/seed in the generator if any changes were made
        self.generator.update(self.model, seed)
        # get isometrized positions and the resulting field-shape
        iso_pos, shape = self.pre_pos(pos, mesh_type)
        # generate the field
        field = np.reshape(self.generator(iso_pos), shape)
        # upscaled variance
        if not np.isscalar(point_volumes) or not np.isclose(point_volumes, 0):
            scaled_var = self.upscaling_func(self.model, point_volumes)
            if np.size(scaled_var) > 1:
                scaled_var = np.reshape(scaled_var, shape)
            field *= np.sqrt(scaled_var / self.model.sill)
        return self.post_field(field, name, post_process, save)

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
            raise ValueError(f"gstools.SRF: Unknown generator: {generator}")
        for val in [self.mean, self.trend]:
            if not callable(val) and val is not None:
                if np.size(val) > 1 and self.value_type == "scalar":
                    raise ValueError(f"Mean/Trend: Wrong size ({val})")

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
            raise ValueError(f"SRF: Unknown upscaling method: {upscaling}")

    def __repr__(self):
        """Return String representation."""
        return "{0}(model={1}{2}, generator={3})".format(
            self.name,
            self.model.name,
            self._fmt_mean_norm_trend(),
            self.generator.name,
        )
