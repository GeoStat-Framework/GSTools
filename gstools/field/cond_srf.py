# -*- coding: utf-8 -*-
"""
GStools subpackage providing a class for conditioned spatial random fields.

.. currentmodule:: gstools.field.cond_srf

The following classes are provided

.. autosummary::
   CondSRF
"""
# pylint: disable=C0103

import numpy as np
from gstools.field.generator import RandMeth
from gstools.field.base import Field
from gstools.krige import Krige

__all__ = ["CondSRF"]

GENERATOR = {
    "RandMeth": RandMeth,
}
"""dict: Standard generators for conditioned spatial random fields."""


class CondSRF(Field):
    """A class to generate conditioned spatial random fields (SRF).

    Parameters
    ----------
    model : :any:`CovModel`
        Covariance Model of the spatial random field.
    generator : :class:`str`, optional
        Name of the field generator to be used.
        At the moment, only the following generator is provided:

            * "RandMeth" : The Randomization Method.
              See: :any:`RandMeth`

        Default: "RandMeth"
    **generator_kwargs
        Keyword arguments that are forwarded to the generator in use.
        Have a look at the provided generators for further information.
    """

    def __init__(self, krige, generator="RandMeth", **generator_kwargs):
        if not isinstance(krige, Krige):
            raise ValueError("CondSRF: krige should be an instance of Krige.")
        self.krige = krige
        # initialize attributes
        self.pos = None
        self.mesh_type = None
        self.field = None
        self.raw_field = None
        # initialize private attributes
        self._value_type = None
        self._generator = None
        # initialize attributes
        self.set_generator(generator, **generator_kwargs)

    def __call__(self, pos, seed=np.nan, mesh_type="unstructured", **kwargs):
        """Generate the conditioned spatial random field.

        The field is saved as `self.field` and is also returned.

        Parameters
        ----------
        pos : :class:`list`
            the position tuple, containing main direction and transversal
            directions
        seed : :class:`int`, optional
            seed for RNG for reseting. Default: keep seed from generator
        mesh_type : :class:`str`
            'structured' / 'unstructured'
        **kwargs
            keyword arguments that are forwarded to the kriging routine in use.

        Returns
        -------
        field : :class:`numpy.ndarray`
            the conditioned SRF
        """
        kwargs["mesh_type"] = mesh_type
        kwargs["only_mean"] = False  # overwrite if given
        kwargs["return_var"] = True  # overwrite if given
        # update the model/seed in the generator if any changes were made
        self.generator.update(self.model, seed)
        # get isometrized positions and the resulting field-shape
        iso_pos, shape = self.pre_pos(pos, mesh_type)
        # generate the field
        self.raw_field = np.reshape(
            self.generator(iso_pos, add_nugget=False), shape
        )
        field, krige_var = self.krige(pos, **kwargs)
        var_scale, nugget = self.get_scaling(krige_var, shape)
        self.field = field + var_scale * self.raw_field + nugget
        return self.field

    def get_scaling(self, krige_var, shape):
        """
        Get scaling coefficients for the random field.

        Parameters
        ----------
        krige_var : :class:`numpy.ndarray`
            Kriging variance.
        shape : :class:`tuple` of :class:`int`
            Field shape.

        Returns
        -------
        var_scale : :class:`numpy.ndarray`
            Variance scaling factor for the random field.
        nugget : :class:`numpy.ndarray` or `class:`int
            Nugget to be added to the field.
        """
        if self.model.nugget > 0:
            var_scale = np.maximum(krige_var - self.model.nugget, 0)
            nug_scale = np.sqrt((krige_var - var_scale) / self.model.nugget)
            var_scale = np.sqrt(var_scale / self.model.var)
            nugget = nug_scale * self.generator.get_nugget(shape)
        else:
            var_scale = np.sqrt(krige_var / self.model.var)
            nugget = 0
        return var_scale, nugget

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
            raise ValueError("gstools.CondSRF: Unknown generator " + generator)
        if self.value_type != "scalar":
            raise ValueError("CondSRF: only scalar field value type allowed.")

    @property
    def generator(self):
        """:any:`callable`: The generator of the field."""
        return self._generator

    @property
    def model(self):
        """:any:`CovModel`: The covariance model of the field."""
        return self.krige.model

    @model.setter
    def model(self, model):
        pass

    def __repr__(self):
        """Return String representation."""
        return "CondSRF(krige={0}, generator={1})".format(
            self.krige, self.generator
        )
