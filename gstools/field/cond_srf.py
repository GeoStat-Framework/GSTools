# -*- coding: utf-8 -*-
"""
GStools subpackage providing a class for conditioned spatial random fields.

.. currentmodule:: gstools.field.cond_srf

The following classes are provided

.. autosummary::
   CondSRF
"""
# pylint: disable=C0103, W0231, W0221, W0222, E1102
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
    krige : :any:`Krige`
        Kriging setup to condition the spatial random field.
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
        self._krige = krige
        # initialize attributes
        self.pos = None
        self.mesh_type = None
        self.field = None
        self.raw_field = None
        # initialize private attributes
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
        kwargs["post_process"] = False  # overwrite if given
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
        # need to use a copy to not alter "field" by reference
        self.krige.post_field(self.krige.field.copy())
        return self.post_field(field + var_scale * self.raw_field + nugget)

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
        nugget : :class:`numpy.ndarray` or :class:`int`
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
            self.value_type = self.generator.value_type
        else:
            raise ValueError(f"gstools.CondSRF: Unknown generator {generator}")

    @property
    def krige(self):
        """:any:`Krige`: The underlying kriging class."""
        return self._krige

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
        self.krige.model = model

    @property
    def mean(self):
        """:class:`float` or :any:`callable`: The mean of the field."""
        return self.krige.mean

    @mean.setter
    def mean(self, mean):
        self.krige.mean = mean

    @property
    def normalizer(self):
        """:any:`Normalizer`: Normalizer of the field."""
        return self.krige.normalizer

    @normalizer.setter
    def normalizer(self, normalizer):
        self.krige.normalizer = normalizer

    @property
    def trend(self):
        """:class:`float` or :any:`callable`: The trend of the field."""
        return self.krige.trend

    @trend.setter
    def trend(self, trend):
        self.krige.trend = trend

    @property
    def value_type(self):
        """:class:`str`: Type of the field values (scalar, vector)."""
        return self.krige.value_type

    @value_type.setter
    def value_type(self, value_type):
        self.krige.value_type = value_type

    def __repr__(self):
        """Return String representation."""
        return "CondSRF(krige={0}, generator={1})".format(
            self.krige, self.generator.name
        )
