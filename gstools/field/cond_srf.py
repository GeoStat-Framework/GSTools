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

    default_field_names = ["field", "raw_field", "raw_krige"]
    """:class:`list`: Default field names."""

    def __init__(self, krige, generator="RandMeth", **generator_kwargs):
        if not isinstance(krige, Krige):
            raise ValueError("CondSRF: krige should be an instance of Krige.")
        self._krige = krige
        # initialize attributes
        self._field_names = []
        # initialize private attributes
        self._generator = None
        # initialize attributes
        self.set_generator(generator, **generator_kwargs)

    def __call__(
        self,
        pos=None,
        seed=np.nan,
        mesh_type="unstructured",
        post_process=True,
        store=True,
        krige_store=True,
        **kwargs,
    ):
        """Generate the conditioned spatial random field.

        The field is saved as `self.field` and is also returned.

        Parameters
        ----------
        pos : :class:`list`, optional
            the position tuple, containing main direction and transversal
            directions
        seed : :class:`int`, optional
            seed for RNG for reseting. Default: keep seed from generator
        mesh_type : :class:`str`
            'structured' / 'unstructured'
        post_process : :class:`bool`, optional
            Whether to apply mean, normalizer and trend to the field.
            Default: `True`
        store : :class:`str` or :class:`bool` or :class:`list`, optional
            Whether to store fields (True/False) with default names
            or with specified names.
            The default is :any:`True` for default names
            ["field", "raw_field", "raw_krige"].
        krige_store : :class:`str` or :class:`bool` or :class:`list`, optional
            Whether to store kriging fields (True/False) with default name
            or with specified names.
            The default is :any:`True` for default names
            ["field", "krige_var"].
        **kwargs
            keyword arguments that are forwarded to the kriging routine in use.

        Returns
        -------
        field : :class:`numpy.ndarray`
            the conditioned SRF
        """
        name, save = self.get_store_config(store=store, fld_cnt=3)
        krige_name, krige_save = self.krige.get_store_config(
            store=krige_store, fld_cnt=2
        )
        kwargs["mesh_type"] = mesh_type
        kwargs["only_mean"] = False  # overwrite if given
        kwargs["return_var"] = True  # overwrite if given
        kwargs["post_process"] = False  # overwrite if given
        kwargs["store"] = [False, krige_name[1] if krige_save[1] else False]
        # update the model/seed in the generator if any changes were made
        self.generator.update(self.model, seed)
        # get isometrized positions and the resulting field-shape
        iso_pos, shape, info = self.pre_pos(pos, mesh_type, info=True)
        # generate the field
        rawfield = np.reshape(self.generator(iso_pos, add_nugget=False), shape)
        # call krige on already set pos (reuse already calculated fields)
        if (
            not info["deleted"]
            and name[2] in self.field_names
            and krige_name[1] in self.krige.field_names
        ):
            reuse = True
            rawkrige, krige_var = self[name[2]], self.krige[krige_name[1]]
        else:
            reuse = False
            rawkrige, krige_var = self.krige(**kwargs)
        var_scale, nugget = self.get_scaling(krige_var, shape)
        # store krige field (need a copy to not alter field by reference)
        if not reuse or krige_name[0] not in self.krige.field_names:
            self.krige.post_field(
                rawkrige.copy(), krige_name[0], post_process, krige_save[0]
            )
        # store raw krige field
        if not reuse:
            self.post_field(rawkrige, name[2], False, save[2])
        # store raw random field
        self.post_field(rawfield, name[1], False, save[1])
        # store cond random field
        return self.post_field(
            field=rawkrige + var_scale * rawfield + nugget,
            name=name[0],
            process=post_process,
            save=save[0],
        )

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

    def set_pos(self, pos, mesh_type="unstructured", info=False):
        """
        Set positions and mesh_type.

        Parameters
        ----------
        pos : :any:`iterable`
            the position tuple, containing main direction and transversal
            directions
        mesh_type : :class:`str`, optional
            'structured' / 'unstructured'
            Default: `"unstructured"`
        info : :class:`bool`, optional
            Whether to return information

        Returns
        -------
        info : :class:`dict`, optional
            Information about settings.

        Warnings
        --------
        When setting a new position tuple that differs from the present one,
        all stored fields will be deleted.
        """
        info_ret = super().set_pos(pos, mesh_type, info=True)
        if info_ret["deleted"]:
            self.krige.delete_fields()
        return info_ret if info else None

    @property
    def pos(self):
        """:class:`tuple`: The position tuple of the field."""
        return self.krige.pos

    @pos.setter
    def pos(self, pos):
        self.krige.pos = pos

    @property
    def field_shape(self):
        """:class:`tuple`: The shape of the field."""
        return self.krige.field_shape

    @property
    def mesh_type(self):
        """:class:`str`: The mesh type of the field."""
        return self.krige.mesh_type

    @mesh_type.setter
    def mesh_type(self, mesh_type):
        self.krige.mesh_type = mesh_type

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
        return "{0}(krige={1}, generator={2})".format(
            self.name, self.krige, self.generator.name
        )
