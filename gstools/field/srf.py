# -*- coding: utf-8 -*-
"""
GStools subpackage providing a class for standard spatial random fields.

.. currentmodule:: gstools.field.srf

The following classes are provided

.. autosummary::
   SRF
"""
from __future__ import division, absolute_import, print_function

import numpy as np
from gstools.covmodel.base import CovModel
from gstools.field.generator import RandMeth
from gstools.field.tools import (
    reshape_input,
    check_mesh,
    make_isotropic,
    unrotate_mesh,
    reshape_axis_from_struct_to_unstruct,
    reshape_field_from_unstruct_to_struct,
)

__all__ = ["SRF"]

GENERATOR = {"RandMeth": RandMeth}


class SRF(object):
    """A class to generate a spatial random field (SRF).
    """

    def __init__(
        self, model, mean=0.0, generator="RandMeth", **generator_kwargs
    ):
        """Initialize a spatial random field

        Parameters
        ----------
            model : :any:`CovModel`
                Covariance Model to use for the field.
            mean : :class:`float`, optional
                mean value of the SRF
            generator : :class:`str`, optional
                Name of the generator to use for field generation.
                Default: "RandMeth"
        """
        self._mean = mean
        if isinstance(model, CovModel):
            self._model = model
        else:
            raise ValueError(
                "gstools.SRF: 'model' is not an instance of 'gstools.CovModel'"
            )
        self._do_rotation = not np.all(np.isclose(self._model.angles, 0.0))
        if generator in GENERATOR:
            gen = GENERATOR[generator]
            self._generator = gen(self._model, **generator_kwargs)
        else:
            raise ValueError("gstools.SRF: Unknown generator: " + generator)
        # initialize attributes
        self.field = None

    def __call__(
        self,
        x,
        y=None,
        z=None,
        seed=np.nan,
        mesh_type="unstructured",
        force_moments=False,
        point_volumes=0.0,
    ):
        """Generate an SRF and return it without saving it internally.

        Parameters
        ----------
            x : :class:`numpy.ndarray`
                grid axis in x-direction if structured, or first components of
                position vectors if unstructured
            y : :class:`numpy.ndarray`, optional
                analog to x
            z : :class:`numpy.ndarray`, optional
                analog to x
            seed : :class:`int`, optional
                seed for RNG for reseting. Default: keep seed from generator
            mesh_type : :class:`str`
                'structured' / 'unstructured'
            force_moments : :class:`bool`
                Force the generator to exactly match mean and variance.
                Default: False
        Returns
        -------
            field : :class:`numpy.ndarray`
                the SRF
        """
        # reset seed if wanted
        if seed is None or not np.isnan(seed):
            self.generator.seed = seed

        # format the positional arguments of the mesh
        check_mesh(self.dim, x, y, z, mesh_type)
        mesh_type_changed = False
        if self._do_rotation:
            if mesh_type == "structured":
                mesh_type_changed = True
                mesh_type_old = mesh_type
                mesh_type = "unstructured"
                x, y, z, axis_lens = reshape_axis_from_struct_to_unstruct(
                    self.dim, x, y, z
                )
            x, y, z = unrotate_mesh(self.dim, self.model.angles, x, y, z)
        y, z = make_isotropic(self.dim, self.model.anis, y, z)
        x, y, z = reshape_input(x, y, z, mesh_type)

        # generate the field
        field = self.generator(x, y, z)

        # reshape field if we got an unstructured mesh
        if mesh_type_changed:
            mesh_type = mesh_type_old
            field = reshape_field_from_unstruct_to_struct(
                self.dim, field, axis_lens
            )

        # force variance and mean to be exactly as given (if wanted)
        if force_moments:
            var_in = np.var(field)
            mean_in = np.mean(field)
            rescale = np.sqrt((self.model.var + self.model.nugget) / var_in)
            field = rescale * (field - mean_in)

        # interprete volume as a hypercube and calculate the edge length
        edge = point_volumes ** (1.0 / self.dim)

        # coarse-grained variance-factor
        var_factor = (
            self.model.len_scale ** 2
            / (self.model.len_scale ** 2 + edge ** 2 / 4)
        ) ** (self.dim / 2.0)

        # shift the field to the mean
        self.field = np.sqrt(var_factor) * field + self.mean

        return self.field

    def structured(self, x, y=None, z=None, seed=None):
        """Generate an SRF on a structured mesh

        See SRF.__call__
        """
        return self(x, y, z, seed, "structured")

    def unstructured(self, x, y=None, z=None, seed=None):
        """Generate an SRF on an unstructured mesh

        See SRF.__call__
        """
        return self(x, y, z, seed)

    def generate(self, **kwargs):
        """Generate an SRF and save it as an attribute self.field.

        See SRF.__call__
        """
        return self(**kwargs)

    @property
    def model(self):
        """ The covariance model of the spatial random field."""
        return self._model

    @property
    def generator(self):
        """ The generator-class of the spatial random field."""
        return self._generator

    @property
    def dim(self):
        """ The dimension of the spatial random field."""
        return self._model.dim

    @property
    def mean(self):
        """ The mean of the spatial random field."""
        return self._mean

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "SRF(model={0}, mean={1}, generator={2}".format(
            self.model, self.mean, self.generator
        )


if __name__ == "__main__":
    import doctest

    doctest.testmod()
