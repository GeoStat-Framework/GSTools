# -*- coding: utf-8 -*-
"""
GStools subpackage providing a class for standard spatial random fields.

.. currentmodule:: gstools.field.srf

The following classes are provided

.. autosummary::
   SRF
"""
# pylint: disable=C0103
from __future__ import division, absolute_import, print_function

from functools import partial

import numpy as np
from gstools.covmodel.base import CovModel
from gstools.field.generator import RandMeth, IncomprRandMeth
from gstools.field.tools import (
    check_mesh,
    make_isotropic,
    unrotate_mesh,
    reshape_axis_from_struct_to_unstruct,
    reshape_field_from_unstruct_to_struct,
)
from gstools.tools.geometric import pos2xyz
from gstools.field.upscaling import var_coarse_graining, var_no_scaling
from gstools.field.condition import condition_ok

__all__ = ["SRF"]

GENERATOR = {
    "RandMeth": RandMeth,
    "IncomprRandMeth": IncomprRandMeth,
    "VectorField": IncomprRandMeth,
    "VelocityField": IncomprRandMeth,
}
UPSCALING = {
    "coarse_graining": var_coarse_graining,
    "no_scaling": var_no_scaling,
}
CONDITION = {"ordinary": condition_ok}


class SRF(object):
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
        # initialize private attributes
        self._model = None
        self._generator = None
        self._upscaling = None
        self._upscaling_func = None
        # condition related
        self._cond_pos = None
        self._cond_val = None
        self._krige_type = None
        # initialize attributes
        self.field = None
        self.raw_field = None
        self.krige_field = None
        self.err_field = None
        self.krige_var = None
        self.mean = mean
        self.model = model
        self.set_generator(generator, **generator_kwargs)
        self.upscaling = upscaling

    def __call__(
        self,
        pos,
        seed=np.nan,
        force_moments=False,
        point_volumes=0.0,
        mesh_type="unstructured",
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
        force_moments : :class:`bool`
            Force the generator to exactly match the given mean and variance.
            Default: ``False``
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
        # internal conversation
        x, y, z = pos2xyz(pos)
        # update the model/seed in the generator if any changes were made
        self.generator.update(self.model, seed)
        # format the positional arguments of the mesh
        check_mesh(self.model.dim, x, y, z, mesh_type)
        mesh_type_changed = False
        if self.do_rotation:
            if mesh_type == "structured":
                mesh_type_changed = True
                mesh_type_old = mesh_type
                mesh_type = "unstructured"
                x, y, z, axis_lens = reshape_axis_from_struct_to_unstruct(
                    self.model.dim, x, y, z
                )
            x, y, z = unrotate_mesh(self.model.dim, self.model.angles, x, y, z)
        y, z = make_isotropic(self.model.dim, self.model.anis, y, z)

        # generate the field
        field = self.generator.__call__(self.model.dim, x, y, z, mesh_type)

        # reshape field if we got an unstructured mesh
        if mesh_type_changed:
            mesh_type = mesh_type_old
            field = reshape_field_from_unstruct_to_struct(
                self.model.dim, field, axis_lens
            )

        # force variance and mean to be exactly as given (if wanted)
        if force_moments:
            var_in = np.var(field)
            mean_in = np.mean(field)
            rescale = np.sqrt(self.model.sill / var_in)
            field = rescale * (field - mean_in)

        # upscaled variance
        scaled_var = self.upscaling_func(self.model, point_volumes)

        # rescale and shift the field to the mean
        self.raw_field = np.sqrt(scaled_var / self.model.sill) * field

        # apply given conditions to the field
        if self.condition:
            cond_field, krige_field, err_field, krige_var = self.cond_func(
                pos,
                self.raw_field,
                self._cond_pos,
                self._cond_val,
                self.model,
                mesh_type=mesh_type,
            )
            # store everything in the class
            self.field = cond_field
            self.krige_field = krige_field
            self.err_field = err_field
            self.krige_var = krige_var
        else:
            self.field = self.raw_field + self.mean

        return self.field

    def set_condition(
        self, cond_pos=None, cond_val=None, krige_type="ordinary"
    ):
        """Condition a given spatial random field with measurements.

        Parameters
        ----------
        cond_pos : :class:`list`
            the position tuple of the conditions
        cond_pos : :class:`numpy.ndarray`
            the values of the conditions
        krige_type : :class:`str`, optional
            Used kriging type for conditioning.
            Only "ordinary" provided at the moment.
            Default: 'ordinary'
        """
        self._cond_pos = cond_pos
        self._cond_val = cond_val
        self._krige_type = krige_type
        if krige_type not in CONDITION:
            raise ValueError(
                "gstools.SRF: Unknown kriging method: " + krige_type
            )

    def del_condition(self):
        """Delete Conditions."""
        self._cond_pos = None
        self._cond_val = None
        self._krige_type = None

    @property
    def condition(self):
        """:any:`bool`: State if conditions ar given"""
        return self._cond_pos is not None

    def cond_func(self, *args, **kwargs):
        """The conditioning method applied to the field"""
        if self.condition:
            return CONDITION[self._krige_type](*args, **kwargs)

    def structured(self, *args, **kwargs):
        """Generate an SRF on a structured mesh

        See :any:`SRF.__call__`
        """
        call = partial(self.__call__, mesh_type="structured")
        return call(*args, **kwargs)

    def unstructured(self, *args, **kwargs):
        """Generate an SRF on an unstructured mesh

        See :any:`SRF.__call__`
        """
        call = partial(self.__call__, mesh_type="unstructured")
        return call(*args, **kwargs)

    def upscaling_func(self, *args, **kwargs):
        """The upscaling method applied to the field variance"""
        return self._upscaling_func(*args, **kwargs)

    def set_generator(self, generator, **generator_kwargs):
        """Set the generator for the field

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
        """:class:`str`: Name of the upscaling method for the variance at each
        point depending on the related element volume.

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
    def model(self):
        """:any:`CovModel`: The covariance model of the spatial random field.
        """
        return self._model

    @model.setter
    def model(self, model):
        if isinstance(model, CovModel):
            self._model = model
        else:
            raise ValueError(
                "gstools.SRF: 'model' is not an instance of 'gstools.CovModel'"
            )

    @property
    def do_rotation(self):
        """:any:`bool`: State if a rotation should be performed
        depending on the model.
        """
        return not np.all(np.isclose(self.model.angles, 0.0))

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "SRF(model={0}, mean={1}, generator={2}".format(
            self.model, self.mean, self.generator
        )


if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod()
