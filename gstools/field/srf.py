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
from gstools.tools.geometric import pos2xyz, xyz2pos
from gstools.tools.export import vtk_export as vtk_ex
from gstools.field.upscaling import var_coarse_graining, var_no_scaling
from gstools.field.condition import ordinary, simple

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
CONDITION = {"ordinary": ordinary, "simple": simple}


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
        self.pos = None
        self.mesh_type = None
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
        self.pos = xyz2pos(x, y, z)
        self.mesh_type = mesh_type
        # update the model/seed in the generator if any changes were made
        self.generator.update(self.model, seed)
        # format the positional arguments of the mesh
        check_mesh(self.model.dim, x, y, z, mesh_type)
        mesh_type_changed = False
        if self.model.do_rotation:
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
        self.raw_field = self.generator.__call__(x, y, z, mesh_type)

        # reshape field if we got an unstructured mesh
        if mesh_type_changed:
            mesh_type = mesh_type_old
            self.raw_field = reshape_field_from_unstruct_to_struct(
                self.model.dim, self.raw_field, axis_lens
            )

        # apply given conditions to the field
        if self.condition:
            cond_field, krige_field, err_field, krigevar = self.cond_func(self)
            # store everything in the class
            self.field = cond_field
            self.krige_field = krige_field
            self.err_field = err_field
            self.krige_var = krigevar
        else:
            self.field = self.raw_field + self.mean

        # upscaled variance
        if not np.isscalar(point_volumes) or not np.isclose(point_volumes, 0):
            scaled_var = self.upscaling_func(self.model, point_volumes)
            if self.condition and self._krige_type != "simple":
                mean = self.field.mean()
            else:
                mean = self.mean
            self.field -= mean
            self.field *= np.sqrt(scaled_var / self.model.sill)
            self.field += mean

        return self.field

    def vtk_export(self, filename, fieldname="field"):
        """Export the stored field to vtk.

        Parameters
        ----------
        filename : :class:`str`
            Filename of the file to be saved, including the path. Note that an
            ending (.vtr or .vtu) will be added to the name.
        fieldname : :class:`str`, optional
            Name of the field in the VTK file. Default: "field"
        """
        if not (
            self.pos is None or self.field is None or self.mesh_type is None
        ):
            vtk_ex(filename, self.pos, self.field, fieldname, self.mesh_type)
        else:
            print("gstools.SRF.vtk_export: No field stored in the srf class.")

    def plot(self, fig=None, ax=None):
        """
        Plot the spatial random field.

        Parameters
        ----------
        fig : :any:`Figure` or :any:`None`
            Figure to plot the axes on. If `None`, a new one will be created.
            Default: `None`
        ax : :any:`Axes` or :any:`None`
            Axes to plot on. If `None`, a new one will be added to the figure.
            Default: `None`
        """
        # just import if needed; matplotlib is not required by setup
        from gstools.field.plot import plot_srf
        plot_srf(self, fig, ax)

    def set_condition(
        self, cond_pos=None, cond_val=None, krige_type="ordinary"
    ):
        """Condition a given spatial random field with measurements.

        Parameters
        ----------
        cond_pos : :class:`list`
            the position tuple of the conditions
        cond_val : :class:`numpy.ndarray`
            the values of the conditions
        krige_type : :class:`str`, optional
            Used kriging type for conditioning.
            Either 'ordinary' or 'simple'.
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
    def cond_pos(self):
        """:class:`list`: The position tuple of the conditions."""
        return self._cond_pos

    @property
    def cond_val(self):
        """:class:`list`: The values of the conditions."""
        return self._cond_val

    @property
    def condition(self):
        """:any:`bool`: State if conditions ar given."""
        return self._cond_pos is not None

    def cond_func(self, *args, **kwargs):
        """Conditioning method applied to the field."""
        if self.condition:
            return CONDITION[self._krige_type](*args, **kwargs)
        return None

    def structured(self, *args, **kwargs):
        """Generate an SRF on a structured mesh.

        See :any:`SRF.__call__`
        """
        call = partial(self.__call__, mesh_type="structured")
        return call(*args, **kwargs)

    def unstructured(self, *args, **kwargs):
        """Generate an SRF on an unstructured mesh.

        See :any:`SRF.__call__`
        """
        call = partial(self.__call__, mesh_type="unstructured")
        return call(*args, **kwargs)

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
    def model(self):
        """:any:`CovModel`: The covariance model of the random field."""
        return self._model

    @model.setter
    def model(self, model):
        if isinstance(model, CovModel):
            self._model = model
        else:
            raise ValueError(
                "gstools.SRF: 'model' is not an instance of 'gstools.CovModel'"
            )

    def __str__(self):
        """Return String representation."""
        return self.__repr__()

    def __repr__(self):
        """Return String representation."""
        return "SRF(model={0}, mean={1}, generator={2}".format(
            self.model, self.mean, self.generator
        )


if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod()
