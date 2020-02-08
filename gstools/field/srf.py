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
from gstools.field.tools import reshape_field_from_unstruct_to_struct
from gstools.field.base import Field
from gstools.field.upscaling import var_coarse_graining, var_no_scaling
from gstools.field.condition import ordinary, simple
from gstools.krige.tools import set_condition

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
        super().__init__(model, mean)
        # initialize private attributes
        self._generator = None
        self._upscaling = None
        self._upscaling_func = None
        # condition related
        self._cond_pos = None
        self._cond_val = None
        self._krige_type = None
        # initialize attributes
        self.raw_field = None
        self.krige_field = None
        self.err_field = None
        self.krige_var = None
        self.set_generator(generator, **generator_kwargs)
        self.upscaling = upscaling
        if self._value_type is None:
            raise ValueError(
                "Unknown field value type, "
                + "specify 'scalar' or 'vector' before calling SRF."
            )

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
        self.mesh_type = mesh_type
        # update the model/seed in the generator if any changes were made
        self.generator.update(self.model, seed)
        # internal conversation
        x, y, z, self.pos, mt_gen, mt_changed, axis_lens = self._pre_pos(
            pos, mesh_type
        )
        # generate the field
        self.raw_field = self.generator.__call__(x, y, z, mt_gen)
        # reshape field if we got an unstructured mesh
        if mt_changed:
            self.raw_field = reshape_field_from_unstruct_to_struct(
                self.model.dim, self.raw_field, axis_lens
            )
        # apply given conditions to the field
        if self.condition:
            (
                cond_field,
                krige_field,
                err_field,
                krigevar,
                info,
            ) = self.cond_func(self)
            # store everything in the class
            self.field = cond_field
            self.krige_field = krige_field
            self.err_field = err_field
            self.krige_var = krigevar
            if "mean" in info:  # ordinary krging estimates mean
                self.mean = info["mean"]
        else:
            self.field = self.raw_field + self.mean
        # upscaled variance
        if not np.isscalar(point_volumes) or not np.isclose(point_volumes, 0):
            scaled_var = self.upscaling_func(self.model, point_volumes)
            self.field -= self.mean
            self.field *= np.sqrt(scaled_var / self.model.sill)
            self.field += self.mean
        return self.field

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

        Notes
        -----
        When using "ordinary" as ``krige_type``, the ``mean`` attribute of the
        spatial random field will be overwritten with the estimated mean.
        """
        if cond_pos is not None:
            self._cond_pos, self._cond_val = set_condition(
                cond_pos, cond_val, self.model.dim
            )
        else:
            self._cond_pos = self._cond_val = None
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
            self._value_type = self._generator.value_type
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

    def __repr__(self):
        """Return String representation."""
        return "SRF(model={0}, mean={1}, generator={2}".format(
            self.model, self.mean, self.generator
        )


if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod()
