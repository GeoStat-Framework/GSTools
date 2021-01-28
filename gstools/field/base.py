# -*- coding: utf-8 -*-
"""
GStools subpackage providing a base class for spatial fields.

.. currentmodule:: gstools.field.base

The following classes are provided

.. autosummary::
   Field
"""
# pylint: disable=C0103
from functools import partial
import numpy as np
from gstools.covmodel.base import CovModel
from gstools.tools.geometric import format_struct_pos_dim, gen_mesh
from gstools.tools.misc import eval_func
from gstools.normalize import Normalizer
from gstools.field.tools import mesh_call, to_vtk_helper

__all__ = ["Field"]

VALUE_TYPES = ["scalar", "vector"]
""":class:`list` of :class:`str`: valid field value types."""


class Field:
    """A base class for random fields, kriging fields, etc.

    Parameters
    ----------
    model : :any:`CovModel`
        Covariance Model related to the field.
    value_type : :class:`str`, optional
        Value type of the field. Either "scalar" or "vector".
        The default is "scalar".
    mean : :any:`None` or :class:`float` or :any:`callable`, optional
        Mean of the field if wanted. Could also be a callable.
        The default is None.
    normalizer : :any:`None` or :any:`Normalizer`, optional
        Normalizer to be applied to the field.
        The default is None.
    trend : :any:`None` or :class:`float` or :any:`callable`, optional
        Trend of the denormalized fields. If no normalizer is applied,
        this behaves equal to 'mean'.
        The default is None.
    """

    def __init__(
        self,
        model,
        value_type="scalar",
        mean=None,
        normalizer=None,
        trend=None,
    ):
        # initialize attributes
        self.pos = None
        self.mesh_type = None
        # initialize private attributes
        self._model = None
        self._value_type = None
        self._mean = None
        self._normalizer = None
        self._trend = None
        # set properties
        self.model = model
        self.value_type = value_type
        self.mean = mean
        self.normalizer = normalizer
        self.trend = trend

    def __call__(*args, **kwargs):
        """Generate the field."""
        pass

    def structured(self, *args, **kwargs):
        """Generate a field on a structured mesh.

        See :any:`Field.__call__`
        """
        call = partial(self.__call__, mesh_type="structured")
        return call(*args, **kwargs)

    def unstructured(self, *args, **kwargs):
        """Generate a field on an unstructured mesh.

        See :any:`Field.__call__`
        """
        call = partial(self.__call__, mesh_type="unstructured")
        return call(*args, **kwargs)

    def mesh(
        self, mesh, points="centroids", direction="all", name="field", **kwargs
    ):
        """Generate a field on a given meshio or ogs5py mesh.

        Parameters
        ----------
        mesh : meshio.Mesh or ogs5py.MSH or PyVista mesh
            The given meshio, ogs5py, or PyVista mesh
        points : :class:`str`, optional
            The points to evaluate the field at.
            Either the "centroids" of the mesh cells
            (calculated as mean of the cell vertices) or the "points"
            of the given mesh.
            Default: "centroids"
        direction : :class:`str` or :class:`list`, optional
            Here you can state which direction should be choosen for
            lower dimension. For example, if you got a 2D mesh in xz direction,
            you have to pass "xz". By default, all directions are used.
            One can also pass a list of indices.
            Default: "all"
        name : :class:`str` or :class:`list` of :class:`str`, optional
            Name(s) to store the field(s) in the given mesh as point_data or
            cell_data. If to few names are given, digits will be appended.
            Default: "field"
        **kwargs
            Keyword arguments forwareded to `Field.__call__`.

        Notes
        -----
        This will store the field in the given mesh under the given name,
        if a meshio or PyVista mesh was given.

        See: https://github.com/nschloe/meshio
        See: https://github.com/pyvista/pyvista

        See: :any:`Field.__call__`
        """
        return mesh_call(self, mesh, points, direction, name, **kwargs)

    def pre_pos(self, pos, mesh_type="unstructured"):
        """
        Preprocessing positions and mesh_type.

        Parameters
        ----------
        pos : :any:`iterable`
            the position tuple, containing main direction and transversal
            directions
        mesh_type : :class:`str`, optional
            'structured' / 'unstructured'
            Default: `"unstructured"`

        Returns
        -------
        iso_pos : (d, n), :class:`numpy.ndarray`
            the isometrized position tuple
        shape : :class:`tuple`
            Shape of the resulting field.
        """
        # save mesh-type
        self.mesh_type = mesh_type
        dim = self.model.field_dim
        # save pos tuple
        if mesh_type != "unstructured":
            pos, shape = format_struct_pos_dim(pos, dim)
            self.pos = pos
            pos = gen_mesh(pos)
        else:
            pos = np.array(pos, dtype=np.double).reshape(dim, -1)
            self.pos = pos
            shape = np.shape(pos[0])
        # prepend dimension if we have a vector field
        if self.value_type == "vector":
            shape = (self.model.dim,) + shape
            if self.model.latlon:
                raise ValueError("Field: Vector fields not allowed for latlon")
        # return isometrized pos tuple and resulting field shape
        return self.model.isometrize(pos), shape

    def post_field(self, field, name="field", process=True, save=True):
        """
        Postprocessing field values.

        Parameters
        ----------
        field : :class:`numpy.ndarray`
            Field values.
        name : :class:`str`, optional
            Name. to store the field.
            The default is "field".
        process : :class:`bool`, optional
            Whether to process field to apply mean, normalizer and trend.
            The default is True.
        save : :class:`bool`, optional
            Whether to store the field under the given name.
            The default is True.

        Returns
        -------
        field : :class:`numpy.ndarray`
            Processed field values.
        """
        if process:
            if self.pos is None:
                raise ValueError("post_field: no 'pos' tuple set for field.")
            kwargs = dict(
                pos=self.pos,
                dim=self.model.dim,
                mesh_type=self.mesh_type,
                value_type=self.value_type,
                broadcast=True,
            )
            # apply mean - normalizer - trend
            field += eval_func(func_val=self.mean, **kwargs)
            field = self.normalizer.denormalize(field)
            field += eval_func(self.trend, **kwargs)
        if save:
            setattr(self, name, field)
        return field

    def to_pyvista(
        self, field_select="field", fieldname="field"
    ):  # pragma: no cover
        """Create a VTK/PyVista grid of the stored field.

        Parameters
        ----------
        field_select : :class:`str`, optional
            Field that should be stored. Can be:
            "field", "raw_field", "krige_field", "err_field" or "krige_var".
            Default: "field"
        fieldname : :class:`str`, optional
            Name of the field in the VTK file. Default: "field"
        """
        grid = to_vtk_helper(
            self, filename=None, field_select=field_select, fieldname=fieldname
        )
        return grid

    def vtk_export(
        self, filename, field_select="field", fieldname="field"
    ):  # pragma: no cover
        """Export the stored field to vtk.

        Parameters
        ----------
        filename : :class:`str`
            Filename of the file to be saved, including the path. Note that an
            ending (.vtr or .vtu) will be added to the name.
        field_select : :class:`str`, optional
            Field that should be stored. Can be:
            "field", "raw_field", "krige_field", "err_field" or "krige_var".
            Default: "field"
        fieldname : :class:`str`, optional
            Name of the field in the VTK file. Default: "field"
        """
        if not isinstance(filename, str):
            raise TypeError("Please use a string filename.")
        return to_vtk_helper(
            self,
            filename=filename,
            field_select=field_select,
            fieldname=fieldname,
        )

    def plot(self, field="field", fig=None, ax=None):  # pragma: no cover
        """
        Plot the spatial random field.

        Parameters
        ----------
        field : :class:`str`, optional
            Field that should be plotted.
            Default: "field"
        fig : :class:`Figure` or :any:`None`
            Figure to plot the axes on. If `None`, a new one will be created.
            Default: `None`
        ax : :class:`Axes` or :any:`None`
            Axes to plot on. If `None`, a new one will be added to the figure.
            Default: `None`
        """
        # just import if needed; matplotlib is not required by setup
        from gstools.field.plot import plot_field, plot_vec_field

        if self.value_type is None:
            raise ValueError(
                "Field value type not set! "
                "Specify 'scalar' or 'vector' before plotting."
            )

        elif self.value_type == "scalar":
            r = plot_field(self, field, fig, ax)

        elif self.value_type == "vector":
            if self.model.dim == 2:
                r = plot_vec_field(self, field, fig, ax)
            else:
                raise NotImplementedError(
                    "Streamflow plotting only supported for 2d case."
                )
        else:
            raise ValueError(
                "Unknown field value type: {}".format(self.value_type)
            )

        return r

    @property
    def model(self):
        """:any:`CovModel`: The covariance model of the field."""
        return self._model

    @model.setter
    def model(self, model):
        if isinstance(model, CovModel):
            self._model = model
        else:
            raise ValueError(
                "Field: 'model' is not an instance of 'gstools.CovModel'"
            )

    @property
    def mean(self):
        """:class:`float` or :any:`callable`: The mean of the field."""
        return self._mean

    @mean.setter
    def mean(self, mean):
        self._mean = mean if (callable(mean) or mean is None) else float(mean)

    @property
    def normalizer(self):
        """:any:`Normalizer`: Normalizer of the field."""
        return self._normalizer

    @normalizer.setter
    def normalizer(self, normalizer):
        if isinstance(normalizer, Normalizer):
            self._normalizer = normalizer
        elif normalizer is None:
            self._normalizer = Normalizer()
        else:
            raise ValueError("Field: 'normalizer' not of type 'Normalizer'.")

    @property
    def trend(self):
        """:class:`float` or :any:`callable`: The trend of the field."""
        return self._trend

    @trend.setter
    def trend(self, tren):
        self._trend = tren if (callable(tren) or tren is None) else float(tren)

    @property
    def value_type(self):
        """:class:`str`: Type of the field values (scalar, vector)."""
        return self._value_type

    @value_type.setter
    def value_type(self, value_type):
        """:class:`str`: Type of the field values (scalar, vector)."""
        if value_type not in VALUE_TYPES:
            raise ValueError("Field: value type not in {}".format(VALUE_TYPES))
        self._value_type = value_type

    def _fmt_func_val(self, func_val):
        if func_val is None:
            return str(None)
        if callable(func_val):
            return "<function>"
        return "{0:.{p}}".format(float(func_val), p=self.model._prec)

    def _fmt_normalizer(self):
        norm = self.normalizer
        return str(None) if norm.__class__ is Normalizer else norm.name

    @property
    def name(self):
        """:class:`str`: The name of the class."""
        return self.__class__.__name__

    def __repr__(self):
        """Return String representation."""
        return (
            "{0}(model={1}, value_type='{2}', "
            "mean={3}, normalizer={4}, trend={5})".format(
                self.name,
                self.model.name,
                self.value_type,
                self._fmt_func_val(self.mean),
                self._fmt_normalizer(),
                self._fmt_func_val(self.trend),
            )
        )
