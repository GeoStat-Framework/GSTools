# -*- coding: utf-8 -*-
"""
GStools subpackage providing a base class for spatial fields.

.. currentmodule:: gstools.field.base

The following classes are provided

.. autosummary::
   Field
"""
# pylint: disable=C0103, C0415
from functools import partial
import numpy as np
from gstools.covmodel.base import CovModel
from gstools.tools.geometric import format_struct_pos_dim, generate_grid
from gstools.field.tools import (
    generate_on_mesh,
    to_vtk_helper,
    fmt_mean_norm_trend,
)
from gstools.normalizer.tools import apply_mean_norm_trend, _check_normalizer

__all__ = ["Field"]

VALUE_TYPES = ["scalar", "vector"]
""":class:`list` of :class:`str`: valid field value types."""


def _set_mean_trend(value, dim):
    if callable(value) or value is None:
        return value
    value = np.array(value, dtype=np.double).ravel()
    if value.size > 1 and value.size != dim:  # vector mean
        raise ValueError(f"Mean/Trend: Wrong size ({value})")
    return value if value.size > 1 else value.item()


class Field:
    """A base class for random fields, kriging fields, etc.

    Parameters
    ----------
    model : :any:`CovModel`, optional
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
    dim : :any:`None` or :class:`int`, optional
        Dimension of the field if no model is given.
    """

    def __init__(
        self,
        model=None,
        value_type="scalar",
        mean=None,
        normalizer=None,
        trend=None,
        dim=None,
    ):
        # initialize attributes
        self.pos = None
        self.mesh_type = None
        self.field = None
        # initialize private attributes
        self._model = None
        self._value_type = None
        self._mean = None
        self._normalizer = None
        self._trend = None
        self._dim = dim if dim is None else int(dim)
        # set properties
        self.model = model
        self.value_type = value_type
        self.mean = mean
        self.normalizer = normalizer
        self.trend = trend

    def __call__(
        self, pos, field=None, mesh_type="unstructured", post_process=True
    ):
        """Generate the field.

        Parameters
        ----------
        pos : :class:`list`
            the position tuple, containing main direction and transversal
            directions
        field : :class:`numpy.ndarray` or :any:`None`, optional
            the field values. Will be all zeros if :any:`None` is given.
        mesh_type : :class:`str`, optional
            'structured' / 'unstructured'. Default: 'unstructured'
        post_process : :class:`bool`, optional
            Whether to apply mean, normalizer and trend to the field.
            Default: `True`

        Returns
        -------
        field : :class:`numpy.ndarray`
            the field values.
        """
        pos, shape = self.pre_pos(pos, mesh_type)
        if field is None:
            field = np.zeros(shape, dtype=np.double)
        else:
            field = np.array(field, dtype=np.double).reshape(shape)
        return self.post_field(field, process=post_process)

    def structured(self, *args, **kwargs):
        """Generate a field on a structured mesh.

        See :any:`__call__`
        """
        call = partial(self.__call__, mesh_type="structured")
        return call(*args, **kwargs)

    def unstructured(self, *args, **kwargs):
        """Generate a field on an unstructured mesh.

        See :any:`__call__`
        """
        call = partial(self.__call__, mesh_type="unstructured")
        return call(*args, **kwargs)

    def mesh(
        self, mesh, points="centroids", direction="all", name="field", **kwargs
    ):
        """Generate a field on a given meshio, ogs5py or PyVista mesh.

        Parameters
        ----------
        mesh : meshio.Mesh or ogs5py.MSH or PyVista mesh
            The given mesh
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
            Keyword arguments forwarded to :any:`__call__`.

        Notes
        -----
        This will store the field in the given mesh under the given name,
        if a meshio or PyVista mesh was given.

        See:
            - meshio: https://github.com/nschloe/meshio
            - ogs5py: https://github.com/GeoStat-Framework/ogs5py
            - PyVista: https://github.com/pyvista/pyvista
        """
        return generate_on_mesh(self, mesh, points, direction, name, **kwargs)

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
        # save pos tuple
        if mesh_type != "unstructured":
            pos, shape = format_struct_pos_dim(pos, self.dim)
            self.pos = pos
            pos = generate_grid(pos)
        else:
            pos = np.array(pos, dtype=np.double).reshape(self.dim, -1)
            self.pos = pos
            shape = np.shape(pos[0])
        # prepend dimension if we have a vector field
        if self.value_type == "vector":
            shape = (self.dim,) + shape
            if self.latlon:
                raise ValueError("Field: Vector fields not allowed for latlon")
        # return isometrized pos tuple and resulting field shape
        if self.model is None:
            return pos, shape
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
            field = apply_mean_norm_trend(
                pos=self.pos,
                field=field,
                mesh_type=self.mesh_type,
                value_type=self.value_type,
                mean=self.mean,
                normalizer=self.normalizer,
                trend=self.trend,
                check_shape=False,
                stacked=False,
            )
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

    def plot(
        self, field="field", fig=None, ax=None, **kwargs
    ):  # pragma: no cover
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
        **kwargs
            Forwarded to the plotting routine.
        """
        # just import if needed; matplotlib is not required by setup
        from gstools.field.plot import plot_field, plot_vec_field

        if self.value_type is None:
            raise ValueError(
                "Field value type not set! "
                "Specify 'scalar' or 'vector' before plotting."
            )

        if self.value_type == "scalar":
            r = plot_field(self, field, fig, ax, **kwargs)
        elif self.value_type == "vector":
            if self.dim == 2:
                r = plot_vec_field(self, field, fig, ax, **kwargs)
            else:
                raise NotImplementedError(
                    "Streamflow plotting only supported for 2d case."
                )
        else:
            raise ValueError(f"Unknown field value type: {self.value_type}")

        return r

    @property
    def model(self):
        """:any:`CovModel`: The covariance model of the field."""
        return self._model

    @model.setter
    def model(self, model):
        if model is not None:
            if not isinstance(model, CovModel):
                raise ValueError(
                    "Field: 'model' is not an instance of 'gstools.CovModel'"
                )
            self._model = model
            self._dim = None
        elif self._dim is None:
            raise ValueError("Field: either needs 'model' or 'dim'.")
        else:
            self._model = None

    @property
    def mean(self):
        """:class:`float` or :any:`callable`: The mean of the field."""
        return self._mean

    @mean.setter
    def mean(self, mean):
        self._mean = _set_mean_trend(mean, self.dim)

    @property
    def normalizer(self):
        """:any:`Normalizer`: Normalizer of the field."""
        return self._normalizer

    @normalizer.setter
    def normalizer(self, normalizer):
        self._normalizer = _check_normalizer(normalizer)

    @property
    def trend(self):
        """:class:`float` or :any:`callable`: The trend of the field."""
        return self._trend

    @trend.setter
    def trend(self, trend):
        self._trend = _set_mean_trend(trend, self.dim)

    @property
    def value_type(self):
        """:class:`str`: Type of the field values (scalar, vector)."""
        return self._value_type

    @value_type.setter
    def value_type(self, value_type):
        if value_type not in VALUE_TYPES:
            raise ValueError(f"Field: value type not in {VALUE_TYPES}")
        self._value_type = value_type

    @property
    def dim(self):
        """:class:`int`: Dimension of the field."""
        return self._dim if self.model is None else self.model.field_dim

    @property
    def latlon(self):
        """:class:`bool`: Whether the field depends on geographical coords."""
        return False if self.model is None else self.model.latlon

    @property
    def name(self):
        """:class:`str`: The name of the class."""
        return self.__class__.__name__

    def _fmt_mean_norm_trend(self):
        # fmt_mean_norm_trend for all child classes
        return fmt_mean_norm_trend(self)

    def __repr__(self):
        """Return String representation."""
        if self.model is None:
            dim_str = f"dim={self.dim}"
        else:
            dim_str = f"model={self.model.name}"
        return "{0}({1}, value_type='{2}'{3})".format(
            self.name,
            dim_str,
            self.value_type,
            self._fmt_mean_norm_trend(),
        )
