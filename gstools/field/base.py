# -*- coding: utf-8 -*-
"""
GStools subpackage providing a base class for spatial fields.

.. currentmodule:: gstools.field.base

The following classes are provided

.. autosummary::
   Field
   FieldData
"""
# pylint: disable=C0103

from functools import partial
from typing import List, Dict

import numpy as np

from gstools.covmodel.base import CovModel
from gstools.tools.export import to_vtk, vtk_export
from gstools.field.tools import _get_select

__all__ = ["Field", "FieldData"]


class Data:
    """A data class mainly storing the specific values of an individual field.

    Instances of this class are stored in a dictionary owned by FieldBase.
    The positions on which these field values are defined are also saved in
    FieldBase.
    """

    def __init__(
        self,
        values: np.ndarray = None,
        mean: float = 0.0,
        value_type: str = "scalar",
    ):
        self.values = values
        self.mean = mean
        self.value_type = value_type


class FieldData:
    """A base class encapsulating field data.

    It holds a position array, which define the spatial locations of the
    field values.
    It can hold multiple fields in the :any:`self.fields` dict. This assumes
    that each field is defined on the same positions.
    The mesh type must also be specified.

    Parameters
    ----------
    pos : :class:`numpy.ndarray`, optional
        positions of the field values
    name : :any:`str`, optional
        key of the field values
    values : :any:`list`, optional
        a list of the values of the fields
    mean : :any:`float`, optional
        mean of the field
    value_type : :any:`str`, optional
        the value type of the default field, can be
        * scalar
        * vector
    mesh_type : :any:`str`, optional
        the type of mesh on which the field is defined on, can be
        * unstructured
        * structured

    Examples
    --------
    >>> import numpy as np
    >>> pos = np.random.random((100, 100))
    >>> z = np.random.random((100, 100))
    >>> z2 = np.random.random((100, 100))
    >>> field_data = FieldData(pos)
    >>> field_data.add_field("test_field1", z)
    >>> field_data.add_field("test_field2", z2)
    >>> field_data.set_default_field("test_field2")
    >>> print(field.values)

    """

    def __init__(
        self,
        pos: np.ndarray = None,
        name: str = "default_field",
        values: np.ndarray = None,
        *,
        mean: float = 0.0,
        value_type: str = "scalar",
        mesh_type: str = "unstructured",
    ):
        # initialize attributes
        self.pos = pos
        self.fields: Dict[str, np.ndarray] = {}
        self._default_field = "default_field"
        if mesh_type != "unstructured" and mesh_type != "structured":
            raise ValueError("Unknown 'mesh_type': {}".format(mesh_type))
        self.mesh_type = mesh_type
        self.add_field(name, values, mean=mean, value_type=value_type)

    def add_field(
        self,
        name: str,
        values: np.ndarray,
        *,
        mean: float = 0.0,
        value_type: str = "scalar",
        default_field: bool = False,
    ):
        values = np.array(values)
        self._check_field(values)
        self.fields[name] = Data(values, mean, value_type)
        # set the default_field to the first field added
        if len(self.fields) == 1 or default_field:
            self._default_field = name

    def get_data(self, key):
        """:class:`Data`: The field data class."""
        return self.fields[key]

    def __getitem__(self, key):
        """:any:`numpy.ndarray`: The values of the field."""
        return self.fields[key].values

    def __setitem__(self, key, value):
        self.fields[key].values = value

    @property
    def default_field(self):
        """:any:`str`: The key of the default field."""
        return self._default_field

    @default_field.setter
    def default_field(self, value):
        self._default_field = value

    @property
    def field(self):
        """:any:`Data`: The Data instance of the default field."""
        return self.fields[self.default_field]

    @property
    def values(self):
        """:any:`numpy.ndarray`: The values of the default field."""
        return self.fields[self.default_field].values

    @values.setter
    def values(self, values):
        self.fields[self.default_field].values = values

    @property
    def value_type(self):
        """:any:`str`: The value type of the default field."""
        return self.fields[self.default_field].value_type

    @property
    def mean(self):
        """:any:`float`: The mean of the default field."""
        return self.fields[self.default_field].mean

    @mean.setter
    def mean(self, value):
        self.fields[self.default_field].mean = value

    def _check_field(self, values: np.ndarray):
        """Compare field shape to pos shape.

        Parameters
        ----------
        values : :class:`numpy.ndarray`
            the values of the field to be checked
        """
        # TODO
        if self.mesh_type == "unstructured":
            pass
        elif self.mesh_type == "structured":
            pass
        else:
            raise ValueError("Unknown 'mesh_type': {}".format(mesh_type))


class Field(FieldData):
    """A field base class for random and kriging fields ect.

    Parameters
    ----------
    model : :any:`CovModel`
        Covariance Model related to the field.
    """

    def __init__(self, model):
        # initialize attributes
        super().__init__(self)
        # initialize private attributes
        self._model = None
        self.model = model
        self._value_type = None

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
        self, mesh, points="centroids", direction="xyz", name="field", **kwargs
    ):
        """Generate a field on a given meshio or ogs5py mesh.

        Parameters
        ----------
        mesh : meshio.Mesh or ogs5py.MSH
            The given meshio or ogs5py mesh
        points : :class:`str`, optional
            The points to evaluate the field at.
            Either the "centroids" of the mesh cells
            (calculated as mean of the cell vertices) or the "points"
            of the given mesh.
            Default: "centroids"
        direction : :class:`str`, optional
            Here you can state which direction should be choosen for
            lower dimension. For example, if you got a 2D mesh in xz direction,
            you have to pass "xz"
            Default: "xyz"
        name : :class:`str`, optional
            Name to store the field in the given mesh as point_data or
            cell_data. Default: "field"
        **kwargs
            Keyword arguments forwareded to `Field.__call__`.

        Notes
        -----
        This will store the field in the given mesh under the given name,
        if a meshio mesh was given.

        See: https://github.com/nschloe/meshio

        See: :any:`Field.__call__`
        """
        select = _get_select(direction)
        if len(select) < self.model.dim:
            raise ValueError(
                "Field.mesh: need at least {} direction(s), got '{}'".format(
                    self.model.dim, direction
                )
            )
        if hasattr(mesh, "centroids_flat"):
            if points == "centroids":
                pnts = mesh.centroids_flat.T[select]
            else:
                pnts = mesh.NODES.T[select]
            out = self.unstructured(pos=pnts, **kwargs)
        else:
            if points == "centroids":
                # define unique order of cells
                cells = list(mesh.cells)
                offset = []
                length = []
                pnts = np.empty((0, 3), dtype=np.double)
                for cell in cells:
                    pnt = np.mean(mesh.points[mesh.cells[cell]], axis=1)
                    offset.append(pnts.shape[0])
                    length.append(pnt.shape[0])
                    pnts = np.vstack((pnts, pnt))
                # generate pos for __call__
                pnts = pnts.T[select]
                out = self.unstructured(pos=pnts, **kwargs)
                if isinstance(out, np.ndarray):
                    field = out
                else:
                    # if multiple values are returned, take the first one
                    field = out[0]
                field_dict = {}
                for i, cell in enumerate(cells):
                    field_dict[cell] = field[offset[i] : offset[i] + length[i]]
                mesh.cell_data[name] = field_dict
            else:
                out = self.unstructured(pos=mesh.points.T[select], **kwargs)
                if isinstance(out, np.ndarray):
                    field = out
                else:
                    # if multiple values are returned, take the first one
                    field = out[0]
                mesh.point_data[name] = field
        return out

    def _to_vtk_helper(
        self, filename=None, field_select="field", fieldname="field"
    ):  # pragma: no cover
        """Create a VTK/PyVista grid of the field or save it as a VTK file.

        This is an internal helper that will handle saving or creating objects

        Parameters
        ----------
        filename : :class:`str`
            Filename of the file to be saved, including the path. Note that an
            ending (.vtr or .vtu) will be added to the name. If ``None`` is
            passed, a PyVista dataset of the appropriate type will be returned.
        field_select : :class:`str`, optional
            Field that should be stored. Can be:
            "field", "raw_field", "krige_field", "err_field" or "krige_var".
            Default: "field"
        fieldname : :class:`str`, optional
            Name of the field in the VTK file. Default: "field"
        """
        if self.value_type is None:
            raise ValueError(
                "Field value type not set! "
                + "Specify 'scalar' or 'vector' before plotting."
            )
        elif self.value_type == "vector":
            if hasattr(self, field_select):
                field = getattr(self, field_select)
            else:
                field = None
            if not (
                self.pos is None or field is None or self.mesh_type is None
            ):
                suf = ["_X", "_Y", "_Z"]
                fields = {}
                for i in range(self.model.dim):
                    fields[fieldname + suf[i]] = field[i]
                if filename is None:
                    return to_vtk(self.pos, fields, self.mesh_type)
                else:
                    return vtk_export(
                        filename, self.pos, fields, self.mesh_type
                    )
        elif self.value_type == "scalar":
            if hasattr(self, field_select):
                field = getattr(self, field_select)
            else:
                field = None
            if not (
                self.pos is None or field is None or self.mesh_type is None
            ):
                if filename is None:
                    return to_vtk(self.pos, {fieldname: field}, self.mesh_type)
                else:
                    return vtk_export(
                        filename, self.pos, {fieldname: field}, self.mesh_type
                    )
            else:
                print(
                    "Field.to_vtk: No "
                    + field_select
                    + " stored in the class."
                )
        else:
            raise ValueError(
                "Unknown field value type: {}".format(self.value_type)
            )

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
        grid = self._to_vtk_helper(
            filename=None, field_select=field_select, fieldname=fieldname
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
        return self._to_vtk_helper(
            filename=filename, field_select=field_select, fieldname=fieldname
        )

    def plot(self, field="field", fig=None, ax=None):  # pragma: no cover
        """
        Plot the spatial random field.

        Parameters
        ----------
        field : :class:`str`, optional
            Field that should be plotted. Can be:
            "field", "raw_field", "krige_field", "err_field" or "krige_var".
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
                + "Specify 'scalar' or 'vector' before plotting."
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
    def value_type(self):
        """:class:`str`: Type of the field values (scalar, vector)."""
        return self._value_type

    def __str__(self):
        """Return String representation."""
        return self.__repr__()

    def __repr__(self):
        """Return String representation."""
        return "Field(model={0})".format(self.model)


if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod()
