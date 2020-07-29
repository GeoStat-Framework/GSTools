# -*- coding: utf-8 -*-
"""
GStools subpackage providing a base class for spatial fields.

.. currentmodule:: gstools.field.base

The following classes are provided

.. autosummary::
   Mesh
"""
# pylint: disable=C0103

import numpy as np

from pyevtk.hl import gridToVTK, pointsToVTK

from gstools.tools.geometric import pos2xyz
from gstools.field.tools import check_mesh, check_point_data

__all__ = ["Mesh"]


def value_type(mesh_type, shape):
    """Determine the value type ("scalar" or "vector")"""
    r = "scalar"
    if mesh_type == "unstructured":
        # TODO is this the right place for doing these checks?!
        if len(shape) == 2 and 2 <= shape[0] <= 3:
            r = "vector"
    else:
        # for very small (2-3 points) meshes, this could break
        # a 100% solution would require dim. information.
        if len(shape) == shape[0] + 1:
            r = "vector"
    return r

# TODO figure out what this is all about
def convert_points(dim, points):
    points = np.array(points)
    shape = points.shape
    if dim == 1:
        #points = points.flatten()
        if len(points.shape) > 1:
            raise ValueError(
                "points shape {} does not match dim {}".format(shape, dim)
            )
    else:
        pass

    return points


class Mesh:
    """A base class encapsulating field data.

    It holds a position array, which define the spatial locations of the
    field values.
    It can hold multiple fields in the :any:`self.point_data` dict. This assumes
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
    mesh_type : :any:`str`, optional
        the type of mesh on which the field is defined on, can be
        * unstructured
        * structured

    Examples
    --------
    >>> import numpy as np
    >>> from gstools import Mesh
    >>> pos = np.linspace(0., 100., 40)
    >>> z = np.random.random(40)
    >>> z2 = np.random.random(40)
    >>> mesh = Mesh(1, (pos,))
    >>> mesh.add_field(z, "test_field1")
    >>> mesh.add_field(z2, "test_field2")
    >>> mesh.set_default_field("test_field2")
    >>> print(mesh.field)

    """
    def __init__(
        self, dim, points=None, name="field", values=None, mesh_type="unstructured",
    ):
        self._dim = dim

        # the pos/ points of the mesh
        if points is not None:
            check_mesh(dim, points, mesh_type)
        self._points = points

        # data stored at each pos/ point, the "fields"
        if values is not None:
            self.point_data = {name: values}
        else:
            self.point_data = {}

        # data valid for the global field
        self.field_data = {}

        # do following line manually in order to satisfy the linters
        # self.set_field_data(name, "default_field")
        self.default_field = name
        self.field_data["default_field"] = name

        self.mesh_type = mesh_type

    def set_field_data(self, value, name):
        """Add an attribute to this instance and add it the `field_data`

        This helper method is used to create attributes for easy access
        while using this class, but it also adds an entry to the dictionary
        `field_data`, which is used for exporting the data.
        """
        setattr(self, name, value)
        self.field_data[name] = value

    def add_field(
        self, values, name="field", is_default_field=False,
    ):
        """Add a field (point data) to the mesh

        .. note::
            If no field has existed before calling this method,
            the given field will be set to the default field.

        .. warning::
            If point data with the same `name` already exists, it will be
            overwritten.

        Parameters
        ----------
        values : :class:`numpy.ndarray`
            the point data, has to be the same shape as the mesh
        name : :class:`str`, optional
            the name of the point data
        is_default_field : :class:`bool`, optional
            is this the default field of the mesh?

        """
        values = np.array(values)
        check_point_data(
            self.dim,
            self.points,
            values,
            self.mesh_type,
            value_type(self.mesh_type, values.shape),
        )
        # set the default_field to the first field added
        if not self.point_data:
            is_default_field = True
        elif self.point_data.get(self.default_field, False) is None:
            del(self.point_data[self.default_field])
            is_default_field = True
        if is_default_field:
            self.set_field_data(name, "default_field")
        self.point_data[name] = values

    def del_field_data(self):
        """Delete the field data.

        Deleting the field data resets the field data dictionary and deletes
        the attributes too.
        """
        for attr in self.field_data.keys():
            if attr == "default_field" or attr == "mesh_type":
                continue
            try:
                delattr(self, attr)
            except AttributeError:
                pass
        self.field_data = {"default_field": "field", "mesh_type": self.mesh_type}

    def __getitem__(self, key):
        """:any:`numpy.ndarray`: The values of the field."""
        return self.point_data[key]

    def __setitem__(self, key, value):
        self.point_data[key] = value

    @property
    def dim(self):
        """:any:`int`: The dimension of the mesh."""
        return self._dim

    @property
    def points(self):
        """:any:`numpy.ndarray`: The pos. on which the field is defined."""
        return self._points

    @points.setter
    def points(self, value):
        """
        Warning: setting new positions deletes all previously stored fields.
        """
        self.point_data = {self.default_field: None}
        check_mesh(self.dim, value, self.mesh_type)
        self._points = value

    @property
    def field(self):
        """:class:`numpy.ndarray`: The point data of the default field."""
        return self.point_data[self.default_field]

    @field.setter
    def field(self, values):
        check_point_data(
            self.dim,
            self.points,
            values,
            self.mesh_type,
            value_type(self.mesh_type, values.shape),
        )
        self.point_data[self.default_field] = values

    @property
    def value_type(self):
        """:any:`str`: The value type of the default field."""
        if (
            self.default_field in self.point_data and
            self.point_data[self.default_field] is not None
        ):
            r = value_type(
                self.mesh_type, self.point_data[self.default_field].shape
            )
        else:
            r = None
        return r

    @property
    def mesh_type(self):
        """:any:`str`: The mesh type of the fields."""
        return self._mesh_type

    @mesh_type.setter
    def mesh_type(self, value):
        """
        Warning: setting a new mesh type deletes all previously stored fields.
        """
        self._check_mesh_type(value)
        if value == "structured":
            self._vtk_export_fct = self._vtk_export_structured
            self._to_vtk_fct = self._to_vtk_structured
        else:
            self._vtk_export_fct = self._vtk_export_unstructured
            self._to_vtk_fct = self._to_vtk_unstructured
        self.point_data = {}
        self._mesh_type = value
        self.field_data["mesh_type"] = value

    def _vtk_naming_helper(
        self, field_select="field", fieldname="field"
    ):  # pragma: no cover
        """Prepare the field names for export.

        Parameters
        ----------
        field_select : :class:`str`, optional
            Field that should be stored. Can be:
            "field", "raw_field", "krige_field", "err_field" or "krige_var".
            Default: "field"
        fieldname : :class:`str`, optional
            Name of the field in the VTK file. Default: "field"

        Returns
        -------
        fields : :class:`dict`
            a dictionary containing the fields to be exported
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
                self.points is None or field is None or self.mesh_type is None
            ):
                suf = ["_X", "_Y", "_Z"]
                fields = {}
                for i in range(self.dim):
                    fields[fieldname + suf[i]] = field[i]
                return fields
        elif self.value_type == "scalar":
            if hasattr(self, field_select):
                field = getattr(self, field_select)
            else:
                field = None
            if not (
                self.points is None or field is None or self.mesh_type is None
            ):
                return {fieldname: field}
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

    def convert(
        self, data_format="pyvista", field_select="field", fieldname="field"
    ):  # pragma: no cover
        """Create a VTK/PyVista grid of the stored field.

        Parameters
        ----------
        data_format : :class:`str`
            the format in which to convert the data, possible choices:
            * 'pyvista'
        field_select : :class:`str`, optional
            Field that should be stored. Can be:
            "field", "raw_field", "krige_field", "err_field" or "krige_var".
            Default: "field"
        fieldname : :class:`str`, optional
            Name of the field in the VTK file. Default: "field"
        """
        if data_format.lower() != "pyvista":
            raise NotImplementedError(
                "Only pyvista format is supported at the moment."
            )
        field_names = self._vtk_naming_helper(
            field_select=field_select, fieldname=fieldname
        )

        grid = self._to_vtk_fct(field_names)
        return grid

    def export(
        self, filename, data_format="vtk", field_select="field", fieldname="field"
    ):  # pragma: no cover
        """Export the stored field to vtk.

        Parameters
        ----------
        filename : :class:`str`
            Filename of the file to be saved, including the path. Note that an
            ending (.vtr or .vtu) will be added to the name.
        data_format : :class:`str`
            the format in which to export the data, possible choices:
            * 'vtk'
        field_select : :class:`str`, optional
            Field that should be stored. Can be:
            "field", "raw_field", "krige_field", "err_field" or "krige_var".
            Default: "field"
        fieldname : :class:`str`, optional
            Name of the field in the VTK file. Default: "field"
        """
        if data_format.lower() != "vtk":
            raise NotImplementedError(
                "Only VTK format is supported at the moment."
            )
        if not isinstance(filename, str):
            raise TypeError("Please use a string as a filename.")
        field_names = self._vtk_naming_helper(
            field_select=field_select, fieldname=fieldname
        )
        return self._vtk_export_fct(filename, field_names)

    def _vtk_export_unstructured(self, filename, fields):
        """Export a field to vtk unstructured grid file.

        Parameters
        ----------
        filename : :class:`str`
            Filename of the file to be saved, including the path. Note that an
            ending (.vtu) will be added to the name.
        fields : :class:`dict` or :class:`numpy.ndarray`
            Unstructured fields to be saved.
            Either a single numpy array as returned by SRF,
            or a dictionary of fields with theirs names as keys.
        """
        x, y, z, fields = self._vtk_unstructured_reshape(fields)
        return pointsToVTK(filename, x, y, z, data=fields)

    def _vtk_export_structured(self, filename, fields):
        """Export a field to vtk structured rectilinear grid file.

        Parameters
        ----------
        filename : :class:`str`
            Filename of the file to be saved, including the path. Note that an
            ending (.vtr) will be added to the name.
        fields : :class:`dict` or :class:`numpy.ndarray`
            Structured fields to be saved.
            Either a single numpy array as returned by SRF,
            or a dictionary of fields with theirs names as keys.
        """
        x, y, z, fields = self._vtk_structured_reshape(fields=fields)
        return gridToVTK(filename, x, y, z, pointData=fields)

    def _to_vtk_unstructured(self, fields):  # pragma: no cover
        """Export a field to vtk structured rectilinear grid file.

        Parameters
        ----------
        fields : :class:`dict` or :class:`numpy.ndarray`
            Unstructured fields to be saved.
            Either a single numpy array as returned by SRF,
            or a dictionary of fields with theirs names as keys.

        Returns
        -------
        :class:`pyvista.UnstructuredGrid`
            A PyVista unstructured grid of the unstructured field data. Data arrays
            live on the point data of this PyVista dataset. This is essentially
            a point cloud with no topology.
        """
        x, y, z, fields = self._vtk_unstructured_reshape(fields)
        try:
            import pyvista as pv

            grid = pv.PolyData(np.c_[x, y, z]).cast_to_unstructured_grid()
            grid.point_arrays.update(fields)
        except ImportError:
            raise ImportError("Please install PyVista to create VTK datasets.")
        return grid

    def _to_vtk_structured(self, fields):  # pragma: no cover
        """Create a vtk structured rectilinear grid from a field.

        Parameters
        ----------
        fields : :class:`dict` or :class:`numpy.ndarray`
            Structured fields to be saved.
            Either a single numpy array as returned by SRF,
            or a dictionary of fields with theirs names as keys.

        Returns
        -------
        :class:`pyvista.RectilinearGrid`
            A PyVista rectilinear grid of the structured field data. Data arrays
            live on the point data of this PyVista dataset.
        """
        x, y, z, fields = self._vtk_structured_reshape(fields=fields)
        try:
            import pyvista as pv

            grid = pv.RectilinearGrid(x, y, z)
            grid.point_arrays.update(fields)
        except ImportError:
            raise ImportError("Please install PyVista to create VTK datasets.")
        return grid

    def _vtk_unstructured_reshape(self, fields):
        if not isinstance(fields, dict):
            fields = {"field": fields}
        x, y, z = pos2xyz(self.points)
        if y is None:
            y = np.zeros_like(x)
        if z is None:
            z = np.zeros_like(x)
        for field in fields:
            fields[field] = fields[field].reshape(-1)
            if (
                len(fields[field]) != len(x)
                or len(fields[field]) != len(y)
                or len(fields[field]) != len(z)
            ):
                raise ValueError(
                    "gstools.vtk_export_unstructured: "
                    "field shape doesn't match the given mesh"
                )
        return x, y, z, fields

    def _vtk_structured_reshape(self, fields):
        """An internal helper to extract what is needed for the vtk rectilinear grid
        """
        if not isinstance(fields, dict):
            fields = {"field": fields}
        x, y, z = pos2xyz(self.points)
        if y is None:
            y = np.array([0])
        if z is None:
            z = np.array([0])
        # need fortran order in VTK
        for field in fields:
            fields[field] = fields[field].reshape(-1, order="F")
            if len(fields[field]) != len(x) * len(y) * len(z):
                raise ValueError(
                    "gstools.vtk_export_structured: "
                    "field shape doesn't match the given mesh"
                )
        return x, y, z, fields

    def _check_mesh_type(self, mesh_type):
        if mesh_type != "unstructured" and mesh_type != "structured":
            raise ValueError("Unknown 'mesh_type': {}".format(mesh_type))


#class Mesh(MeshBase):
#    def __init__(self):
#        super().__init__()
#
#    def _vtk_reshape(self, fields):
#        if not isinstance(fields, dict):
#            fields = {"field": fields}
#        x, y, z = pos2xyz(self.pos)
#        if y is None:
#            y = np.zeros_like(x)
#        if z is None:
#            z = np.zeros_like(x)
#        for field in fields:
#            fields[field] = fields[field].reshape(-1)
#            if (
#                len(fields[field]) != len(x)
#                or len(fields[field]) != len(y)
#                or len(fields[field]) != len(z)
#            ):
#                raise ValueError(
#                    "gstools.vtk_export_unstructured: "
#                    "field shape doesn't match the given mesh"
#                )
#        return x, y, z, fields
#
#    def _vtk_export(self, filename, fields):
#        """Export a field to vtk unstructured grid file.
#
#        Parameters
#        ----------
#        filename : :class:`str`
#            Filename of the file to be saved, including the path. Note that an
#            ending (.vtu) will be added to the name.
#        fields : :class:`dict` or :class:`numpy.ndarray`
#            Unstructured fields to be saved.
#            Either a single numpy array as returned by SRF,
#            or a dictionary of fields with theirs names as keys.
#        """
#        x, y, z, fields = self._vtk_reshape(fields=fields)
#        return pointsToVTK(filename, x, y, z, data=fields)
#
#    def _to_vtk(self, fields):
#        """Export a field to vtk structured rectilinear grid file.
#
#        Parameters
#        ----------
#        fields : :class:`dict` or :class:`numpy.ndarray`
#            Unstructured fields to be saved.
#            Either a single numpy array as returned by SRF,
#            or a dictionary of fields with theirs names as keys.
#
#        Returns
#        -------
#        :class:`pyvista.UnstructuredGrid`
#            A PyVista unstructured grid of the unstructured field data. Data arrays
#            live on the point data of this PyVista dataset. This is essentially
#            a point cloud with no topology.
#        """
#        x, y, z, fields = self._vtk_unstructured_reshape(fields=fields)
#        try:
#            import pyvista as pv
#
#            grid = pv.PolyData(np.c_[x, y, z]).cast_to_unstructured_grid()
#            grid.point_arrays.update(fields)
#        except ImportError:
#            raise ImportError("Please install PyVista to create VTK datasets.")
#        return grid


if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod()
