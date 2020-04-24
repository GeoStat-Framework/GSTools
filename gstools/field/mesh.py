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
    >>> mesh = Mesh((pos,))
    >>> mesh.add_field(z, "test_field1")
    >>> mesh.add_field(z2, "test_field2")
    >>> mesh.set_default_field("test_field2")
    >>> print(mesh.field)

    """

    def __init__(
        self, pos=None, name="field", values=None, mesh_type="unstructured",
    ):
        self._vtk_export_fct = self._vtk_export_unstructured
        self._to_vtk_fct = self._to_vtk_unstructured
        # mesh_type needs a special setter, therefore, `set_field_data` is not
        # used here
        self.mesh_type = mesh_type

        # the pos/ points of the mesh
        self._pos = pos

        # data stored at each pos/ point, the "fields"
        if values is not None:
            self.point_data = {name: values}
        else:
            self.point_data = {}

        # data valid for the global field
        self.field_data = {}

        self.set_field_data(name, "default_field")

        self.field_data["mesh_type"] = mesh_type

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
        self._check_point_data(values)
        self.point_data[name] = values
        # set the default_field to the first field added
        if len(self.point_data) == 1 or is_default_field:
            self.set_field_data(name, "default_field")

    def __getitem__(self, key):
        """:any:`numpy.ndarray`: The values of the field."""
        return self.point_data[key]

    def __setitem__(self, key, value):
        self.point_data[key] = value

    @property
    def pos(self):
        """:any:`numpy.ndarray`: The pos. on which the field is defined."""
        return self._pos

    @pos.setter
    def pos(self, value):
        """
        Warning: setting new positions deletes all previously stored fields.
        """
        self.point_data = {self.default_field: None}
        self._pos = value

    @property
    def field(self):
        """:class:`numpy.ndarray`: The point data of the default field."""
        return self.point_data[self.default_field]

    @field.setter
    def field(self, values):
        self._check_point_data(values)
        self.point_data[self.default_field] = values

    @property
    def value_type(self):
        """:any:`str`: The value type of the default field."""
        if self.default_field in self.point_data:
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
                self.pos is None or field is None or self.mesh_type is None
            ):
                suf = ["_X", "_Y", "_Z"]
                fields = {}
                for i in range(self.model.dim):
                    fields[fieldname + suf[i]] = field[i]
                return fields
        elif self.value_type == "scalar":
            if hasattr(self, field_select):
                field = getattr(self, field_select)
            else:
                field = None
            if not (
                self.pos is None or field is None or self.mesh_type is None
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
        field_names = self._vtk_naming_helper(
            field_select=field_select, fieldname=fieldname
        )

        grid = self._to_vtk_fct(field_names)
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
            raise TypeError("Please use a string as a filename.")
        field_names = self._vtk_naming_helper(
            field_select=field_select, fieldname=fieldname
        )
        return self._vtk_export_fct(filename, field_names)

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
        x, y, z, fields = self._vtk_unstructured_reshape(fields=fields)
        return pointsToVTK(filename, x, y, z, data=fields)

    def _to_vtk_structured(self, fields):  # pragma: no cover
        """Create a vtk structured rectilinear grid from a field.

        Parameters
        ----------
        pos : :class:`list`
            the position tuple, containing main direction and transversal
            directions
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

    def _to_vtk_unstructured(self, fields):
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
        x, y, z, fields = self._vtk_unstructured_reshape(fields=fields)
        try:
            import pyvista as pv

            grid = pv.PolyData(np.c_[x, y, z]).cast_to_unstructured_grid()
            grid.point_arrays.update(fields)
        except ImportError:
            raise ImportError("Please install PyVista to create VTK datasets.")
        return grid

    def _vtk_structured_reshape(self, fields):
        """An internal helper to extract what is needed for the vtk rectilinear grid
        """
        if not isinstance(fields, dict):
            fields = {"field": fields}
        x, y, z = pos2xyz(self.pos)
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

    def _vtk_unstructured_reshape(self, fields):
        if not isinstance(fields, dict):
            fields = {"field": fields}
        x, y, z = pos2xyz(self.pos)
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

    def _check_mesh_type(self, mesh_type):
        if mesh_type != "unstructured" and mesh_type != "structured":
            raise ValueError("Unknown 'mesh_type': {}".format(mesh_type))

    def _check_point_data(self, values):
        """Compare field shape to pos shape.

        Parameters
        ----------
        values : :class:`numpy.ndarray`
            the values of the field to be checked
        """
        err = True
        if self.mesh_type == "unstructured":
            # scalar
            if len(values.shape) == 1:
                if values.shape[0] == len(self.pos[0]):
                    err = False
            # vector
            elif len(values.shape) == 2:
                if (
                    values.shape[1] == len(self.pos[0])
                    and 2 <= values.shape[0] <= 3
                ):
                    err = False
            if err:
                raise ValueError(
                    "Wrong field shape: {0} does not match mesh shape ({1},)".format(
                        values.shape, len(self.pos[0])
                    )
                )
        else:
            # scalar
            if len(values.shape) == len(self.pos):
                if all(
                    [
                        values.shape[i] == len(self.pos[i])
                        for i in range(len(self.pos))
                    ]
                ):
                    err = False
            # vector
            elif len(values.shape) == len(self.pos) + 1:
                if all(
                    [
                        values.shape[i + 1] == len(self.pos[i])
                        for i in range(len(self.pos))
                    ]
                ) and values.shape[0] == len(self.pos):
                    err = False
            if err:
                raise ValueError(
                    "Wrong field shape: {0} does not match mesh shape [0/2/3]{1}".format(
                        list(values.shape),
                        [len(self.pos[i]) for i in range(len(self.pos))],
                    )
                )


if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod()
