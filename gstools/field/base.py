# -*- coding: utf-8 -*-
"""
GStools subpackage providing a base class for spatial fields.

.. currentmodule:: gstools.field.base

The following classes are provided

.. autosummary::
   Field
   Mesh
"""
# pylint: disable=C0103

from functools import partial
from typing import List, Dict, Tuple

import numpy as np

from gstools.covmodel.base import CovModel
from gstools.tools.export import to_vtk, vtk_export
from gstools.field.tools import (
    _get_select,
    check_mesh,
    make_isotropic,
    unrotate_mesh,
    reshape_axis_from_struct_to_unstruct,
)
from gstools.tools.geometric import pos2xyz, xyz2pos

__all__ = ["Field", "Mesh"]


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
        self,
        pos=None,
        name: str = "field",
        values: np.ndarray = None,
        *,
        mesh_type: str = "unstructured",
    ) -> None:
        # mesh_type needs a special setter, therefore, `set_field_data` is not
        # used here
        self.mesh_type = mesh_type

        # the pos/ points of the mesh
        self._pos = pos

        # data stored at each pos/ point, the "fields"
        if values is not None:
            self.point_data: Dict[str, np.ndarray] = {name: values}
        else:
            self.point_data: Dict[str, np.ndarray] = {}

        # data valid for the global field
        self.field_data = {}

        self.set_field_data("default_field", name)

        self.field_data["mesh_type"] = mesh_type

    def set_field_data(self, name: str, value) -> None:
        """Add an attribute to this instance and add it the `field_data`

        This helper method is used to create attributes for easy access
        while using this class, but it also adds an entry to the dictionary
        `field_data`, which is used for exporting the data.
        """
        setattr(self, name, value)
        self.field_data[name] = value

    def add_field(
        self,
        values: np.ndarray,
        name: str = "field",
        *,
        is_default_field: bool = False,
    ) -> None:
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
            self.set_field_data("default_field", name)

    def __getitem__(self, key: str) -> np.ndarray:
        """:any:`numpy.ndarray`: The values of the field."""
        return self.point_data[key]

    def __setitem__(self, key: str, value):
        self.point_data[key] = value

    @property
    def pos(self) -> Tuple[np.ndarray]:
        """:any:`numpy.ndarray`: The pos. on which the field is defined."""
        return self._pos

    @pos.setter
    def pos(self, value: Tuple[np.ndarray]):
        """
        Warning: setting new positions deletes all previously stored fields.
        """
        self.point_data = {self.default_field: None}
        self._pos = value

    @property
    def field(self) -> np.ndarray:
        """:class:`numpy.ndarray`: The point data of the default field."""
        return self.point_data[self.default_field]

    @field.setter
    def field(self, values: np.ndarray):
        self._check_point_data(values)
        self.point_data[self.default_field] = values

    @property
    def value_type(self, field="field") -> str:
        """:any:`str`: The value type of the default field."""
        if field in self.point_data:
            r = value_type(self.mesh_type, self.point_data[field].shape)
        else:
            r = None
        return r

    @property
    def mesh_type(self) -> str:
        """:any:`str`: The mesh type of the fields."""
        return self._mesh_type

    @mesh_type.setter
    def mesh_type(self, value: str):
        """
        Warning: setting a new mesh type deletes all previously stored fields.
        """
        self._check_mesh_type(value)
        self.point_data = {}
        self._mesh_type = value

    def _check_mesh_type(self, mesh_type: str) -> None:
        if mesh_type != "unstructured" and mesh_type != "structured":
            raise ValueError("Unknown 'mesh_type': {}".format(mesh_type))

    def _check_point_data(self, values: np.ndarray):
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


class Field(Mesh):
    """A field base class for random and kriging fields, etc.

    Parameters
    ----------
    model : :any:`CovModel`
        Covariance Model related to the field.
    """

    def __init__(
        self,
        model,
        *,
        pos=None,
        name: str = "field",
        values: np.ndarray = None,
        mesh_type: str = "unstructured",
        mean: float = 0.0,
    ) -> None:
        # initialize attributes
        super().__init__(
            pos=pos,
            name=name,
            values=values,
            mesh_type=mesh_type
        )
        # initialize private attributes
        self._model = None
        self.model = model
        self.mean = mean

    def __call__(self, *args, **kwargs):
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

    def _pre_pos(self, pos, mesh_type="unstructured", make_unstruct=False):
        """
        Preprocessing positions and mesh_type.

        Parameters
        ----------
        pos : :any:`iterable`
            the position tuple, containing main direction and transversal
            directions
        mesh_type : :class:`str`
            'structured' / 'unstructured'
        make_unstruct: :class:`bool`
            State if mesh_type should be made unstructured.

        Returns
        -------
        x : :class:`numpy.ndarray`
            first components of unrotated and isotropic position vectors
        y : :class:`numpy.ndarray` or None
            analog to x
        z : :class:`numpy.ndarray` or None
            analog to x
        pos : :class:`tuple` of :class:`numpy.ndarray`
            the normalized position tuple
        mesh_type_gen : :class:`str`
            'structured' / 'unstructured' for the generator
        mesh_type_changed : :class:`bool`
            State if the mesh_type was changed.
        axis_lens : :class:`tuple` or :any:`None`
            axis lengths of the structured mesh if mesh type was changed.
        """
        x, y, z = pos2xyz(pos, max_dim=self.model.dim)
        pos = xyz2pos(x, y, z)
        mesh_type_gen = mesh_type
        # format the positional arguments of the mesh
        check_mesh(self.model.dim, x, y, z, mesh_type)
        mesh_type_changed = False
        axis_lens = None
        if (
            self.model.do_rotation or make_unstruct
        ) and mesh_type == "structured":
            mesh_type_changed = True
            mesh_type_gen = "unstructured"
            x, y, z, axis_lens = reshape_axis_from_struct_to_unstruct(
                self.model.dim, x, y, z
            )
        if self.model.do_rotation:
            x, y, z = unrotate_mesh(self.model.dim, self.model.angles, x, y, z)
        if not self.model.is_isotropic:
            y, z = make_isotropic(self.model.dim, self.model.anis, y, z)
        return x, y, z, pos, mesh_type_gen, mesh_type_changed, axis_lens

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
    def mean(self):
        """:class:`float`: The mean of the field."""
        return self._mean

    @mean.setter
    def mean(self, mean):
        self._mean = float(mean)

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

    def __str__(self):
        """Return String representation."""
        return self.__repr__()

    def __repr__(self):
        """Return String representation."""
        return "Field(model={0})".format(self.model)


if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod()
