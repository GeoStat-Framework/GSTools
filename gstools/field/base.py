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
import meshio

from gstools.covmodel.base import CovModel
from gstools.tools.geometric import format_struct_pos_dim, gen_mesh
from gstools.tools.export import to_vtk, vtk_export


__all__ = ["Field"]


class Field:
    """A base class for random fields, kriging fields, etc.

    Parameters
    ----------
    model : :any:`CovModel`
        Covariance Model related to the field.
    mean : :class:`float`, optional
        Mean value of the field.
    """

    def __init__(self, model, mean=0.0):
        # initialize attributes
        self.pos = None
        self.mesh_type = None
        self.field = None
        # initialize private attributes
        self._mean = None
        self._model = None
        self.mean = mean
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
        has_pyvista = False
        has_ogs5py = False

        try:
            import pyvista as pv

            has_pyvista = True
        except ImportError:
            pass
        try:
            import ogs5py as ogs

            has_ogs5py = True
        except ImportError:
            pass

        if isinstance(direction, str) and direction == "all":
            select = list(range(self.model.field_dim))
        elif isinstance(direction, str):
            select = _get_select(direction)[: self.model.field_dim]
        else:
            select = direction[: self.model.field_dim]
        if len(select) < self.model.field_dim:
            raise ValueError(
                "Field.mesh: need at least {} direction(s), got '{}'".format(
                    self.model.field_dim, direction
                )
            )
        # convert pyvista mesh
        if has_pyvista and pv.is_pyvista_dataset(mesh):
            if points == "centroids":
                pnts = mesh.cell_centers().points.T[select]
            else:
                pnts = mesh.points.T[select]
            out = self.unstructured(pos=pnts, **kwargs)
            # Deal with the output
            fields = [out] if isinstance(out, np.ndarray) else out
            for f_name, field in zip(_names(name, len(fields)), fields):
                mesh[f_name] = field
        # convert ogs5py mesh
        elif has_ogs5py and isinstance(mesh, ogs.MSH):
            if points == "centroids":
                pnts = mesh.centroids_flat.T[select]
            else:
                pnts = mesh.NODES.T[select]
            out = self.unstructured(pos=pnts, **kwargs)
        # convert meshio mesh
        elif isinstance(mesh, meshio.Mesh):
            if points == "centroids":
                # define unique order of cells
                offset = []
                length = []
                mesh_dim = mesh.points.shape[1]
                if mesh_dim < self.model.field_dim:
                    raise ValueError("Field.mesh: mesh dimension too low!")
                pnts = np.empty((0, mesh_dim), dtype=np.double)
                for cell in mesh.cells:
                    pnt = np.mean(mesh.points[cell[1]], axis=1)
                    offset.append(pnts.shape[0])
                    length.append(pnt.shape[0])
                    pnts = np.vstack((pnts, pnt))
                # generate pos for __call__
                pnts = pnts.T[select]
                out = self.unstructured(pos=pnts, **kwargs)
                fields = [out] if isinstance(out, np.ndarray) else out
                f_lists = []
                for field in fields:
                    f_list = []
                    for of, le in zip(offset, length):
                        f_list.append(field[of : of + le])
                    f_lists.append(f_list)
                for f_name, f_list in zip(_names(name, len(f_lists)), f_lists):
                    mesh.cell_data[f_name] = f_list
            else:
                out = self.unstructured(pos=mesh.points.T[select], **kwargs)
                fields = [out] if isinstance(out, np.ndarray) else out
                for f_name, field in zip(_names(name, len(fields)), fields):
                    mesh.point_data[f_name] = field
        else:
            raise ValueError("Field.mesh: Unknown mesh format!")
        return out

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
                print("Field.to_vtk: '{}' not available.".format(field_select))
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

    @property
    def value_type(self):
        """:class:`str`: Type of the field values (scalar, vector)."""
        return self._value_type

    def __str__(self):
        """Return String representation."""
        return self.__repr__()

    def __repr__(self):
        """Return String representation."""
        return "Field(model={0}, mean={1:.{p}})".format(
            self.model, self.mean, p=self.model._prec
        )


def _names(name, cnt):
    name = [name] if isinstance(name, str) else list(name)[:cnt]
    if len(name) < cnt:
        name += [name[-1] + str(i + 1) for i in range(cnt - len(name))]
    return name


def _get_select(direction):
    select = []
    if not (0 < len(direction) < 4):
        raise ValueError(
            "Field.mesh: need 1 to 3 direction(s), got '{}'".format(direction)
        )
    for axis in direction:
        if axis == "x":
            if 0 in select:
                raise ValueError(
                    "Field.mesh: got duplicate directions {}".format(direction)
                )
            select.append(0)
        elif axis == "y":
            if 1 in select:
                raise ValueError(
                    "Field.mesh: got duplicate directions {}".format(direction)
                )
            select.append(1)
        elif axis == "z":
            if 2 in select:
                raise ValueError(
                    "Field.mesh: got duplicate directions {}".format(direction)
                )
            select.append(2)
        else:
            raise ValueError(
                "Field.mesh: got unknown direction {}".format(axis)
            )
    return select
