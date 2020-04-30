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

from gstools.field import Mesh
from gstools.covmodel.base import CovModel
from gstools.field.tools import (
    _get_select,
    check_mesh,
    make_isotropic,
    unrotate_mesh,
    reshape_axis_from_struct_to_unstruct,
)
from gstools.tools.geometric import pos2xyz, xyz2pos

__all__ = ["Field"]


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
        pos=None,
        name="field",
        values=None,
        mesh_type="unstructured",
        mean=0.0,
    ):
        # initialize attributes
        super().__init__(
            dim=model.dim,
            pos=pos,
            name=name,
            values=values,
            mesh_type=mesh_type,
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
