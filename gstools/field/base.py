# -*- coding: utf-8 -*-
"""
GStools subpackage providing a base class for spatial fields.

.. currentmodule:: gstools.field.base

The following classes are provided

.. autosummary::
   Field
"""
# pylint: disable=C0103
from __future__ import division, absolute_import, print_function

from functools import partial

import numpy as np

from gstools.covmodel.base import CovModel
from gstools.tools.export import vtk_export as vtk_ex

__all__ = ["Field"]


class Field(object):
    """A field base class for random and kriging fields ect.

    Parameters
    ----------
    model : :any:`CovModel`
        Covariance Model related to the field.
    mean : :class:`float`, optional
        Mean value of the field.
    """

    def __init__(
        self,
        model,
        mean=0.0,
    ):
        # initialize attributes
        self.pos = None
        self.mesh_type = None
        self.field = None
        # initialize private attributes
        self._mean = mean
        self._model = None
        self.model = model

    def __call__(*args, mesh_type="unstructured", **kwargs):
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

    def mesh(self, mesh, name="field", points="centriods", **kwargs):
        """Generate a field on a given meshio mesh.

        Parameters
        ----------
        mesh : meshio.Mesh
            The given meshio.Mesh
        field : :class:`str`, optional
            Name to store the field in the given mesh as point_data or
            cell_data. Default: "field"
        points : :class:`str`, optional
            The points to evaluate the field at.
            Either the "centroids" of the mesh cells
            (calculated as mean of the cell vertices) or the "points"
            of the given mesh.
            Default: "centroids"
        **kwargs
            Keyword arguments forwareded to `Field.__call__`.

        Notes
        -----
        This will store the field in the given mesh under the given name.

        See: https://github.com/nschloe/meshio

        See: :any:`Field.__call__`
        """
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
            pnts = list(pnts.T)
            out = self.unstructured(pos=pnts, **kwargs)
            if isinstance(out, np.ndarray):
                field = out
            else:
                # if multiple values are returned, take the first one
                field = out[0]
            field_dict = {}
            for i, cell in enumerate(cells):
                field_dict[cell] = field[offset[i]: offset[i] + length[i]]
            mesh.cell_data[name] = field_dict
        else:
            out = self.unstructured(pos=list(mesh.points.T), **kwargs)
            if isinstance(out, np.ndarray):
                field = out
            else:
                # if multiple values are returned, take the first one
                field = out[0]
            mesh.point_data[name] = field
        return out

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
        if hasattr(self, field_select):
            field = getattr(self, field_select)
        else:
            field = None
        if not (self.pos is None or field is None or self.mesh_type is None):
            vtk_ex(filename, self.pos, field, fieldname, self.mesh_type)
        else:
            print(
                "Field.vtk_export: No "
                + field_select
                + " stored in the class."
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
        from gstools.field.plot import plot_field

        return plot_field(self, field, fig, ax)

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
        return "Field(model={0}, mean={1})".format(self.model, self.mean)


if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod()
