"""
GStools subpackage providing a facies label field.

.. currentmodule:: gstools.geometry.field

The following classes are provided

.. autosummary::
   FaciesField
"""

import numpy as np

from gstools.field.base import Field
from gstools.geometry.base import Composite, Region

__all__ = ["FaciesField"]


class FaciesField(Field):
    """A categorical field rendering labels from a :any:`Region`.

    Holds a separate, inspectable geometry-state object (composition,
    mirroring how :any:`CondSRF` holds a :any:`Krige`).

    Parameters
    ----------
    geometry : :any:`Region` or :class:`list` of :any:`Region`
        The geometry to render. A list is wrapped in a
        :any:`Composite`.
    background : :class:`float`, optional
        Label used where the geometry assigns no label. Default: 0
    dtype : :class:`numpy.dtype`, optional
        Output dtype. Default: ``int64`` if every finite label is
        integral, else ``float64``.
    names : :class:`dict`, optional
        Mapping ``{label: name}`` used only for plot legends, never affects
        values. Default: derived from ``Layer.name``/``Surface.name``.
    """

    valid_value_types = ["scalar"]
    default_field_names = ["field"]

    def __init__(self, geometry, background=0, dtype=None, names=None):
        geometry = (
            geometry if isinstance(geometry, Region) else Composite(geometry)
        )
        super().__init__(model=None, value_type="scalar", dim=geometry.dim)
        self._geometry = geometry
        self.background = background
        self.dtype = dtype
        self._names = names

    def __call__(self, pos=None, mesh_type="unstructured", store=True):
        """Generate the label field.

        Parameters
        ----------
        pos : :class:`list`, optional
            the position tuple, containing main direction and transversal
            directions. Default: the present ``pos``
        mesh_type : :class:`str`, optional
            'structured' / 'unstructured'. Default: 'unstructured'
        store : :class:`str` or :class:`bool`, optional
            Whether to store field (True/False) with default name
            or with specified name.
            The default is :any:`True` for default name "field".

        Returns
        -------
        field : :class:`numpy.ndarray`
            the labels, ``background`` where the geometry assigns no label.
        """
        name, save = self.get_store_config(store)
        if pos is not None:
            self.set_pos(pos, mesh_type)
        elif self.pos is None:
            raise ValueError("FaciesField: no position tuple 'pos' present")
        lab = self._geometry.labels(self.pos, self.mesh_type)
        lab = np.where(np.isfinite(lab), lab, self.background)
        return self.post_field(lab, name, process=False, save=save)

    def set_pos(self, pos, mesh_type="unstructured", info=False):
        """Set positions and mesh_type.

        Clears the geometry cache if the positions changed.

        Parameters
        ----------
        pos : :any:`iterable`
            the position tuple, containing main direction and transversal
            directions
        mesh_type : :class:`str`, optional
            'structured' / 'unstructured'
            Default: `"unstructured"`
        info : :class:`bool`, optional
            Whether to return information

        Returns
        -------
        info : :class:`dict`, optional
            Information about settings.
        """
        # Note: deliberately does not call `pre_pos` -- for structured
        # meshes that would materialise the full (d, n) grid only to
        # discard it, and its other job (model.isometrize) is a no-op
        # here since model is None.
        info_ret = super().set_pos(pos, mesh_type, info=True)
        if info_ret["deleted"]:
            self._geometry.clear_cache()
        return info_ret if info else None

    def post_field(self, field, name="field", process=False, save=True):
        """Store a label field, preserving integer labels.

        Unlike :any:`Field.post_field`, this never applies
        mean/normalizer/trend processing (``process`` is ignored).

        Parameters
        ----------
        field : :class:`numpy.ndarray`
            Label values.
        name : :class:`str`, optional
            Name to store the field. The default is "field".
        process : :class:`bool`, optional
            Ignored, labels are never processed.
        save : :class:`bool`, optional
            Whether to store the field under the given name.
            The default is True.

        Returns
        -------
        field : :class:`numpy.ndarray`
            Label values.
        """
        if self.field_shape is None:
            raise ValueError("post_field: no 'field_shape' present.")
        field = np.asarray(field).reshape(self.field_shape)
        dtype = self.dtype
        if dtype is None:
            fin = np.isfinite(field)
            integral = np.all(fin) and np.all(field == np.round(field))
            dtype = np.int64 if integral else np.double
        field = field.astype(dtype)
        if save:
            name = str(name)
            if not name.isidentifier() or (
                name not in self.field_names and name in dir(self)
            ):
                raise ValueError(
                    f"FaciesField: given field name '{name}' is not valid"
                )
            if name not in self._field_names:
                self._field_names.append(name)
            setattr(self, name, field)
        return field

    @property
    def geometry(self):
        """:any:`Region`: The underlying geometry-state object."""
        return self._geometry

    @property
    def names(self):
        """:class:`dict`: Mapping ``{label: name}`` for plot legends."""
        if self._names is not None:
            return dict(self._names)
        return dict(getattr(self._geometry, "names", {}))

    def __repr__(self):
        return f"{self.name}(geometry={self._geometry!r}, dim={self.dim})"
