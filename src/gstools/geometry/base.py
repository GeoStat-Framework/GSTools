"""
GStools subpackage providing the base classes for stochastic geometry.

.. currentmodule:: gstools.geometry.base

The following classes are provided

.. autosummary::
   Region
   Composite
   HalfSpace
   StratColumn
"""

import warnings
from abc import ABC, abstractmethod
from functools import partial

import numpy as np

from gstools.field.base import Field
from gstools.field.cond_srf import CondSRF
from gstools.krige import Krige
from gstools.normalizer.base import Normalizer
from gstools.tools.geometric import generate_grid

__all__ = [
    "NO_LABEL",
    "RELATION",
    "Region",
    "Composite",
    "HalfSpace",
    "StratColumn",
]


NO_LABEL = np.nan
""":class:`float`: Sentinel meaning 'this region assigns no label here'."""

RELATION = ["erode", "onlap"]
""":class:`list` of :class:`str`: Supported relations between groups."""


class Region(ABC):
    """An implicit region: a containment test on ``(dim, n)`` positions.

    Parameters
    ----------
    dim : :class:`int`
        Spatial dimension of the region.
    label : :class:`float`, optional
        Label assigned to points contained in the region. Default: 1.0
    priority : :class:`float`, optional
        Priority used by :any:`Composite` to order overlapping regions;
        higher priority (younger) overwrites lower priority (older).
        Default: 0.0

    Notes
    -----
    :any:`Region.labels` must stay valid under a position transform: no
    cached state may be keyed on anything but ``np.array_equal(pos)``. This
    reserves the seam for a future ``Faulted(region, fault_surface, throw)``
    wrapper that evaluates ``region.labels(pos - throw)`` on the
    hanging-wall side.
    """

    def __init__(self, dim, label=1.0, priority=0.0):
        self._dim = int(dim)
        self.label = float(label)
        self.priority = float(priority)

    @abstractmethod
    def contains(self, pos):
        """Check containment of the given positions.

        Parameters
        ----------
        pos : :class:`numpy.ndarray`
            Positions with shape ``(dim, n)``.

        Returns
        -------
        :class:`numpy.ndarray`
            Boolean array with shape ``(n,)``.
        """

    def labels(self, pos, mesh_type="unstructured"):
        """Evaluate labels at the given positions.

        Parameters
        ----------
        pos : :class:`numpy.ndarray` or :class:`tuple`
            Positions with shape ``(dim, n)``, or a tuple of ``dim`` axes
            for ``mesh_type="structured"``.
        mesh_type : :class:`str`, optional
            'structured' / 'unstructured'. Default: 'unstructured'

        Returns
        -------
        :class:`numpy.ndarray`
            Labels with shape ``(n,)`` (C-order flattened for structured
            grids); ``NO_LABEL`` outside the region.
        """
        if mesh_type == "structured":
            return self._labels_structured(pos)
        pos = np.asarray(pos, dtype=np.double).reshape(self.dim, -1)
        return self._labels_unstructured(pos)

    def _labels_unstructured(self, pos):
        res = np.full(pos.shape[1], NO_LABEL)
        res[self.contains(pos)] = self.label
        return res

    def _labels_structured(self, pos):
        return self._labels_unstructured(generate_grid(pos))

    def clear_cache(self):
        """Drop any cached position-dependent state. No-op by default."""

    @property
    def dim(self):
        """:class:`int`: Spatial dimension of the region."""
        return self._dim

    @property
    def name(self):
        """:class:`str`: Class name of the region."""
        return self.__class__.__name__

    def __repr__(self):
        return f"{self.name}(dim={self.dim}, label={self.label})"


class Composite(Region):
    """Priority-ordered union of regions.

    Higher priority (younger) regions overwrite lower priority (older)
    regions wherever they overlap.

    Parameters
    ----------
    parts : :any:`Region` or :class:`list` of :any:`Region`
        The regions to compose.
    dim : :class:`int`, optional
        Spatial dimension. Default: dimension of the first part.
    priority : :class:`float`, optional
        Priority of the composite itself, for nesting. Default: 0.0
    """

    def __init__(self, parts, dim=None, priority=0.0):
        parts = [parts] if isinstance(parts, Region) else list(parts)
        if not parts:
            raise ValueError("Composite: needs at least one region.")
        if len({p.dim for p in parts}) > 1:
            raise ValueError("Composite: all regions need the same dim.")
        super().__init__(
            parts[0].dim if dim is None else dim,
            label=NO_LABEL,
            priority=priority,
        )
        self._parts = tuple(parts)

    @property
    def parts(self):
        """:class:`tuple` of :any:`Region`: Parts in ascending priority.

        Ties keep insertion order.
        """
        return tuple(sorted(self._parts, key=lambda p: p.priority))

    def contains(self, pos):
        """Check containment of the given positions.

        See :any:`Region.contains`
        """
        return np.isfinite(self.labels(pos))

    def _labels_unstructured(self, pos):
        return self._compose(lambda p: p.labels(pos), pos.shape[1])

    def _labels_structured(self, pos):
        size = int(np.prod([len(np.atleast_1d(p)) for p in pos]))
        return self._compose(lambda p: p.labels(pos, "structured"), size)

    def clear_cache(self):
        """Drop cached position-dependent state of all parts."""
        for part in self._parts:
            part.clear_cache()

    def _compose(self, evaluate, size):
        """Evaluate and stack all parts in ascending priority order."""
        res = np.full(size, NO_LABEL)
        for part in self.parts:  # ascending priority
            lab = evaluate(part)
            fin = np.isfinite(lab)
            res[fin] = lab[fin]
        return res


class HalfSpace(Region):
    """The region above (or below) a lateral surface.

    Contains all points with ``z >= h(x)`` (or ``z < h(x)`` if
    ``below=True``), where ``z`` is the last component of ``pos``.
    Serves as topography/air, as an unconditional erosion surface and as
    the building block of :class:`gstools.geometry.SurfaceStack`.

    Parameters
    ----------
    surface : :any:`Field`, :any:`callable`, :class:`float` or :class:`numpy.ndarray`
        Elevation ``h`` of the surface on the ``(dim - 1)``-dimensional
        lateral domain. A random :any:`Field` must have
        ``model.nugget == 0``.
    dim : :class:`int`, optional
        Spatial dimension of the region. Default: 3
    label : :class:`float`, optional
        Label assigned inside the half space. Default: 1.0
    priority : :class:`float`, optional
        Priority for :any:`Composite`. Default: 0.0
    below : :class:`bool`, optional
        Whether the region lies below the surface. Default: False
    """

    def __init__(self, surface, dim=3, label=1.0, priority=0.0, below=False):
        super().__init__(dim, label=label, priority=priority)
        if self.dim < 2:
            raise ValueError("HalfSpace: needs dim >= 2.")
        self._surface = surface
        self._below = bool(below)
        self._cache = _LateralCache()
        _check_lateral(surface, self.dim - 1, "HalfSpace surface")

    def contains(self, pos):
        """Check containment of the given positions.

        See :any:`Region.contains`
        """
        lat, vert = _split_pos(pos, self.dim, structured=False)
        return self._contains(lat, vert)

    def _labels_structured(self, pos):
        """The surface is evaluated at the lateral grid only."""
        lat, vert = _split_pos(pos, self.dim, structured=True)
        cont = self._contains(lat, vert[None, :], lat_axis=True)
        res = np.full(cont.shape, NO_LABEL)
        res[cont] = self.label
        return res.ravel()

    def clear_cache(self):
        """Drop the cached surface elevations."""
        self._cache.clear()

    def _contains(self, lat, vert, lat_axis=False):
        _check_lateral(self._surface, self.dim - 1, "HalfSpace surface")
        dedupe = not lat_axis and _pointwise(self._surface)
        surf = self._cache(partial(_eval_lateral, self._surface), lat, dedupe)
        surf = surf[:, None] if lat_axis else surf
        above = vert >= surf
        return ~above if self._below else above

    @property
    def surface(self):
        """:any:`Field` or :any:`callable`: The lateral surface elevation.

        Can also be a :class:`float` or :class:`numpy.ndarray`.
        """
        return self._surface

    @property
    def below(self):
        """:class:`bool`: Whether the region lies below the surface."""
        return self._below


class StratColumn(Region):
    """Groups from oldest to youngest, combined by stratigraphic relations.

    Every point is classified per group as *below*, *inside* or *above*
    the group, and the groups are applied from oldest to youngest:

    ======== ==========================================================
    Relation Effect of the younger group
    ======== ==========================================================
    (oldest) inside: its labels. below: basement. above: free space
    erode    inside: overwrites everything. above: erases to free space
    onlap    inside: fills free space only. older units are kept
    ======== ==========================================================

    Free space remaining at the end gets ``air_label``. Points above an
    optional ``topography`` are set to ``air_label`` as well.

    Labels are assigned consecutively across groups (``0 .. n - 1``,
    oldest group first) unless a layer or surface carries an explicit
    ``label``. A generic :any:`Region` group is *inside* where it assigns
    a label and *below* elsewhere.

    Parameters
    ----------
    groups : :class:`list` of :any:`Region`
        Groups (e.g. :class:`gstools.geometry.LayerStack` or
        :class:`gstools.geometry.SurfaceStack`), oldest first.
    relations : :class:`list` of :class:`str`
        One relation from ``RELATION`` per consecutive pair of
        groups, i.e. ``len(groups) - 1`` entries.
    topography : :any:`Field`, :any:`callable`, :class:`float` or :class:`numpy.ndarray`, optional
        Surface above which everything is air. Default: None
    basement_label : :class:`float`, optional
        Label below the oldest group. Default: -1
    air_label : :class:`float`, optional
        Label of free space and air. Default: total number of layers
    priority : :class:`float`, optional
        Priority of the column itself, for nesting. Default: 0.0

    Raises
    ------
    ValueError
        If ``len(relations) != len(groups) - 1``, a relation is unknown or
        the groups have different dimensions.

    Notes
    -----
    Plain :any:`Composite` priorities with ``NO_LABEL`` outside labels
    express a single erode or onlap contact between two groups, but not
    mixed chains: an onlap group above an erosive one would let the eroded
    older group reappear above the erosive group's top.
    """

    def __init__(
        self,
        groups,
        relations=None,
        topography=None,
        basement_label=-1,
        air_label=None,
        priority=0.0,
    ):
        groups = [groups] if isinstance(groups, Region) else list(groups)
        if not groups:
            raise ValueError("StratColumn: needs at least one group.")
        if len({g.dim for g in groups}) > 1:
            raise ValueError("StratColumn: all groups need the same dim.")
        relations = [] if relations is None else list(relations)
        if len(relations) != len(groups) - 1:
            raise ValueError(
                "StratColumn: need exactly len(groups) - 1 relations, "
                f"got {len(relations)} for {len(groups)} groups."
            )
        for rel in relations:
            if rel not in RELATION:
                raise ValueError(
                    f"StratColumn: unknown relation '{rel}', "
                    f"use one of {RELATION}."
                )
        super().__init__(groups[0].dim, label=NO_LABEL, priority=priority)
        self._groups = tuple(groups)
        self._relations = tuple(relations)
        self._topography = topography
        self._topo_cache = _LateralCache()
        if topography is not None:
            _check_lateral(topography, self.dim - 1, "topography")
        # consecutive labels across groups
        self._group_labels = []
        offset = 0
        for grp in groups:
            n_lay = getattr(grp, "n_layers", None)
            if n_lay is None:
                self._group_labels.append(None)
                continue
            labs = [
                offset + i if lab is None else lab
                for i, lab in enumerate(grp._explicit_labels())
            ]
            self._group_labels.append(np.array(labs, dtype=np.double))
            offset += n_lay
        self._basement_label = basement_label
        self._air_label = offset if air_label is None else air_label

    def contains(self, pos):
        """Check containment of the given positions.

        See :any:`Region.contains`
        """
        return np.isfinite(self.labels(pos))

    def _labels_unstructured(self, pos):
        return self._compose(pos, structured=False)

    def _labels_structured(self, pos):
        return self._compose(pos, structured=True)

    def clear_cache(self):
        """Drop cached position-dependent state of all groups."""
        self._topo_cache.clear()
        for grp in self._groups:
            grp.clear_cache()

    def _zones(self, grp, labs, pos, structured):
        """Classify points into (inside, below, above) and inside labels."""
        if labs is None:  # generic region
            mesh_type = "structured" if structured else "unstructured"
            lab = grp.labels(pos, mesh_type)
            inside = np.isfinite(lab)
            return inside, ~inside, np.zeros_like(inside), lab
        idx, outside = grp._index(pos, structured)
        idx, outside = idx.ravel(), outside.ravel()
        n_lay = grp.n_layers
        inside = (idx >= 0) & (idx < n_lay) & ~outside
        below = (idx < 0) | outside
        above = (idx >= n_lay) & ~outside
        lab = np.full(idx.shape, NO_LABEL)
        lab[inside] = labs[idx[inside]]
        return inside, below, above, lab

    def _compose(self, pos, structured):
        res = None
        free = None
        rels = (None,) + self._relations
        for grp, labs, rel in zip(self._groups, self._group_labels, rels):
            inside, below, above, lab = self._zones(grp, labs, pos, structured)
            if res is None:  # oldest group
                res = np.full(inside.shape, NO_LABEL)
                free = np.ones(inside.shape, dtype=bool)
                res[inside] = lab[inside]
                res[below] = self._basement_label
                free[inside | below] = False
            elif rel == "erode":
                res[inside] = lab[inside]
                free[inside] = False
                free[above] = True
            else:  # onlap
                sel = inside & free
                res[sel] = lab[sel]
                free[sel] = False
        res[free] = self._air_label
        if self._topography is not None:
            lat, vert = _split_pos(pos, self.dim, structured)
            dedupe = not structured and _pointwise(self._topography)
            topo = self._topo_cache(
                partial(_eval_lateral, self._topography), lat, dedupe
            )
            if structured:
                air = (vert[None, :] >= topo[:, None]).ravel()
            else:
                air = vert >= topo
            res[air] = self._air_label
        return res

    @property
    def groups(self):
        """:class:`tuple` of :any:`Region`: The groups, oldest first."""
        return self._groups

    @property
    def relations(self):
        """:class:`tuple` of :class:`str`: Relations between groups."""
        return self._relations

    @property
    def names(self):
        """:class:`dict`: Mapping ``{label: name}`` of all named layers."""
        res = {}
        for grp, labs in zip(self._groups, self._group_labels):
            if labs is None:
                continue
            for lab, name in zip(labs.tolist(), grp._layer_names()):
                if name is not None:
                    res[lab] = name
        return res

    def __repr__(self):
        return (
            f"{self.name}(groups={len(self._groups)}, "
            f"relations={list(self._relations)})"
        )


def _split_pos(pos, dim, structured):
    """Split positions into lateral positions and vertical coordinates.

    Returns ``(lat, vert)`` with ``lat`` of shape ``(dim - 1, n_lat)``.
    For structured input ``vert`` holds the vertical axis only and
    ``n_lat`` is the number of lateral grid points (C-order).
    """
    if structured:
        lat = generate_grid(tuple(pos[:-1]))
        return lat, np.asarray(pos[-1], dtype=np.double).ravel()
    pos = np.asarray(pos, dtype=np.double).reshape(dim, -1)
    return pos[:-1], pos[-1]


def _is_random(obj):
    """Whether the object is a random field with a generator."""
    return isinstance(obj, Field) and hasattr(obj, "generator")


def _check_lateral(obj, lat_dim, name):
    """Validate a lateral field/base object (dim, nugget, temporal, latlon).

    Called at construction and again on every evaluation, since
    :any:`CovModel` attributes are mutable.
    """
    if not isinstance(obj, Field):
        return
    if obj.dim != lat_dim:
        raise ValueError(
            f"{name}: field has dim={obj.dim}, but the lateral domain "
            f"needs dim={lat_dim}."
        )
    model = obj.model
    if model is None:
        return
    if _is_random(obj) and model.nugget > 0:
        raise ValueError(
            f"{name}: model.nugget must be 0 (got {model.nugget}). A nugget "
            "draws new white noise on every evaluation, which breaks "
            "pointwise consistency of the realization, and in a height "
            "function it is an interface that jumps at every point."
        )
    if model.temporal:
        raise ValueError(f"{name}: spatio-temporal models are not supported.")
    if model.latlon:
        raise ValueError(
            f"{name}: latlon models are not supported, since the vertical "
            "unit is undefined relative to geo_scale."
        )
    if type(obj.normalizer) is not Normalizer:
        warnings.warn(
            f"{name}: a normalizer is set: layer and surface fields should "
            "live in pure Gaussian space (normalizer=None)."
        )


def _eval_lateral(obj, lat):
    """Evaluate a lateral field/callable/scalar at lateral positions.

    Parameters
    ----------
    obj : :any:`Field`, :any:`callable`, :class:`float` or :class:`numpy.ndarray`
        Object to evaluate.
    lat : :class:`numpy.ndarray`
        Lateral positions, shape ``(dim - 1, n)``.

    Returns
    -------
    :class:`numpy.ndarray`
        Values, shape ``(n,)``.
    """
    lat = np.asarray(lat, dtype=np.double)
    if isinstance(obj, Field):
        if lat.shape[1] == 0:
            return np.empty(0, dtype=np.double)
        # keyword arguments only: the second positional one of SRF is seed
        kw = {"pos": lat, "mesh_type": "unstructured", "store": False}
        if isinstance(obj, CondSRF):
            kw["krige_store"] = False
        elif isinstance(obj, Krige):
            kw["return_var"] = False
        return np.asarray(obj(**kw), dtype=np.double).reshape(-1)
    if callable(obj):
        res = np.asarray(obj(*lat), dtype=np.double)
        return np.broadcast_to(res, lat.shape[1]).astype(np.double)
    return np.broadcast_to(obj, lat.shape[1]).astype(np.double)


def _pointwise(obj):
    """Whether a lateral object can be evaluated at any subset of positions.

    Arrays with one value per lateral position are tied to the given
    positions and must not be evaluated at a deduplicated subset.
    """
    return callable(obj) or np.ndim(obj) == 0


def _eval_distinct(func, lat):
    """Evaluate ``func`` at the distinct lateral positions only.

    Parameters
    ----------
    func : :any:`callable`
        Maps lateral positions ``(dim - 1, m)`` to values ``(..., m)``.
    lat : :class:`numpy.ndarray`
        Lateral positions, shape ``(dim - 1, n)``.

    Returns
    -------
    :class:`numpy.ndarray`
        Values, shape ``(..., n)``.
    """
    if lat.shape[1] < 2:
        return func(lat)
    uni, inv = np.unique(lat, axis=1, return_inverse=True)
    if uni.shape[1] == lat.shape[1]:
        return func(lat)
    return func(uni)[..., inv.reshape(-1)]


class _LateralCache:
    """One-entry cache of values on lateral positions.

    Keyed by ``np.array_equal`` of the lateral positions only, as required
    by :any:`Region`. With ``dedupe``, values are computed at the distinct
    lateral positions only, e.g. once per vertical column of an
    unstructured mesh.
    """

    def __init__(self):
        self.clear()

    def clear(self):
        """Drop the cached values."""
        self._lat = None
        self._val = None

    def __call__(self, func, lat, dedupe=False):
        if self._lat is not None and np.array_equal(self._lat, lat):
            return self._val
        val = _eval_distinct(func, lat) if dedupe else func(lat)
        self._lat, self._val = np.array(lat, copy=True), val
        return val
