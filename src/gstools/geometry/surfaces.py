"""
GStools subpackage providing elevation-parameterised stochastic geometry.

.. currentmodule:: gstools.geometry.surfaces

The following classes are provided

.. autosummary::
   Surface
   SurfaceStack
"""

import numpy as np

from gstools.field import SRF
from gstools.geometry.base import RELATION
from gstools.geometry.layers import _identity, _Lateral, _Stack

__all__ = ["Surface", "SurfaceStack"]


class Surface(_Lateral):
    """One stratigraphic surface: an elevation field.

    Parameters
    ----------
    field : :any:`Field`, :any:`callable` or :class:`float`
        Elevation of the surface in pure Gaussian space (``mean`` = mean
        elevation). A :any:`Field` must have ``dim == d - 1``,
        ``model.nugget == 0`` and ``normalizer=None``. Only
        :any:`SRF`/:any:`CondSRF` backed surfaces can be conditioned.
    label : :class:`float`, optional
        Label of the layer *above* this surface (ignored for the top
        surface). Default: index in the stack.
    name : :class:`str`, optional
        Name of the layer above this surface, for error messages, legends
        and ``__repr__``.
    """

    def __init__(self, field, label=None, name=None):
        super().__init__(field, name=name)
        self.label = label

    @classmethod
    def from_model(
        cls, model, mean=0.0, seed=None, generator="RandMeth", **kwargs
    ):
        """Build a :any:`Surface` from a :any:`CovModel`.

        Constructs an :any:`SRF` with ``dim = model.dim``,
        ``normalizer=None`` and the given ``mean`` (mean elevation).

        Parameters
        ----------
        model : :any:`CovModel`
            Covariance model for the elevation field. Must have
            ``nugget == 0``.
        mean : :class:`float` or :any:`callable`, optional
            Mean elevation. Default: 0.0
        seed : :class:`int`, optional
            Seed for the field generator.
        generator : :class:`str`, optional
            Generator passed through to :any:`SRF`. Default: "RandMeth"
        **kwargs
            Passed through to :any:`Surface`.

        Returns
        -------
        :any:`Surface`
            The surface holding the new :any:`SRF`.
        """
        srf = SRF(model, mean=mean, seed=seed, generator=generator)
        return cls(srf, **kwargs)

    def elevation(self, lat):
        """Evaluate the underlying elevation field.

        Parameters
        ----------
        lat : :class:`numpy.ndarray`
            Lateral positions, shape ``(dim - 1, n)``.

        Returns
        -------
        :class:`numpy.ndarray`
            Elevations, shape ``(n,)``.
        """
        return self.evaluate(lat)

    def __repr__(self):
        return f"Surface(name={self.name!r})"


class SurfaceStack(_Stack):
    """A stochastic stack of stratigraphic surfaces.

    Surfaces ``h_0 .. h_K`` bound ``K`` layers, ``h_0`` is the base.
    Ordering is exact (pure comparisons, no arithmetic):

    - ``relation="onlap"``: ``f_0 = h_0``, ``f_i = max(f_{i-1}, h_i)``
    - ``relation="erode"``: ``f_K = h_K``, ``f_i = min(f_{i+1}, h_i)``

    Complementary to :class:`gstools.geometry.LayerStack`, which puts the
    covariance on thicknesses. Use this class when the data are contact
    elevations. Every such datum is then an exact equality or a box
    constraint on a single surface, and pinch-outs are true zeros.
    Surfaces are independent fields. Correlated (parallel) bedding is
    better modelled by a :class:`gstools.geometry.LayerStack`.

    Realization identity is ``(derived per-surface seeds, sampled
    inequality values, frozen measurement-error draws)``. :any:`reset` is
    the only mutator that changes it.

    Parameters
    ----------
    surfaces : :class:`list` of :any:`Surface`
        Surfaces, ordered from bottom (base) to top. Other objects are
        wrapped into a :any:`Surface`.
    relation : :class:`str`, optional
        Either "onlap" or "erode". Default: "onlap"
    dim : :class:`int`, optional
        Spatial dimension of the stack (lateral dim is ``dim - 1``). The
        vertical axis is the last component of ``pos``. Default: 3
    below_label : :class:`float`, optional
        Label below the base. ``NO_LABEL`` makes it transparent.
        Default: -1
    above_label : :class:`float`, optional
        Label above the top. ``NO_LABEL`` makes it transparent.
        Default: ``n_layers``
    mask : :class:`numpy.ndarray` or :any:`callable`, optional
        Lateral mask, outside the mask evaluates to ``outside_label``.
    outside_label : :class:`float`, optional
        Label outside ``mask``. Default: ``NO_LABEL``
    seed : :class:`int`, optional
        Master seed for per-surface seed derivation. If :any:`None`, the
        fields keep their own seeds.
    priority : :class:`float`, optional
        Priority of the stack as a whole, for nesting inside a
        :any:`gstools.geometry.base.Composite`. Default: 0.0

    Warnings
    --------
    Mutating a surface directly (e.g. ``surface.field.model.len_scale =
    ...``, or calling ``srf(pos, seed=...)``) is outside this class's
    control and silently changes the realization.
    """

    _kind = "elevation"

    def __init__(
        self,
        surfaces,
        relation="onlap",
        dim=3,
        below_label=-1,
        above_label=None,
        mask=None,
        outside_label=None,
        seed=None,
        priority=0.0,
    ):
        self._surfaces = [
            srf if isinstance(srf, Surface) else Surface(srf)
            for srf in surfaces
        ]
        if len(self._surfaces) < 2:
            raise ValueError("SurfaceStack: needs at least two surfaces.")
        if relation not in RELATION:
            raise ValueError(
                f"SurfaceStack: relation must be one of {RELATION}."
            )
        self._relation = relation
        super().__init__(
            dim,
            below_label,
            above_label,
            mask,
            outside_label,
            seed,
            priority,
        )

    def elevations(self, lat):
        """Evaluate all raw surface elevations at lateral positions.

        Parameters
        ----------
        lat : :class:`numpy.ndarray`
            Lateral positions, shape ``(dim - 1, n)``.

        Returns
        -------
        :class:`numpy.ndarray`
            Raw elevations ``h``, shape ``(n_layers + 1, n)``, unordered.
        """
        lat = np.asarray(lat, dtype=np.double).reshape(self.dim - 1, -1)
        self._check()
        return np.array([srf.elevation(lat) for srf in self._surfaces])

    def residuals(self, data):
        """Residuals of the current realization per datum.

        Equalities give ``|f_j - z_obs|`` on the *ordered* interfaces, so
        an inconsistency with the ordering shows up. Inequalities give the
        bound violation of the raw surface ``h_j`` (``0`` if honoured).
        Offending wells: ``data.well[res > tol]``.

        Parameters
        ----------
        data : :any:`gstools.geometry.observations.BoreholeData`
            Conditioning data of the stack, e.g. the data given to
            :any:`condition`.

        Returns
        -------
        :class:`numpy.ndarray`
            Residual per datum, shape ``(len(data),)``.
        """
        res = np.zeros(len(data))
        for j in np.unique(data.layer):
            sel = data.layer == j
            pos, val = data.pos[:, sel], data.value[sel]
            eq = np.isfinite(val)
            ordered = self.interfaces(pos)[j]
            raw = self._surfaces[j].elevation(pos)
            viol = np.maximum(
                np.maximum(data.lower[sel] - raw, raw - data.upper[sel]), 0.0
            )
            res[sel] = np.where(eq, np.abs(ordered - val), viol)
        return res

    def _compute_ifaces(self, lat):
        return self._order(self.elevations(lat))

    def _order(self, h):
        """Order raw elevations by cumulative max (onlap) or min (erode).

        Parameters
        ----------
        h : :class:`numpy.ndarray`
            Raw elevations, shape ``(n_layers + 1, n)``.

        Returns
        -------
        :class:`numpy.ndarray`
            Interfaces, same shape, increasing along axis 0.
        """
        if self._relation == "onlap":
            return np.maximum.accumulate(h, axis=0)
        return np.minimum.accumulate(h[::-1], axis=0)[::-1]

    def _laterals(self):
        return self._surfaces

    def _units(self):
        return self._surfaces[:-1]

    def _targets(self):
        return {j: (srf, _identity) for j, srf in enumerate(self._surfaces)}

    def _target_name(self, i):
        return f"surface {i}"

    @property
    def n_layers(self):
        """:class:`int`: Number of layers in the stack."""
        return len(self._surfaces) - 1

    @property
    def surfaces(self):
        """:class:`list` of :any:`Surface`: The surfaces, bottom to top."""
        return self._surfaces

    @property
    def relation(self):
        """:class:`str`: Either "onlap" or "erode"."""
        return self._relation
