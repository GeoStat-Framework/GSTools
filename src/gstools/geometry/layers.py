"""
GStools subpackage providing layered stochastic geometry.

.. currentmodule:: gstools.geometry.layers

The following classes are provided

.. autosummary::
   Layer
   LayerStack
"""

import warnings
from abc import abstractmethod

import numpy as np
from scipy.spatial.distance import cdist

from gstools.field import SRF, CondSRF
from gstools.field.base import Field
from gstools.field.generator import RandMeth
from gstools.geometry.base import (
    NO_LABEL,
    Region,
    _check_lateral,
    _eval_lateral,
    _is_random,
    _LateralCache,
    _pointwise,
    _split_pos,
)
from gstools.geometry.gibbs import gibbs_sample
from gstools.geometry.thickness import LogLink, set_link
from gstools.krige import Krige
from gstools.random import MasterRNG
from gstools.tools.misc import eval_func

__all__ = ["Layer", "LayerStack"]

COND_TOL = 1e10
""":class:`float`: Condition number of the data covariance to warn above."""

TAIL_WARN = 5.0
""":class:`float`: Prior standard deviations beyond which a bound warns."""


class _Lateral:
    """A lateral object of a stack (layer, surface or base).

    Holds the user-given object (the *prior*) and the object currently
    evaluated, which is a :any:`CondSRF` built from the prior after
    conditioning.

    Conditioning never mutates the prior, but replaces the evaluated
    object. Keeping both allows re-conditioning on new data from the
    prior (instead of stacking conditioned fields), dropping the data
    again (:any:`uncondition`) and reseeding prior and conditioned field
    together, so that the conditioned field stays the conditioned version
    of the same realization. The stored data allow redrawing the Gibbs
    imputation (:any:`resample`) without passing them again.

    It also gives :any:`LayerStack` and :any:`SurfaceStack` one interface
    (evaluate, check, seed, purge, condition) for all kinds of lateral
    objects: fields, callables, scalars and arrays.
    """

    def __init__(self, field, name=None):
        self._prior = field
        self._field = field
        self._cond = None
        self.name = name

    def evaluate(self, lat):
        """Evaluate the current field at lateral positions."""
        return _eval_lateral(self._field, lat)

    def check(self, lat_dim, name):
        """Validate prior and current field."""
        _check_lateral(self._prior, lat_dim, name)
        if self._field is not self._prior:
            _check_lateral(self._field, lat_dim, name)

    def set_seed(self, seed):
        """Set the generator seed of prior and current field."""
        for fld in self._objects():
            if _is_random(fld):
                fld.generator.seed = seed
        self.purge()

    def purge(self):
        """Delete stored fields on the fields and their kriging setups."""
        for fld in self._objects():
            if isinstance(fld, Field):
                fld.delete_fields()
            if isinstance(fld, CondSRF):
                fld.krige.delete_fields()

    def _objects(self):
        if self._field is self._prior:
            return [self._prior]
        return [self._prior, self._field]

    @property
    def is_random(self):
        """:class:`bool`: Whether the prior is a random field."""
        return _is_random(self._prior)

    @property
    def seed(self):
        """:class:`int` or :any:`None`: Generator seed of the prior."""
        return self._prior.generator.seed if self.is_random else None

    @property
    def conditioned(self):
        """:class:`bool`: Whether conditioning data is set."""
        return self._cond is not None

    def condition(
        self, pos, val, low, upp, wells, label, gibbs, seed, krige_kw
    ):
        """Condition on equality and inequality data (Gaussian space)."""
        prior = self._prior
        if not isinstance(prior, (SRF, CondSRF)):
            raise ValueError(
                f"{label}: only SRF or CondSRF backed objects can be "
                f"conditioned, got {type(prior).__name__}."
            )
        krige_kw = dict(krige_kw)
        ineq = ~np.isfinite(val)
        if isinstance(prior, CondSRF):
            unbiased = prior.krige.unbiased
            drift = prior.krige.int_drift_no > 0
        else:
            unbiased = bool(krige_kw.get("unbiased", False))
            drift = bool(krige_kw.get("drift_functions")) or (
                krige_kw.get("ext_drift") is not None
            )
        if np.any(ineq) and (unbiased or drift):
            raise ValueError(
                f"{label}: inequality data require simple kriging with an "
                "explicit mean (unbiased=False, no drift). With an unknown "
                "mean the prior law at the data is not the covariance used "
                "by the Gibbs sampler."
            )
        self._cond = {
            "pos": pos,
            "val": val,
            "low": low,
            "upp": upp,
            "wells": wells,
            "label": label,
            "gibbs": dict(gibbs or {}),
            "krige_kw": krige_kw,
        }
        self._apply(seed)

    def resample(self, seed):
        """Redraw the Gibbs imputation of the inequality data."""
        if self._cond is not None and np.any(~np.isfinite(self._cond["val"])):
            self._apply(seed)

    def uncondition(self):
        """Drop conditioning data (a user-given CondSRF is kept as is)."""
        self._cond = None
        if not isinstance(self._prior, CondSRF):
            self._field = self._prior
        self.purge()

    def _apply(self, seed):
        cond = self._cond
        prior = self._prior
        pos, val = cond["pos"], cond["val"]
        _warn_conditioning(prior.model, pos, cond["wells"], cond["label"])
        if np.any(~np.isfinite(val)):
            dim = prior.model.field_dim
            mean = eval_func(prior.mean, pos, dim) + eval_func(
                prior.trend, pos, dim
            )
            _warn_extreme_bounds(prior.model, mean, val, cond, TAIL_WARN)
            val = gibbs_sample(
                prior.model,
                pos,
                val,
                cond["low"],
                cond["upp"],
                mean=mean,
                seed=seed,
                **cond["gibbs"],
            )
        if isinstance(prior, CondSRF):
            prior.krige.set_condition(pos, val)
            if prior.cond_method != "error":
                warnings.warn(
                    f"{cond['label']}: CondSRF uses cond_method="
                    f"'{prior.cond_method}', which does not reproduce the "
                    "data exactly. Use cond_method='error'."
                )
            field = prior
        else:
            gen = prior.generator
            if type(gen) is not RandMeth:
                raise NotImplementedError(
                    f"{cond['label']}: conditioning is only supported for "
                    "the RandMeth generator."
                )
            kw = {"unbiased": False}
            kw.update(cond["krige_kw"])
            krige = Krige(
                prior.model,
                pos,
                val,
                mean=prior.mean,
                trend=prior.trend,
                **kw,
            )
            field = CondSRF(
                krige,
                cond_method="error",
                seed=gen.seed,
                mode_no=gen.mode_no,
                sampling=gen.sampling,
            )
        field.delete_fields()
        field.krige.delete_fields()
        if field.krige.cond_no != len(val):
            raise RuntimeError(
                f"{cond['label']}: {len(val)} data given, but the kriging "
                f"system holds {field.krige.cond_no} non-finite values "
                "must never reach Krige."
            )
        self._field = field

    @property
    def field(self):
        """:any:`Field`, :any:`callable` or :class:`float`: Evaluated object.

        A :any:`CondSRF` built from the prior if data is set.
        """
        return self._field

    @property
    def prior(self):
        """:any:`Field`, :any:`callable` or :class:`float`: The prior."""
        return self._prior


class Layer(_Lateral):
    """One stratigraphic layer: a log-thickness field plus a link.

    Parameters
    ----------
    field : :any:`Field`, :any:`callable` or :class:`float`
        Log-thickness of the layer. A :any:`Field` must have
        ``dim == d - 1``, ``model.nugget == 0`` and ``normalizer=None``.
        Only :any:`SRF`/:any:`CondSRF` backed layers can be conditioned.
    link : :class:`str` or :any:`gstools.geometry.thickness.ThicknessLink`, optional
        Thickness link. Default: "log"
    label : :class:`float`, optional
        Label assigned to this layer. Default: index in the stack.
    name : :class:`str`, optional
        Human readable name for error messages and ``__repr__``.

    Notes
    -----
    There is deliberately no per-layer erosion flag: thicknesses are
    ``>= 0`` and cumulated, so a layer can never cut below itself. Erosion
    and onlap are relations *between groups*, see
    :class:`gstools.geometry.StratColumn`.
    """

    def __init__(self, field, link="log", label=None, name=None):
        super().__init__(field, name=name)
        self._link = set_link(link)
        self.label = label

    @classmethod
    def from_model(
        cls, model, mean=0.0, seed=None, generator="RandMeth", **kwargs
    ):
        """Build a :any:`Layer` from a :any:`CovModel`.

        Constructs an :any:`SRF` with ``dim = model.dim``,
        ``normalizer=None`` and the given Gaussian ``mean``, living in
        pure Gaussian (log-thickness) space.

        Parameters
        ----------
        model : :any:`CovModel`
            Covariance model for the log-thickness field. Must have
            ``nugget == 0``.
        mean : :class:`float`, optional
            Gaussian mean of the log-thickness field, see
            :any:`LogLink.from_moments` or :any:`LogLink.from_percentiles`.
            Default: 0.0
        seed : :class:`int`, optional
            Seed for the field generator.
        generator : :class:`str`, optional
            Generator passed through to :any:`SRF`. Default: "RandMeth"
        **kwargs
            Passed through to :any:`Layer`.

        Returns
        -------
        :any:`Layer`
            The layer holding the new :any:`SRF`.
        """
        srf = SRF(model, mean=mean, seed=seed, generator=generator)
        return cls(srf, **kwargs)

    def gauss(self, lat):
        """Evaluate the underlying log-thickness field.

        Parameters
        ----------
        lat : :class:`numpy.ndarray`
            Lateral positions, shape ``(dim - 1, n)``.

        Returns
        -------
        :class:`numpy.ndarray`
            Gaussian (log-thickness) values, shape ``(n,)``.
        """
        return self.evaluate(lat)

    def thickness(self, lat):
        """Evaluate the thickness of the layer.

        Parameters
        ----------
        lat : :class:`numpy.ndarray`
            Lateral positions, shape ``(dim - 1, n)``.

        Returns
        -------
        :class:`numpy.ndarray`
            Thicknesses, shape ``(n,)``, all ``>= 0``.
        """
        return np.maximum(self._link.thickness(self.gauss(lat)), 0.0)

    @property
    def link(self):
        """:any:`ThicknessLink`: The thickness link."""
        return self._link

    def _fmt_thickness(self):
        """Both parameterisations, to catch 'mean=5 meaning 5 m' errors."""
        prior = self._prior
        if not isinstance(self._link, LogLink):
            return ""
        if isinstance(prior, Field) and prior.model is not None:
            if callable(prior.mean) or np.size(prior.mean) != 1:
                return ""
            mu = 0.0 if prior.mean is None else float(prior.mean)
            var = prior.model.var
            mean_t = np.exp(mu + var / 2.0)
            return (
                f", mean_thickness={mean_t:.4g} "
                f"(mu_g={mu:.4g}, var_g={var:.4g})"
            )
        if not callable(prior) and np.size(prior) == 1:
            return f", thickness={np.exp(float(prior)):.4g} (g={prior})"
        return ""

    def __repr__(self):
        return (
            f"Layer(name={self.name!r}, link={self._link.name}"
            f"{self._fmt_thickness()})"
        )


class _LayerRegion(Region):
    """A :any:`Region` view of a single layer of a stack."""

    def __init__(self, stack, i):
        super().__init__(
            stack.dim, label=stack._unit_labels()[i], priority=stack.priority
        )
        self._stack = stack
        self._i = int(i)

    def contains(self, pos):
        idx, out = self._stack._index(pos, structured=False)
        return (idx == self._i) & ~out

    def _labels_structured(self, pos):
        idx, out = self._stack._index(pos, structured=True)
        cont = ((idx == self._i) & ~out).ravel()
        res = np.full(cont.shape, NO_LABEL)
        res[cont] = self.label
        return res

    def clear_cache(self):
        self._stack.clear_cache()


class _Stack(Region):
    """Common machinery of :any:`LayerStack` and :any:`SurfaceStack`."""

    _kind = None

    def __init__(
        self,
        dim,
        below_label,
        above_label,
        mask,
        outside_label,
        seed,
        priority,
        sign=1.0,
    ):
        super().__init__(dim, label=NO_LABEL, priority=priority)
        if self.dim < 2:
            raise ValueError(f"{self.name}: needs dim >= 2.")
        self._sign = float(sign)
        self._below_label = below_label
        self._above_label = (
            self.n_layers if above_label is None else above_label
        )
        self._mask = mask
        self._outside_label = (
            NO_LABEL if outside_label is None else outside_label
        )
        self._vertical_axis = -1
        self._iface_cache = _LateralCache()
        self._seed = seed
        self._check()
        if seed is None:
            self._warn_equal_seeds()
        else:
            self._apply_seeds()
        self._purge()

    @abstractmethod
    def _laterals(self):
        """All lateral objects in seeding order."""

    @abstractmethod
    def _targets(self):
        """Mapping ``data index -> (lateral, to_gauss)`` for conditioning."""

    @abstractmethod
    def _units(self):
        """Objects carrying ``label``/``name`` for layers ``0 .. K - 1``."""

    @abstractmethod
    def _compute_ifaces(self, lat):
        """Compute the interfaces in signed space ``s = sign * z``, uncached.

        Increasing along axis 0, shape ``(n_layers + 1, n)``.
        """

    # geometry ################################################################

    def interfaces(self, lat):
        """Evaluate all interface elevations at lateral positions.

        Parameters
        ----------
        lat : :class:`numpy.ndarray`
            Lateral positions, shape ``(dim - 1, n)``.

        Returns
        -------
        :class:`numpy.ndarray`
            Interface elevations, shape ``(n_layers + 1, n)``, monotone
            along axis 0 in world units (increasing for upward stacking).
        """
        lat = np.asarray(lat, dtype=np.double).reshape(self.dim - 1, -1)
        return self._sign * self._ifaces(lat)

    def _labels_unstructured(self, pos):
        idx, out = self._index(pos, structured=False)
        lab = self._from_index(idx)
        lab[out] = self._outside_label
        return lab

    def _labels_structured(self, pos):
        """Fast path: the interfaces are evaluated at the lateral grid only."""
        idx, out = self._index(pos, structured=True)
        lab = self._from_index(idx)
        lab[out] = self._outside_label
        return lab.ravel()

    def contains(self, pos):
        """Check containment of the given positions.

        See :any:`Region.contains`
        """
        return np.isfinite(self.labels(pos))

    def strat_coords(self, pos, mesh_type="unstructured"):
        """Stratigraphic coordinates of the given positions.

        The height-function counterpart of a potential field: the layer
        index and the relative position within that layer,
        ``w = (z - f_i) / (f_{i+1} - f_i)``. A property field evaluated on
        ``(x, y, w)`` follows the layering (proportional gridding).

        Parameters
        ----------
        pos : :class:`numpy.ndarray` or :class:`tuple`
            Positions with shape ``(dim, n)``, or a tuple of axes for
            ``mesh_type="structured"``.
        mesh_type : :class:`str`, optional
            'structured' / 'unstructured'. Default: 'unstructured'

        Returns
        -------
        idx : :class:`numpy.ndarray`
            Layer indices, shape ``(n,)``, ``-1`` below the base,
            ``n_layers`` above the top.
        w : :class:`numpy.ndarray`
            Relative position in ``[0, 1)``, shape ``(n,)``, NaN outside
            the stack or the mask.
        """
        structured = mesh_type != "unstructured"
        lat, vert = _split_pos(pos, self.dim, structured)
        ifaces = self._ifaces(lat, dedupe=not structured)
        s_val = self._sign * vert
        if structured:
            idx = self._layer_index(ifaces[:, :, None], s_val[None, :])
            s_val = np.broadcast_to(s_val[None, :], idx.shape)
            ifaces = ifaces.T[:, :, None]  # (n_lat, K + 1, 1)
            take = np.take_along_axis
            idx_c = np.clip(idx, 0, self.n_layers - 1)[:, None, :]
            low = take(ifaces, idx_c, axis=1)[:, 0, :]
            upp = take(ifaces, idx_c + 1, axis=1)[:, 0, :]
        else:
            idx = self._layer_index(ifaces, s_val)
            idx_c = np.clip(idx, 0, self.n_layers - 1)
            cols = np.arange(idx.size)
            low, upp = ifaces[idx_c, cols], ifaces[idx_c + 1, cols]
        thick = upp - low
        with np.errstate(divide="ignore", invalid="ignore"):
            w_val = np.where(thick > 0, (s_val - low) / thick, 0.0)
        out = self._outside(lat)
        out = out[:, None] if structured else out
        valid = (idx >= 0) & (idx < self.n_layers) & ~out
        w_val = np.where(valid, w_val, np.nan)
        return idx.ravel(), w_val.ravel()

    def region(self, i):
        """A :any:`Region` view of a single layer.

        Parameters
        ----------
        i : :class:`int`
            Layer index.

        Returns
        -------
        :any:`Region`
            Region labelling layer ``i`` only, ``NO_LABEL`` elsewhere.
        """
        if not 0 <= i < self.n_layers:
            raise ValueError(f"{self.name}.region: layer {i} out of range.")
        return _LayerRegion(self, i)

    # state ###################################################################

    def reset(self, seed=np.nan):
        """Draw a fresh realization: new per-field seeds, cleared caches.

        Calls ``delete_fields()`` on every field and its ``.krige`` (if
        any), then re-derives the per-field seeds from a single
        :class:`gstools.random.MasterRNG`. Inequality imputations are kept,
        redraw them with :any:`resample_inequalities`.

        Parameters
        ----------
        seed : :class:`int` or :any:`None` or :any:`numpy.nan`, optional
            Master seed. :any:`None` draws a random master seed,
            :any:`numpy.nan` keeps the current one (if the stack was
            created without a seed, the fields keep their own seeds).
            Default: :any:`numpy.nan`
        """
        keep = isinstance(seed, float) and np.isnan(seed)
        if not keep:
            self._seed = seed
            self._apply_seeds()
        elif self._seed is not None:
            self._apply_seeds()
        self._purge()

    def clear_cache(self):
        """Drop the cached interfaces. Does not change the realization."""
        self._iface_cache.clear()

    def _apply_seeds(self):
        master = MasterRNG(self._seed)
        # one seed per slot, also for deterministic objects, so the
        # seeding scheme does not depend on which objects are random
        for lateral in self._laterals():
            lateral.set_seed(master())

    def _warn_equal_seeds(self):
        seeds = [lat.seed for lat in self._laterals() if lat.is_random]
        seeds = [s for s in seeds if s is not None]
        if len(seeds) != len(set(seeds)):
            warnings.warn(
                f"{self.name}: several fields share the same seed. With "
                "equal models they are perfectly correlated. Pass 'seed' "
                "to derive distinct per-field seeds."
            )

    def _purge(self):
        for lateral in self._laterals():
            lateral.purge()
        self.clear_cache()

    # conditioning ############################################################

    def condition(self, data, gibbs=None, seed=None, **krige_kw):
        """Condition the stack on borehole data.

        Equalities are honoured exactly. Inequality data are imputed once
        by ``gstools.geometry.gibbs.gibbs_sample`` and frozen into the
        realization (see :any:`resample_inequalities`). Objects without
        data are reset to their unconditioned prior.

        Parameters
        ----------
        data : :any:`gstools.geometry.observations.BoreholeData`
            Aligned per-datum conditioning records.
        gibbs : :class:`dict`, optional
            Keyword arguments passed to
            ``gstools.geometry.gibbs.gibbs_sample`` (e.g. ``burn_in``,
            ``sweeps``).
        seed : :class:`int`, optional
            Master seed for the Gibbs imputation.
        **krige_kw
            Passed to :any:`Krige` when a conditioned field is built from
            an :any:`SRF` (default ``unbiased=False``: simple kriging with
            the field mean).

        Raises
        ------
        ValueError
            If the data kind does not fit the stack, a layer index is out
            of range, or data target an object that can't be conditioned.
        """
        if data.kind != self._kind:
            other = (
                "SurfaceStack" if self._kind == "thickness" else "LayerStack"
            )
            raise ValueError(
                f"{self.name}.condition: needs kind='{self._kind}' data, "
                f"got kind='{data.kind}'. Use {other} for that data."
            )
        if data.err is not None and np.any(data.err > 0):
            raise NotImplementedError(
                f"{self.name}.condition: measurement errors are not "
                "supported yet. Exact conditioning would need a frozen "
                "error draw per realization."
            )
        data = data.merge_duplicates()
        targets = self._targets()
        bad = sorted(set(np.unique(data.layer).tolist()) - set(targets))
        if bad:
            raise ValueError(f"{self.name}.condition: unknown layer(s) {bad}")
        master = MasterRNG(seed)
        for i, (lateral, to_gauss) in targets.items():
            sub_seed = master()
            sub = data.for_layer(i)
            label = f"{self.name} {self._target_name(i)}"
            if len(sub) == 0:
                if lateral.conditioned:
                    lateral.uncondition()
                continue
            if not lateral.is_random and not isinstance(
                lateral.prior, CondSRF
            ):
                warnings.warn(
                    f"{label}: deterministic, {len(sub)} datum/data ignored. "
                    "Check residuals()."
                )
                continue
            val, low, upp = to_gauss(sub)
            lateral.condition(
                sub.pos,
                val,
                low,
                upp,
                sub.well,
                label,
                gibbs,
                sub_seed,
                krige_kw,
            )
        self._purge()

    def resample_inequalities(self, seed=None):
        """Redraw the Gibbs imputation for inequality data.

        Parameters
        ----------
        seed : :class:`int`, optional
            Master seed for the resample (same seed as in :any:`condition`
            reproduces its draw).
        """
        master = MasterRNG(seed)
        for lateral, _ in self._targets().values():
            sub_seed = master()
            lateral.resample(sub_seed)
        self._purge()

    # internals ###############################################################

    def _check(self):
        for i, (lateral, _) in self._targets().items():
            name = f"{self.name} {self._target_name(i)}"
            if lateral.name is not None:
                name += f" ({lateral.name})"
            lateral.check(self.dim - 1, name)

    def _ifaces(self, lat, dedupe=True):
        """Cached signed-space interfaces, keyed by ``np.array_equal``.

        With ``dedupe``, the fields are evaluated at the distinct lateral
        positions only (skipped for structured grids, which have none).
        """
        self._check()
        dedupe = dedupe and all(_pointwise(o.field) for o in self._laterals())
        return self._iface_cache(self._compute_ifaces, lat, dedupe)

    def _outside(self, lat):
        """Boolean ``(n_lat,)``: lateral positions outside the mask."""
        n_lat = lat.shape[1]
        if self._mask is None:
            return np.zeros(n_lat, dtype=bool)
        if callable(self._mask):
            inside = np.asarray(self._mask(*lat), dtype=bool)
        else:
            inside = np.asarray(self._mask, dtype=bool)
        return ~np.broadcast_to(inside.ravel(), (n_lat,))

    def _index(self, pos, structured):
        """Layer index per point and outside-mask flags.

        Unstructured: shapes ``(n,)``. Structured: ``(n_lat, nz)``.
        """
        lat, vert = _split_pos(pos, self.dim, structured)
        ifaces = self._ifaces(lat, dedupe=not structured)
        out = self._outside(lat)
        if structured:
            idx = self._layer_index(
                ifaces[:, :, None], self._sign * vert[None, :]
            )
            return idx, np.broadcast_to(out[:, None], idx.shape)
        return self._layer_index(ifaces, self._sign * vert), out

    @staticmethod
    def _layer_index(ifaces, s):
        """Map signed vertical coordinates to layer indices.

        ``ifaces`` increasing along axis 0, ``-1`` means below base,
        ``n_layers`` means above the top. Intervals are half-open
        ``[f_i, f_{i+1})``. The label is decided by a single monotone
        count, never by two independent comparisons.

        Parameters
        ----------
        ifaces : :class:`numpy.ndarray`
            Interface elevations.
        s : :class:`numpy.ndarray`
            Signed vertical coordinates.

        Returns
        -------
        :class:`numpy.ndarray`
            Integer layer indices.
        """
        idx = np.full(np.broadcast(ifaces[0], s).shape, -1, dtype=np.int64)
        for i in range(ifaces.shape[0]):
            idx += ifaces[i] <= s
        return idx

    def _explicit_labels(self):
        return [unit.label for unit in self._units()]

    def _unit_labels(self):
        return [
            i if lab is None else lab
            for i, lab in enumerate(self._explicit_labels())
        ]

    def _layer_names(self):
        return [unit.name for unit in self._units()]

    def _from_index(self, idx):
        """Map layer indices to labels (below/above/layer/NO_LABEL)."""
        table = [self._below_label] + self._unit_labels() + [self._above_label]
        table = np.array(
            [NO_LABEL if lab is None else lab for lab in table],
            dtype=np.double,
        )
        return table[idx + 1]

    def _target_name(self, i):
        return f"layer {i}"

    @property
    def names(self):
        """:class:`dict`: Mapping ``{label: name}`` of all named layers."""
        return {
            lab: name
            for lab, name in zip(self._unit_labels(), self._layer_names())
            if name is not None
        }

    @property
    def seed(self):
        """:class:`int` or :any:`None`: The master seed."""
        return self._seed

    @property
    @abstractmethod
    def n_layers(self):
        """:class:`int`: Number of layers in the stack."""

    def __repr__(self):
        return f"{self.name}(dim={self.dim}, n_layers={self.n_layers})"


class LayerStack(_Stack):
    """A stochastic stack of stratigraphic layers.

    Interfaces are ``f_i = base + cumsum(t)`` with thicknesses
    ``t_i = link(g_i) >= 0``, so they touch but never cross.

    Realization identity is ``(derived per-layer seeds, sampled
    inequality values, frozen measurement-error draws)``. :any:`reset` is
    the only mutator that changes it.

    Parameters
    ----------
    layers : :class:`list` of :any:`Layer`
        Layers, ordered from bottom to top (in stacking direction). Other
        objects are wrapped into a :any:`Layer`.
    base : :class:`float`, :any:`callable`, :class:`numpy.ndarray` or :any:`Field`, optional
        Elevation of the base of the stack. Default: 0.0
    stacking : :class:`str`, optional
        Either "up" or "down". Default: "up"
    dim : :class:`int`, optional
        Spatial dimension of the stack (lateral dim is ``dim - 1``). The
        vertical axis is the last component of ``pos``. Default: 3
    below_label : :class:`float`, optional
        Label below the base. ``NO_LABEL`` makes it transparent, so a
        lower-priority region shows through. Default: -1
    above_label : :class:`float`, optional
        Label above the top. ``NO_LABEL`` makes it transparent, so a
        lower-priority region shows through. ``None`` means the default,
        never transparent. Default: ``n_layers``
    mask : :class:`numpy.ndarray` or :any:`callable`, optional
        Lateral mask (bool per lateral position, or a callable on the
        lateral coordinates), outside the mask evaluates to
        ``outside_label``.
    outside_label : :class:`float`, optional
        Label outside ``mask``. Default: ``NO_LABEL``
    seed : :class:`int`, optional
        Master seed for per-field seed derivation (base first, then the
        layers). If :any:`None`, the fields keep their own seeds.
    priority : :class:`float`, optional
        Priority of the stack as a whole, for nesting inside a
        :any:`gstools.geometry.base.Composite`. Default: 0.0

    Warnings
    --------
    Mutating a layer directly (e.g. ``layer.field.model.len_scale = ...``,
    or calling ``srf(pos, seed=...)``) is outside this class's control and
    silently changes the realization.
    """

    _kind = "thickness"

    def __init__(
        self,
        layers,
        base=0.0,
        stacking="up",
        dim=3,
        below_label=-1,
        above_label=None,
        mask=None,
        outside_label=None,
        seed=None,
        priority=0.0,
    ):
        self._layers = [
            lay if isinstance(lay, Layer) else Layer(lay) for lay in layers
        ]
        if not self._layers:
            raise ValueError("LayerStack: needs at least one layer.")
        self._base = _Lateral(base, name="base")
        if stacking not in ("up", "down"):
            raise ValueError("LayerStack: stacking must be 'up' or 'down'.")
        self._stacking = stacking
        super().__init__(
            dim,
            below_label,
            above_label,
            mask,
            outside_label,
            seed,
            priority,
            sign=1.0 if stacking == "up" else -1.0,
        )

    def thicknesses(self, lat):
        """Evaluate all layer thicknesses at lateral positions.

        Parameters
        ----------
        lat : :class:`numpy.ndarray`
            Lateral positions, shape ``(dim - 1, n)``.

        Returns
        -------
        :class:`numpy.ndarray`
            Thicknesses, shape ``(n_layers, n)``, all ``>= 0``.
        """
        lat = np.asarray(lat, dtype=np.double).reshape(self.dim - 1, -1)
        self._check()
        return np.array([lay.thickness(lat) for lay in self._layers])

    def residuals(self, data):
        """Misfit of the current realization per datum.

        Equalities give ``|t_sim - t_obs|`` (``|z_sim - z_obs|`` for base
        data), inequalities the bound violation in thickness space
        (``0`` if honoured). Offending wells: ``data.well[res > tol]``.

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
        for i in np.unique(data.layer):
            sel = data.layer == i
            pos, val = data.pos[:, sel], data.value[sel]
            eq = np.isfinite(val)
            if i < 0:
                sim = self._base.evaluate(pos)
                low, upp = data.lower[sel], data.upper[sel]
            else:
                lay = self._layers[i]
                sim = lay.thickness(pos)
                glo, gup = lay.link.gauss_bounds(
                    data.lower[sel], data.upper[sel]
                )
                low, upp = lay.link.thickness(glo), lay.link.thickness(gup)
            viol = np.maximum(np.maximum(low - sim, sim - upp), 0.0)
            res[sel] = np.where(eq, np.abs(sim - val), viol)
        return res

    def _compute_ifaces(self, lat):
        t = self.thicknesses(lat)
        out = np.empty((self.n_layers + 1, t.shape[1]))
        out[0] = self._sign * self._base.evaluate(lat)
        np.cumsum(t, axis=0, out=out[1:])
        out[1:] += out[0]
        return out

    def _laterals(self):
        return [self._base] + self._layers

    def _units(self):
        return self._layers

    def _targets(self):
        res = {-1: (self._base, _identity)}
        for i, lay in enumerate(self._layers):
            res[i] = (lay, _link_to_gauss(lay.link))
        return res

    def _target_name(self, i):
        return "base" if i < 0 else f"layer {i}"

    @property
    def n_layers(self):
        """:class:`int`: Number of layers in the stack."""
        return len(self._layers)

    @property
    def layers(self):
        """:class:`list` of :any:`Layer`: The layers, bottom to top."""
        return self._layers

    @property
    def base(self):
        """:any:`Field`, :any:`callable` or :class:`float`: The base object.

        A :any:`CondSRF` built from the given base if base data were given.
        """
        return self._base.field

    @property
    def stacking(self):
        """:class:`str`: Either "up" or "down"."""
        return self._stacking


def _identity(sub):
    """Data values are already Gaussian (elevations)."""
    return sub.value.copy(), sub.lower.copy(), sub.upper.copy()


def _link_to_gauss(link):
    """Thickness data -> Gaussian values and bounds via the link."""

    def to_gauss(sub):
        val = link.gauss_value(sub.value)
        low, upp = link.gauss_bounds(sub.lower, sub.upper)
        # an equality on a censored atom must be handled as a bound
        atom = np.isfinite(sub.value) & ~np.isfinite(val)
        if np.any(atom):
            low = np.where(atom, -np.inf, low)
            upp = np.where(atom, link.gauss_bounds(0.0, 0.0)[1], upp)
        return val, low, upp

    return to_gauss


def _warn_conditioning(model, pos, wells, label):
    """Warn if the data covariance is ill-conditioned (near duplicates)."""
    n = pos.shape[1]
    if n < 2:
        return
    iso = model.isometrize(pos)
    dist = cdist(iso.T, iso.T)
    cond = np.linalg.cond(model.covariance(dist))
    if cond <= COND_TOL:
        return
    dist[np.diag_indices(n)] = np.inf
    order = np.argsort(dist, axis=None)[: 2 * min(3, n)]
    pairs = []
    for flat in order:
        i, j = np.unravel_index(flat, dist.shape)
        if i < j:
            pairs.append(f"#{wells[i]}-#{wells[j]} ({dist[i, j]:.3g})")
    warnings.warn(
        f"{label}: data covariance is ill-conditioned (cond={cond:.2e}), "
        f"near-duplicate positions are averaged by the pseudo-inverse. "
        f"Closest pairs (isotropic distance): {', '.join(pairs)}"
    )


def _warn_extreme_bounds(model, mean, val, cond, tail):
    """Warn if inequality bounds lie far in the tail of the prior.

    Typical cause: a "layer absent" datum under :any:`LogLink` becomes the
    bound ``g <= log(absent)``, many standard deviations below the mean.
    The imputed value then drags the conditioned field far down and,
    with smooth covariances, makes it overshoot elsewhere.
    """
    ineq = ~np.isfinite(val)
    std = np.sqrt(model.var)
    z_low = (cond["low"][ineq] - mean[ineq]) / std
    z_upp = (cond["upp"][ineq] - mean[ineq]) / std
    worst = np.max(np.maximum(z_low, -z_upp), initial=-np.inf)
    if worst > tail:
        warnings.warn(
            f"{cond['label']}: an inequality bound lies {worst:.1f} prior "
            "standard deviations in the tail. The conditioned field may be "
            "pulled far off and overshoot elsewhere. For 'layer absent' "
            "data consider a larger LogLink(absent=...)."
        )
