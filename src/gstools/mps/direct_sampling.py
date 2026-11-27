"""
GStools subpackage providing the Direct Sampling MPS simulation class.

.. currentmodule:: gstools.mps

The following classes and functions are provided

.. autosummary::
   DirectSampling
"""

import warnings

import numpy as np

from gstools.field.base import Field
from gstools.mps._validation import check_float64_integers, exact_count
from gstools.mps.model import MPSModel
from gstools.mps.nonstationary import build_zone_selector, resolve_spec
from gstools.mps.simulate import _Run, _simulate
from gstools.mps.training_image import _check_category_codes
from gstools.normalizer.base import Normalizer
from gstools.normalizer.tools import apply_mean_norm_trend
from gstools.random.rng import RNG
from gstools.tools.geometric import generate_grid, no_of_angles

__all__ = ["DirectSampling"]


def _check_penalty_matrix_conditioning(
    var_name, penalty_matrix, values, context
):
    """Check conditioning values against a penalty matrix (if set); ``context`` prefixes the error.

    A bad code would otherwise be a raw ``IndexError`` deep in the scan loop.
    """
    if penalty_matrix is None:
        return
    _check_category_codes(
        values,
        penalty_matrix.shape[0],
        f"{context}: conditioning values for variable {var_name!r}",
    )


def _check_finite_positions(pos):
    """Raise ``ValueError`` if a conditioning position is NaN or +-inf."""
    if not np.isfinite(pos).all():
        raise ValueError(
            "DirectSampling.set_condition: conditioning positions must be "
            "finite."
        )


def _check_num_threads(value):
    """Return ``None`` or ``value`` as an exact int >= 1."""
    if value is None:
        return None
    return exact_count(value, minimum=1, context="DirectSampling: num_threads")


def _unset(seed, name="seed"):
    """``True`` for ``None`` or NaN; ints of any size are explicit seeds. ``name`` labels the error."""
    if seed is None:
        return True
    if isinstance(seed, (int, np.integer)):
        return False
    if isinstance(seed, (float, np.floating)):
        return bool(np.isnan(seed))
    raise TypeError(
        f"DirectSampling: {name} must be an int, None or NaN, not {type(seed).__name__}."
    )


def _exact_seed(seed, name="seed"):
    """Return a set ``seed`` as an int; reject fractional, infinite or negative values."""
    return exact_count(seed, minimum=0, context=f"DirectSampling: {name}")


def _check_finite_axes(axes):
    """Raise ``ValueError`` if a simulation-grid coordinate is NaN or +-inf."""
    if not all(np.isfinite(ax).all() for ax in axes):
        raise ValueError("DirectSampling: grid coordinates must be finite.")


def _warn_nonuniform_axes(axes):
    """Warn when grid spacing is non-uniform: DS works in grid-index space.

    Direct Sampling works in cells, so physical spacing is ignored; rotation,
    scale, zones and ``max_radius`` all act on cell offsets.
    """
    steps = []
    for d, ax in enumerate(axes):
        ax = np.asarray(ax, dtype=np.double)
        if len(ax) < 2:
            continue
        diffs = np.diff(ax)
        if not np.allclose(diffs, diffs[0]):
            warnings.warn(
                f"DirectSampling: axis {d} is not uniformly spaced. DS works "
                "in grid-index space (cells) and ignores physical spacing, so "
                "patterns are not isotropic in physical units.",
                UserWarning,
                stacklevel=3,
            )
            return
        # a reflected (decreasing) axis has the same spacing magnitude
        steps.append(abs(diffs[0]))
    if len(steps) > 1 and not np.allclose(steps, steps[0]):
        warnings.warn(
            "DirectSampling: grid axes have unequal spacing. DS works in "
            "grid-index space (cells) and ignores physical spacing, so "
            "patterns are stretched in physical units.",
            UserWarning,
            stacklevel=3,
        )


# The transforms that need a covariance model, as dispatched by
# ``gstools.transform.field.apply``.
_MODEL_TRANSFORMS = ("binary", "discrete", "zinnharvey")
_MODEL_TRANSFORM_SUFFIXES = (
    "force_moments",
    "uniform",
    "arcsin",
    "uquad",
)


def _check_entry(kind, name, value):
    """Validate one per-variable entry; return it (normalizer classes instantiated)."""
    if kind == "normalizer":
        if isinstance(value, type) and issubclass(value, Normalizer):
            value = value()
        if not isinstance(value, Normalizer):
            raise ValueError(
                f"DirectSampling: normalizer for {name!r} must be a "
                "Normalizer instance or subclass."
            )
        return value
    if not callable(value) and (
        isinstance(value, (str, bytes, bool)) or not np.isscalar(value)
    ):
        raise ValueError(
            f"DirectSampling: {kind} for {name!r} must be a number or callable."
        )
    return value


def _processing_property(kind, base):
    """Field property that also accepts a ``{variable name: value}`` dict."""

    def fset(self, value):
        if isinstance(value, dict):
            value = {k: _check_entry(kind, k, v) for k, v in value.items()}
            self._check_processing_names(value)
            setattr(self, "_" + kind, value)
        else:
            base.fset(self, value)

    def fget(self):
        return getattr(self, "_" + kind)

    return property(fget, fset, doc=base.__doc__)


class DirectSampling(Field):
    """Multiple Point Statistics simulation using Direct Sampling.

    Subclasses :class:`gstools.field.base.Field`. Takes an :class:`MPSModel`
    that bundles a :class:`TrainingImage` and all algorithm parameters, and
    produces fields on structured grids.

    Parameters
    ----------
    model : MPSModel
        Algorithm configuration: the training image and every search
        parameter (see :class:`MPSModel`). Per-variable settings live on the
        :class:`Variable` objects of its :class:`TrainingImage`.
    seed : int or None or nan, optional
        Master RNG seed; ``None`` or nan draws a fresh random seed.
        Default: nan.
    mean : None or float or callable, optional
        Mean added to every continuous variable when ``post_process=True``.
        Default: None.
    normalizer : None or Normalizer, optional
        Normalizer applied to continuous variables; the simulated values are
        treated as normalized values and back-transformed. Default: None.
    trend : None or float or callable, optional
        Trend of the denormalized continuous variables. Default: None.

    Notes
    -----
    ``mean``, ``normalizer`` and ``trend`` are applied by the inherited
    :class:`Field` post-processing. Each also accepts a dict
    ``{variable name: value}`` (number or callable for mean and trend,
    ``Normalizer`` instance or subclass for the normalizer); variables missing
    from the dict are not processed by that argument. Keys must be continuous
    variables of the training image. :meth:`transform` with ``process=True``
    is not available while a dict is set. Categorical variables are never processed,
    so their values stay within the training-image codes; with
    ``post_process=True`` and a mean, trend or normalizer set, a
    ``UserWarning`` names the skipped variables. :meth:`transform` works as in
    :class:`Field` with ``field=<variable name>``, except for transformations
    that need a covariance model.

    Resolution is per argument: a dict applies only to its listed variables,
    a single value to every continuous variable.

    The thread count is a call argument (``num_threads``).

    Raises
    ------
    TypeError
        If ``model`` is not an :class:`MPSModel` or ``seed`` is not an int,
        None or nan.
    ValueError
        If a variable name of a TI built from a list of Variables collides
        with an attribute of the instance, or ``seed`` is fractional or
        negative.

    See Also
    --------
    MPSModel : holds the algorithm parameters.
    TrainingImage : training data holder.

    Examples
    --------
    >>> import numpy as np
    >>> import gstools as gs
    >>> ti = gs.mps.TrainingImage(np.indices((20, 20)).sum(axis=0) % 3.0)
    >>> model = gs.mps.MPSModel(ti, scan_fraction=0.5, threshold=0.05)
    >>> dsim = gs.mps.DirectSampling(model, seed=42)
    >>> field = dsim([np.arange(10.), np.arange(10.)])
    >>> field.shape
    (10, 10)

    Per-variable mean for a two-variable training image:

    >>> ti2 = gs.mps.TrainingImage(
    ...     [gs.mps.Variable("a", ti.variable().data, categorical=False),
    ...      gs.mps.Variable("b", ti.variable().data, categorical=False)]
    ... )
    >>> dsim = gs.mps.DirectSampling(
    ...     gs.mps.MPSModel(ti2, scan_fraction=0.5), mean={"a": 10.0}
    ... )
    >>> a, b = dsim([np.arange(10.), np.arange(10.)], seed=1)
    >>> bool(a.min() >= 10.0 and b.max() < 10.0)
    True
    """

    default_field_names = ["field"]

    def __init__(
        self,
        model,
        seed=np.nan,
        *,
        mean=None,
        normalizer=None,
        trend=None,
    ):
        if not isinstance(model, MPSModel):
            raise TypeError(
                "DirectSampling requires an MPSModel as its first argument. "
                "Wrap a TrainingImage first: "
                "DirectSampling(MPSModel(ti, scan_fraction=…))"
            )

        # DirectSampling has no covariance model; the MPSModel is kept apart.
        super().__init__(
            model=None,
            dim=model.ti.ndim,
            value_type="scalar",
            mean=mean,
            normalizer=normalizer,
            trend=trend,
        )
        self._mps_model = model
        self._cond_pos = None
        self._cond_val = None
        self._stored_vars = {}  # stored field name -> variable name
        self.rng = RNG(
            None if _unset(seed) else _exact_seed(seed), seed_sequence=True
        )
        self._checked_names = None
        self._check_ti_names()
        for kind in ("mean", "normalizer", "trend"):
            self._check_processing_names(getattr(self, kind))

    mean = _processing_property("mean", Field.mean)
    normalizer = _processing_property("normalizer", Field.normalizer)
    trend = _processing_property("trend", Field.trend)

    def _check_processing_names(self, value):
        """Raise if a dict argument names a missing or categorical variable."""
        ti = getattr(getattr(self, "_mps_model", None), "ti", None)
        if not isinstance(value, dict) or ti is None:
            return
        cont = {v.name for v in ti.variables if not v.categorical}
        bad = sorted(set(value) - cont, key=str)
        if bad:
            raise ValueError(
                f"DirectSampling: mean/normalizer/trend dict names {bad} are "
                f"not continuous variables of the TI (continuous: {sorted(cont)})."
            )

    def _per_variable(self, kind, name):
        """Value of ``kind`` for variable ``name`` (a dict is looked up)."""
        value = getattr(self, "_" + kind)
        return value.get(name) if isinstance(value, dict) else value

    def __call__(
        self,
        pos=None,
        seed=np.nan,
        *,
        path_seed=np.nan,
        node_seed=np.nan,
        path="random",
        mesh_type="structured",
        post_process=True,
        store=True,
        num_threads=None,
    ):
        """Generate the spatial random field via Direct Sampling.

        One variable gives an array (stored as ``self.field``); several give
        a tuple of arrays in variable order, each stored under its own name.

        Parameters
        ----------
        pos : :class:`list`, optional
            The position tuple, containing main direction and transversal
            directions. Only structured grids are supported.
        seed : :class:`int` or :any:`None`, optional
            Seed for the RNG. If ``np.nan``, the master RNG stream continues,
            so each call draws a new realization (pass a seed to reproduce
            one); ``None`` draws a fresh random seed (see also the
            constructor's ``seed``). Default: ``np.nan``
        path_seed : :class:`int` or :any:`numpy.nan`, optional
            Seed controlling the order in which simulation grid nodes are
            visited. If ``np.nan`` (default), derived from the master RNG
            together with ``node_seed``. Fix this while varying ``node_seed``
            to study the effect of TI search randomness under a constant
            visit order, or vice versa. Only has an effect when a random visit
            order is drawn: ``path="random"``, or post-processing passes whose
            ``path`` resolves to random (see :class:`PostProcessing`).
            Default: ``np.nan``
        node_seed : :class:`int` or :any:`numpy.nan`, optional
            Seed controlling the TI scan entry point and fallback cell for
            every node. If ``np.nan`` (default), derived from the master RNG.
            Default: ``np.nan``
        path : :class:`str` or array-like, optional
            Node visit order.  ``"random"`` (default) shuffles the unknown
            nodes uniformly using the ``path_seed``-controlled RNG — the
            standard DS behaviour.  ``"sequential"`` visits nodes in raster
            (lexicographic) order, which is deterministic regardless of
            ``path_seed``.  An explicit integer array of shape ``(N, dim)``
            provides a caller-supplied visit order: it must contain every
            unknown node once. Conditioned nodes in it are dropped, while a
            missing unknown node, a duplicate row or an out-of-bounds
            coordinate raises :class:`ValueError`. Default: ``"random"``
        mesh_type : :class:`str`, optional
            Grid type. Must be ``"structured"``.
            Default: ``"structured"``
        post_process : :class:`bool`, optional
            Whether to apply post-processing transformations (mean,
            normalizer, trend) to the field. Categorical variables are never
            processed. Default: :any:`True`
        store : :class:`str` or :class:`bool` or :class:`list`, optional
            Whether to store the fields (``True``/``False``) under the default
            names (``"field"`` for one variable, else the variable names) or
            under given names (a string for one variable, a list for several).
            Default: :any:`True`
        num_threads : :class:`int` or :any:`None`, optional
            Threads for the node dependency graph. The output is identical for
            any thread count. Default: :any:`None` (``config.NUM_THREADS``,
            else 1)

        Returns
        -------
        field : :class:`numpy.ndarray` or :class:`tuple` of :class:`numpy.ndarray`
            The simulated field for one variable, else a tuple of fields in
            variable order; each is also stored under its name
            (``self[name]`` / :attr:`all_fields`).

        Raises
        ------
        ValueError
            If ``mesh_type`` is not ``"structured"``, ``path`` is
            invalid, ``num_threads`` is not an integer >= 1, the TI
            dimension differs from the instance's, a grid axis is non-finite,
            stored conditioning refers to variables absent from the TI or
            fails a penalty-matrix check, a variable name collides with an
            attribute, a rotation, scale or zone spec does not fit the
            grid, or a seed is fractional or negative.
        TypeError
            If ``seed``, ``path_seed`` or ``node_seed`` is not an int, None
            or nan.

        Notes
        -----
        DS post-processing passes (``MPSModel(post_processing=PostProcessing(...))``,
        Meerschman2013 §4) run after the main simulation pass, over the same
        ``num_threads``-parallel dependency DAG. They are distinct from this
        method's ``post_process`` kwarg (Field mean/normalizer/trend pipeline).
        """
        if mesh_type != "structured":
            raise ValueError(
                "DirectSampling: only structured grids are supported."
            )
        variables = self._ti.variables
        for kind in ("mean", "normalizer", "trend"):
            self._check_processing_names(getattr(self, kind))
        if len(variables) == 1:
            name, save = self.get_store_config(store)
            names, saves = [name], [save]
        else:
            names, saves = self.get_store_config(
                store, [v.name for v in variables], len(variables)
            )
        pos, shape = self.pre_pos(pos, mesh_type)
        dim = len(shape)
        flat_pos = generate_grid(self.pos)
        rotation_map, rot_stat = resolve_spec(
            self._mps_model.rotation,
            shape,
            no_of_angles(dim),
            flat_pos,
            "rotation",
            pad=True,
        )
        scale_map, scale_stat = resolve_spec(
            self._mps_model.scale,
            shape,
            dim,
            flat_pos,
            "scale",
            positive=True,
        )
        # Identity transforms are dropped so they match the no-transform path.
        if rot_stat and rotation_map is not None and not np.any(rotation_map):
            rotation_map = None
        if scale_stat and scale_map is not None and np.all(scale_map == 1):
            scale_map = None
        zones = self._mps_model.zones
        selector = (
            build_zone_selector(zones, shape, flat_pos) if zones else None
        )
        _check_finite_axes(self.pos)
        _warn_nonuniform_axes(self.pos)
        if self._ti.ndim != self.dim:
            raise ValueError(
                f"DirectSampling: the model's TI is {self._ti.ndim}-D, but "
                f"this instance was built for {self.dim}-D."
            )
        self._check_ti_names()
        self._check_conditioning_names()
        # The model is mutable: its penalty matrices may differ from set time.
        self._check_condition_codes()
        conditions = self._conditions_to_grid(self.pos)
        if seed is None:
            self.rng.seed = None  # fresh random seed, as in gs.SRF
        elif not _unset(seed):
            self.rng.seed = _exact_seed(seed)
        rng_path = (
            self.rng.random
            if _unset(path_seed, "path_seed")
            else RNG(
                _exact_seed(path_seed, "path_seed"), seed_sequence=True
            ).random
        )
        rng_nodes = (
            self.rng.random
            if _unset(node_seed, "node_seed")
            else RNG(
                _exact_seed(node_seed, "node_seed"), seed_sequence=True
            ).random
        )
        n_threads = (
            None if num_threads is None else _check_num_threads(num_threads)
        )
        result = _simulate(
            self._mps_model,
            shape,
            _Run(
                rng_path=rng_path,
                rng_nodes=rng_nodes,
                conditions=conditions,
                path=path,
                rotation_map=rotation_map,
                scale_map=scale_map,
                stationary_transform=rot_stat and scale_stat,
                zone_selector=selector,
                num_threads=n_threads,
            ),
        )
        # Every variable is a named field; the output is unwrapped in one place.
        # Categorical variables are never mean/normalizer/trend-processed.
        skipped = [v.name for v in self._ti.variables if v.categorical]
        if (
            post_process
            and skipped
            and (
                not isinstance(self.mean, dict)
                and self.mean is not None
                or not isinstance(self.trend, dict)
                and self.trend is not None
                or not isinstance(self.normalizer, dict)
                and type(self.normalizer) is not Normalizer
            )
        ):
            warnings.warn(
                "DirectSampling: mean/normalizer/trend are not applied to "
                f"categorical variable(s) {skipped}.",
                UserWarning,
                stacklevel=2,
            )
        out = []
        self._stored_vars = {}
        per_var = any(
            isinstance(getattr(self, k), dict)
            for k in ("mean", "normalizer", "trend")
        )
        for v, name, save in zip(variables, names, saves):
            process = post_process and not v.categorical
            field = result[v.name]
            if process and per_var:
                field = apply_mean_norm_trend(
                    self.pos,
                    np.array(field, dtype=np.double).reshape(self.field_shape),
                    mean=self._per_variable("mean", v.name),
                    normalizer=self._per_variable("normalizer", v.name),
                    trend=self._per_variable("trend", v.name),
                    mesh_type=self.mesh_type,
                    check_shape=False,
                )
                process = False
            out.append(
                self.post_field(field, name=name, process=process, save=save)
            )
            if save:
                self._stored_vars[name] = v.name
        return out[0] if len(out) == 1 else tuple(out)

    def _conditions_to_grid(self, axes):
        """Snap conditioning points to nearest grid nodes (Mariethoz2010 §3 ¶12).

        Returns ``{idx: {variable: value}}`` with NaN entries skipped. When
        two points snap to the same node, the conflict is resolved
        **per variable**: for each variable the closest colliding point with a
        finite value wins. A farther point therefore still contributes its
        finite values for any variable the closer point left ``NaN``, so valid
        conditioning data is never silently discarded.

        Parameters
        ----------
        axes : tuple of numpy.ndarray
            Raw grid axis arrays (``self.pos``).

        Returns
        -------
        dict
        """
        if self._cond_pos is None:
            return {}
        # Points beyond half a cell outside the grid snap to its edge: warn.
        outside = 0
        for d, ax in enumerate(axes):
            ax = np.asarray(ax, dtype=np.double)
            tol = 0.5 * np.abs(np.diff(ax)).max() if len(ax) > 1 else 0.0
            p = np.asarray(self._cond_pos[d], dtype=np.double)
            outside |= (p < ax.min() - tol) | (p > ax.max() + tol)
        if np.any(outside):
            warnings.warn(
                f"DirectSampling: {int(np.sum(outside))} conditioning "
                "point(s) lie outside the simulation grid and were snapped "
                "to its nearest edge node.",
                UserWarning,
                stacklevel=3,
            )
        n_collided = 0  # finite values landing on an occupied (node, variable)
        # idx -> {variable: (value, dist_sq)}; the closest finite value per
        # variable wins, so a farther point fills variables the closer one
        # left NaN (per-variable collision resolution).
        candidates = {}
        n_cond = self._cond_pos.shape[1]
        for k in range(n_cond):
            idx, dist_sq = self._snap_to_nearest(axes, k)
            var_best = candidates.setdefault(idx, {})
            for v in self._cond_val:
                val = self._cond_val[v][k]
                if not np.isfinite(val):
                    continue
                if v in var_best:
                    n_collided += 1
                if v not in var_best or dist_sq < var_best[v][1]:
                    var_best[v] = (float(val), dist_sq)
        result = {
            idx: {v: val for v, (val, _) in var_best.items()}
            for idx, var_best in candidates.items()
        }
        if n_collided:
            warnings.warn(
                f"DirectSampling: {n_collided} conditioning value(s) collided "
                "on an occupied grid node; the one closest to the node was "
                "kept (the first one on ties).",
                UserWarning,
                stacklevel=3,
            )
        return result

    def _snap_to_nearest(self, axes, k):
        """Return (grid_index_tuple, dist_sq) for the k-th conditioning point."""
        idx = tuple(
            int(np.argmin(np.abs(axes[d] - self._cond_pos[d][k])))
            for d in range(self.dim)
        )
        dist_sq = sum(
            (axes[d][idx[d]] - self._cond_pos[d][k]) ** 2
            for d in range(self.dim)
        )
        return idx, dist_sq

    def set_condition(self, cond_pos, cond_val):
        """Set the conditioning data for the simulation.

        Conditioning weight is configured per :class:`Variable` (``cond_weight=``).
        Points beyond half a cell outside the grid snap to its edge at call
        time and warn.

        Parameters
        ----------
        cond_pos : :class:`list`
            The position tuple of the conditioning data ``(x, [y, z])``.
        cond_val : :class:`numpy.ndarray` or :class:`dict`
            A ``{variable: numpy.ndarray}`` mapping; for a one-variable
            training image a plain array is also accepted. ``numpy.nan`` marks a
            variable as not conditioned at a point; ``+-inf`` raises
            :class:`ValueError`.

        Raises
        ------
        ValueError
            If ``cond_val`` is empty, names an unknown variable, is not a
            dict for a multi-variable TI, has a length different from
            ``cond_pos``, or holds ``+-inf`` or integers not exact in
            float64; if positions are not finite; or if a categorical value
            is not a valid code of a variable's ``penalty_matrix``.
        """
        names = [v.name for v in self._ti.variables]
        if not isinstance(cond_val, dict):
            if len(names) != 1:
                raise ValueError(
                    "DirectSampling: cond_val must be a dict {variable: array} "
                    "for a training image with several variables."
                )
            cond_val = {names[0]: cond_val}
        if not cond_val:
            raise ValueError("DirectSampling: cond_val must not be empty")
        unknown = set(cond_val) - set(names)
        if unknown:
            raise ValueError(
                f"cond_val contains unknown variable(s): {sorted(unknown)}. "
                f"TI variables are: {sorted(names)}"
            )
        ctx = "DirectSampling.set_condition: cond_val"
        categorical = {v.name: v.categorical for v in self._ti.variables}
        for name, a in cond_val.items():
            check_float64_integers(a, categorical[name], ctx)
        pos = np.asarray(cond_pos, dtype=np.double).reshape(self.dim, -1)
        n_cond = pos.shape[1]
        vals = {v: np.asarray(a, dtype=np.double) for v, a in cond_val.items()}
        for v, arr in vals.items():
            if arr.shape != (n_cond,):
                raise ValueError(
                    "DirectSampling: cond_pos and cond_val length mismatch; "
                    "each cond_val array must be 1-D with the same length "
                    f"as cond_pos ({n_cond}), got {arr.shape} for {v!r}."
                )
        _check_finite_positions(pos)
        if any(np.isinf(a).any() for a in vals.values()):
            raise ValueError(
                "DirectSampling.set_condition: conditioning values must not "
                "be +-inf; use NaN for a variable that is not conditioned."
            )
        # NaN means "not conditioned": drop points with no value at all.
        keep = np.any([np.isfinite(a) for a in vals.values()], axis=0)
        n_dropped = int(np.count_nonzero(~keep))
        pos = pos[:, keep]
        vals = {v: a[keep] for v, a in vals.items()}
        uncovered = {v: self._count_uncovered(v, a) for v, a in vals.items()}
        self._check_condition_codes(vals)
        self._cond_pos, self._cond_val = pos, vals
        if n_dropped:
            warnings.warn(
                f"DirectSampling.set_condition: {n_dropped} conditioning "
                "point(s) without any finite value were dropped.",
                UserWarning,
                stacklevel=2,
            )
        bad = {v: n for v, n in uncovered.items() if n}
        if bad:
            warnings.warn(
                f"DirectSampling.set_condition: {sum(bad.values())} "
                "conditioning value(s) of variable(s) "
                f"{list(bad)} cannot be reproduced by the training image(s) "
                "(absent category, or outside the TI range).",
                UserWarning,
                stacklevel=2,
            )

    def _count_uncovered(self, name, vals):
        """Count finite conditioning values the TI(s) cannot reproduce."""
        vals = np.asarray(vals, dtype=np.double)
        vals = vals[np.isfinite(vals)]
        data = [
            t.variable(name).data
            for t in [self._ti] + [z.ti for z in self._mps_model.zones]
        ]
        pool = np.concatenate([d[np.isfinite(d)] for d in data])
        if vals.size == 0 or pool.size == 0:
            return 0
        if self._ti.variable(name).categorical:
            return int(np.count_nonzero(~np.isin(vals, pool)))
        return int(np.count_nonzero((vals < pool.min()) | (vals > pool.max())))

    def _check_ti_names(self):
        """Reject variable names that collide with an attribute (once per distinct name set)."""
        names = tuple(v.name for v in self._ti.variables)
        if names == self._checked_names:
            return
        if len(names) > 1:
            for name in names:
                if name in dir(self):
                    raise ValueError(
                        f"DirectSampling: variable name {name!r} collides "
                        "with an existing attribute."
                    )
        self._checked_names = names

    def _check_conditioning_names(self):
        """Raise if stored conditioning names a variable the TI does not have."""
        names = {v.name for v in self._ti.variables}
        extra = set(self._cond_val or {}) - names
        if extra:
            raise ValueError(
                f"DirectSampling: conditioning refers to variable(s) "
                f"{sorted(extra)} that are not in the model's TI "
                f"({sorted(names)}). The TI was replaced after it was "
                "set; call set_condition again."
            )

    def _check_condition_codes(self, cond_val=None):
        """Check conditioning codes (default the stored ones) against the primary
        and post-pass penalty matrices; run at set time and again at call time."""
        context = "DirectSampling.set_condition"
        if cond_val is None:
            cond_val = self._cond_val
            context = "DirectSampling (checked at call time)"
        if cond_val is None:
            return
        post_ti = self._mps_model._resolved.post_ti
        for v in self._ti.variables:
            vals = cond_val.get(v.name)
            if vals is None:
                continue
            mats = [v.penalty_matrix]
            if post_ti is not None:
                mats.append(post_ti.variable(v.name).penalty_matrix)
            for mat in mats:
                _check_penalty_matrix_conditioning(v.name, mat, vals, context)

    def transform(self, method, field="field", **kwargs):
        """Apply a field transformation to the stored field ``field``.

        Same as :meth:`Field.transform`; use the variable name as ``field``
        for multivariate output. Transformations that read the covariance
        model (``binary``, ``discrete``, ``zinnharvey``, ``normal_force_moments``,
        ``normal_to_uniform``, ``normal_to_arcsin``, ``normal_to_uquad``)
        raise :class:`ValueError`.

        Raises
        ------
        ValueError
            If ``method`` needs a covariance model.
        """
        if str(method) in _MODEL_TRANSFORMS or str(method).endswith(
            _MODEL_TRANSFORM_SUFFIXES
        ):
            raise ValueError(
                f"DirectSampling.transform: {method!r} needs a covariance "
                "model, which DirectSampling does not have."
            )
        if kwargs.get("process") and any(
            isinstance(getattr(self, k), dict)
            for k in ("mean", "normalizer", "trend")
        ):
            raise ValueError(
                "DirectSampling.transform: process=True is not available with "
                "per-variable mean/normalizer/trend dicts; process the field "
                "yourself or use process=False."
            )
        var = self._stored_vars.get(field)
        if var is not None and self._ti.variable(var).categorical:
            warnings.warn(
                f"DirectSampling.transform: variable {var!r} is categorical; "
                "the result leaves the training-image code set.",
                UserWarning,
                stacklevel=2,
            )
        return super().transform(method, field=field, **kwargs)

    @property
    def mps_model(self):
        """:any:`MPSModel`: the search-parameter configuration (training image + parameters)."""
        return self._mps_model

    @property
    def _ti(self):
        return self._mps_model.ti

    @property
    def ti(self):
        """:any:`TrainingImage`: The training image model."""
        return self._ti

    def __repr__(self):
        args = [repr(self._mps_model)]
        if self.rng.seed is not None:
            args.append(f"seed={self.rng.seed!r}")
        for name in ("mean", "trend"):
            val = getattr(self, name)
            if val is not None:
                shown = (
                    getattr(val, "__name__", "callable")
                    if callable(val)
                    else repr(val)
                )
                args.append(f"{name}={shown}")
        if type(self.normalizer) is not Normalizer:
            args.append(f"normalizer={self.normalizer!r}")
        return f"DirectSampling({', '.join(args)})"
