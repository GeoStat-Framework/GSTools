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
from gstools.mps._validation import _validate_real_data, exact_count
from gstools.mps.model import MPSModel
from gstools.mps.nonstationary import build_zone_selector, resolve_spec
from gstools.mps.simulate import _simulate_model
from gstools.normalizer.tools import apply_mean_norm_trend
from gstools.random.rng import RNG
from gstools.tools.geometric import no_of_angles

__all__ = ["DirectSampling"]


def _is_unset(value):
    """True for an unset seed control (None or a real floating NaN)."""
    return value is None or (
        isinstance(value, (float, np.floating)) and np.isnan(value)
    )


def _seed_value(value, name):
    """Validate a Direct Sampling seed before creating or changing streams."""
    if _is_unset(value):
        return None
    return exact_count(value, minimum=0, context=f"DirectSampling.{name}")


def _check_finite_positions(pos):
    """Raise ``ValueError`` if a conditioning position is NaN or +-inf."""
    if not np.isfinite(pos).all():
        raise ValueError(
            "DirectSampling.set_condition: conditioning positions must be "
            "finite."
        )


def _condition_values(values, variable):
    """Convert hard data without silently dropping infinities or integer bits."""
    _validate_real_data(
        values, variable.categorical, "DirectSampling conditioning values"
    )
    result = np.asarray(values, dtype=np.double)
    if np.isinf(result).any():
        raise ValueError(
            "DirectSampling: conditioning values must not contain infinity; "
            "use NaN for unconditioned values."
        )
    return result


def _warn_nonuniform_axes(axes):
    """Warn when index-space searches differ from physical-space distances."""
    steps = []
    for dim, axis in enumerate(axes):
        axis = np.asarray(axis, dtype=np.double)
        if axis.size < 2:
            continue
        diffs = np.diff(axis)
        if not np.allclose(diffs, diffs[0]):
            warnings.warn(
                f"DirectSampling: axis {dim} is not uniformly spaced. DS "
                "works in grid-index space (cells), ignoring physical spacing.",
                UserWarning,
                stacklevel=3,
            )
            return
        steps.append(abs(diffs[0]))
    if len(steps) > 1 and not np.allclose(steps, steps[0]):
        warnings.warn(
            "DirectSampling: grid axes have unequal spacing. DS works in "
            "grid-index space (cells), ignoring physical spacing.",
            UserWarning,
            stacklevel=3,
        )


def _nearest_axis_index(ax, p):
    """Index of the ``ax`` entry nearest to each of ``p`` (lower index on ties).

    Equals ``np.argmin(np.abs(ax - p_k))`` for every point. A strictly
    increasing axis is bisected; any other axis uses a chunked brute force.
    """
    if len(ax) == 1:
        return np.zeros(p.size, dtype=np.intp)
    if np.all(np.diff(ax) > 0):
        right = np.clip(np.searchsorted(ax, p), 1, len(ax) - 1)
        left = right - 1
        return np.where(
            np.abs(ax[left] - p) <= np.abs(ax[right] - p), left, right
        )
    idx = np.empty(p.size, dtype=np.intp)
    step = max(1, 2**22 // len(ax))
    for s in range(0, p.size, step):
        idx[s : s + step] = np.argmin(
            np.abs(ax[None, :] - p[s : s + step, None]), axis=1
        )
    return idx


def _closest_per_node(flat, dist_sq):
    """Pick one point per node: the closest, the first one on ties.

    Returns the sorted unique node indices and, for each, the position of the
    winning point in ``flat``. ``np.lexsort`` is stable, so among equally
    close points the earliest keeps its place.
    """
    order = np.lexsort((dist_sq, flat))
    sorted_flat = flat[order]
    first = np.ones(sorted_flat.size, dtype=bool)
    first[1:] = sorted_flat[1:] != sorted_flat[:-1]
    return sorted_flat[first], order[first]


class DirectSampling(Field):
    """Multiple Point Statistics simulation using Direct Sampling.

    Subclasses :class:`gstools.field.base.Field`. Takes an :class:`MPSModel`
    that bundles a :class:`TrainingImage` and all algorithm parameters, and
    produces fields on structured grids.

    Parameters
    ----------
    model : MPSModel
        Algorithm configuration: training image, scan fraction, threshold,
        conditioning weight, and boundary strategy. Neighbour count and
        maximum search radius live on each :class:`Variable` inside the
        :class:`TrainingImage`. Build one with
        ``MPSModel(ti, scan_fraction=…, threshold=…)``.
    seed : int, None or nan, optional
        Master RNG seed. ``None`` or the default ``nan`` initializes from
        fresh entropy. DS uses spawned SeedSequences and MT19937 streams.

    Notes
    -----
    Thread resolution uses call-time ``num_threads``, then the instance
    setting, then :any:`gstools.config.NUM_THREADS`, then one thread when
    unset/falsy. Paths shorter than 512 nodes run serially. Within a fixed
    supported build and dependency environment, fixed inputs and seed controls
    are intended to give identical results across DS thread counts. This does
    not promise bitwise identity across platforms or algorithm releases.

    See Also
    --------
    MPSModel : holds the algorithm parameters.
    TrainingImage : training data holder.
    """

    default_field_names = ["field"]

    def __init__(
        self,
        model,
        seed=np.nan,
    ):
        if not isinstance(model, MPSModel):
            raise TypeError(
                "DirectSampling requires an MPSModel as its first argument. "
                "Wrap a TrainingImage first: "
                "DirectSampling(MPSModel(ti, scan_fraction=…))"
            )

        super().__init__(model=None, dim=model.ti.ndim, value_type="scalar")
        # Retain the MPSModel as the single source of truth for search
        # parameters. It is NOT stored in Field._model (which stays None) —
        # Field.pre_pos/field_dim/latlon/temporal assume a CovModel there.
        self._mps_model = model
        self._ti = model.ti
        self._num_threads = None
        self._cond_pos = None
        self._cond_val = None
        self._mv_mean = {}
        self._mv_normalizer = {}
        self._mv_trend = {}
        self.rng = RNG(_seed_value(seed, "seed"), seed_sequence=True)
        if model.ti.multivariate:
            for v in model.ti.variables:
                if v.name in dir(self):
                    raise ValueError(
                        f"DirectSampling: variable name {v.name!r} collides with an "
                        f"existing attribute."
                    )

    def __call__(
        self,
        pos=None,
        seed=np.nan,
        path_seed=np.nan,
        node_seed=np.nan,
        path="random",
        mesh_type="structured",
        post_process=True,
        store=True,
        num_threads=None,
    ):
        """Generate the spatial random field via Direct Sampling.

        For a univariate training image the field is saved as ``self.field``
        and is also returned. For a multivariate training image no single
        ``self.field`` is set — each variable is stored under its own name
        (accessible via ``self[variable]`` / :attr:`all_fields`) and a
        ``{variable: numpy.ndarray}`` dict is returned.

        Parameters
        ----------
        pos : :class:`list`, optional
            The position tuple, containing main direction and transversal
            directions. Only structured grids are supported.
        seed : :class:`int`, :any:`None` or nan, optional
            Reset the master sequence with an integer. Omitted/``np.nan``
            continues its existing sequence; ``None`` resets it from fresh
            entropy. This differs from the constructor, where both unset
            forms initialize a fresh sequence.
            Default: ``np.nan``
        path_seed : :class:`int` or :any:`numpy.nan`, optional
            Seed controlling the order in which simulation grid nodes are
            visited. If ``np.nan`` (default) or ``None``, derived from the
            master RNG together with ``node_seed``. Fix this while varying
            ``node_seed`` to study the effect of TI search randomness under a
            constant visit order, or vice versa. Only has an effect when
            ``path="random"``; inert for ``"sequential"`` and explicit arrays.
            Default: ``np.nan``
        node_seed : :class:`int` or :any:`numpy.nan`, optional
            Seed controlling the TI scan entry point and fallback cell for
            every node. If ``np.nan`` (default) or ``None``, derived from the
            master RNG. Equal explicit path/node seeds intentionally initialize
            equal control streams; separate controls do not guarantee statistical
            independence for every user-chosen pair.
            Default: ``np.nan``
        path : :class:`str` or array-like, optional
            Node visit order.  ``"random"`` (default) shuffles the unknown
            nodes uniformly using the ``path_seed``-controlled RNG — the
            standard DS behaviour.  ``"sequential"`` visits nodes in raster
            (lexicographic) order, which is deterministic regardless of
            ``path_seed``.  An explicit integer array of shape ``(N, dim)``
            provides a caller-supplied visit order and must include every
            unknown node (conditioned nodes are ignored; missing unknown
            nodes and duplicate rows raise :class:`ValueError`).
            Default: ``"random"``
        mesh_type : :class:`str`, optional
            Grid type. Must be ``"structured"``.
            Default: ``"structured"``
        post_process : :class:`bool`, optional
            Whether to apply post-processing transformations (mean,
            normalizer, trend) to the raw field. This does not disable DS
            cleanup passes configured by ``MPSModel(post_processing=...)``.
            Hard-data preservation refers to the raw DS field; these downstream
            transformations can change its displayed numeric values.
            Default: :any:`True`
        store : :class:`bool` or :class:`str`, optional
            Whether to store the field (``True``), not store it (``False``),
            or store it under a custom name (string).
            Default: :any:`True`
        Returns
        -------
        field : :class:`numpy.ndarray` or :class:`dict`
            For a univariate training image, the simulated field (also saved as
            ``self.field``). For a multivariate training image, a
            ``{variable: numpy.ndarray}`` dict with all variables on equal
            footing — each is also stored as a named field accessible via
            ``self[variable]`` / :attr:`all_fields`.
        """
        if mesh_type != "structured":
            raise ValueError(
                "DirectSampling: only structured grids are supported."
            )
        if self._ti.multivariate and isinstance(store, str):
            # A multivariate run produces one field per variable; there is no
            # single privileged field to store under a custom name. Reject it
            # explicitly rather than silently dropping the name (the fields are
            # always stored under their variable names).
            raise ValueError(
                "DirectSampling: a custom store name is not supported for "
                "multivariate training images; each variable is stored under "
                "its own name. Use store=True/False instead."
            )
        name, save = self.get_store_config(store)
        pos, shape = self.pre_pos(pos, mesh_type)
        # Reject non-finite grid coordinates before snapping, coordinate maps or
        # stored field metadata can use them (audit DS15). Nonuniform *finite*
        # axes remain allowed; their spacing is interpreted in index space.
        for axis in self.pos:
            if not np.all(np.isfinite(np.asarray(axis, dtype=np.double))):
                raise ValueError(
                    "DirectSampling: simulation grid coordinates must be finite "
                    "(NaN/inf axis values are not allowed)."
                )
        _warn_nonuniform_axes(self.pos)
        # Resolve the MPSModel's grid-independent rotation/scale specs to
        # canonical per-node maps now that the grid is known (Mariethoz2010
        # §6.2). Non-callable arrays are re-validated here against the grid
        # shape, and callables are materialized and finiteness-checked.
        # Coordinate callables receive one flat array per axis (meshgridded in
        # C-order), so ``f(*grid_coords)`` yields one value per simulation node.
        grid_coords = [
            g.ravel() for g in np.meshgrid(*self.pos, indexing="ij")
        ]
        rotation_map, _ = resolve_spec(
            self._mps_model.rotation,
            shape,
            no_of_angles(len(shape)),
            grid_coords,
            "rotation",
            angles=True,
        )
        scale_map, _ = resolve_spec(
            self._mps_model.scale,
            shape,
            len(shape),
            grid_coords,
            "scale",
            positive=True,
        )
        # Resolve the zone masks against this grid into an integer selector
        # (0 = primary TI, i+1 = zones[i]); overlap/shape/dtype errors surface
        # here. The native engine maps it to a per-node domain and scans each
        # node's TI, with neighbours still gathered across zone boundaries.
        zones = self._mps_model.zones
        zone_selector = (
            build_zone_selector(zones, shape, grid_coords) if zones else None
        )
        active_mask = self._mps_model.active_mask
        if active_mask is not None:
            if active_mask.shape != tuple(shape):
                raise ValueError(
                    "DirectSampling: active_mask shape must match the simulation grid"
                )
            if active_mask.dtype != bool:
                raise TypeError(
                    "DirectSampling: active_mask must remain boolean"
                )
        # Settings may have changed since set_condition. Reject invalid active
        # matrices before acquiring streams or entering native simulation.
        if self._cond_val is not None:
            values = (
                self._cond_val
                if self._ti.multivariate
                else {self._ti.variables[0].name: self._cond_val}
            )
            self._check_penalty_codes(
                values,
                include_cleanup=self._mps_model.post_processing > 0,
            )
        conditions = self._conditions_to_grid(self.pos)
        if active_mask is not None and conditions is not None:
            if not active_mask[conditions[0]].all():
                raise ValueError(
                    "DirectSampling: hard conditions outside active_mask are not allowed"
                )
        # Validate the effective thread setting before acquiring control streams.
        n_threads = (
            num_threads if num_threads is not None else self._num_threads
        )
        if n_threads is not None:
            n_threads = exact_count(
                n_threads, minimum=0, context="DirectSampling.num_threads"
            )
        master_seed = _seed_value(seed, "seed")
        path_seed = _seed_value(path_seed, "path_seed")
        node_seed = _seed_value(node_seed, "node_seed")
        if seed is None or master_seed is not None:
            self.rng.seed = master_seed
        rng_path = (
            self.rng.random
            if path_seed is None
            else RNG(path_seed, seed_sequence=True).random
        )
        rng_nodes = (
            self.rng.random
            if node_seed is None
            else RNG(node_seed, seed_sequence=True).random
        )
        result = _simulate_model(
            self._mps_model,
            shape,
            rng_path,
            rng_nodes,
            conditions=conditions,
            num_threads=n_threads,
            rotation_map=rotation_map,
            scale_map=scale_map,
            path=path,
            zone_selector=zone_selector,
        )
        # Branch only on the return type: multivariate → dict of named arrays;
        # univariate → bare array (unwrap the single None key).
        if self._ti.multivariate:
            # Equal treatment: every variable is a first-class named field
            # (no privileged primary). Returned as a dict keyed by variable name.
            # Per-variable transforms (mean/normalizer/trend) are applied here
            # using the dicts set via set_mv_transforms(); variables absent from
            # those dicts fall back to the inherited self.mean/normalizer/trend.
            # post_field is then called with process=False to handle storage only.
            out = {}
            for v in self._ti.variables:
                fld = result[v.name]
                if post_process and not v.categorical:
                    mv_mean = self._mv_mean.get(v.name, self.mean)
                    mv_norm = self._mv_normalizer.get(v.name, self.normalizer)
                    mv_trend = self._mv_trend.get(v.name, self.trend)
                    fld = apply_mean_norm_trend(
                        pos=self.pos,
                        field=fld,
                        mesh_type=self.mesh_type,
                        value_type=self.value_type,
                        mean=mv_mean,
                        normalizer=mv_norm,
                        trend=mv_trend,
                        check_shape=False,
                        stacked=False,
                    )
                out[v.name] = self.post_field(
                    fld, name=v.name, process=False, save=save
                )
            return out
        return self.post_field(
            result[self._ti.variables[0].name],
            name,
            post_process and not self._ti.variables[0].categorical,
            save,
        )

    def _conditions_to_grid(self, axes):
        """Snap conditioning points to nearest grid nodes (Mariethoz2010 §3 ¶12).

        Each point snaps to the nearest node along every axis (the lower index
        on ties). When two points snap to the same node, the conflict is
        resolved **per variable**: for each variable the closest colliding
        point with a finite value wins (the first one on ties). A farther point
        therefore still contributes its finite values for any variable the
        closer point left ``NaN``, so valid conditioning data is never silently
        discarded. All steps are vectorized, so exhaustive auxiliary data (one
        point per grid node) stays cheap on large grids.

        Parameters
        ----------
        axes : tuple of numpy.ndarray
            Raw grid axis arrays (``self.pos``).

        Returns
        -------
        tuple or None
            ``(nodes, values)``: ``nodes`` is a tuple of per-axis index arrays
            of the conditioned nodes, ``values`` maps each variable name to one
            value per node (``NaN`` where that variable is not conditioned).
            ``None`` if there is no conditioning data.
        """
        if self._cond_pos is None:
            return None
        if self._ti.multivariate and not self._cond_val:
            raise ValueError("DirectSampling: cond_val must not be empty")
        axes = [np.asarray(ax, dtype=np.double) for ax in axes]
        points = [
            np.asarray(self._cond_pos[d], dtype=np.double)
            for d in range(self.dim)
        ]
        if not points[0].size:
            return None
        # Points beyond half a cell outside the grid snap to its edge: warn.
        outside = 0
        for ax, p in zip(axes, points):
            tol = 0.5 * np.abs(np.diff(ax)).max() if len(ax) > 1 else 0.0
            outside |= (p < ax.min() - tol) | (p > ax.max() + tol)
        if np.any(outside):
            warnings.warn(
                f"DirectSampling: {int(np.sum(outside))} conditioning "
                "point(s) lie outside the simulation grid and were snapped "
                "to its nearest edge node.",
                UserWarning,
                stacklevel=3,
            )
        nearest = [_nearest_axis_index(ax, p) for ax, p in zip(axes, points)]
        dist_sq = sum(
            (ax[i] - p) ** 2 for ax, i, p in zip(axes, nearest, points)
        )
        shape = tuple(len(ax) for ax in axes)
        flat = np.ravel_multi_index(nearest, shape)
        if self._ti.multivariate:
            # Fill each variable from its own closest finite point.
            node_flat = np.unique(flat)
            values = {}
            # finite values landing on an occupied (node, variable)
            n_collided = 0
            for v, val in self._cond_val.items():
                val = np.asarray(val, dtype=np.double)
                finite = np.flatnonzero(np.isfinite(val))
                won_flat, won = _closest_per_node(
                    flat[finite], dist_sq[finite]
                )
                n_collided += finite.size - won_flat.size
                column = np.full(node_flat.size, np.nan)
                column[np.searchsorted(node_flat, won_flat)] = val[finite[won]]
                values[v] = column
            informed = np.logical_or.reduce(
                [np.isfinite(column) for column in values.values()]
            )
            if not informed.any():
                return None
            node_flat = node_flat[informed]
            values = {v: column[informed] for v, column in values.items()}
        else:
            # univariate (NaN values were already dropped in set_condition)
            node_flat, won = _closest_per_node(flat, dist_sq)
            n_collided = flat.size - node_flat.size
            val = np.asarray(self._cond_val, dtype=np.double)
            values = {self._ti.variables[0].name: val[won]}
        if n_collided:
            warnings.warn(
                f"DirectSampling: {n_collided} conditioning value(s) collided "
                "on an occupied grid node; the one closest to the node was "
                "kept (the first one on ties).",
                UserWarning,
                stacklevel=3,
            )
        return np.unravel_index(node_flat, shape), values

    def set_condition(self, cond_pos, cond_val):
        """Set the conditioning data for the simulation.

        Conditioning weight is configured per variable on the :class:`Variable`
        (``cond_weight=``), or via the univariate :class:`TrainingImage` sugar.

        Parameters
        ----------
        cond_pos : :class:`list`
            The position tuple of the conditioning data ``(x, [y, z])``.
        cond_val : :class:`numpy.ndarray` or :class:`dict`
            Univariate: values at the conditioning positions. Multivariate: a
            ``{variable: numpy.ndarray}`` mapping (use ``numpy.nan`` for a
            variable that is not conditioned at a given point).
        """
        # Conditioning updates are transactional (audit DS08): everything is
        # converted and validated into locals first, and ``self._cond_pos`` /
        # ``self._cond_val`` are reassigned only after every check passes, so a
        # rejected update never corrupts the state a later simulation reuses.
        _validate_real_data(
            cond_pos, False, "DirectSampling conditioning positions"
        )
        if self._ti.multivariate and isinstance(cond_val, dict):
            if not cond_val:
                raise ValueError("DirectSampling: cond_val must not be empty")
            unknown = set(cond_val) - {v.name for v in self._ti.variables}
            if unknown:
                raise ValueError(
                    f"cond_val contains unknown variable(s): {sorted(unknown)}. "
                    f"TI variables are: {sorted(v.name for v in self._ti.variables)}"
                )
            cond_pos_arr = np.asarray(cond_pos, dtype=np.double).reshape(
                self.dim, -1
            )
            _check_finite_positions(cond_pos_arr)
            n_cond = cond_pos_arr.shape[1]
            local_val = {}
            for v, a in cond_val.items():
                arr = _condition_values(a, self._ti.variable(v))
                # Require exactly one scalar per conditioning position; reject an
                # (N, k) array up front instead of failing later in snapping with
                # an ambiguous-truth error (audit DS08).
                if arr.ndim != 1:
                    raise ValueError(
                        f"DirectSampling: cond_val[{v!r}] must be a 1-D array "
                        f"with one value per conditioning position; got shape "
                        f"{arr.shape!r}."
                    )
                if arr.shape[0] != n_cond:
                    raise ValueError(
                        "DirectSampling: all cond_val arrays must match the "
                        f"conditioning-position count ({n_cond}); got "
                        f"{arr.shape[0]} for {v!r}."
                    )
                local_val[v] = arr
            self._check_penalty_codes(local_val)  # DS09
            self._cond_pos = cond_pos_arr
            self._cond_val = local_val
        elif self._ti.multivariate:
            raise ValueError(
                "DirectSampling: cond_val must be a dict {variable: array} "
                "for multivariate TrainingImages."
            )
        else:
            from gstools.krige.tools import set_condition as _gs_set_condition

            # Validate into locals BEFORE storing so a non-finite position does
            # not leave invalid conditioning installed (audit DS08 univariate).
            local_pos, local_val = _gs_set_condition(
                cond_pos,
                _condition_values(cond_val, self._ti.variables[0]),
                self.dim,
            )
            _check_finite_positions(local_pos)
            uv_name = self._ti.variables[0].name
            self._check_penalty_codes(
                {uv_name: np.asarray(local_val, dtype=np.double)}
            )  # DS09
            self._cond_pos, self._cond_val = local_pos, local_val

    def _check_penalty_codes(self, values_by_name, include_cleanup=True):
        """Screen hard category codes against active penalty matrices (DS09).

        A categorical penalty matrix is indexed by the hard code during the
        scan, so a non-integer, negative, or out-of-range code would silently
        become a full-mismatch (or, for the raw kernel, an invalid index). The
        effective matrices are the primary TI's and any cleanup
        ``post_processing_variables`` override's; a finite hard value must be a
        valid code ``[0, C)`` of each.
        """
        from gstools.mps.training_image import _check_category_codes

        for vals in values_by_name.values():
            if np.isinf(np.asarray(vals, dtype=np.double)).any():
                raise ValueError(
                    "DirectSampling: conditioning values must not contain "
                    "infinity; use NaN for unconditioned values."
                )
        checks = {}
        for var in self._ti.variables:
            if var.penalty_matrix is not None:
                checks.setdefault(var.name, []).append(
                    (var.penalty_matrix.shape[0], "penalty_matrix")
                )
        overrides = (
            self._mps_model.post_processing_variables
            if include_cleanup
            else []
        )
        for ov in overrides:
            if ov.penalty_matrix is not None:
                checks.setdefault(ov.name, []).append(
                    (
                        ov.penalty_matrix.shape[0],
                        "post_processing penalty_matrix",
                    )
                )
        for name, matrix_checks in checks.items():
            if name not in values_by_name:
                continue
            vals = np.asarray(values_by_name[name], dtype=np.double)
            finite = vals[np.isfinite(vals)]
            for ncat, label in matrix_checks:
                _check_category_codes(
                    finite,
                    ncat,
                    f"DirectSampling.set_condition: conditioning code for "
                    f"{name!r} (checked against {label})",
                )

    def set_mv_transforms(self, mean=None, normalizer=None, trend=None):
        """Set per-variable post-processing transforms for multivariate simulations.

        Only meaningful when the training image is multivariate. Variables not
        listed here fall back to the instance-level ``self.mean``,
        ``self.normalizer``, and ``self.trend``.

        Parameters
        ----------
        mean : dict of {str: scalar or callable}, optional
            Per-variable mean.
        normalizer : dict of {str: Normalizer}, optional
            Per-variable normalizer.
        trend : dict of {str: scalar or callable}, optional
            Per-variable trend (applied after denormalization).
        """
        from gstools.field.base import _set_mean_trend
        from gstools.normalizer.tools import _check_normalizer

        if mean is not None:
            self._mv_mean = {
                v: _set_mean_trend(m, self.dim) for v, m in mean.items()
            }
        if normalizer is not None:
            self._mv_normalizer = {
                v: _check_normalizer(n) for v, n in normalizer.items()
            }
        if trend is not None:
            self._mv_trend = {
                v: _set_mean_trend(t, self.dim) for v, t in trend.items()
            }

    @property
    def mps_model(self):
        """:any:`MPSModel`: the search-parameter configuration (training image + parameters)."""
        return self._mps_model

    @property
    def ti(self):
        """:any:`TrainingImage`: The training image model."""
        return self._ti

    def _collapse_per_var(self, attr):
        """Read ``attr`` from every TI Variable; scalar if all equal, else a dict.

        Read-only view over the variables' values. Used by the
        :attr:`n_neighbors` and :attr:`max_radius` getters.
        """
        vars_ = self._ti.variables
        vals = [getattr(v, attr) for v in vars_]
        if len(set(vals)) == 1:
            return vals[0]
        return {v.name: getattr(v, attr) for v in vars_}

    @property
    def n_neighbors(self):
        """:class:`int` or :class:`dict`: n_neighbors per variable (scalar if all equal).

        Read-only. Configure on the :class:`Variable` (``ti.variable(name).n_neighbors = …``)
        or at :class:`TrainingImage` construction.
        """
        return self._collapse_per_var("n_neighbors")

    @property
    def scan_fraction(self):
        """:class:`float`: Fraction of the TI scanned per node. Read-only; set on the MPSModel."""
        return self._mps_model.scan_fraction

    @property
    def threshold(self):
        """:class:`float`: Distance threshold (0.0 → DSBC mode). Read-only; set on the MPSModel."""
        return self._mps_model.threshold

    @property
    def cond_weight(self):
        """:class:`float` or :class:`dict`: Conditioning-node weight per variable (scalar if all equal).

        Read-only. Configure on the :class:`Variable` (``ti.variable(name).cond_weight``)
        or via the univariate :class:`TrainingImage` ``cond_weight=`` sugar.
        """
        return self._collapse_per_var("cond_weight")

    @property
    def boundary(self):
        """:class:`str`: Search-window strategy. Read-only; set on the MPSModel."""
        return self._mps_model.boundary

    @property
    def max_radius(self):
        """:class:`float` or :any:`None` or :class:`dict`: Euclidean neighbour cap (scalar if all equal).

        Read-only. Configure at :class:`Variable`/:class:`TrainingImage` construction.
        """
        return self._collapse_per_var("max_radius")

    @property
    def num_threads(self):
        """:class:`int` or :any:`None`: Number of threads for outer DAG parallelism."""
        return self._num_threads

    @num_threads.setter
    def num_threads(self, value):
        self._num_threads = (
            None
            if value is None
            else exact_count(
                value, minimum=0, context="DirectSampling.num_threads"
            )
        )

    def __repr__(self):
        parts = [
            f"DirectSampling(dim={self.dim}, ",
            f"n_neighbors={self.n_neighbors}, ",
            f"scan_fraction={self.scan_fraction}, ",
            f"threshold={self.threshold}, ",
            f"boundary={self.boundary!r}",
        ]
        if self._mps_model.rotation is not None:
            parts.append(f", rotation={self._mps_model.rotation!r}")
        if self._mps_model.scale is not None:
            parts.append(f", scale={self._mps_model.scale!r}")
        return "".join(parts) + ")"
