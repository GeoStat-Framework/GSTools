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
from gstools.mps.model import MPSModel
from gstools.mps.nonstationary import build_zone_selector, resolve_spec
from gstools.mps.simulate import ds_simulate
from gstools.mps.training_image import _check_category_codes
from gstools.normalizer.tools import apply_mean_norm_trend
from gstools.random.rng import RNG
from gstools.tools.geometric import generate_grid, no_of_angles

__all__ = ["DirectSampling"]


def _check_penalty_matrix_conditioning(var_name, penalty_matrix, values):
    """Validate conditioning values against a Variable's penalty_matrix, if set.

    Fails fast at ``set_condition()`` time with a clear ``ValueError`` rather
    than raising a raw ``IndexError`` deep in the scan loop when the matrix
    is later indexed by an out-of-range or non-integer category code.
    """
    if penalty_matrix is None:
        return
    label = "variable" if var_name is None else f"variable {var_name!r}"
    _check_category_codes(
        values,
        penalty_matrix.shape[0],
        f"DirectSampling.set_condition: conditioning values for {label}",
    )


def _warn_nonuniform_axes(axes):
    """Warn when a transform/zonation is active on non-index-like axes.

    Rotation/scale act on **index lags** and zone masks are per-cell; for
    uniformly and equally spaced axes index-space and physical-space geometry
    coincide (the spacing matrix h·I commutes with rotation). Otherwise a
    physical-space rotation is not an index-space rotation — warn (MPS-local;
    Mariethoz2010 §6.2 geometry is defined on cells).
    """
    steps = []
    for d, ax in enumerate(axes):
        ax = np.asarray(ax, dtype=np.double)
        if len(ax) < 2:
            continue
        diffs = np.diff(ax)
        if not np.allclose(diffs, diffs[0]):
            warnings.warn(
                f"DirectSampling: axis {d} is not uniformly spaced while a "
                "geometric transform or zonation is active. Rotation/scale "
                "act on index lags (cell offsets); on non-uniform axes the "
                "result is not a physical-space transform.",
                UserWarning,
                stacklevel=3,
            )
            return
        steps.append(diffs[0])
    if len(steps) > 1 and not np.allclose(steps, steps[0]):
        warnings.warn(
            "DirectSampling: grid axes have unequal spacing while a "
            "geometric transform or zonation is active. Rotation/scale act "
            "on index lags (cell offsets); with anisotropic spacing the "
            "result is not a physical-space transform.",
            UserWarning,
            stacklevel=3,
        )


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
    seed : int or nan, optional
        Master RNG seed. Default: nan.

    Notes
    -----
    Runtime concerns (thread count, progress reporting) are set on the
    instance via :attr:`num_threads` or passed as keyword arguments to
    :meth:`__call__`.

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
        self.rng = RNG(None if np.isnan(seed) else int(seed))
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
        progress=None,
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
        seed : :class:`int`, optional
            Seed for the RNG. If ``np.nan``, the current seed is kept.
            Default: ``np.nan``
        path_seed : :class:`int` or :any:`numpy.nan`, optional
            Seed controlling the order in which simulation grid nodes are
            visited. If ``np.nan`` (default), derived from the master RNG
            together with ``node_seed``. Fix this while varying ``node_seed``
            to study the effect of TI search randomness under a constant
            visit order, or vice versa.  Only has an effect when
            ``path="random"``; inert for ``"sequential"`` and explicit arrays.
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
            provides a caller-supplied visit order and must be a permutation
            of exactly the unknown-node set (missing nodes, extra nodes, and
            duplicate rows all raise :class:`ValueError`).
            Default: ``"random"``
        mesh_type : :class:`str`, optional
            Grid type. Must be ``"structured"``.
            Default: ``"structured"``
        post_process : :class:`bool`, optional
            Whether to apply post-processing transformations (mean,
            normalizer, trend) to the field. Default: :any:`True`
        store : :class:`bool` or :class:`str`, optional
            Whether to store the field (``True``), not store it (``False``),
            or store it under a custom name (string).
            Default: :any:`True`
        progress : :class:`bool` or callable or None, optional
            Show simulation progress. ``True`` prints a plain percentage line
            (no third-party dependency); a callable is invoked as
            ``progress(n_done, n_total)`` once per completed node.
            ``None``/``False`` (default) disables it.

        Returns
        -------
        field : :class:`numpy.ndarray` or :class:`dict`
            For a univariate training image, the simulated field (also saved as
            ``self.field``). For a multivariate training image, a
            ``{variable: numpy.ndarray}`` dict with all variables on equal
            footing — each is also stored as a named field accessible via
            ``self[variable]`` / :attr:`all_fields`.

        Notes
        -----
        DS post-processing passes (``MPSModel(post_processing=..., post_processing_factor=...)``,
        Me13 §4) run after the main simulation pass, over the same
        ``num_threads``-parallel dependency DAG. They are distinct from this
        method's ``post_process`` kwarg (Field mean/normalizer/trend pipeline).
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
        dim = len(shape)
        flat_pos = generate_grid(self.pos)
        rotation_map, rot_stat = resolve_spec(
            self._mps_model.rotation,
            shape,
            no_of_angles(dim),
            flat_pos,
            "rotation",
        )
        scale_map, scale_stat = resolve_spec(
            self._mps_model.scale,
            shape,
            dim,
            flat_pos,
            "scale",
            positive=True,
        )
        zones = self._mps_model.zones
        selector = (
            build_zone_selector(zones, shape, flat_pos) if zones else None
        )
        if (
            rotation_map is not None
            or scale_map is not None
            or selector is not None
        ):
            _warn_nonuniform_axes(self.pos)
        conditions = self._conditions_to_grid(self.pos)
        if not np.isnan(seed):
            self.rng.seed = int(seed)
        rng_path = (
            self.rng.random
            if np.isnan(path_seed)
            else RNG(int(path_seed)).random
        )
        rng_nodes = (
            self.rng.random
            if np.isnan(node_seed)
            else RNG(int(node_seed)).random
        )
        # Call-time num_threads overrides the instance default.
        n_threads = (
            num_threads if num_threads is not None else self._num_threads
        )
        # ds_simulate now accepts both univariate and multivariate TIs directly.
        # Univariate TIs produce {None: array}; multivariate produce {name: array}.
        # Conditions for univariate are {idx: value}; ds_simulate expects
        # {idx: {var: value}} so we remap using the sentinel key None.
        if self._ti.multivariate:
            sim_conditions = conditions  # already {idx: {var: val}}
        else:
            sim_conditions = (
                {idx: {None: val} for idx, val in conditions.items()}
                if conditions
                else None
            )
        result = ds_simulate(
            training_image=self._ti,
            sim_shape=shape,
            threshold=self.threshold,
            scan_fraction=self.scan_fraction,
            rng_path=rng_path,
            rng_nodes=rng_nodes,
            conditions=sim_conditions,
            cond_weight=self.cond_weight,
            boundary=self.boundary,
            num_threads=n_threads,
            rotation_map=rotation_map,
            scale_map=scale_map,
            stationary_transform=rot_stat and scale_stat,
            progress=progress,
            path=path,
            zone_tis=[z.ti for z in zones] if zones else None,
            zone_selector=selector,
            post_processing=self._mps_model.post_processing,
            post_processing_factor=self._mps_model.post_processing_factor,
            post_processing_path=self._mps_model.post_processing_path,
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
                if post_process:
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
        return self.post_field(result[None], name, post_process, save)

    def _conditions_to_grid(self, axes):
        """Snap conditioning points to nearest grid nodes (Mariethoz2010 §3 ¶12).

        Univariate returns ``{idx: value}``; multivariate returns
        ``{idx: {variable: value}}`` with non-finite (NaN) entries skipped. When
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
        if self._ti.multivariate and isinstance(self._cond_val, dict):
            if not self._cond_val:
                raise ValueError("DirectSampling: cond_val must not be empty")
            # idx -> {variable: (value, dist_sq)}; the closest finite value per
            # variable wins, so a farther point fills variables the closer one
            # left NaN (per-variable collision resolution).
            candidates = {}
            n_cond = len(next(iter(self._cond_val.values())))
            for k in range(n_cond):
                idx, dist_sq = self._snap_to_nearest(axes, k)
                var_best = candidates.setdefault(idx, {})
                for v in self._cond_val:
                    val = self._cond_val[v][k]
                    if not np.isfinite(val):
                        continue
                    if v not in var_best or dist_sq < var_best[v][1]:
                        var_best[v] = (float(val), dist_sq)
            return {
                idx: {v: val for v, (val, _) in var_best.items()}
                for idx, var_best in candidates.items()
            }
        # univariate (unchanged)
        candidates = {}  # idx -> (val, dist_sq)
        for k in range(self._cond_val.shape[0]):
            idx, dist_sq = self._snap_to_nearest(axes, k)
            if idx not in candidates or dist_sq < candidates[idx][1]:
                candidates[idx] = (self._cond_val[k], dist_sq)
        return {idx: val for idx, (val, _) in candidates.items()}

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

        Conditioning weight is configured on the :class:`MPSModel` (``cond_weight=``).

        Parameters
        ----------
        cond_pos : :class:`list`
            The position tuple of the conditioning data ``(x, [y, z])``.
        cond_val : :class:`numpy.ndarray` or :class:`dict`
            Univariate: values at the conditioning positions. Multivariate: a
            ``{variable: numpy.ndarray}`` mapping (use ``numpy.nan`` for a
            variable that is not conditioned at a given point).
        """
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
            n_cond = len(next(iter(cond_val.values())))
            for v, arr in cond_val.items():
                if len(arr) != n_cond:
                    raise ValueError(
                        "DirectSampling: all cond_val arrays must have the same "
                        f"length; got {n_cond} for {next(iter(cond_val))!r} but "
                        f"{len(arr)} for {v!r}."
                    )
            if cond_pos_arr.shape[1] != n_cond:
                raise ValueError(
                    "DirectSampling: cond_pos and cond_val length mismatch."
                )
            self._cond_pos = cond_pos_arr
            self._cond_val = {
                v: np.asarray(a, dtype=np.double) for v, a in cond_val.items()
            }
            for v_name, arr in self._cond_val.items():
                _check_penalty_matrix_conditioning(
                    v_name, self._ti.variable(v_name).penalty_matrix, arr
                )
        elif self._ti.multivariate:
            raise ValueError(
                "DirectSampling: cond_val must be a dict {variable: array} "
                "for multivariate TrainingImages."
            )
        else:
            from gstools.krige.tools import set_condition as _gs_set_condition

            self._cond_pos, self._cond_val = _gs_set_condition(
                cond_pos, cond_val, self.dim
            )
            _check_penalty_matrix_conditioning(
                None, self._ti.variable().penalty_matrix, self._cond_val
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
        """:class:`float`: Conditioning-node weight. Read-only; set on the MPSModel."""
        return self._mps_model.cond_weight

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
        self._num_threads = None if value is None else int(value)

    def __repr__(self):
        parts = [
            f"DirectSampling(dim={self.dim}, ",
            f"n_neighbors={self.n_neighbors}, ",
            f"scan_fraction={self.scan_fraction}, ",
            f"threshold={self.threshold}, ",
            f"boundary={self.boundary!r}",
        ]
        return "".join(parts) + ")"
