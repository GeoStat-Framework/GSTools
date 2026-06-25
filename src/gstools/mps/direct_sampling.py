"""
GStools subpackage providing the Direct Sampling MPS simulation class.

.. currentmodule:: gstools.mps

The following classes and functions are provided

.. autosummary::
   DirectSampling
"""

import numpy as np

from gstools.field.base import Field
from gstools.mps.model import MPSModel
from gstools.mps.model import _validate_boundary as _mv_validate_boundary
from gstools.mps.model import _validate_max_radius as _mv_validate_max_radius
from gstools.mps.model import _validate_n_neighbors as _mv_validate_n_neighbors
from gstools.mps.model import (
    _validate_scan_fraction as _mv_validate_scan_fraction,
)
from gstools.mps.model import _validate_threshold as _mv_validate_threshold
from gstools.mps.simulate import _MV_VAR, _univar_as_mv_ti, ds_simulate
from gstools.normalizer.tools import apply_mean_norm_trend
from gstools.random.rng import RNG
from gstools.tools.geometric import no_of_angles

__all__ = ["DirectSampling"]


def _resolve_nonstationary_map(param, sim_shape, n_stationary):
    """Return ``None`` or an array broadcastable to ``sim_shape``.

    Scalars and 0-d arrays are broadcast to the grid. A 1-D vector is a
    *stationary* multi-component value (e.g. a 3-element Tait–Bryan angle triple
    for a 3-D grid) and is accepted only when its length equals
    ``n_stationary`` — the number of components a stationary value has for this
    parameter and grid dimension. A *per-node* map must have leading dimensions
    equal to ``sim_shape`` (i.e. a full-shape array, not a flattened vector).

    Parameters
    ----------
    param : scalar or array-like or None
        User-supplied rotation/anis value.
    sim_shape : tuple
        Simulation grid shape.
    n_stationary : int
        Component count of a stationary value for this parameter/dimension
        (``no_of_angles(dim)`` for rotation, ``dim - 1`` for anis).

    Raises
    ------
    ValueError
        If a 1-D vector matches neither a stationary value nor the grid — the
        common cause is a per-node map passed *flattened* instead of shaped
        like the grid, which would otherwise silently apply only element [0].
    """
    if param is None:
        return None
    arr = np.asarray(param, dtype=np.float64)
    if arr.ndim == 0 or arr.size == 1:
        return np.full(sim_shape, float(arr.flat[0]))
    if arr.ndim == 1:
        # Only a genuine stationary multi-component value is accepted as 1-D.
        # A per-node map is multi-dimensional (leading dims == grid); a 1-D
        # array of any other length is almost certainly a flattened per-node
        # map and would silently degrade to its first element — reject it.
        if arr.size == n_stationary:
            return arr
        raise ValueError(
            f"1-D non-stationary value of length {arr.size} matches neither a "
            f"stationary value ({n_stationary} component(s) for a "
            f"{len(sim_shape)}-D grid) nor a per-node map. For per-node values "
            f"pass an array shaped like the grid {tuple(sim_shape)!r}, not a "
            f"flattened vector."
        )
    if arr.shape[: len(sim_shape)] == tuple(sim_shape):
        return arr
    raise ValueError(
        f"Non-stationary map shape {arr.shape!r} is incompatible with "
        f"simulation grid shape {tuple(sim_shape)!r}. Pass a scalar or "
        f"1-D vector for a stationary value, or an array whose leading "
        f"dimensions match the grid."
    )


class DirectSampling(Field):
    """Multiple Point Statistics simulation using Direct Sampling.

    Subclasses :class:`gstools.field.base.Field`. Takes an :class:`MPSModel`
    that bundles a :class:`TrainingImage` and all algorithm parameters, and
    produces fields on structured grids.

    Parameters
    ----------
    model : MPSModel
        Algorithm configuration: training image, neighbour count, scan
        fraction, threshold, conditioning weight, boundary strategy, and
        maximum search radius. Build one with
        ``MPSModel(ti, n_neighbors=…, scan_fraction=…, …)``.
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

    @staticmethod
    def _validate_variation_n_neighbors(n_neighbors, ti):
        """Reject variation distance with a single-neighbour data event.

        The variation distance (Mariethoz2010 Eq. 9) compares each event to its
        own local mean. With ``n_neighbors == 1`` the local mean equals the sole
        value, so every deviation is zero and every TI candidate matches — the
        scan degenerates to a random draw. Fail fast instead.
        """
        dist_type = ti.distance_type
        if isinstance(dist_type, dict):
            dist_map = dist_type
            n_map = (
                n_neighbors
                if isinstance(n_neighbors, dict)
                else {v: n_neighbors for v in dist_map}
            )
        else:
            dist_map = {None: dist_type}
            n_map = {None: n_neighbors}
        for var, dt in dist_map.items():
            if isinstance(dt, str) and dt.lower().startswith("variation"):
                if int(n_map[var]) < 2:
                    where = "" if var is None else f" for variable {var!r}"
                    raise ValueError(
                        f"distance='variation' requires n_neighbors >= 2{where}: "
                        "a single-point data event has no local mean deviation."
                    )

    def __init__(
        self,
        model,
        seed=np.nan,
    ):
        if not isinstance(model, MPSModel):
            raise TypeError(
                "DirectSampling requires an MPSModel as its first argument. "
                "Wrap a TrainingImage first: "
                "DirectSampling(MPSModel(ti, n_neighbors=…, …))"
            )

        self._model = model
        super().__init__(model=None, dim=model.ti.ndim, value_type="scalar")
        self._ti = model.ti
        self._n_neighbors = model.n_neighbors
        DirectSampling._validate_variation_n_neighbors(
            self._n_neighbors, self._ti
        )
        self._scan_fraction = model.scan_fraction
        self._threshold = model.threshold
        self._cond_weight = model.cond_weight
        self._boundary = model.boundary
        self._max_radius = model.max_radius
        self._num_threads = None
        self._cond_pos = None
        self._cond_val = None
        self._mv_mean = {}
        self._mv_normalizer = {}
        self._mv_trend = {}
        self._rotation = None
        self._anis = None
        self.rng = RNG(None if np.isnan(seed) else int(seed))
        if model.ti.multivariate:
            for v in model.ti.variables:
                if not v.isidentifier() or v in dir(self):
                    raise ValueError(
                        f"DirectSampling: variable name {v!r} cannot be used as "
                        f"a field name; use a valid Python identifier that does "
                        f"not collide with an existing attribute."
                    )

    def __call__(
        self,
        pos=None,
        seed=np.nan,
        path_seed=np.nan,
        node_seed=np.nan,
        mesh_type="structured",
        post_process=True,
        store=True,
        progress=None,
        num_threads=None,
    ):
        """Generate the spatial random field via Direct Sampling.

        The field is saved as ``self.field`` and is also returned.

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
            visit order, or vice versa.
            Default: ``np.nan``
        node_seed : :class:`int` or :any:`numpy.nan`, optional
            Seed controlling the TI scan entry point and fallback cell for
            every node. If ``np.nan`` (default), derived from the master RNG.
            Default: ``np.nan``
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
            Show simulation progress. ``True`` displays a :mod:`tqdm` bar (or a
            plain percentage line if ``tqdm`` is not installed); a callable is
            invoked as ``progress(n_done, n_total)`` once per completed node.
            ``None``/``False`` (default) disables it.

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
        # Stationary component counts disambiguate a 1-D value from a flattened
        # per-node map: rotation has no_of_angles(dim) angles, anis has dim-1.
        rotation_map = _resolve_nonstationary_map(
            self._rotation, shape, no_of_angles(len(shape))
        )
        anis_map = _resolve_nonstationary_map(
            self._anis, shape, max(len(shape) - 1, 1)
        )
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
        if self._ti.multivariate:
            result = ds_simulate(
                training_image=self._ti,
                sim_shape=shape,
                n_neighbors=self._n_neighbors,
                threshold=self._threshold,
                scan_fraction=self._scan_fraction,
                rng_path=rng_path,
                rng_nodes=rng_nodes,
                conditions=conditions,
                cond_weight=self._cond_weight,
                boundary=self._boundary,
                max_radius=self._max_radius,
                num_threads=n_threads,
                rotation_map=rotation_map,
                anis_map=anis_map,
                progress=progress,
            )
            # Equal treatment: every variable is a first-class named field
            # (no privileged primary). Returned as a dict keyed by variable name.
            # Per-variable transforms (mean/normalizer/trend) are applied here
            # using the dicts set via set_mv_transforms(); variables absent from
            # those dicts fall back to the inherited self.mean/normalizer/trend.
            # post_field is then called with process=False to handle storage only.
            out = {}
            for v in self._ti.variables:
                fld = result[v]
                if post_process:
                    mv_mean = self._mv_mean.get(v, self.mean)
                    mv_norm = self._mv_normalizer.get(v, self.normalizer)
                    mv_trend = self._mv_trend.get(v, self.trend)
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
                out[v] = self.post_field(fld, name=v, process=False, save=save)
            return out
        mv_ti = _univar_as_mv_ti(self._ti)
        mv_cond = (
            {idx: {_MV_VAR: val} for idx, val in conditions.items()}
            if conditions
            else None
        )
        mv_result = ds_simulate(
            training_image=mv_ti,
            sim_shape=shape,
            n_neighbors=self._n_neighbors,
            threshold=self._threshold,
            scan_fraction=self._scan_fraction,
            rng_path=rng_path,
            rng_nodes=rng_nodes,
            conditions=mv_cond,
            cond_weight=self._cond_weight,
            boundary=self._boundary,
            max_radius=self._max_radius,
            num_threads=n_threads,
            rotation_map=rotation_map,
            anis_map=anis_map,
            progress=progress,
        )
        return self.post_field(mv_result[_MV_VAR], name, post_process, save)

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
                raise ValueError("cond_val must not be empty")
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

    def set_condition(self, cond_pos, cond_val, cond_weight=None):
        """Set the conditioning data for the simulation.

        Parameters
        ----------
        cond_pos : :class:`list`
            The position tuple of the conditioning data ``(x, [y, z])``.
        cond_val : :class:`numpy.ndarray` or :class:`dict`
            Univariate: values at the conditioning positions. Multivariate: a
            ``{variable: numpy.ndarray}`` mapping (use ``numpy.nan`` for a
            variable that is not conditioned at a given point).
        cond_weight : :class:`float`, optional
            Conditioning weight delta. If given, overrides the ``cond_weight`` set
            at construction. Default: :any:`None` (keep existing weight)
        """
        if cond_weight is not None:
            self._cond_weight = float(cond_weight)
        if self._ti.multivariate and isinstance(cond_val, dict):
            if not cond_val:
                raise ValueError("cond_val must not be empty")
            unknown = set(cond_val) - set(self._ti.variables)
            if unknown:
                raise ValueError(
                    f"cond_val contains unknown variable(s): {sorted(unknown)}. "
                    f"TI variables are: {sorted(self._ti.variables)}"
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

    def set_nonstationary(self, rotation=None, anis=None):
        """Set per-node geometric transform for non-stationary simulation.

        Transforms lag vectors from the simulation-grid frame into the training
        image's own frame before each TI scan (Mariethoz2010 §6.2). Enables
        spatially varying orientation and anisotropy without modifying the TI.

        Parameters
        ----------
        rotation : float or numpy.ndarray, optional
            Rotation angle(s) in radians. Scalar → stationary (same angle at
            every node). Array whose leading dimensions match the simulation
            grid shape → per-node angles. Convention matches
            :class:`gstools.CovModel` ``angles`` (2-D: one angle; 3-D:
            Tait–Bryan yaw/pitch/roll). ``None`` → no rotation applied.
        anis : float or numpy.ndarray, optional
            Anisotropy ratio(s). Scalar → stationary. Array → per-node. Values
            less than 1 compress the TI search in transversal directions.
            Convention matches :class:`gstools.CovModel` ``anis``.
            ``None`` → isotropic.
        """
        if rotation is not None:
            self._rotation = np.asarray(rotation, dtype=np.float64)
        if anis is not None:
            anis_arr = np.asarray(anis, dtype=np.float64)
            if np.any(anis_arr <= 0):
                raise ValueError(
                    f"DirectSampling: anis must be positive everywhere, "
                    f"got minimum value {float(anis_arr.min())!r}"
                )
            self._anis = anis_arr

    @property
    def ti(self):
        """TrainingImage: The training image model."""
        return self._ti

    @property
    def n_neighbors(self):
        """:class:`int` or :class:`dict`: Maximum neighbours in the data event (per-variable dict for multivariate TIs)."""
        return self._n_neighbors

    @n_neighbors.setter
    def n_neighbors(self, value):
        validated = _mv_validate_n_neighbors(value, self._ti)
        DirectSampling._validate_variation_n_neighbors(validated, self._ti)
        self._n_neighbors = validated

    @property
    def scan_fraction(self):
        """:class:`float`: Fraction of the TI to scan per node (capped at the search window)."""
        return self._scan_fraction

    @scan_fraction.setter
    def scan_fraction(self, value):
        self._scan_fraction = _mv_validate_scan_fraction(value)

    @property
    def threshold(self):
        """:class:`float`: Distance threshold (0.0 → DSBC mode)."""
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        self._threshold = _mv_validate_threshold(value)

    @property
    def cond_weight(self):
        """:class:`float`: Weight for conditioning nodes in distance."""
        return self._cond_weight

    @cond_weight.setter
    def cond_weight(self, value):
        self._cond_weight = float(value)

    @property
    def boundary(self):
        """:class:`str`: Search-window strategy (``"strict"`` or ``"partial"``)."""
        return self._boundary

    @boundary.setter
    def boundary(self, value):
        self._boundary = _mv_validate_boundary(value)

    @property
    def max_radius(self):
        """:class:`float` or :any:`None`: Euclidean cap on SG neighbour selection.

        Values in ``(0, 1)`` disable all neighbours (nearest grid cell is
        at distance 1.0), causing every node to fall back to a random TI
        sample.
        """
        return self._max_radius

    @max_radius.setter
    def max_radius(self, value):
        self._max_radius = _mv_validate_max_radius(value)

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
        if self._rotation is not None:
            rot_val = (
                float(self._rotation)
                if self._rotation.ndim == 0
                else self._rotation
            )
            parts.append(f", rotation={rot_val!r}")
        if self._anis is not None:
            anis_val = (
                float(self._anis) if self._anis.ndim == 0 else self._anis
            )
            parts.append(f", anis={anis_val!r}")
        return "".join(parts) + ")"
