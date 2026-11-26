"""Direct Sampling simulation engine.

`ds_simulate` is the entry point; `_DirectSamplingEngine` holds one run's
state (output grids, conditioning masks, config) explicitly. The complete node
path executes in the Rust engine.
"""

import warnings

import numpy as np

from gstools import config

# The complete engine comes from gstools.mps._core, which checks the installed
# gstools_core package and its supported MPS API revision.
from gstools.mps._core import mps_cleanup_pass as _mps_cleanup_pass_gsc
from gstools.mps._core import mps_simulate as _mps_simulate_gsc
from gstools.mps._validation import _validate_real_data
from gstools.mps.neighbors import _lag_transform_matrix, _precompute_offsets
from gstools.mps.training_image import (
    _post_pass_n_neighbors,
    _resolve_post_ti,
)

# Python precomputes optional per-node lag transform matrices before the
# GIL-free Rust call.
_MPS_RUST_ENGINE_STATS_HOOK = None
# Called once per post-processing (cleanup) pass with {"used_threads": int};
# lets tests confirm the native parallel branch engaged. ``None`` disables it.
_MPS_CLEANUP_STATS_HOOK = None


def _build_path(unknown, path, rng_path, sim_shape):
    """Build the (N, dim) node visit order.

    Parameters
    ----------
    unknown : numpy.ndarray of bool, shape sim_shape
        Mask of nodes that require simulation (at least one uninformed variable).
    path : str or array-like
        ``"random"`` — uniformly shuffled raster order (default DS behaviour).
        ``"sequential"`` — lexicographic (raster) order; ``rng_path`` is not
        consumed, making the visit order deterministic without a ``path_seed``.
        An explicit ``(N, dim)`` integer array — the caller-supplied visit
        order.  Must include every unknown node; conditioned nodes present in
        the array are silently skipped so that a full-grid path (e.g. a spiral
        over all nodes) works unchanged with conditioning data.  Duplicate rows
        and missing unknown nodes are still errors.
    rng_path : numpy.random.RandomState
        Controls shuffling when ``path="random"``; unused for the other modes.
    sim_shape : tuple of int
        Simulation grid shape.

    Returns
    -------
    numpy.ndarray, shape (N, dim), dtype numpy.intp
        Node visit order.

    Raises
    ------
    ValueError
        For an unrecognised string or an invalid explicit array.
    """
    base = np.argwhere(
        unknown
    )  # raster (lexicographic) order over unknown nodes
    if isinstance(path, str):
        if path == "sequential":
            return base
        if path == "random":
            return base[rng_path.permutation(len(base))]
        raise ValueError(
            f"DirectSampling: path must be 'random', 'sequential', or an "
            f"(N, dim) array; got {path!r}"
        )
    arr = np.asarray(path)
    dim = len(sim_shape)
    # Shape check
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise ValueError(
            f"DirectSampling: explicit path shape must be (N, dim={dim}); "
            f"got {arr.shape!r}"
        )
    # Integer check and bounds check
    if not np.issubdtype(arr.dtype, np.integer):
        # Allow float arrays with integer values (e.g. from argwhere stored as float)
        if not np.all(arr == np.floor(arr)):
            raise ValueError(
                "DirectSampling: explicit path must contain integer coordinates"
            )
        arr = arr.astype(np.intp)
    else:
        arr = arr.astype(np.intp, copy=False)
    for d in range(dim):
        if np.any(arr[:, d] < 0) or np.any(arr[:, d] >= sim_shape[d]):
            raise ValueError(
                f"DirectSampling: explicit path has out-of-bounds coordinates "
                f"along axis {d} for grid shape {sim_shape!r}"
            )
    # Validation: duplicates → error; conditioned extras → silently drop;
    # missing unknown nodes → error.
    flat_path = np.ravel_multi_index(arr.T, sim_shape)
    flat_base = (
        np.ravel_multi_index(base.T, sim_shape)
        if len(base)
        else np.empty(0, dtype=np.intp)
    )
    if len(np.unique(flat_path)) != len(flat_path):
        raise ValueError(
            "DirectSampling: explicit path contains duplicate nodes"
        )
    # Keep only unknown-set nodes (conditioned entries are silently dropped)
    unknown_mask = np.isin(flat_path, flat_base)
    arr = arr[unknown_mask]
    flat_path = flat_path[unknown_mask]
    # Every unknown node must appear in the (filtered) path
    if not np.array_equal(np.sort(flat_path), np.sort(flat_base)):
        raise ValueError(
            "DirectSampling: explicit path is missing one or more unknown nodes "
            "(every unset grid node must appear in the path)"
        )
    return arr


def _condition_arrays(conditions, dim):
    """Return conditioning data as ``(nodes, {variable: values})`` arrays.

    The array form passes through unchanged; a ``{node_index: {variable:
    value}}`` dict is converted, with ``NaN`` where a node lacks a variable.
    """
    if not isinstance(conditions, dict):
        return conditions
    idx = np.array(list(conditions), dtype=np.intp).reshape(
        len(conditions), dim
    )
    values = {}
    for k, vd in enumerate(conditions.values()):
        for v, val in vd.items():
            _validate_real_data(
                val, False, f"DirectSampling conditioning {v!r}"
            )
            values.setdefault(v, np.full(len(conditions), np.nan))[k] = val
    return tuple(idx.T), values


class _DirectSamplingEngine:
    """Holds one Direct Sampling run's state and orchestrates it."""

    def __init__(
        self,
        model,
        sim_shape,
        rng_path,
        rng_nodes,
        conditions=None,
        rotation_map=None,
        scale_map=None,
        path="random",
        zone_selector=None,
    ):
        training_image = model.ti
        self.training_image = training_image
        self.zones = model.zones
        self.zone_selector = zone_selector
        self.variables = [v.name for v in training_image.variables]
        # A variable with a penalty matrix uses metric kind 3; its CxC matrix
        # is concatenated into penalty_flat with a per-variable start offset.
        self.var_penalty = {
            v.name: v.penalty_matrix for v in training_image.variables
        }
        self.weights = (
            {
                v.name: training_image.weights[v.name]
                for v in training_image.variables
            }
            if training_image.multivariate
            # Univariate: key by the single variable's actual name (``None`` for
            # the anonymous bare-array sugar, or a user-given name).
            else {training_image.variables[0].name: 1.0}
        )
        self.ti_shape = np.array(training_image.shape)
        self.sim_shape = sim_shape
        self.dim = len(sim_shape)
        self.threshold = model.threshold
        self.scan_fraction = model.scan_fraction
        self.boundary = model.boundary
        self.random_scan = model.scan_path == "random"
        # Per-variable spatial-decay exponent and conditioning-node weight,
        # keyed by variable name.
        self.var_distance_power = {
            v.name: v.distance_power for v in training_image.variables
        }
        self.var_cond_weight = {
            v.name: v.cond_weight for v in training_image.variables
        }
        # An exact identity transform (no rotation, unit scale everywhere) must
        # reduce events the same way as an absent transform (audit DS07). The
        # transformed route uses a geometric reduce-to-fit while the stationary
        # route keeps the nearest-neighbour prefix, so they can retain different
        # events despite the identity matrix. Collapsing a globally-identity map
        # to ``None`` routes it through stationary handling and keeps identity
        # bit-identical to no transform.
        if rotation_map is not None and np.all(rotation_map == 0.0):
            rotation_map = None
        if scale_map is not None and np.all(scale_map == 1.0):
            scale_map = None
        self.rotation_map = rotation_map
        self.scale_map = scale_map

        self.n_k = {v.name: v.n_neighbors for v in training_image.variables}

        self.ti_matrix_f64 = np.ascontiguousarray(
            np.stack(
                [
                    np.asarray(v.data, dtype=np.float64).ravel()
                    for v in training_image.variables
                ],
                axis=0,
            )
        )

        active_mask = model.active_mask
        self.active_mask = None
        if active_mask is not None:
            mask = np.asarray(active_mask)
            if mask.dtype != bool:
                raise TypeError("DirectSampling: active_mask must be boolean")
            if mask.shape != tuple(sim_shape):
                raise ValueError(
                    "DirectSampling: active_mask shape must match the simulation grid"
                )
            self.active_mask = mask.copy()
        self.sg = {v: np.full(sim_shape, np.nan) for v in self.variables}
        self.is_cond = {
            v: np.zeros(sim_shape, dtype=bool) for v in self.variables
        }
        if conditions:
            nodes, values = _condition_arrays(conditions, self.dim)
            for v, column in values.items():
                _validate_real_data(
                    column, False, f"DirectSampling conditioning {v!r}"
                )
                column = np.asarray(column, dtype=np.float64)
                if np.isinf(column).any():
                    raise ValueError(
                        "DirectSampling: conditioning values must be finite or NaN"
                    )
                known = ~np.isnan(column)
                at = tuple(n[known] for n in nodes)
                if (
                    self.active_mask is not None
                    and not self.active_mask[at].all()
                ):
                    raise ValueError(
                        "DirectSampling: hard conditions outside active_mask are not allowed"
                    )
                self.sg[v][at] = column[known]
                self.is_cond[v][at] = True

        # Post-processing (Meerschman et al. 2013, Sect. 4): resolved once here.
        # The cleanup passes run after the main path through the Rust
        # ``mps_cleanup_pass`` engine (serial Gauss-Seidel reopen/resimulate).
        self.post_processing = model.post_processing
        self.post_processing_factor = model.post_processing_factor
        post_processing_path = model.post_processing_path
        post_processing_mask = model.post_processing_mask
        self._rng_path = rng_path
        self._rng_nodes = rng_nodes
        # None inherits the main-path mode (an explicit main-path array has no
        # meaning for a full-grid re-visit -> random); Me13 §4 does not
        # prescribe the post-pass order, so the explicit option is an extension.
        if post_processing_path is None:
            post_processing_path = (
                "sequential"
                if isinstance(path, str) and path == "sequential"
                else "random"
            )
        self._post_path = post_processing_path
        # Post-pass TI: the primary's variables with any override's explicitly
        # set search fields substituted in (data/kind preserved).
        self._post_ti = _resolve_post_ti(
            training_image, model.post_processing_variables
        )
        if self.post_processing > 0:  # fail before the main pass, not after
            _post_pass_n_neighbors(self._post_ti, self.post_processing_factor)
        self._post_max_radius_per_var = {
            v.name: v.max_radius for v in self._post_ti.variables
        }
        # The cleanup mask must match the grid exactly: NumPy would otherwise
        # broadcast a (2,), (2,1) or scalar mask against the eligibility grid
        # and silently select a column/row/everything, and a non-broadcastable
        # shape would only fail after the main pass had consumed randomness
        # (audit DS11). Validate here, before the main pass runs.
        if post_processing_mask is not None and tuple(
            np.shape(post_processing_mask)
        ) != tuple(sim_shape):
            raise ValueError(
                "DirectSampling: post_processing_mask shape "
                f"{np.shape(post_processing_mask)!r} must match the simulation "
                f"grid shape {tuple(sim_shape)!r}."
            )
        self._post_processing_mask = post_processing_mask

        self.max_radius_per_var = {
            v.name: v.max_radius for v in training_image.variables
        }
        # The offset table is shared by the main and cleanup passes, so when
        # post-processing runs it must also cover any (possibly larger) post-pass
        # radius. Extending it only ever adds farther candidates the main path
        # never reaches, so the post_processing=0 table is unchanged.
        _radius_values = list(self.max_radius_per_var.values())
        if self.post_processing > 0:
            _radius_values += list(self._post_max_radius_per_var.values())
        # Any unlimited (None) effective radius makes the shared envelope
        # unlimited — bounded only per axis by the grid extent in
        # ``_precompute_offsets`` (audit DS06). Dropping None and taking the max
        # of the finite siblings would silently cap the unbounded variable's
        # neighbourhood to a smaller radius.
        if any(r is None for r in _radius_values):
            max_off_int = None
        else:
            max_off_int = int(np.ceil(max(_radius_values)))
        self.offset_arr = _precompute_offsets(sim_shape, max_off_int)

        # Node path: every node with >= 1 uninformed variable.  Fully-conditioned
        # nodes need no simulation and are excluded (they stay -1 / always available).
        unknown = np.zeros(sim_shape, dtype=bool)
        for v in self.variables:
            unknown |= np.isnan(self.sg[v])
        if self.active_mask is not None:
            unknown &= self.active_mask
        self.path = _build_path(unknown, path, rng_path, sim_shape)
        n_nodes = len(self.path)
        self.u_start = rng_nodes.uniform(size=n_nodes)
        self.u_fallback = rng_nodes.uniform(size=(n_nodes, self.dim))

        # A joint fallback draw must have all variables defined.
        if training_image.has_nan:
            finite_all = np.ones(self.ti_shape, dtype=bool)
            for variable in training_image.variables:
                data = variable.data
                if np.issubdtype(data.dtype, np.floating):
                    finite_all &= ~np.isnan(data)
            if not finite_all.any():
                raise ValueError(
                    "TrainingImage has no cell defined in all variables (every "
                    "cell is NaN in at least one variable); cannot simulate."
                )

        self.var_categorical = {
            v.name: v.categorical for v in training_image.variables
        }
        self.var_has_nan = {
            v.name: v.has_nan for v in training_image.variables
        }
        self.var_p_norm = {v.name: v.p_norm for v in training_image.variables}
        self.var_d_max = {v.name: v.d_max for v in training_image.variables}
        self.var_variation_p = {
            v.name: v.variation_p_norm for v in training_image.variables
        }

    def _lag_matrices_for(self, path):
        """Per-node SG→TI transform matrices for ``path`` (empty if stationary)."""
        if (self.rotation_map is None and self.scale_map is None) or not len(
            path
        ):
            return np.empty((0, self.dim, self.dim), dtype=np.float64)
        return np.ascontiguousarray(
            np.stack(
                [
                    _lag_transform_matrix(
                        self.dim, self.rotation_map, self.scale_map, node
                    )
                    for node in path
                ]
            )
        )

    def _node_domain_for(self, path):
        """Per-path-node domain selector (0 = primary, k = zones[k-1])."""
        if self.zones and self.zone_selector is not None:
            return np.ascontiguousarray(
                self.zone_selector[tuple(np.asarray(path).T)], dtype=np.int64
            )
        return np.zeros(len(path), dtype=np.int64)

    def _pack_zones(self):
        """Pack zone TIs column-wise (self.variables order) + offsets/shapes/d_max.

        Zones share the primary's search config; only data/shape/d_max differ
        per domain. Pass-invariant, so the main and cleanup passes reuse it.
        """
        n_zones = len(self.zones)
        n_vars = len(self.variables)
        zone_offsets = np.zeros(n_zones + 1, dtype=np.int64)
        zone_shapes = np.zeros((n_zones, self.dim), dtype=np.int64)
        zone_d_max = np.ones((n_zones, n_vars), dtype=np.float64)
        zone_ti_parts = []
        running = 0
        for z, zone in enumerate(self.zones):
            zti = zone.ti
            rows = np.stack(
                [
                    np.asarray(zti.variable(v).data, dtype=np.float64).ravel()
                    for v in self.variables
                ],
                axis=0,
            )
            zone_ti_parts.append(rows)
            running += rows.shape[1]
            zone_offsets[z + 1] = running
            zone_shapes[z] = np.asarray(zti.shape, dtype=np.int64)
            for vi, v in enumerate(self.variables):
                dm = zti.variable(v).d_max
                zone_d_max[z, vi] = 1.0 if dm is None else dm
        zone_ti = (
            np.ascontiguousarray(np.concatenate(zone_ti_parts, axis=1))
            if zone_ti_parts
            else np.empty((n_vars, 0), dtype=np.float64)
        )
        return zone_ti, zone_offsets, zone_shapes, zone_d_max

    def _run_rust_engine(self, n_threads):
        """Run the complete node path inside one GIL-free Rust call."""
        fields = np.ascontiguousarray(
            np.stack([self.sg[v].reshape(-1) for v in self.variables])
        )
        conditioned = np.ascontiguousarray(
            np.stack(
                [self.is_cond[v].reshape(-1) for v in self.variables]
            ).astype(np.uint8)
        )
        metric_kinds = np.fromiter(
            (
                3
                if self.var_categorical[v] and self.var_penalty[v] is not None
                else 0
                if self.var_categorical[v]
                else 1
                if self.var_p_norm[v] is not None
                else 2
                for v in self.variables
            ),
            dtype=np.int64,
            count=len(self.variables),
        )
        # Pack per-variable penalty matrices into a flat buffer + offsets/ncat.
        penalty_mats = [self.var_penalty[v] for v in self.variables]
        penalty_ncat = np.fromiter(
            (0 if m is None else m.shape[0] for m in penalty_mats),
            dtype=np.int64,
            count=len(self.variables),
        )
        penalty_offsets = np.zeros(len(self.variables), dtype=np.int64)
        _flat_parts = []
        _running = 0
        for _i, _m in enumerate(penalty_mats):
            penalty_offsets[_i] = _running
            if _m is not None:
                _flat_parts.append(
                    np.ascontiguousarray(_m, dtype=np.float64).ravel()
                )
                _running += _m.size
        penalty_flat = (
            np.concatenate(_flat_parts)
            if _flat_parts
            else np.empty(0, dtype=np.float64)
        )
        p_norm = np.fromiter(
            (
                1.0
                if self.var_categorical[v]
                else self.var_p_norm[v]
                if self.var_p_norm[v] is not None
                else self.var_variation_p[v]
                for v in self.variables
            ),
            dtype=np.float64,
            count=len(self.variables),
        )
        lag_matrices = self._lag_matrices_for(self.path)
        node_domain = self._node_domain_for(self.path)
        zone_ti, zone_offsets, zone_shapes, zone_d_max = self._pack_zones()
        (
            result,
            strict_fallbacks,
            level_count,
            max_ready_width,
            used_threads,
            collapsed_lags,
        ) = _mps_simulate_gsc(
            self.ti_matrix_f64,
            np.asarray(self.ti_shape, dtype=np.int64),
            fields,
            conditioned,
            np.asarray(self.sim_shape, dtype=np.int64),
            np.asarray(self.path, dtype=np.int64),
            lag_matrices,
            np.asarray(self.u_start, dtype=np.float64),
            np.asarray(self.u_fallback, dtype=np.float64),
            np.asarray(self.offset_arr, dtype=np.int64),
            np.fromiter(
                (self.n_k[v] for v in self.variables),
                dtype=np.int64,
                count=len(self.variables),
            ),
            np.fromiter(
                (
                    np.nan
                    if self.max_radius_per_var[v] is None
                    else self.max_radius_per_var[v]
                    for v in self.variables
                ),
                dtype=np.float64,
                count=len(self.variables),
            ),
            np.fromiter(
                (self.weights[v] for v in self.variables),
                dtype=np.float64,
                count=len(self.variables),
            ),
            metric_kinds,
            np.fromiter(
                (self.var_has_nan[v] for v in self.variables),
                dtype=np.uint8,
                count=len(self.variables),
            ),
            np.fromiter(
                (
                    1.0 if self.var_d_max[v] is None else self.var_d_max[v]
                    for v in self.variables
                ),
                dtype=np.float64,
                count=len(self.variables),
            ),
            p_norm,
            penalty_flat,
            penalty_offsets,
            penalty_ncat,
            self.threshold,
            self.scan_fraction,
            self.random_scan,
            # Per-variable values, indexed in self.variables order.
            np.fromiter(
                (self.var_distance_power[v] for v in self.variables),
                dtype=np.float64,
                count=len(self.variables),
            ),
            np.fromiter(
                (self.var_cond_weight[v] for v in self.variables),
                dtype=np.float64,
                count=len(self.variables),
            ),
            self.boundary == "partial",
            n_threads,
            zone_ti,
            zone_offsets,
            zone_shapes,
            zone_d_max,
            node_domain,
            active_mask=None
            if self.active_mask is None
            else self.active_mask.astype(np.uint8).ravel(),
        )
        for row, variable in enumerate(self.variables):
            self.sg[variable][...] = result[row].reshape(self.sim_shape)
        self._rust_engine_stats = {
            "level_count": int(level_count),
            "max_ready_width": int(max_ready_width),
            "requested_threads": int(n_threads),
            "used_threads": int(used_threads),
            "visited_sim_nodes": int(len(self.path)),
            "active_cells": int(np.prod(self.sim_shape))
            if self.active_mask is None
            else int(self.active_mask.sum()),
        }
        if _MPS_RUST_ENGINE_STATS_HOOK is not None:
            _MPS_RUST_ENGINE_STATS_HOOK(self._rust_engine_stats.copy())
        if strict_fallbacks:
            warnings.warn(
                "gstools.mps: boundary='strict' could not fit the full data "
                "event inside the TI; falling back to partial mode (dropping "
                "the furthest neighbour(s)). Ensure the TI is at least as "
                "large as the data event extent to enforce strict mode.",
                RuntimeWarning,
                stacklevel=2,
            )
        if collapsed_lags:
            warnings.warn(
                "gstools.mps: anisotropy/rotation transform collapsed "
                "neighbour lag(s) onto duplicate TI positions; the "
                "duplicates are excluded from the data event. Reduce the "
                "anisotropy ratio or rotation to retain them.",
                RuntimeWarning,
                stacklevel=2,
            )
        return self.sg

    def run(self, num_threads=None):
        # GSTools-Core is the only numerical backend (Cython was dropped
        # upstream), so there is no backend flag to check — MPS always runs
        # through the Rust engine. The required core version/exports are
        # enforced at import time by gstools.mps._core.
        n_threads = (
            num_threads
            if num_threads is not None
            else (config.NUM_THREADS or 1)
        )
        self._run_rust_engine(n_threads)
        self._run_post_passes(n_threads)
        return self.sg

    def _cleanup_var_arrays(self, ordered_vars, pass_n_k, weights):
        """Pack the per-variable *effective* post-pass config as the arrays the
        Rust cleanup engine consumes, in ``self.variables`` order.

        Mirrors the per-variable packing in :meth:`_run_rust_engine`, but reads
        from the resolved post-processing TI's :class:`Variable` objects so any
        ``post_processing_variables`` override (distance kernel, penalty,
        ``distance_power``/``cond_weight``, weight) and the divided
        ``n_neighbors`` take effect.
        """
        names = self.variables
        count = len(names)
        metric_kinds = np.fromiter(
            (
                3
                if v.categorical and v.penalty_matrix is not None
                else 0
                if v.categorical
                else 1
                if v.p_norm is not None
                else 2
                for v in ordered_vars
            ),
            dtype=np.int64,
            count=count,
        )
        p_norm = np.fromiter(
            (
                1.0
                if v.categorical
                else v.p_norm
                if v.p_norm is not None
                else v.variation_p_norm
                for v in ordered_vars
            ),
            dtype=np.float64,
            count=count,
        )
        penalty_mats = [v.penalty_matrix for v in ordered_vars]
        penalty_ncat = np.fromiter(
            (0 if m is None else m.shape[0] for m in penalty_mats),
            dtype=np.int64,
            count=count,
        )
        penalty_offsets = np.zeros(count, dtype=np.int64)
        flat_parts = []
        running = 0
        for i, m in enumerate(penalty_mats):
            penalty_offsets[i] = running
            if m is not None:
                flat_parts.append(
                    np.ascontiguousarray(m, dtype=np.float64).ravel()
                )
                running += m.size
        penalty_flat = (
            np.concatenate(flat_parts)
            if flat_parts
            else np.empty(0, dtype=np.float64)
        )
        return {
            "n_neighbors": np.fromiter(
                (pass_n_k[name] for name in names),
                dtype=np.int64,
                count=count,
            ),
            "max_radius": np.fromiter(
                (
                    np.nan
                    if self._post_max_radius_per_var[name] is None
                    else self._post_max_radius_per_var[name]
                    for name in names
                ),
                dtype=np.float64,
                count=count,
            ),
            "variable_weights": np.fromiter(
                (weights[name] for name in names),
                dtype=np.float64,
                count=count,
            ),
            "metric_kinds": metric_kinds,
            "has_nan": np.fromiter(
                (v.has_nan for v in ordered_vars),
                dtype=np.uint8,
                count=count,
            ),
            "d_max": np.fromiter(
                (1.0 if v.d_max is None else v.d_max for v in ordered_vars),
                dtype=np.float64,
                count=count,
            ),
            "p_norm": p_norm,
            "penalty_flat": penalty_flat,
            "penalty_offsets": penalty_offsets,
            "penalty_ncat": penalty_ncat,
            "distance_power": np.fromiter(
                (v.distance_power for v in ordered_vars),
                dtype=np.float64,
                count=count,
            ),
            "cond_weight": np.fromiter(
                (v.cond_weight for v in ordered_vars),
                dtype=np.float64,
                count=count,
            ),
        }

    def _run_cleanup_pass(
        self, order, u_start, u_fallback, pass_n_k, scan_frac, n_threads
    ):
        """One deterministic Gauss-Seidel cleanup pass through the Rust engine.

        Reopens each node of ``order`` (its non-conditioned variables), scans
        the fully-informed field, and writes the result back into ``self.sg``.
        The native engine orders the sweep by a position dependency DAG, so the
        result is bit-identical for any ``n_threads`` (serial == parallel).
        """
        names = self.variables
        by_name = {v.name: v for v in self._post_ti.variables}
        ordered_vars = [by_name[name] for name in names]
        weights = (
            {name: self._post_ti.weights[name] for name in names}
            if self._post_ti.multivariate
            else {names[0]: 1.0}
        )
        post_ti_matrix = np.ascontiguousarray(
            np.stack(
                [
                    np.asarray(v.data, dtype=np.float64).ravel()
                    for v in ordered_vars
                ],
                axis=0,
            )
        )
        fields = np.ascontiguousarray(
            np.stack([self.sg[v].reshape(-1) for v in names])
        )
        conditioned = np.ascontiguousarray(
            np.stack([self.is_cond[v].reshape(-1) for v in names]).astype(
                np.uint8
            )
        )
        arrays = self._cleanup_var_arrays(ordered_vars, pass_n_k, weights)
        lag_matrices = self._lag_matrices_for(order)
        node_domain = self._node_domain_for(order)
        zone_ti, zone_offsets, zone_shapes, zone_d_max = self._pack_zones()
        result, strict_fallbacks, collapsed_lags, _used = (
            _mps_cleanup_pass_gsc(
                post_ti_matrix,
                np.asarray(self.ti_shape, dtype=np.int64),
                fields,
                conditioned,
                np.asarray(self.sim_shape, dtype=np.int64),
                np.asarray(order, dtype=np.int64),
                lag_matrices,
                np.asarray(u_start, dtype=np.float64),
                np.asarray(u_fallback, dtype=np.float64),
                np.asarray(self.offset_arr, dtype=np.int64),
                arrays["n_neighbors"],
                arrays["max_radius"],
                arrays["variable_weights"],
                arrays["metric_kinds"],
                arrays["has_nan"],
                arrays["d_max"],
                arrays["p_norm"],
                arrays["penalty_flat"],
                arrays["penalty_offsets"],
                arrays["penalty_ncat"],
                self.threshold,
                scan_frac,
                self.random_scan,
                arrays["distance_power"],
                arrays["cond_weight"],
                self.boundary == "partial",
                n_threads,
                zone_ti,
                zone_offsets,
                zone_shapes,
                zone_d_max,
                node_domain,
                active_mask=None
                if self.active_mask is None
                else self.active_mask.astype(np.uint8).ravel(),
            )
        )
        for row, variable in enumerate(names):
            self.sg[variable][...] = result[row].reshape(self.sim_shape)
        # Effective thread count of the most recent pass (after the native
        # small-path cutoff); lets callers/tests confirm the parallel branch ran.
        self._cleanup_used_threads = int(_used)
        if _MPS_CLEANUP_STATS_HOOK is not None:
            _MPS_CLEANUP_STATS_HOOK({"used_threads": int(_used)})
        if strict_fallbacks:
            warnings.warn(
                "gstools.mps: boundary='strict' could not fit the full data "
                "event inside the TI during a post-processing pass; falling "
                "back to partial mode (dropping the furthest neighbour(s)).",
                RuntimeWarning,
                stacklevel=2,
            )
        if collapsed_lags:
            warnings.warn(
                "gstools.mps: anisotropy/rotation transform collapsed "
                "neighbour lag(s) during a post-processing pass; the "
                "duplicates are excluded from the data event.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _run_post_passes(self, n_threads):
        """Run ``post_processing`` deterministic cleanup passes (Me13 §4).

        Eligible nodes are those with at least one non-conditioned variable,
        intersected with ``post_processing_mask`` when given. ``n_neighbors``
        and ``scan_fraction`` are divided by ``post_processing_factor``. Each
        pass is bit-identical for any ``n_threads``. ``post_processing=0`` is a
        true no-op: it draws no extra random numbers.
        """
        if self.post_processing <= 0:
            return
        resim = np.zeros(self.sim_shape, dtype=bool)
        for v in self.variables:
            resim |= ~self.is_cond[v]
        if self.active_mask is not None:
            resim &= self.active_mask
        if self._post_processing_mask is not None:
            resim &= self._post_processing_mask
        if not resim.any():
            return
        p_f = self.post_processing_factor
        pass_n_k = _post_pass_n_neighbors(self._post_ti, p_f)
        scan_frac = self.scan_fraction / p_f
        for _ in range(self.post_processing):
            if isinstance(self._post_path, str) and self._post_path == "same":
                # Without a mask, resim is exactly the main path's unknown set,
                # so filtering self.path preserves its order. With a mask, this
                # narrows to the eligible subset, still in main-path order.
                order = self.path[resim[tuple(self.path.T)]]
            else:
                order = _build_path(
                    resim, self._post_path, self._rng_path, self.sim_shape
                )
            u_start = self._rng_nodes.uniform(size=len(order))
            u_fallback = self._rng_nodes.uniform(size=(len(order), self.dim))
            self._run_cleanup_pass(
                order, u_start, u_fallback, pass_n_k, scan_frac, n_threads
            )


def ds_simulate(
    training_image,
    sim_shape,
    threshold,
    scan_fraction,
    rng_path,
    rng_nodes,
    conditions=None,
    boundary="strict",
    num_threads=None,
    rotation_map=None,
    scale_map=None,
    path="random",
    scan_path="sequential",
    zones=None,
    zone_selector=None,
    post_processing=0,
    post_processing_factor=1.0,
    post_processing_path=None,
    post_processing_variables=None,
    post_processing_mask=None,
    active_mask=None,
):
    """Node-wise multivariate Direct Sampling (Mariethoz2010 §3, Eq. 8).

    ``active_mask`` is an optional boolean array matching ``sim_shape``. False
    cells are never targets or neighbors and remain NaN, including in cleanup.
    Hard conditions outside the mask are rejected. ``None`` preserves the full
    rectangular domain.

    Co-simulates every variable of a dict-valued :class:`TrainingImage` on one
    structured grid, treating all variables equally. The path visits every
    *node* with at least one unknown variable; at each node a single joint scan
    finds the TI cell ``y`` minimising the weighted distance ``Σ_k w_k d_k`` over
    all variables, and that one cell's whole vector ``{TI[v][y]}`` is copied to
    the node's *uninformed* variables. Because every variable at a node is drawn
    from the same TI cell, the collocated cross-variable tuple pasted at that
    node is exactly a tuple that occurs in the TI (when copied without variation
    correction or retained partial conditioning). This preserves the local joint
    relationship cell-by-cell; it does not by itself guarantee exact finite-
    sample joint frequencies or higher-order multipoint statistics over the
    realization (audit DS16). Variables already known at the node (only ever via
    conditioning data) act as collocated ``h = 0`` constraints, weighted by
    ``cond_weight``.

    Parameters
    ----------
    training_image : TrainingImage
        Training image; per-variable ``n_neighbors`` and ``max_radius`` are
        read from each :class:`Variable` object.
    sim_shape : tuple
        Simulation grid shape.
    threshold : float
        Distance threshold (Juda2022 §2). ``0.0`` -> DSBC mode.
    scan_fraction : float
        Fraction of the TI to scan per node, capped at the valid search window.
    rng_path : numpy.random.RandomState
        RandomState controlling the simulation path (node visit order).
        Only consumed when ``path="random"``; unused otherwise.
    rng_nodes : numpy.random.RandomState
        RandomState controlling TI scan entry points and fallback cells.
    path : str or array-like, optional
        Node visit order.  ``"random"`` (default) shuffles the unknown nodes
        uniformly using ``rng_path`` — the standard DS behaviour.
        ``"sequential"`` visits nodes in raster (lexicographic) order, which
        is deterministic and does not consume ``rng_path``.  An explicit
        integer array of shape ``(N, dim)`` provides a caller-supplied order
        and must include every unknown node (conditioned nodes are ignored;
        missing unknown nodes and duplicate rows raise ``ValueError``).
        Default: ``"random"``.
    conditions : dict or tuple, optional
        ``{node_index: {variable: value}}`` conditioning data, or the
        equivalent array form ``(nodes, {variable: values})`` with ``nodes`` a
        tuple of per-axis index arrays and ``NaN`` marking a variable that is
        not conditioned at a node. Per-variable conditioning-node weights are
        read from each :class:`Variable`.
    active_mask : numpy.ndarray of bool or None, optional
        Boolean array with shape ``sim_shape``. False cells remain NaN and are
        excluded as simulation targets and neighbors in the main run and
        cleanup passes. Conditions on inactive cells raise ``ValueError``.
        ``None`` uses the full rectangular domain.
    boundary : str, optional
        ``"strict"`` (default) or ``"partial"`` search-window strategy.
    num_threads : int or None, optional
        Threads for the node-wise DAG. ``None`` -> ``config.NUM_THREADS``.
    rotation_map : numpy.ndarray or None, optional
        Canonical per-node rotation map, shape ``grid_shape +
        (no_of_angles(dim),)`` from :func:`gstools.mps.nonstationary.resolve_spec`.
        ``None`` → no rotation (stationary).
    scale_map : numpy.ndarray or None, optional
        Canonical per-node scale map, shape ``grid_shape + (dim,)`` from
        :func:`~gstools.mps.nonstationary.resolve_spec`. ``None`` → no dilation.
        All values must be positive.
    Returns
    -------
    dict
        ``{variable: numpy.ndarray}`` — one simulated field per variable.
    """
    from gstools.mps.model import MPSModel

    model = MPSModel(
        training_image,
        threshold=threshold,
        scan_fraction=scan_fraction,
        boundary=boundary,
        scan_path=scan_path,
        zones=zones,
        post_processing=post_processing,
        post_processing_factor=post_processing_factor,
        post_processing_path=post_processing_path,
        post_processing_variables=post_processing_variables,
        post_processing_mask=post_processing_mask,
        active_mask=active_mask,
    )
    return _simulate_model(
        model,
        sim_shape,
        rng_path,
        rng_nodes,
        conditions=conditions,
        rotation_map=rotation_map,
        scale_map=scale_map,
        path=path,
        zone_selector=zone_selector,
        num_threads=num_threads,
    )


def _simulate_model(
    model, sim_shape, rng_path, rng_nodes, *, num_threads, **preparation
):
    """Simulate with a validated configuration and resolved grid settings."""
    return _DirectSamplingEngine(
        model, sim_shape, rng_path, rng_nodes, **preparation
    ).run(num_threads=num_threads)
