"""Direct Sampling simulation engine.

`ds_simulate` is the entry point; `_DirectSamplingEngine` holds one run's
state (output grids, informed masks, config) explicitly — what used to be
captured by closures inside `ds_simulate`. The engine orchestrates per-node
simulation by calling the stateless `neighbors`, `scan`, and `runner` modules.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import prod

import numpy as np

from gstools import config
from gstools.mps.data_event import DataEvent
from gstools.mps.neighbors import (
    _lag_transform_matrix,
    _precompute_offsets,
    _reduce_to_fit,
    _select_neighbors,
    _transform_lags,
    _window_bounds,
)
from gstools.mps.runner import _make_progress, _run_path
from gstools.mps.scan import _scan_for_match, _ScanConfig


@dataclass(frozen=True)
class _Pass:
    """Everything that varies per pass — the main path or one post-processing round.

    Threaded through the node-simulation seams as an argument, so the engine's
    own ``n_k``/``scan_fraction``/``domains`` stay un-mutated base configuration
    and any further phase is just another ``_Pass``.

    Parameters
    ----------
    order : numpy.ndarray, shape (N, dim)
        This pass's visit order.
    vmap : dict of {str: numpy.ndarray}
        Per-variable availability map (see :func:`_build_vmap`).  All ``-1`` for
        a post-pass — Me13 §4's "fully informed neighbourhood".
    pos_map : numpy.ndarray or None
        Visit-order orientation for the DAG (see :func:`runner._build_dag_base`).
        ``None`` for the main path, whose vmap already carries the order.
    n_k : dict of {str: int}
        Per-variable neighbour count (post-passes divide by
        ``post_processing_factor``, floored at 1).
    scan_fraction : float
        Likewise divided by ``post_processing_factor`` on a post-pass.
    neighbor_cache : dict
        One-slot box for the DAG-build neighbour cache, filled under key
        ``"coords"``.  Fresh per pass, so no cache can leak between passes;
        stays empty in serial mode.
    """

    order: object
    vmap: dict
    pos_map: object
    n_k: dict
    scan_fraction: float
    neighbor_cache: dict


@dataclass(frozen=True)
class _Domain:
    """One scan domain: the primary TI or a zone TI, with its precomputations.

    Search hyper-parameters (n_neighbors, max_radius, weights, distance
    kernel) always come from the PRIMARY TI; the domain contributes the data
    arrays, shape, per-variable d_max, and NaN bookkeeping (spec §3).
    """

    ti: object  # TrainingImage
    ti_shape: object  # numpy.ndarray (dim,)
    ti_vars: dict  # {var: ndarray}
    ti_flat: dict
    ti_strides: dict
    has_nan: bool
    finite_flat: object  # ndarray or None
    scan_config: object  # _ScanConfig


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


def _order_positions(order, sim_shape):
    """Flat-cell → index in ``order`` (``-1`` for cells absent from it)."""
    m = np.full(int(np.prod(sim_shape)), -1, dtype=np.intp)
    if len(order):
        m[np.ravel_multi_index(order.T, sim_shape)] = np.arange(len(order))
    return m


def _build_vmap(order, is_cond, variables, sim_shape):
    """Per-variable availability map: visit index, or ``-1`` = always available.

    A cell conditioned in variable ``v`` is ``-1`` in ``vmap[v]`` so its known
    value is never gated by visit order — essential for a partially-conditioned
    node. Cells absent from ``order`` are already ``-1``.
    """
    base = _order_positions(order, sim_shape)
    vmap = {}
    for v in variables:
        m = base.copy()
        m[np.flatnonzero(is_cond[v].reshape(-1))] = -1
        vmap[v] = m
    return vmap


class _DirectSamplingEngine:
    """Holds one Direct Sampling run's state and orchestrates it."""

    def __init__(
        self,
        training_image,
        sim_shape,
        threshold,
        scan_fraction,
        rng_path,
        rng_nodes,
        conditions=None,
        cond_weight=1.0,
        boundary="strict",
        rotation_map=None,
        scale_map=None,
        stationary_transform=False,
        path="random",
        zone_tis=None,
        zone_selector=None,
        post_processing=0,
        post_processing_factor=1.0,
        post_processing_path=None,
    ):
        self.training_image = training_image
        self.variables = [v.name for v in training_image.variables]
        self.weights = (
            {
                v.name: training_image.weights[v.name]
                for v in training_image.variables
            }
            if training_image.multivariate
            else {None: 1.0}
        )
        self.ti_shape = np.array(training_image.shape)
        self.sim_shape = sim_shape
        self.sim_shape_arr = np.array(sim_shape)
        self.dim = len(sim_shape)
        self.threshold = threshold
        self.scan_fraction = scan_fraction
        self.cond_weight = cond_weight
        self.boundary = boundary
        self.rotation_map = rotation_map
        self.scale_map = scale_map
        # Stationary transform: compute M once per simulation (today's fast
        # path) instead of rebuilding it per node. For per-node maps, cache
        # matrices keyed on the (θ..., s...) value tuple — real-world maps
        # are often piecewise constant, collapsing construction to a dict
        # lookup (worst case: per-node cost, dominated by the TI scan).
        self._m_cache = {}
        self._stationary_M = None
        if (
            rotation_map is not None or scale_map is not None
        ) and stationary_transform:
            self._stationary_M = _lag_transform_matrix(
                self.dim,
                rotation_map,
                scale_map,
                np.zeros(self.dim, dtype=int),
            )

        self.n_k = {v.name: v.n_neighbors for v in training_image.variables}

        self.sg = {v: np.full(sim_shape, np.nan) for v in self.variables}
        self.informed = {
            v: np.zeros(sim_shape, dtype=bool) for v in self.variables
        }
        self.is_cond = {
            v: np.zeros(sim_shape, dtype=bool) for v in self.variables
        }
        if conditions:
            for idx, vd in conditions.items():
                for v, val in vd.items():
                    self.sg[v][idx] = val
                    self.is_cond[v][idx] = True
                    self.informed[v][idx] = True

        self.post_processing = int(post_processing)
        self.post_processing_factor = float(post_processing_factor)
        self._rng_path = rng_path
        self._rng_nodes = rng_nodes
        # Post-pass visit order: None inherits the main-path mode (an
        # explicit main path array has no meaning for a full-grid re-visit
        # -> random). Me13 §4 does not prescribe the post-pass order; the
        # explicit option is an implementation extension.
        if post_processing_path is None:
            post_processing_path = (
                "sequential"
                if isinstance(path, str) and path == "sequential"
                else "random"
            )
        self._post_path = post_processing_path

        self.max_radius_per_var = {
            v.name: v.max_radius for v in training_image.variables
        }
        _radii = list(self.max_radius_per_var.values())
        # None means "unbounded" for that variable (docstring: "None -> no
        # limit"). offset_arr is shared across variables, so if any variable
        # is unbounded the shared candidate set must be too -- capping it to
        # a finite sibling's radius would silently truncate the unbounded
        # variable's neighbourhood.
        if any(r is None for r in _radii):
            global_max_radius = None
        else:
            global_max_radius = max(_radii) if _radii else None
        max_off_int = (
            int(np.ceil(global_max_radius))
            if global_max_radius is not None
            else None
        )
        self.offset_arr = _precompute_offsets(sim_shape, max_off_int)
        self.max_radius = global_max_radius

        # Node path: every node with >= 1 uninformed variable.  Fully-conditioned
        # nodes need no simulation and are excluded (they stay -1 / always available).
        unknown = np.zeros(sim_shape, dtype=bool)
        for v in self.variables:
            unknown |= np.isnan(self.sg[v])
        self.path = _build_path(unknown, path, rng_path, sim_shape)
        n_nodes = len(self.path)
        self.u_start = rng_nodes.uniform(size=n_nodes)
        self.u_fallback = rng_nodes.uniform(size=(n_nodes, self.dim))

        self.vmap = _build_vmap(
            self.path, self.is_cond, self.variables, sim_shape
        )

        # Domains: index 0 is always the primary TI; zone TIs (if any) follow
        # in `zones` order (spec: 0 = primary, i+1 = zone_tis[i]). Search
        # hyper-parameters (n_k, max_radius, weights, distance kernel) stay
        # engine-level (always the primary's) — only the search domain (TI
        # shape/data/d_max/NaN bookkeeping) varies per zone.
        self.zone_selector = zone_selector
        self.domains = [self._build_domain(training_image)]
        for zti in zone_tis or []:
            self.domains.append(self._build_domain(zti))
        if zone_selector is not None:
            if zone_selector.shape != tuple(sim_shape):
                raise ValueError(
                    f"ds_simulate: zone_selector shape "
                    f"{zone_selector.shape!r} does not match the simulation "
                    f"grid shape {tuple(sim_shape)!r}."
                )
            smin, smax = int(zone_selector.min()), int(zone_selector.max())
            if smin < 0 or smax >= len(self.domains):
                raise ValueError(
                    f"ds_simulate: zone_selector values must be in "
                    f"[0, {len(self.domains) - 1}] (0 = primary TI, i + 1 = "
                    f"zone_tis[i]), got range [{smin}, {smax}]."
                )

    def _build_domain(self, ti):
        """Precompute one scan domain (flat arrays, strides, d_max, NaN)."""
        ti_shape = np.array(ti.shape)
        ti_shape_tuple = tuple(int(s) for s in ti.shape)
        _dim = len(ti_shape_tuple)
        ti_vars = {v.name: v.data for v in ti.variables}
        ti_strides = {
            v.name: np.array(
                [prod(ti_shape_tuple[d + 1 :]) for d in range(_dim)],
                dtype=np.intp,
            )
            for v in ti.variables
        }
        ti_flat = {
            v.name: np.ascontiguousarray(v.data).ravel() for v in ti.variables
        }
        has_nan = ti.has_nan
        if has_nan:
            finite_all = np.ones(ti_shape_tuple, dtype=bool)
            for a in ti_vars.values():
                if np.issubdtype(a.dtype, np.floating):
                    finite_all &= ~np.isnan(a)
            finite_flat = np.flatnonzero(finite_all.reshape(-1))
            if finite_flat.size == 0:
                raise ValueError(
                    "TrainingImage has no cell defined in all variables "
                    "(every cell is NaN in at least one variable); cannot "
                    "simulate."
                )
        else:
            finite_flat = None
        # Per-domain d_max: the selected zone TI's value range normalizes
        # continuous distances (spec §3 — threshold/scan_fraction semantics
        # follow the selected TI). Kernel type stays the primary's
        # (vec_distance_var is bound to self.training_image).
        d_max = {v.name: v.d_max for v in ti.variables}
        scan_config = _ScanConfig(
            variables=self.variables,
            weights=self.weights,
            ti_vars=ti_vars,
            ti_flat=ti_flat,
            ti_strides=ti_strides,
            ti_shape=ti_shape,
            threshold=self.threshold,
            cond_weight=self.cond_weight,
            distance_power=self.training_image.distance_power,
            vec_distance_var=self.training_image.vec_distance_var,
            d_max=d_max,
            ti_has_nan=has_nan,
        )
        return _Domain(
            ti=ti,
            ti_shape=ti_shape,
            ti_vars=ti_vars,
            ti_flat=ti_flat,
            ti_strides=ti_strides,
            has_nan=has_nan,
            finite_flat=finite_flat,
            scan_config=scan_config,
        )

    def _rand_fallback(self, targets, u_fb_i, dom):
        # Single random TI cell supplies the whole node-vector (preserves the
        # joint relationship); never an independent draw per variable. On a
        # masked TI the draw is restricted to cells defined in every variable.
        # Draws from the SELECTED zone TI's domain, never the primary — this
        # is what keeps the subset property true on rare fallback draws.
        if dom.has_nan:
            flat = dom.finite_flat[int(u_fb_i[0] * len(dom.finite_flat))]
            cell = tuple(
                int(c) for c in np.unravel_index(flat, tuple(dom.ti_shape))
            )
        else:
            cell = tuple(
                int(u_fb_i[d] * s) for d, s in enumerate(dom.ti_shape)
            )
        return {v: float(dom.ti_vars[v][cell]) for v in targets}

    def _node_transform_matrix(self, x_i):
        """Per-node M with value-tuple caching (dict get/set are GIL-atomic;
        a rare duplicate computation under threads is benign and value-equal)."""
        if self._stationary_M is not None:
            return self._stationary_M
        idx = tuple(int(c) for c in x_i)
        ang = self.rotation_map[idx] if self.rotation_map is not None else None
        sc = self.scale_map[idx] if self.scale_map is not None else None
        key = (
            None if ang is None else tuple(ang.tolist()),
            None if sc is None else tuple(sc.tolist()),
        )
        M = self._m_cache.get(key)
        if M is None:
            M = _lag_transform_matrix(
                self.dim, self.rotation_map, self.scale_map, x_i
            )
            self._m_cache[key] = M
        return M

    # ------------------------------------------------------------------
    # Simulation pipeline helpers (called only from _simulate_node)
    # ------------------------------------------------------------------

    def _gather_neighborhood(self, pss, x_i, x_i_t, curr_idx):
        """Seam 1 — build per-variable data-event arrays for node ``x_i``.

        For each variable, selects the ``n_k[var]`` closest already-informed
        neighbours, converts them to lag vectors, and prepends the collocated
        ``h=0`` row when the variable is already known at this node (only
        possible via conditioning).

        Parameters
        ----------
        pss : _Pass
            The current pass (main path or one post-processing round).
        x_i : numpy.ndarray, shape (dim,)
            Integer grid coordinates of the current node.
        x_i_t : tuple
            ``tuple(int(c) for c in x_i)`` — precomputed for indexing.
        curr_idx : int
            Path index of the current node (neighbours must have index < this).

        Returns
        -------
        events : dict[str, DataEvent]
            One per variable, ``lags_ti`` unset (``None``).
        """
        events = {}
        for var in self.variables:
            # Parallel path: the DAG pass already selected neighbours for this
            # node/variable (same call, informed=None, all earlier-path deps
            # guaranteed informed by DAG ordering) using the shared/global
            # radius, so reuse those coords instead of recomputing -- but
            # re-filter by this variable's own radius below (see r).  Serial
            # path (cache absent): compute normally, already per-variable.
            cache = pss.neighbor_cache.get("coords")
            if cache is not None:
                coords = cache[curr_idx][var]
                r = self.max_radius_per_var[var]
                # The DAG cache was built with the shared (global) radius, a
                # superset of any variable's own (possibly smaller) radius.
                # Since _select_neighbors scans offset_arr in strictly
                # increasing distance order, the first n_k[var] valid hits
                # under the global radius are a prefix-superset of what this
                # variable's own radius would find -- re-filtering here
                # reproduces the serial result exactly (Mariethoz2010 ¶15).
                if r is not None and len(coords):
                    d_sq = np.sum((coords - x_i) ** 2, axis=1)
                    coords = coords[d_sq <= r * r]
            else:
                coords, _ = _select_neighbors(
                    x_i,
                    self.offset_arr,
                    self.sim_shape_arr,
                    self.sim_shape,
                    pss.vmap[var],
                    curr_idx,
                    self.informed[var],
                    self.max_radius_per_var[var],
                    pss.n_k[var],
                )
            if len(coords):
                lv = (coords - x_i).astype(np.float64)
                dv = self.sg[var][tuple(coords.T)]
                cv = self.is_cond[var][tuple(coords.T)]
                lnv = np.linalg.norm(lv, axis=1)
            else:
                lv = np.empty((0, self.dim), dtype=np.float64)
                dv = np.empty(0)
                cv = np.empty(0, dtype=bool)
                lnv = np.empty(0)
            # collocated h=0 — a variable known at this very node (only possible
            # via conditioning, as the node's unknown variables are what we fill).
            # Never added for a target variable (uninformed here by definition).
            if self.informed[var][x_i_t]:
                lv = np.concatenate([np.zeros((1, self.dim)), lv], axis=0)
                dv = np.concatenate([[self.sg[var][x_i_t]], dv])
                cv = np.concatenate([[self.is_cond[var][x_i_t]], cv])
                lnv = np.concatenate([[0.0], lnv])
            events[var] = DataEvent(lv, dv, cv, lnv)
        return events

    def _transform_and_reduce_lags(self, x_i, events, dom):
        """Seam 2 — map SG lags into TI frame (non-stationary) or copy (stationary).

        When ``rotation_map`` or ``scale_map`` is set, applies the per-node
        detransform matrix, deduplicates collapsed lags
        (:func:`_transform_lags`), and globally drops lags that cannot fit the
        selected domain's TI (:func:`_reduce_to_fit`, Mariethoz2010 para [43]).

        The stationary fast-path sets ``lags_ti`` to a **copy** of ``lags_sg``
        so partial-mode truncation in seam 3 can never alias back into the SG
        lags.

        Parameters
        ----------
        x_i : numpy.ndarray, shape (dim,)
            Current node coordinates (used to index per-node maps).
        events : dict[str, DataEvent]
            Output of :meth:`_gather_neighborhood` (``lags_ti is None``).
        dom : _Domain
            The domain (primary or zone) selected for this node.

        Returns
        -------
        dict[str, DataEvent]
            New events with ``lags_ti`` populated and SG arrays possibly trimmed
            (non-stationary dedup / reduce-to-fit).
        """
        if self.rotation_map is not None or self.scale_map is not None:
            M = self._node_transform_matrix(x_i)
            out = {}
            for v in self.variables:
                de = events[v]
                lv_ti, lags_sg, values, cond_mask, lag_norms = _transform_lags(
                    de.lags_sg,
                    M,
                    de.lags_sg,
                    de.values,
                    de.cond_mask,
                    de.lag_norms,
                )
                # M10 para [43]: globally reduce an over-large transformed event
                # to fit the TI, dropping the most-outside node first (by TI
                # extent, regardless of SG rank). One reduction per simulation
                # node, independent of the TI scan position.
                lv_ti, lags_sg, values, cond_mask, lag_norms = _reduce_to_fit(
                    lv_ti, dom.ti_shape, lags_sg, values, cond_mask, lag_norms
                )
                out[v] = DataEvent(
                    lags_sg, values, cond_mask, lag_norms, lv_ti
                )
            return out
        # Stationary: TI frame == SG frame. Copy so truncation stays independent.
        return {
            v: events[v].with_lags_ti(events[v].lags_sg.copy())
            for v in self.variables
        }

    def _intersect_search_windows(self, events, dom):
        """Seam 3 — compute and intersect per-variable TI search windows.

        Calls :func:`_window_bounds` for every variable, truncates the event to
        the fitting lag count in partial mode, and intersects the per-variable
        windows into a single ``(win_lo, win_hi)`` pair.

        Returns ``win_lo is None`` (sentinel) when any single-variable window is
        infeasible or the intersection is empty; the caller draws a random
        fallback.

        Parameters
        ----------
        events : dict[str, DataEvent]
            TI-frame events from :meth:`_transform_and_reduce_lags`.
        dom : _Domain
            The domain (primary or zone) selected for this node.

        Returns
        -------
        win_lo : numpy.ndarray or None
        win_hi : numpy.ndarray or None
        events : dict[str, DataEvent]
            Possibly truncated (only when ``win_lo`` is not ``None``).
        """
        win_lo = np.zeros(self.dim, dtype=int)
        win_hi = dom.ti_shape - 1
        for var in self.variables:
            de = events[var]
            lo, hi, valid_count = _window_bounds(
                de.lags_ti, dom.ti_shape, self.boundary
            )
            if valid_count == -1:
                # Infeasible: no subset of this variable's lags fits the TI.
                return None, None, events
            if valid_count < len(de):
                # Truncate TI-frame and SG arrays to the same keep count.
                events[var] = de.truncate(valid_count)
            win_lo = np.maximum(win_lo, lo)
            win_hi = np.minimum(win_hi, hi)

        if np.any(win_lo > win_hi):
            # Infeasible: the per-variable windows do not intersect.
            return None, None, events

        return win_lo, win_hi, events

    def _scan_and_retrieve(
        self,
        pss,
        win_lo,
        win_hi,
        events,
        targets,
        u_start_i,
        u_fallback_i,
        dom,
    ):
        """Seam 4 — scan the TI window and copy the matched cell to targets.

        See module docstring for the scan semantics.  Unpacks ``events`` into
        the per-variable dicts the stateless :func:`_scan_for_match` kernel
        expects (its signature is unchanged).

        Parameters
        ----------
        pss : _Pass
            The current pass (main path or one post-processing round).
        win_lo, win_hi : numpy.ndarray, shape (dim,)
        events : dict[str, DataEvent]
            TI-frame events after truncation.
        targets : list[str]
        u_start_i : float
        u_fallback_i : numpy.ndarray, shape (dim,)
        dom : _Domain
            The domain (primary or zone) selected for this node.

        Returns
        -------
        dict[str, float]
        """
        # Integer lags per variable (already exact integers as float64, incl.
        # the 0.0 h=0 row); reused for the scan and the mean-shift gather.
        int_lags = {
            v: events[v].lags_ti.astype(int)
            for v in self.variables
            if len(events[v])
        }
        de_v = {v: events[v].values for v in self.variables}
        cm_v = {v: events[v].cond_mask for v in self.variables}
        ln_v = {v: events[v].lag_norms for v in self.variables}
        y = _scan_for_match(
            win_lo,
            tuple(win_hi - win_lo + 1),
            int_lags,
            de_v,
            cm_v,
            ln_v,
            u_start_i,
            targets,
            pss.scan_fraction,
            dom.scan_config,
        )
        if y is None:
            # No candidate in the window was defined (masked TI): treat as the
            # empty-neighbourhood case and draw a defined TI cell.
            return self._rand_fallback(targets, u_fallback_i, dom)
        y_t = tuple(int(c) for c in y)
        result = {}
        for v in targets:
            ti_val = float(dom.ti_vars[v][y_t])
            il = int_lags.get(v)
            de_ti_v = (
                dom.ti_vars[v][tuple((y + il).T)]
                if il is not None
                else np.empty(0)
            )
            result[v] = self.training_image.adjust_value(
                ti_val, de_v[v], de_ti_v, var=v
            )
        return result

    def _simulate_node(self, pss, curr_idx, x_i, u_start_i, u_fallback_i):
        """Simulate one node: a short pipeline of four named steps.

        Parameters
        ----------
        pss : _Pass
            The current pass (main path or one post-processing round).
        """
        x_i_t = tuple(int(c) for c in x_i)
        # Zonation selects WHICH search domain this node scans; the data
        # event is built from SG neighbours regardless of their zones,
        # which keeps zone boundaries coherent (spec §3 Mechanism).
        dom = (
            self.domains[int(self.zone_selector[x_i_t])]
            if self.zone_selector is not None
            else self.domains[0]
        )
        targets = [v for v in self.variables if np.isnan(self.sg[v][x_i_t])]

        # Step 1: collect informed neighbours and build per-variable data events.
        events = self._gather_neighborhood(pss, x_i, x_i_t, curr_idx)

        # Step 2: map SG lags into TI frame (stationary path copies so step 3
        # truncation never aliases back into the SG lags).
        events = self._transform_and_reduce_lags(x_i, events, dom)

        if all(len(events[v]) == 0 for v in self.variables):
            return self._rand_fallback(targets, u_fallback_i, dom)

        # Step 3: compute and intersect per-variable TI search windows.
        win_lo, win_hi, events = self._intersect_search_windows(events, dom)
        if win_lo is None:
            return self._rand_fallback(targets, u_fallback_i, dom)

        # Step 4: scan the TI window and paste the matched cell into targets.
        return self._scan_and_retrieve(
            pss, win_lo, win_hi, events, targets, u_start_i, u_fallback_i, dom
        )

    def _write_result(self, node, result):
        for v, val in result.items():
            if np.isnan(val):
                raise ValueError(
                    f"Simulation produced NaN for {(v, node)}. Check TI data."
                )
            self.sg[v][node] = val
            self.informed[v][node] = True

    def run(self, num_threads=None, progress=None):
        n_threads = (
            num_threads
            if num_threads is not None
            else (config.NUM_THREADS or 1)
        )
        executor = (
            ThreadPoolExecutor(max_workers=n_threads)
            if n_threads > 1
            else None
        )
        main_pass = _Pass(
            order=self.path,
            vmap=self.vmap,
            pos_map=None,
            n_k=self.n_k,
            scan_fraction=self.scan_fraction,
            neighbor_cache={},
        )
        try:
            self._dispatch_pass(
                main_pass,
                self.u_start,
                self.u_fallback,
                self._simulate_node,
                executor,
                progress,
                "DS",
            )
            # One executor for the whole run: the post-passes reuse it rather
            # than each spinning up their own pool.
            self._run_post_passes(progress=progress, executor=executor)
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
        return self.sg

    def _dispatch_pass(
        self, pss, u_start, u_fallback, node_fn, executor, progress, desc
    ):
        """Run one pass (main path or post-processing round) through :func:`runner._run_path`.

        Parameters
        ----------
        pss : _Pass
            The pass to run; supplies order, vmap, pos_map and ``n_k``.
        u_start, u_fallback : numpy.ndarray
            Per-node random draws for this pass.
        node_fn : callable
            ``node_fn(pss, i, x_i, u_start_i, u_fallback_i)`` → result dict.
        desc : str
            Progress label.

        Notes
        -----
        In parallel mode the DAG build already calls ``_select_neighbors`` per
        node×variable, so ``on_cache_ready`` hands those coordinates to
        ``pss.neighbor_cache`` before any node future runs and
        :meth:`_gather_neighborhood` skips the redundant call.  Serial mode
        leaves the box empty and recomputes normally.
        """
        update, close = _make_progress(progress, len(pss.order), desc)
        try:
            _run_path(
                pss.order,
                u_start,
                u_fallback,
                lambda i, x_i, u_st, u_fb: node_fn(pss, i, x_i, u_st, u_fb),
                self._write_result,
                update,
                executor,
                self.offset_arr,
                pss.vmap,
                pss.n_k,
                self.sim_shape,
                self.max_radius,
                on_cache_ready=(
                    None
                    if executor is None
                    else lambda cache: pss.neighbor_cache.__setitem__(
                        "coords", cache
                    )
                ),
                pos_map=pss.pos_map,
            )
        finally:
            close()

    def _reopen_and_simulate(self, pss, i, x_i, u_start_i, u_fallback_i):
        """Post-pass node kernel: erase the node's own values, then simulate.

        Erasing the node's ``informed`` flag suppresses the collocated ``h=0``
        self-constraint that would otherwise anchor the re-draw to its own
        stale value, and makes :meth:`_simulate_node` pick the erased
        variables up as targets.  Conditioning data is never erased.

        Safe to run concurrently: every cell this node goes on to read is one
        the DAG pass selected for it, and each such pair carries an edge, so
        no reader of this node's cell is in flight while it is erased
        (see :func:`runner._build_dag_base` Notes).
        """
        x_t = tuple(int(c) for c in x_i)
        for v in self.variables:
            if not self.is_cond[v][x_t]:
                self.sg[v][x_t] = np.nan
                self.informed[v][x_t] = False
        return self._simulate_node(pss, i, x_i, u_start_i, u_fallback_i)

    def _run_post_passes(self, progress=None, executor=None):
        """Post-processing (Meerschman2013 §4 / CASE 3).

        Re-simulates every node with >= 1 non-conditioned variable against a
        **fully informed** neighbourhood: the pass-scoped all-``-1`` vmap makes
        every informed cell available, so unlike the main path a node also sees
        cells it *precedes* in the visit order and reads their
        not-yet-re-simulated values — that is what refines the existing
        realization instead of redrawing it.  ``pos_map`` then gives
        :func:`runner._build_dag_base` the ordering its vmap no longer carries,
        so the DAG encodes the Gauss–Seidel dependency in both directions and
        each pass parallelizes while staying bit-identical to serial.

        ``scan_fraction`` and ``n_neighbors`` are divided by
        ``post_processing_factor`` (Me13 §4) into a per-round ``_Pass``.
        """
        if self.post_processing <= 0:
            return
        resim = np.zeros(self.sim_shape, dtype=bool)
        for v in self.variables:
            resim |= ~self.is_cond[v]
        if not resim.any():
            return
        p_f = self.post_processing_factor
        pass_n_k = {
            v: max(1, int(round(n / p_f))) for v, n in self.n_k.items()
        }
        pass_scan_fraction = self.scan_fraction / p_f
        sg_size = int(np.prod(self.sim_shape))
        # Round-invariant (no order-dependence): every cell is available
        # regardless of visit position, so build it once and reuse.
        pass_vmap = {
            v: np.full(sg_size, -1, dtype=np.intp) for v in self.variables
        }

        for _ in range(self.post_processing):
            if isinstance(self._post_path, str) and self._post_path == "same":
                # resim is exactly the main path's unknown-node set (a node
                # is re-simulatable iff it was not conditioned, and only
                # conditioning informs the SG before the main pass).
                order = self.path
            else:
                order = _build_path(
                    resim, self._post_path, self._rng_path, self.sim_shape
                )
            pss = _Pass(
                order=order,
                vmap=pass_vmap,
                pos_map=_order_positions(order, self.sim_shape),
                n_k=pass_n_k,
                scan_fraction=pass_scan_fraction,
                neighbor_cache={},
            )
            self._dispatch_pass(
                pss,
                self._rng_nodes.uniform(size=len(order)),
                self._rng_nodes.uniform(size=(len(order), self.dim)),
                self._reopen_and_simulate,
                executor,
                progress,
                "DS-post",
            )


def ds_simulate(
    training_image,
    sim_shape,
    threshold,
    scan_fraction,
    rng_path,
    rng_nodes,
    conditions=None,
    cond_weight=1.0,
    boundary="strict",
    num_threads=None,
    rotation_map=None,
    scale_map=None,
    stationary_transform=False,
    progress=None,
    path="random",
    zone_tis=None,
    zone_selector=None,
    post_processing=0,
    post_processing_factor=1.0,
    post_processing_path=None,
):
    """Node-wise multivariate Direct Sampling (Mariethoz2010 §3, Eq. 8).

    Co-simulates every variable of a dict-valued :class:`TrainingImage` on one
    structured grid, treating all variables equally. The path visits every
    *node* with at least one unknown variable; at each node a single joint scan
    finds the TI cell ``y`` minimising the weighted distance ``Σ_k w_k d_k`` over
    all variables, and that one cell's whole vector ``{TI[v][y]}`` is copied to
    the node's *uninformed* variables. Because every variable at a node is drawn
    from the same TI cell, the joint (cross-variable) relationship is reproduced
    exactly. Variables already known at the node (only ever via conditioning
    data) act as collocated ``h = 0`` constraints, weighted by ``cond_weight``.

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
        and must be a permutation of exactly the unknown-node set (missing
        nodes, extra nodes, and duplicate rows all raise ``ValueError``).
        Default: ``"random"``.
    conditions : dict, optional
        ``{node_index: {variable: value}}`` conditioning data.
    cond_weight : float, optional
        Weight delta for conditioning nodes (Mariethoz2010 §3 ¶26).
    boundary : str, optional
        ``"strict"`` (default) or ``"partial"`` search-window strategy.
    num_threads : int or None, optional
        Threads for the node-wise DAG. ``None`` -> ``config.NUM_THREADS``.
    rotation_map : numpy.ndarray or None, optional
        Canonical per-node rotation map, shape
        ``sim_shape + (no_of_angles(dim),)``, as produced by
        :func:`gstools.mps.nonstationary.resolve_spec`. ``None`` → no
        rotation (stationary).
    scale_map : numpy.ndarray or None, optional
        Canonical per-node scale map, shape ``sim_shape + (dim,)``, as
        produced by :func:`gstools.mps.nonstationary.resolve_spec`. ``None``
        → no dilation (stationary). All values must be positive.
    stationary_transform : bool, optional
        When ``True`` (and at least one of ``rotation_map``/``scale_map`` is
        set), the detransform matrix ``M`` is computed once for the whole
        simulation instead of being resolved per node. Set by the caller when
        both maps resolved as node-independent. Default: ``False``.
    progress : bool or callable or None, optional
        Show simulation progress. ``True`` prints a plain percentage line
        (no third-party dependency); a callable is invoked as
        ``progress(n_done, n_total)`` once per completed node.
        ``None``/``False`` (default) disables it.
    zone_tis : list of TrainingImage or None, optional
        Zone training images (Mariethoz2010 para [40]): a node inside a
        zone's region scans that zone's TI instead of the primary one.
        Indexed by ``zone_selector`` (``0`` = primary, ``i + 1`` =
        ``zone_tis[i]``). ``None`` (default) → no zonation, exact single-TI
        behaviour. Produced by :meth:`DirectSampling.__call__` from
        :attr:`MPSModel.zones`. Only a zone TI's data, shape and ``d_max``
        are used — ``n_neighbors``, ``max_radius``, ``distance``, ``weight``
        and ``penalty_matrix`` always come from ``training_image``.
    zone_selector : numpy.ndarray of numpy.intp or None, optional
        Per-node domain selector, shape ``sim_shape``. ``0`` selects the
        primary TI, ``i + 1`` selects ``zone_tis[i]``. ``None`` (default) →
        every node scans the primary TI. Zonation only changes *which* TI a
        node's scan/window/fallback draw uses; RNG consumption and the DAG
        are untouched, so output stays deterministic and thread-count
        invariant regardless of zoning.
    post_processing : int, optional
        Number of post-processing passes (Meerschman2013 §4 / CASE 3) run
        after the main path completes. Each pass re-simulates every node
        that has at least one non-conditioned variable, using a fully
        informed neighbourhood (every other node's current value is an
        eligible neighbour, `n_neighbors`/`max_radius` permitting) drawn in
        a fresh random (or sequential, matching ``path``) order. Passes are
        Gauss–Seidel, but that ordering is carried by the same dependency DAG
        the main path uses, so they honour ``num_threads`` while staying
        bit-identical to serial execution for any thread count. ``0``
        (default) → no post-processing; bit-identical to omitting the
        parameter and consumes no extra RNG draws.
    post_processing_factor : float, optional
        Divides ``scan_fraction`` and every variable's ``n_neighbors``
        (floored at 1) during post-processing passes only (Meerschman2013
        §4): a larger factor trades pattern fidelity for speed on the
        (typically much more numerous) post-pass visits. Ignored when
        ``post_processing`` is ``0``. Default: ``1.0`` (no reduction).
    post_processing_path : str, array-like, or None, optional
        Visit order for the post-processing passes (an implementation
        extension — Me13 §4 does not prescribe the post-pass order).
        ``None`` (default) inherits the main-path mode: a ``"sequential"``
        main ``path`` gives sequential post-passes, anything else gives a
        fresh random permutation per pass, drawn from ``rng_path``
        (unchanged pre-existing behaviour). ``"random"`` and
        ``"sequential"`` behave as for the main ``path``. ``"same"`` reuses
        the main pass's visit order every pass; it does not
        consume ``rng_path``. An explicit ``(N, dim)`` integer array is
        validated like an explicit main ``path`` but against the post-pass
        node set (every node with at least one non-conditioned variable),
        so it may include already-conditioned nodes (silently dropped).

    Returns
    -------
    dict
        ``{variable: numpy.ndarray}`` — one simulated field per variable.
    """
    engine = _DirectSamplingEngine(
        training_image,
        sim_shape,
        threshold,
        scan_fraction,
        rng_path,
        rng_nodes,
        conditions=conditions,
        cond_weight=cond_weight,
        boundary=boundary,
        rotation_map=rotation_map,
        scale_map=scale_map,
        stationary_transform=stationary_transform,
        path=path,
        zone_tis=zone_tis,
        zone_selector=zone_selector,
        post_processing=post_processing,
        post_processing_factor=post_processing_factor,
        post_processing_path=post_processing_path,
    )
    return engine.run(num_threads=num_threads, progress=progress)
