"""Direct Sampling simulation engine.

`ds_simulate` is the entry point; `_DirectSamplingEngine` holds one run's
state (output grids, informed masks, config) explicitly — what used to be
captured by closures inside `ds_simulate`. The engine orchestrates per-node
simulation by calling the stateless `neighbors`, `scan`, and `runner` modules.
"""

import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from gstools import config
from gstools.mps.neighbors import (
    _lag_transform_matrix,
    _precompute_offsets,
    _reduce_to_fit,
    _select_neighbors,
    _transform_lags,
    _window_bounds,
)
from gstools.mps.runner import _make_progress, _run_path
from gstools.mps.scan import _scan_for_match
from gstools.mps.training_image import TrainingImage

# Sentinel variable name used when wrapping a univariate TI for the MV engine.
_MV_VAR = "_v"


def _univar_as_mv_ti(ti):
    """Wrap a univariate TrainingImage as a single-variable multivariate TI.

    The resulting TI is bit-identical to a hand-crafted single-variable MV TI
    using the same data and distance parameters.  Used by ``ds_simulate`` and
    ``DirectSampling.__call__`` to route univariate work through ``ds_simulate``.

    Re-wraps the *same* data the user already constructed, so any NaN/masked-TI
    warning was already raised at the user's construction; suppress the
    duplicate here.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return TrainingImage(
            {_MV_VAR: ti.data},
            categorical={_MV_VAR: ti.categorical},
            distance={_MV_VAR: ti.distance_type},
            distance_power=ti.distance_power,
        )


class _DirectSamplingEngine:
    """Holds one Direct Sampling run's state and orchestrates it."""

    def __init__(
        self,
        training_image,
        sim_shape,
        n_neighbors,
        threshold,
        scan_fraction,
        rng_path,
        rng_nodes,
        conditions=None,
        cond_weight=1.0,
        boundary="strict",
        max_radius=None,
        rotation_map=None,
        anis_map=None,
    ):
        self.training_image = training_image
        self.variables = training_image.variables
        self.weights = (
            training_image.weights
        )  # hoisted once (avoid per-block dict copy)
        self.ti_shape = np.array(training_image.shape)
        self.sim_shape = sim_shape
        self.sim_shape_arr = np.array(sim_shape)
        self.dim = len(sim_shape)
        self.threshold = threshold
        self.scan_fraction = scan_fraction
        self.cond_weight = cond_weight
        self.boundary = boundary
        self.max_radius = max_radius
        self.rotation_map = rotation_map
        self.anis_map = anis_map

        self.n_k = (
            {v: int(n_neighbors[v]) for v in self.variables}
            if isinstance(n_neighbors, dict)
            else {v: int(n_neighbors) for v in self.variables}
        )

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

        max_off_int = (
            int(np.ceil(max_radius)) if max_radius is not None else None
        )
        self.offset_arr = _precompute_offsets(sim_shape, max_off_int)

        # Node path: every node with >= 1 uninformed variable.  Fully-conditioned
        # nodes need no simulation and are excluded (they stay -1 / always available).
        unknown = np.zeros(sim_shape, dtype=bool)
        for v in self.variables:
            unknown |= np.isnan(self.sg[v])
        path = np.argwhere(unknown)
        path = path[rng_path.permutation(len(path))]
        self.path = path
        n_nodes = len(path)
        self.u_start = rng_nodes.uniform(size=n_nodes)
        self.u_fallback = rng_nodes.uniform(size=(n_nodes, self.dim))

        sg_size = int(np.prod(sim_shape))
        path_flat = (
            np.ravel_multi_index(path.T, sim_shape)
            if len(path)
            else np.empty(0, dtype=np.intp)
        )
        # Per-variable position maps over the node path.  A path node carries its
        # node-path index; a cell conditioned in variable v is set to -1 in vmap[v]
        # (always available for v, like univariate conditioning) — essential for a
        # partially-conditioned node, whose known value must not be gated by path
        # order.  Cells absent from the path are already -1.
        self.vmap = {}
        for v in self.variables:
            m = np.full(sg_size, -1, dtype=np.intp)
            m[path_flat] = np.arange(len(path_flat))
            m[np.flatnonzero(self.is_cond[v].reshape(-1))] = -1
            self.vmap[v] = m

        # Hoist TI variable arrays once — avoids repeated method-call overhead
        # inside the per-node scan loop; all inner closures read from this dict.
        self.ti_vars = {v: training_image.variable(v) for v in self.variables}

        # Masked-TI support: NaN cells are undefined. When present, distances
        # exclude them per-position and fallback draws are restricted to cells that
        # are defined in *every* variable (so a joint draw never yields NaN). The
        # ``ti_has_nan`` gate keeps the fully-defined path byte-identical.
        self.ti_has_nan = any(
            np.issubdtype(a.dtype, np.floating) and np.isnan(a).any()
            for a in self.ti_vars.values()
        )
        if self.ti_has_nan:
            finite_all = np.ones(self.ti_shape, dtype=bool)
            for a in self.ti_vars.values():
                if np.issubdtype(a.dtype, np.floating):
                    finite_all &= ~np.isnan(a)
            self.finite_flat = np.flatnonzero(finite_all.reshape(-1))
            if self.finite_flat.size == 0:
                raise ValueError(
                    "TrainingImage has no cell defined in all variables (every "
                    "cell is NaN in at least one variable); cannot simulate."
                )
        else:
            self.finite_flat = None

    def _rand_fallback(self, targets, u_fb_i):
        # Single random TI cell supplies the whole node-vector (preserves the
        # joint relationship); never an independent draw per variable. On a
        # masked TI the draw is restricted to cells defined in every variable.
        if self.ti_has_nan:
            flat = self.finite_flat[int(u_fb_i[0] * len(self.finite_flat))]
            cell = tuple(
                int(c) for c in np.unravel_index(flat, tuple(self.ti_shape))
            )
        else:
            cell = tuple(
                int(u_fb_i[d] * s) for d, s in enumerate(self.ti_shape)
            )
        return {v: float(self.ti_vars[v][cell]) for v in targets}

    def _simulate_node(self, curr_idx, x_i, u_start_i, u_fallback_i):
        x_i_t = tuple(int(c) for c in x_i)
        targets = [v for v in self.variables if np.isnan(self.sg[v][x_i_t])]

        lags_v, de_v, cm_v, ln_v = {}, {}, {}, {}
        for var in self.variables:
            coords, _ = _select_neighbors(
                x_i,
                self.offset_arr,
                self.sim_shape_arr,
                self.sim_shape,
                self.vmap[var],
                curr_idx,
                self.informed[var],
                self.max_radius,
                self.n_k[var],
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
            lags_v[var], de_v[var], cm_v[var], ln_v[var] = lv, dv, cv, lnv

        # Geometric transform: map all per-variable SG lags into TI frame.
        # de_v / cm_v / ln_v stay on original SG geometry for SG-side ops.
        # dict(lags_v) creates a shallow-copy dict so partial-mode truncation
        # of lags_ti_v does not alias back into lags_v.
        if self.rotation_map is not None or self.anis_map is not None:
            M = _lag_transform_matrix(
                self.dim, self.rotation_map, self.anis_map, x_i
            )
            lags_ti_v = {}
            for v in self.variables:
                lv_ti, lags_v[v], de_v[v], cm_v[v], ln_v[v] = _transform_lags(
                    lags_v[v], M, lags_v[v], de_v[v], cm_v[v], ln_v[v]
                )
                # M10 para [43]: globally reduce an over-large transformed event
                # to fit the TI, dropping the most-outside node first (by TI
                # extent, regardless of SG rank). One reduction per simulation
                # node, independent of the TI scan position.
                lv_ti, lags_v[v], de_v[v], cm_v[v], ln_v[v] = _reduce_to_fit(
                    lv_ti, self.ti_shape, lags_v[v], de_v[v], cm_v[v], ln_v[v]
                )
                lags_ti_v[v] = lv_ti
        else:
            lags_ti_v = dict(lags_v)

        if all(len(lags_v[v]) == 0 for v in self.variables):
            return self._rand_fallback(targets, u_fallback_i)

        # Per-variable search window computed via _window_bounds, then intersected.
        # h=0 lags are excluded inside _window_bounds — they map to y itself and
        # never constrain the window.
        win_lo = np.zeros(self.dim, dtype=int)
        win_hi = self.ti_shape - 1
        for var in self.variables:
            lv_ti = lags_ti_v[var]
            lo, hi, valid_count = _window_bounds(
                lv_ti, self.ti_shape, self.boundary
            )
            if valid_count == -1:
                return self._rand_fallback(targets, u_fallback_i)
            if valid_count < len(lv_ti):
                # Truncate TI-frame and original arrays to the same keep count.
                # lags_ti_v uses a separate dict so this does not affect lags_v.
                lags_ti_v[var] = lv_ti[:valid_count]
                lags_v[var] = lags_v[var][:valid_count]
                de_v[var] = de_v[var][:valid_count]
                cm_v[var] = cm_v[var][:valid_count]
                ln_v[var] = ln_v[var][:valid_count]
            win_lo = np.maximum(win_lo, lo)
            win_hi = np.minimum(win_hi, hi)

        if np.any(win_lo > win_hi):
            return self._rand_fallback(targets, u_fallback_i)

        # Integer lags per variable (already exact integers as float64, incl.
        # the 0.0 h=0 row); reused for the scan and the mean-shift gather.
        int_lags = {
            v: lags_ti_v[v].astype(int)
            for v in self.variables
            if len(lags_ti_v[v])
        }
        y = _scan_for_match(
            win_lo,
            tuple(win_hi - win_lo + 1),
            int_lags,
            de_v,
            cm_v,
            ln_v,
            u_start_i,
            targets,
            variables=self.variables,
            weights=self.weights,
            ti_vars=self.ti_vars,
            ti_shape=self.ti_shape,
            scan_fraction=self.scan_fraction,
            threshold=self.threshold,
            cond_weight=self.cond_weight,
            distance_power=self.training_image.distance_power,
            vec_distance_var=self.training_image.vec_distance_var,
            ti_has_nan=self.ti_has_nan,
        )
        if y is None:
            # No candidate in the window was defined (masked TI): treat as the
            # empty-neighbourhood case and draw a defined TI cell. Avoids the
            # ``y + lag`` gather below, which an unconstrained cell could send
            # out of bounds.
            return self._rand_fallback(targets, u_fallback_i)
        y_t = tuple(int(c) for c in y)
        # Copy the single matched cell's vector to every uninformed variable.
        result = {}
        for v in targets:
            ti_val = float(self.ti_vars[v][y_t])
            il = int_lags.get(v)
            de_ti_v = (
                self.ti_vars[v][tuple((y + il).T)]
                if il is not None
                else np.empty(0)
            )
            result[v] = self.training_image.adjust_value(
                ti_val, de_v[v], de_ti_v, var=v
            )
        return result

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
        update_progress, close_progress = _make_progress(
            progress, len(self.path), "DS"
        )

        try:
            _run_path(
                self.path,
                self.u_start,
                self.u_fallback,
                lambda i, x_i, u_st, u_fb: self._simulate_node(
                    i, x_i, u_st, u_fb
                ),
                self._write_result,
                update_progress,
                executor,
                self.offset_arr,
                self.vmap,
                self.n_k,
                self.sim_shape,
                self.max_radius,
            )
        finally:
            close_progress()
            if executor is not None:
                executor.shutdown(wait=True)

        return self.sg


def ds_simulate(
    training_image,
    sim_shape,
    n_neighbors,
    threshold,
    scan_fraction,
    rng_path,
    rng_nodes,
    conditions=None,
    cond_weight=1.0,
    boundary="strict",
    max_radius=None,
    num_threads=None,
    rotation_map=None,
    anis_map=None,
    progress=None,
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
        Multivariate training image (``training_image.multivariate`` is True).
    sim_shape : tuple
        Simulation grid shape.
    n_neighbors : int or dict
        Maximum neighbours per variable. An ``int`` broadcasts to all variables;
        a dict gives one value per variable name.
    threshold : float
        Distance threshold (Juda2022 §2). ``0.0`` -> DSBC mode.
    scan_fraction : float
        Fraction of the TI to scan per node, capped at the valid search window.
    rng_path : numpy.random.RandomState
        RandomState controlling the simulation path (node visit order).
    rng_nodes : numpy.random.RandomState
        RandomState controlling TI scan entry points and fallback cells.
    conditions : dict, optional
        ``{node_index: {variable: value}}`` conditioning data.
    cond_weight : float, optional
        Weight delta for conditioning nodes (Mariethoz2010 §3 ¶26).
    boundary : str, optional
        ``"strict"`` (default) or ``"partial"`` search-window strategy.
    max_radius : float, optional
        Euclidean cap on neighbour selection.
    num_threads : int or None, optional
        Threads for the node-wise DAG. ``None`` -> ``config.NUM_THREADS``.
    rotation_map : numpy.ndarray or None, optional
        Per-node rotation angles, shape matching the simulation grid. ``None``
        → no rotation (stationary). Use ``DirectSampling.set_nonstationary``
        to produce this array from user-facing scalar or array inputs.
    anis_map : numpy.ndarray or None, optional
        Per-node anisotropy ratios, shape matching the simulation grid.
        ``None`` → isotropic (stationary). All values must be positive.
    progress : bool or callable or None, optional
        Show simulation progress. ``True`` displays a :mod:`tqdm` bar (or a
        plain percentage line if ``tqdm`` is not installed); a callable is
        invoked as ``progress(n_done, n_total)`` once per completed node.
        ``None``/``False`` (default) disables it.

    Returns
    -------
    dict
        ``{variable: numpy.ndarray}`` — one simulated field per variable.
    """
    engine = _DirectSamplingEngine(
        training_image,
        sim_shape,
        n_neighbors,
        threshold,
        scan_fraction,
        rng_path,
        rng_nodes,
        conditions=conditions,
        cond_weight=cond_weight,
        boundary=boundary,
        max_radius=max_radius,
        rotation_map=rotation_map,
        anis_map=anis_map,
    )
    return engine.run(num_threads=num_threads, progress=progress)
