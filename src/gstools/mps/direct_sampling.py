"""
GStools subpackage providing the Direct Sampling MPS simulation class.

.. currentmodule:: gstools.mps

The following classes and functions are provided

.. autosummary::
   DirectSampling
"""

import queue
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from gstools import config
from gstools.field.base import Field
from gstools.random.rng import RNG

__all__ = ["DirectSampling"]

_VALID_BOUNDARY = ("strict", "partial")

# DS-mode scan block size.  Large enough that per-call NumPy overhead is
# negligible (essentially full vectorization speed), small enough that the
# greedy DS scan does not overcompute far past the first accepted match.
# This is a call-overhead-amortization constant, not a cache-tuned one.
_SCAN_BLOCK = 4096


def _precompute_offsets(shape, max_offset=None):
    """Neighbour offsets from the origin, sorted by Euclidean distance.

    Parameters
    ----------
    shape : tuple
        Simulation grid shape.
    max_offset : int, optional
        Maximum offset in any dimension.
        Default: ``max(shape)``.

    Returns
    -------
    numpy.ndarray, shape (N, dim)
    """
    dim = len(shape)
    if max_offset is None:
        max_offset = max(shape)
    rng_vals = np.arange(-max_offset, max_offset + 1)
    grid = np.array(np.meshgrid(*[rng_vals] * dim, indexing="ij"))
    offsets = grid.reshape(dim, -1).T
    offsets = offsets[np.any(offsets != 0, axis=1)]
    idx = np.argsort(np.sum(offsets**2, axis=1))
    return offsets[idx]


def _select_neighbors(
    x_i,
    offset_arr,
    sim_shape_arr,
    sim_shape,
    path_pos_map,
    curr_idx,
    informed,
    max_radius,
    n_neighbors,
):
    """Closest valid neighbours of ``x_i``, with their path indices.

    A candidate is valid when it is in bounds, has path index ``< curr_idx``
    (already-simulated in path order) or ``-1`` (conditioning data), and — if
    ``informed`` is given — is marked informed.  ``offset_arr`` is
    distance-sorted, so slicing the first ``n_neighbors`` survivors yields the
    closest ones.

    Passing ``informed=None`` treats every in-bounds lower-index/conditioning
    cell as available; this is correct when building the dependency DAG, where
    all earlier-path nodes are informed by definition.

    Returns
    -------
    coords : numpy.ndarray, shape (m, dim)
        Neighbour coordinates, ``m <= n_neighbors``.
    path_idx : numpy.ndarray, shape (m,)
        Path index of each neighbour (``-1`` for conditioning data).
    """
    # ``offset_arr`` is distance-sorted, so the closest ``n_neighbors`` valid
    # candidates are the first ``n_neighbors`` survivors and we can stop the
    # moment we have them.  Iterating with an early break avoids masking the
    # whole offset array for every node (O(N**2) on large grids without a
    # ``max_radius`` cap).  Tradeoff: when fewer than ``n_neighbors`` valid
    # candidates exist (sparse early-path nodes), the scan still walks the full
    # ``offset_arr`` in Python; this affects only the first few nodes and is
    # bounded by the ``max_radius`` ball when one is set.
    dim = offset_arr.shape[1]
    r_sq = max_radius * max_radius if max_radius is not None else None
    found_coords = []
    found_vidx = []
    for off in offset_arr:
        # Distance-sorted: the first offset beyond the radius ends the scan.
        if r_sq is not None and float(off @ off) > r_sq:
            break
        cand = x_i + off
        if np.any(cand < 0) or np.any(cand >= sim_shape_arr):
            continue
        vi = path_pos_map[int(np.ravel_multi_index(tuple(cand), sim_shape))]
        if not (vi < curr_idx or vi == -1):
            continue
        if informed is not None and not informed[tuple(cand)]:
            continue
        found_coords.append(cand)
        found_vidx.append(vi)
        if len(found_coords) >= n_neighbors:
            break
    if found_coords:
        return (
            np.array(found_coords, dtype=offset_arr.dtype),
            np.array(found_vidx, dtype=path_pos_map.dtype),
        )
    return (
        np.empty((0, dim), dtype=offset_arr.dtype),
        np.empty(0, dtype=path_pos_map.dtype),
    )


def _build_dag(
    path,
    n_neighbors,
    sim_shape,
    offset_arr,
    path_pos_map,
    max_radius=None,
):
    """Build the simulation dependency DAG.

    Edge ``j -> i`` means path node ``j`` (j < i) is among the n-closest
    neighbours used when simulating node ``i``.  Conditioning data carry no
    edge.  Uses the same vectorized neighbour selection as the simulation
    (:func:`_select_neighbors`), so the resulting dependencies match the set
    each node would actually pick at simulation time.
    """
    N = len(path)
    sim_shape_arr = np.array(sim_shape)
    indegree = np.zeros(N, dtype=np.int32)
    out_edges = [[] for _ in range(N)]

    for i in range(N):
        _, vidx = _select_neighbors(
            path[i],
            offset_arr,
            sim_shape_arr,
            sim_shape,
            path_pos_map,
            i,  # build-time: all path nodes with index < i are informed
            None,
            max_radius,
            n_neighbors,
        )
        # path-node neighbours (conditioning data have index -1, no edge)
        for j in vidx[vidx >= 0]:
            indegree[i] += 1
            out_edges[int(j)].append(i)

    return indegree, out_edges


def ds_simulate(
    training_image,
    sim_shape,
    n_neighbors,
    threshold,
    scan_fraction,
    rng,
    conditions=None,
    cond_weight=1.0,
    boundary="strict",
    max_radius=None,
    num_threads=None,
):
    """Direct Sampling univariate simulation (Mariethoz2010, Juda2022).

    Parameters
    ----------
    training_image : TrainingImage
        Training image; provides ``training_image.distance()`` and
        ``training_image.adjust_value()``.
    sim_shape : tuple
        Simulation grid shape.
    n_neighbors : int
        Maximum number of neighbours in the data event (Juda2022 §2).
    threshold : float
        Distance threshold for early acceptance (Juda2022 §2).
        ``0.0`` → DSBC mode.
    scan_fraction : float
        Fraction of the per-node search window to scan (Mariethoz2010 §3 ¶24).
        Evaluates at most ``floor(f · |window|)`` candidates per node.
        ``1.0`` → full window scan.
    rng : numpy.random.RandomState
        Random number generator.
    conditions : dict, optional
        ``{tuple_index: value}`` mapping of conditioning data.
    cond_weight : float, optional
        Weight δ for conditioning nodes (Mariethoz2010 §3 ¶26).
    boundary : str, optional
        Search-window strategy: ``"strict"`` (default) or ``"partial"``.
    max_radius : float, optional
        If set, SG neighbours beyond this Euclidean distance are excluded
        from the data event (Mariethoz2010 §3 ¶19).
    num_threads : int or None, optional
        Number of threads for outer DAG parallelism. ``None`` defaults to
        ``config.NUM_THREADS``.

    Returns
    -------
    numpy.ndarray
    """
    ti_data = training_image.data
    ti_shape = np.array(ti_data.shape)
    sim_shape_arr = np.array(sim_shape)
    sg = np.full(sim_shape, np.nan)
    is_cond = np.zeros(sim_shape, dtype=bool)
    informed = np.zeros(sim_shape, dtype=bool)

    if conditions:
        for idx, val in conditions.items():
            sg[idx] = val
            is_cond[idx] = True
            informed[idx] = True

    n_threads = (
        num_threads if num_threads is not None else (config.NUM_THREADS or 1)
    )
    executor = (
        ThreadPoolExecutor(max_workers=n_threads) if n_threads > 1 else None
    )
    max_off_int = int(np.ceil(max_radius)) if max_radius is not None else None
    offset_arr = _precompute_offsets(sim_shape, max_off_int)

    path = np.argwhere(np.isnan(sg))
    path = path[rng.permutation(len(path))]
    node_seeds = rng.randint(0, 2**32, size=len(path), dtype=np.int64)

    path_flat = np.ravel_multi_index(path.T, sim_shape)
    path_pos_map = np.full(int(np.prod(sim_shape)), -1, dtype=np.intp)
    path_pos_map[path_flat] = np.arange(len(path_flat))

    def _rand_ti(node_rng):
        return ti_data[tuple(node_rng.randint(0, s) for s in ti_shape)]

    def _get_neighbors(x_i, informed_in):
        curr_idx = path_pos_map[
            int(np.ravel_multi_index(tuple(x_i), sim_shape))
        ]
        coords, _ = _select_neighbors(
            x_i,
            offset_arr,
            sim_shape_arr,
            sim_shape,
            path_pos_map,
            curr_idx,
            informed_in,
            max_radius,
            n_neighbors,
        )
        return coords

    def _scan_ti(lo, win_shape, lags, de_sim, cm, ln, node_rng):
        # Precondition: win_size >= 1 (callers guarantee win_lo <= win_hi on all axes).
        # Also captures scan_fraction, ti_data, threshold, cond_weight from outer scope.
        win_size = int(np.prod(win_shape))
        max_scan = max(1, int(scan_fraction * win_size))
        start = int(node_rng.randint(0, win_size))

        # All scan positions in visit order — shape (max_scan,)
        positions = (start + np.arange(max_scan)) % win_size
        # Anchor coordinates for each position — shape (max_scan, dim)
        y_all = lo + np.column_stack(np.unravel_index(positions, win_shape))

        # lags are integer-valued float64; cast once, reuse for all candidates
        int_lags = lags.astype(int)  # (k, dim)

        def _de_ti(y_rows):
            # TI data events for the given anchor rows — shape (len(y_rows), k)
            coords = y_rows[:, None, :] + int_lags[None, :, :]
            return ti_data[tuple(coords.transpose(2, 0, 1))]

        # DSBC (threshold == 0): no early exit is possible — the global minimum
        # over the whole scan is required — so evaluate every candidate in a
        # single vectorized call.  This is the fastest path and stays exact.
        if threshold <= 0:
            all_de_ti = _de_ti(y_all)
            all_dists = training_image.vec_distance(
                de_sim, all_de_ti, cm, cond_weight, ln
            )
            best_k = int(np.argmin(all_dists))
            return ti_data[tuple(y_all[best_k])], all_de_ti[best_k]

        # DS (threshold > 0): chunked vectorized scan with an early-exit
        # checkpoint between blocks.  Each block is a full vectorized distance
        # call (so the per-element cost matches the single-call version); only
        # the threshold test runs per block.  Blocks advance in scan order, so
        # the first under-threshold candidate found is the first one globally —
        # identical to the unchunked argmax(under) result.
        best_d = np.inf
        best_y = None
        best_de = None
        for b0 in range(0, max_scan, _SCAN_BLOCK):
            y_blk = y_all[b0 : b0 + _SCAN_BLOCK]
            de_blk = _de_ti(y_blk)
            d_blk = training_image.vec_distance(
                de_sim, de_blk, cm, cond_weight, ln
            )
            under = d_blk <= threshold
            if np.any(under):
                k = int(np.argmax(under))
                return ti_data[tuple(y_blk[k])], de_blk[k]
            # No acceptable match in this block.  Track the running best with a
            # strict ``<`` test so that, if no candidate ever falls below the
            # threshold, the returned fallback equals the global argmin with the
            # same first-occurrence tie-break as the unchunked version.
            k = int(np.argmin(d_blk))
            if d_blk[k] < best_d:
                best_d = float(d_blk[k])
                best_y = y_blk[k]
                best_de = de_blk[k]
        return ti_data[tuple(best_y)], best_de

    def _simulate_node(x_i, node_rng, sg_in, informed_in):
        nbrs = _get_neighbors(x_i, informed_in)
        if len(nbrs) == 0:
            return _rand_ti(node_rng)

        lags = (nbrs - x_i).astype(np.float64)  # (k, dim)
        data_event_sim = sg_in[tuple(nbrs.T)]  # (k,)
        cond_mask = is_cond[tuple(nbrs.T)]  # (k,)
        lag_norms = np.linalg.norm(lags, axis=1)  # (k,)

        if boundary == "strict":
            # Search window Y(L_i) — Juda2022 Eq. 5, Mariethoz2010 §3 ¶19
            win_lo = np.maximum(0, np.ceil(-lags.min(axis=0))).astype(int)
            win_hi = np.minimum(
                ti_shape - 1, np.floor(ti_shape - 1 - lags.max(axis=0))
            ).astype(int)
            if np.any(win_lo > win_hi):
                return _rand_ti(node_rng)
            best_v, best_de_ti = _scan_ti(
                win_lo,
                tuple(win_hi - win_lo + 1),
                lags,
                data_event_sim,
                cond_mask,
                lag_norms,
                node_rng,
            )
            return training_image.adjust_value(
                best_v, data_event_sim, best_de_ti
            )

        else:  # "partial" — Mariethoz2010 §6.2: global template reduction
            # Lags are distance-sorted (closest first) because offset_arr is.
            # Drop farthest neighbours one at a time until the bounding box of
            # the remaining data event fits inside the TI, per the paper's
            # "ignore until it becomes possible to scan" directive (§6.2).
            valid_count = len(lags)
            while valid_count > 0:
                lags_p = lags[:valid_count]
                sw_lo = np.maximum(0, np.ceil(-lags_p.min(axis=0))).astype(int)
                sw_hi = np.minimum(
                    ti_shape - 1, np.floor(ti_shape - 1 - lags_p.max(axis=0))
                ).astype(int)
                if np.all(sw_lo <= sw_hi):
                    break
                valid_count -= 1
            else:
                # No subset of the data event fits inside the TI (the closest
                # neighbour's lag already exceeds the TI in some dimension).
                # Recover like the empty-window case in strict mode rather than
                # aborting the whole simulation.
                return _rand_ti(node_rng)
            best_v, best_de_ti = _scan_ti(
                sw_lo,
                tuple(sw_hi - sw_lo + 1),
                lags_p,
                data_event_sim[:valid_count],
                cond_mask[:valid_count],
                lag_norms[:valid_count],
                node_rng,
            )
            # For variation distance, adjust_value uses the mean of the
            # truncated data event (valid_count neighbours), not the full
            # neighbourhood mean.  This is intentional — the mean-shift
            # must be consistent with the lags actually used in the scan.
            return training_image.adjust_value(
                best_v, data_event_sim[:valid_count], best_de_ti
            )

    try:
        if executor is not None:
            indegree, out_edges = _build_dag(
                path,
                n_neighbors,
                sim_shape,
                offset_arr,
                path_pos_map,
                max_radius,
            )
            # Running ready-queue: a node is dispatched the instant its last
            # dependency completes (no per-wave barrier).  Workers read the
            # live sg / informed arrays; this is safe because (1) a node is
            # only submitted once all its dependencies are written, so the
            # values it reads are final, and (2) all shared-state mutation
            # (sg, informed, in-degree, submission) happens on this main
            # thread — workers only read.  Each numpy access holds the GIL for
            # its duration, so element reads never tear against the writes.
            # The result of every node depends only on its seed and its
            # (final) neighbour values, so the output is independent of
            # completion order and stays identical to the serial run.
            done_q = queue.Queue()
            counts = {"submitted": 0, "done": 0}

            def _run(i):
                return i, _simulate_node(
                    path[i],
                    RNG(int(node_seeds[i])).random,
                    sg,
                    informed,
                )

            def _submit(i):
                executor.submit(_run, i).add_done_callback(done_q.put)
                counts["submitted"] += 1

            for i in range(len(path)):
                if indegree[i] == 0:
                    _submit(i)

            while counts["done"] < counts["submitted"]:
                i, val = done_q.get().result()
                counts["done"] += 1
                x_i_t = tuple(path[i])
                if np.isnan(val):
                    raise ValueError(
                        f"Simulation produced NaN at {path[i]}. Check TI data."
                    )
                sg[x_i_t] = val
                informed[x_i_t] = True
                for j in out_edges[i]:
                    indegree[j] -= 1
                    if indegree[j] == 0:
                        _submit(j)
            # Defensive guard: every node must have been scheduled and run.
            # The DAG is provably acyclic, so this never trips in practice — but
            # if a future change ever introduced a cycle, no node would reach
            # in-degree 0, the loop above would exit immediately, and an all-NaN
            # field would be returned silently. Fail loudly instead.
            if counts["done"] != len(path):
                raise ValueError(
                    "DirectSampling: parallel scheduler simulated "
                    f"{counts['done']}/{len(path)} nodes; the dependency graph "
                    "is not acyclic (this should never happen)."
                )
        else:
            for i, x_i in enumerate(path):
                x_i_t = tuple(x_i)
                val = _simulate_node(
                    x_i,
                    RNG(int(node_seeds[i])).random,
                    sg,
                    informed,
                )
                if np.isnan(val):
                    raise ValueError(
                        f"Simulation produced NaN at {x_i}. Check TI data."
                    )
                sg[x_i_t] = val
                informed[x_i_t] = True
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    return sg


class DirectSampling(Field):
    """Multiple Point Statistics simulation using Direct Sampling.

    Subclasses :class:`gstools.field.base.Field`. Takes a :class:`TrainingImage`
    (analogous to :class:`CovModel`) and produces fields on structured grids.

    Parameters
    ----------
    ti : TrainingImage
        The training image (the MPS model).
    n_neighbors : int, optional
        Maximum neighbors in data event. Default: 32.
    scan_fraction : float, optional
        Fraction of the per-node search window to scan. Default: 1.
    threshold : float, optional
        Distance threshold for early acceptance. 0.0 -> DSBC mode. Default: 0.0.

        .. note::
            The threshold is **not comparable across distance metrics**. The
            ``"variation"`` distance is normalised by ``2·d_max`` (and clamped
            to ``[0, 1]``), so the same threshold is stricter for
            ``"variation"`` than for ``"l1"``/``"l2"``. Re-tune ``threshold``
            when you change the training image's distance metric.
    cond_weight : float, optional
        Weight for conditioning nodes in distance. Default: 1.0.
    boundary : str, optional
        Search-window strategy: ``"strict"`` (default) or ``"partial"``.
    max_radius : float, optional
        Exclude SG neighbours beyond this Euclidean distance from the
        data event. Default: ``None`` (no limit).
        The minimum effective value is 1.0 (the grid-cell Euclidean
        distance to the nearest neighbour). Values in ``(0, 1)`` accept
        no neighbours at all, making every node fall back to a random TI
        sample.
    num_threads : int or None, optional
        Number of threads for outer DAG parallelism. ``None`` defaults to
        ``config.NUM_THREADS``.
    seed : int or nan, optional
        Master RNG seed. Default: nan.
    """

    default_field_names = ["field"]

    def __init__(
        self,
        ti,
        n_neighbors=32,
        scan_fraction=1,
        threshold=0.0,
        cond_weight=1.0,
        boundary="strict",
        max_radius=None,
        num_threads=None,
        seed=np.nan,
    ):
        if boundary not in _VALID_BOUNDARY:
            raise ValueError(
                f"DirectSampling: boundary must be one of {_VALID_BOUNDARY!r}, "
                f"got {boundary!r}"
            )
        if int(n_neighbors) < 1:
            raise ValueError(
                f"DirectSampling: n_neighbors must be >= 1, got {n_neighbors!r}"
            )
        if not (0 < float(scan_fraction) <= 1):
            raise ValueError(
                f"DirectSampling: scan_fraction must be in (0, 1], "
                f"got {scan_fraction!r}"
            )
        if float(threshold) < 0:
            raise ValueError(
                f"DirectSampling: threshold must be >= 0, got {threshold!r}"
            )
        if float(threshold) > 1.0:
            import warnings

            warnings.warn(
                "threshold > 1.0 guarantees the first candidate is always accepted.",
                stacklevel=2,
            )
        if max_radius is not None and float(max_radius) <= 0:
            raise ValueError(
                f"DirectSampling: max_radius must be a positive float, "
                f"got {max_radius!r}"
            )
        super().__init__(model=None, dim=ti.ndim, value_type="scalar")
        self._ti = ti
        self._n_neighbors = int(n_neighbors)
        self._scan_fraction = float(scan_fraction)
        self._threshold = float(threshold)
        self._cond_weight = float(cond_weight)
        self._boundary = boundary
        self._max_radius = (
            float(max_radius) if max_radius is not None else None
        )
        self._num_threads = num_threads
        self._cond_pos = None
        self._cond_val = None
        self.rng = RNG(None if np.isnan(seed) else int(seed))

    def __call__(
        self,
        pos=None,
        seed=np.nan,
        mesh_type="structured",
        post_process=True,
        store=True,
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

        Returns
        -------
        field : :class:`numpy.ndarray`
            The simulated field.
        """
        if mesh_type != "structured":
            raise ValueError(
                "DirectSampling: only structured grids are supported."
            )
        name, save = self.get_store_config(store)
        pos, shape = self.pre_pos(pos, mesh_type)
        conditions = self._conditions_to_grid(self.pos)
        if not np.isnan(seed):
            self.rng.seed = int(seed)
        rng = np.random.RandomState(
            int(self.rng.random.randint(0, 2**32, dtype=np.int64))
        )
        field = ds_simulate(
            training_image=self._ti,
            sim_shape=shape,
            n_neighbors=self._n_neighbors,
            threshold=self._threshold,
            scan_fraction=self._scan_fraction,
            rng=rng,
            conditions=conditions,
            cond_weight=self._cond_weight,
            boundary=self._boundary,
            max_radius=self._max_radius,
            num_threads=self._num_threads,
        )
        # Categorical + post-processing (R2): mean/normalizer/trend are meant
        # for continuous fields. Applied to categorical output they turn facies
        # codes into meaningless real values, silently. Warn the user.
        if (
            post_process
            and self._ti.categorical
            and (
                self.mean is not None
                or self.normalizer is not None
                or self.trend is not None
            )
        ):
            import warnings

            warnings.warn(
                "DirectSampling: mean/normalizer/trend post-processing is set "
                "on a categorical training image. This will alter the facies "
                "codes and produce meaningless values. Pass post_process=False "
                "or unset mean/normalizer/trend for categorical simulations.",
                stacklevel=2,
            )
        return self.post_field(field, name, post_process, save)

    def _conditions_to_grid(self, axes):
        """Smart snapping: Mariethoz 2010 collision rule."""
        if self._cond_pos is None:
            return {}
        # Axis bounds for the out-of-grid check (R6): points outside the domain
        # snap to the nearest boundary node, which is silently misleading.
        bounds = [(axes[d].min(), axes[d].max()) for d in range(self.dim)]
        n_outside = 0
        candidates = {}  # idx -> (val, dist_sq)
        for k in range(self._cond_val.shape[0]):
            if any(
                self._cond_pos[d][k] < bounds[d][0]
                or self._cond_pos[d][k] > bounds[d][1]
                for d in range(self.dim)
            ):
                n_outside += 1
            idx = tuple(
                int(np.argmin(np.abs(axes[d] - self._cond_pos[d][k])))
                for d in range(self.dim)
            )
            dist_sq = sum(
                (axes[d][idx[d]] - self._cond_pos[d][k]) ** 2
                for d in range(self.dim)
            )
            if idx not in candidates or dist_sq < candidates[idx][1]:
                candidates[idx] = (self._cond_val[k], dist_sq)
        if n_outside:
            import warnings

            warnings.warn(
                f"DirectSampling: {n_outside} conditioning point(s) lie "
                "outside the simulation grid and were snapped to the nearest "
                "boundary node. Check your conditioning positions.",
                stacklevel=2,
            )
        return {idx: val for idx, (val, _) in candidates.items()}

    def set_condition(self, cond_pos, cond_val, cond_weight=None):
        """Set the conditioning data for the simulation.

        Parameters
        ----------
        cond_pos : :class:`list`
            The position tuple of the conditioning data ``(x, [y, z])``.
        cond_val : :class:`numpy.ndarray`
            The values at the conditioning positions.
        cond_weight : :class:`float`, optional
            Conditioning weight δ. If given, overrides the ``cond_weight``
            set at construction. Default: :any:`None` (keep existing weight)
        """
        from gstools.krige.tools import set_condition as _gs_set_condition

        self._cond_pos, self._cond_val = _gs_set_condition(
            cond_pos, cond_val, self.dim
        )
        if cond_weight is not None:
            self._cond_weight = float(cond_weight)

    @property
    def ti(self):
        """TrainingImage: The training image model."""
        return self._ti

    @property
    def n_neighbors(self):
        """:class:`int`: Maximum neighbours in the data event."""
        return self._n_neighbors

    @n_neighbors.setter
    def n_neighbors(self, value):
        if int(value) < 1:
            raise ValueError(
                f"DirectSampling: n_neighbors must be >= 1, got {value!r}"
            )
        self._n_neighbors = int(value)

    @property
    def scan_fraction(self):
        """:class:`float`: Fraction of the per-node search window to scan."""
        return self._scan_fraction

    @scan_fraction.setter
    def scan_fraction(self, value):
        if not (0 < float(value) <= 1):
            raise ValueError(
                f"DirectSampling: scan_fraction must be in (0, 1], got {value!r}"
            )
        self._scan_fraction = float(value)

    @property
    def threshold(self):
        """:class:`float`: Distance threshold (0.0 → DSBC mode)."""
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        if float(value) < 0:
            raise ValueError(
                f"DirectSampling: threshold must be >= 0, got {value!r}"
            )
        if float(value) > 1.0:
            import warnings

            warnings.warn(
                "threshold > 1.0 guarantees the first candidate is always accepted.",
                stacklevel=2,
            )
        self._threshold = float(value)

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
        if value not in _VALID_BOUNDARY:
            raise ValueError(
                f"DirectSampling: boundary must be one of {_VALID_BOUNDARY!r}, "
                f"got {value!r}"
            )
        self._boundary = value

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
        if value is not None and float(value) <= 0:
            raise ValueError(
                f"DirectSampling: max_radius must be a positive float, "
                f"got {value!r}"
            )
        self._max_radius = float(value) if value is not None else None

    @property
    def num_threads(self):
        """:class:`int` or :any:`None`: Number of threads for outer DAG parallelism."""
        return self._num_threads

    @num_threads.setter
    def num_threads(self, value):
        self._num_threads = None if value is None else int(value)

    def __repr__(self):
        return (
            f"DirectSampling(dim={self.dim}, "
            f"n_neighbors={self.n_neighbors}, "
            f"scan_fraction={self.scan_fraction}, "
            f"threshold={self.threshold}, "
            f"boundary={self.boundary!r})"
        )
