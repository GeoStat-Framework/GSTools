"""Execution layer for Direct Sampling.

Drives a node visit order — the main simulation path or one post-processing
pass — either serially or in parallel over a dependency DAG (deterministic
regardless of thread count), plus the optional progress display. Pure stdlib +
NumPy; no TrainingImage, no engine import.
"""

import queue

import numpy as np

from gstools.mps.neighbors import _select_neighbors


def _build_dag_base(
    path,
    sim_shape,
    offset_arr,
    vmap_dict,
    n_k_dict,
    max_radius=None,
    pos_map=None,
):
    """Unified dependency-DAG builder for both univariate and multivariate DS.

    Parameters
    ----------
    vmap_dict : dict of {key: numpy.ndarray}
        One position map per key.  For univariate DS pass ``{"": path_pos_map}``.
    n_k_dict : dict of {key: int}
        Maximum neighbours per key, matching ``vmap_dict``.
    pos_map : numpy.ndarray of numpy.intp or None, optional
        Flat-cell → index in ``path`` (``-1`` for cells absent from it), used
        **only** to orient edges.  ``None`` (default, main simulation path)
        takes the orientation from the gating index returned by
        :func:`neighbors._select_neighbors`, which is the pre-existing
        behaviour.  Pass it when ``vmap_dict`` no longer carries visit order —
        i.e. for a post-processing pass, whose all-``-1`` vmap makes every
        informed cell available (Me13 §4 "fully informed neighbourhood") and
        therefore returns no usable ordering information.  See Notes.
    Returns
    -------
    indegree : dict of {key: numpy.ndarray of int32, shape (N,)}
    out_edges : list of list of (int, key)
        ``out_edges[j]`` → ``(i, key)`` pairs to process when node ``j`` completes.
    coord_cache : dict of {key: (numpy.ndarray, numpy.ndarray, int)}
        ``(flat_buf, counts, stride)`` per variable ``key``. Node ``i``'s
        neighbours are the **raveled cell indices**
        ``flat_buf[i * stride : i * stride + counts[i]]``, in the same
        distance-sorted order ``_select_neighbors`` returned them;
        ``_gather_neighborhood`` unravels them back to ``(m, dim)`` coordinates.

        Stored flat rather than as a list of ``N`` dicts of ``(m, dim)`` arrays
        because that shape dominates memory on large grids: one int32 cell
        index replaces ``dim`` int64 coordinates, and the fixed stride removes
        the per-node dict/ndarray object overhead entirely.

        Selection uses the shared (global) ``max_radius`` -- a superset of any
        variable's own, possibly smaller, radius. The parallel engine reuses
        these directly for the DAG edges, but ``_gather_neighborhood`` must
        re-filter them by each variable's own radius before use; because
        ``_select_neighbors`` scans ``offset_arr`` in strictly increasing
        distance order, that filtered result is bit-identical to what a second,
        per-variable ``_select_neighbors`` call would have produced.

    Notes
    -----
    Dependencies must be computed via the same :func:`neighbors._select_neighbors`
    call the engine uses per node (with the shared radius, a superset of every
    variable's own), so the DAG edges are a superset of the actual simulation
    neighbour relation — extra edges only add scheduling constraints, never
    incorrectness, and this is the invariant that makes parallel execution
    produce bit-identical output to the serial path once
    ``_gather_neighborhood`` re-filters per variable.

    Every edge runs from the lower to the higher visit position, so the graph
    is acyclic with the visit order itself as a topological order.  Both
    dependency directions reduce to that one rule:

    * neighbour at position ``< i`` → node ``i`` reads its **new** value, so it
      must complete first (a read-after-write true dependency — the only kind
      the main path can have, since its vmap gates candidates to earlier
      positions).
    * neighbour at position ``> i`` → node ``i`` reads its **old** value, so
      ``i`` must complete before that neighbour overwrites it (a
      write-after-read anti-dependency).  This only arises with ``pos_map``,
      where an already-simulated grid makes later-positioned cells readable.

    Because the read relation is not exactly symmetric (``n_neighbors``
    truncation lets a boundary node reach farther than its own neighbours
    reach back), an edge may be discovered from either endpoint, so ``seen``
    dedupes before ``indegree`` is counted — counting one pair twice yields a
    single matching decrement and deadlocks the dispatcher.  With
    ``pos_map=None`` every selected neighbour is at an earlier position, no
    duplicate is reachable, and the edge set is identical to the pre-existing
    one.
    """
    N = len(path)
    sim_shape_arr = np.array(sim_shape)

    # Flat cell indices fit int32 for any grid under 2**31 cells, which is
    # every grid the pure-Python engine can traverse in practice.
    n_cells = int(np.prod(sim_shape))
    flat_dt = np.int32 if n_cells < 2**31 else np.intp

    indegree = {k: np.zeros(N, dtype=np.int32) for k in vmap_dict}
    out_edges = [[] for _ in range(N)]
    # Neighbour cache in fixed-stride form: one flat buffer per key holding
    # raveled cell indices at ``i * stride``, plus the per-node count. The
    # stride is exact -- _select_neighbors returns at most ``n_k`` neighbours.
    coord_cache = {
        k: (
            np.empty(N * n_k_dict[k], dtype=flat_dt),
            np.zeros(N, dtype=np.int32),
            n_k_dict[k],
        )
        for k in vmap_dict
    }
    seen = set()
    for i in range(N):
        x_i = path[i]
        for key, vmap in vmap_dict.items():
            coords, vidx = _select_neighbors(
                x_i,
                offset_arr,
                sim_shape_arr,
                sim_shape,
                vmap,
                i,
                None,
                max_radius,
                n_k_dict[key],
            )
            flat_buf, counts, stride = coord_cache[key]
            m = len(coords)
            flat = (
                np.ravel_multi_index(coords.T, sim_shape)
                if m
                else np.empty(0, dtype=np.intp)
            )
            flat_buf[i * stride : i * stride + m] = flat
            counts[i] = m
            # Orientation index: the gating index for the main path, the
            # visit position for a post-processing pass (all--1 vmap).
            # ``flat`` is reused here -- the ravel is computed once per node.
            pos = vidx if pos_map is None or not m else pos_map[flat]
            for j in pos[pos >= 0]:
                j = int(j)
                if j == i:
                    continue
                lo, hi = (j, i) if j < i else (i, j)
                if (lo, hi, key) in seen:
                    continue
                seen.add((lo, hi, key))
                indegree[key][hi] += 1
                out_edges[lo].append((hi, key))
    return indegree, out_edges, coord_cache


def _make_progress(progress, total, desc):
    """Build ``(update, close)`` callbacks for an optional progress display.

    Parameters
    ----------
    progress : bool or callable or None
        ``None``/``False`` disables it. ``True`` prints the percentage at 5 %
        steps on a single overwritten line (no third-party dependency). A
        callable is invoked as ``progress(n_done, total)`` once per completed
        simulation node.
    total : int
        Number of nodes that will be simulated (``len(path)``).
    desc : str
        Short label for the progress line (e.g. ``"DS"``).

    Returns
    -------
    update : callable
        Call (no arguments) once per completed node.
    close : callable
        Call once when the simulation finishes.
    """
    if not progress or total <= 0:
        return (lambda: None), (lambda: None)
    if callable(progress):
        state = {"done": 0}

        def update():
            state["done"] += 1
            progress(state["done"], total)

        return update, (lambda: None)
    # Dependency-free progress: overwrite a single line at 5 % steps.
    state = {"done": 0, "last": -1}

    def update():
        state["done"] += 1
        pct = 5 * (state["done"] * 20 // total)
        if pct != state["last"]:
            state["last"] = pct
            print(
                f"\r{desc}: {pct:3d}% ({state['done']}/{total})",
                end="",
                flush=True,
            )

    def close():
        print()

    return update, close


def _run_path(
    path,
    u_start,
    u_fallback,
    simulate_fn,
    write_fn,
    update_fn,
    executor,
    offset_arr,
    vmap,
    n_k,
    sim_shape,
    max_radius,
    on_cache_ready=None,
    pos_map=None,
):
    """Dispatch the node simulation path: serial or parallel DAG.

    Parameters
    ----------
    path : numpy.ndarray, shape (N, dim)
        Ordered sequence of grid coordinates to simulate.
    u_start : numpy.ndarray, shape (N,)
        Per-node uniform random values controlling TI scan entry points.
    u_fallback : numpy.ndarray, shape (N, dim)
        Per-node uniform random values for fallback cell draws.
    simulate_fn : callable
        ``simulate_fn(i, x_i, u_start_i, u_fallback_i)`` → result dict
        mapping variable names to simulated values for node ``i``.
    write_fn : callable
        ``write_fn(node_tuple, result)`` → None.  Writes the result dict
        into the simulation grid and marks the node informed.
    update_fn : callable
        ``update_fn()`` → None.  Called once per completed node (progress).
    executor : concurrent.futures.ThreadPoolExecutor or None
        If not ``None``, nodes are dispatched in parallel over the DAG built
        by :func:`_build_dag_base`.  ``None`` → strictly serial execution.
    offset_arr : numpy.ndarray, shape (M, dim)
        Distance-sorted neighbour offsets from :func:`neighbors._precompute_offsets`.
    vmap : dict of {str: numpy.ndarray}
        Per-variable position maps (path index or ``-1`` for conditioning).
    n_k : dict of {str: int}
        Maximum neighbour count per variable.
    sim_shape : tuple
        Simulation grid shape.
    max_radius : float or None
        Euclidean cap on neighbour selection; ``None`` → unlimited.
    on_cache_ready : callable or None, optional
        When not ``None`` and ``executor`` is not ``None``, called once as
        ``on_cache_ready(coord_cache)`` immediately after :func:`_build_dag_base`
        returns and before any futures are submitted.  ``coord_cache[key]`` is
        the ``(flat_buf, counts, stride)`` triple of pre-selected neighbour cell
        indices for variable ``key`` from the DAG pass.  The engine stores it as
        ``self._neighbor_cache`` so :meth:`_gather_neighborhood` can skip the
        redundant ``_select_neighbors`` call on the parallel path.  ``None`` in
        serial mode (no DAG built, no cache).
    pos_map : numpy.ndarray of numpy.intp or None, optional
        Forwarded to :func:`_build_dag_base` to orient edges by visit position
        rather than by ``vmap``'s gating index.  Required for a
        post-processing pass, whose all-``-1`` ``vmap`` carries no ordering.
        ``None`` (default) → main-path behaviour.

    Notes
    -----
    Drives both the main simulation path and every post-processing pass; the
    two differ only in the ``vmap``/``pos_map`` pair they hand in (main:
    order-gated reads, orientation from the gate; post-pass: fully informed
    reads, orientation from ``pos_map``).  Note that syn-processing — which
    re-opens *already-decided* nodes mid-path rather than after it — is not
    expressible this way and would need its own dispatcher.
    """
    variables = list(vmap)

    if executor is not None:
        indegree, out_edges, coord_cache = _build_dag_base(
            path,
            sim_shape,
            offset_arr,
            vmap,
            n_k,
            max_radius,
            pos_map,
        )
        if on_cache_ready is not None:
            on_cache_ready(coord_cache)
        remaining = {v: indegree[v].copy() for v in variables}
        submitted = np.zeros(len(path), dtype=bool)
        done_q = queue.Queue()
        counts = {"submitted": 0, "done": 0}

        def _ready(i):
            return all(remaining[v][i] == 0 for v in variables)

        def _run(i):
            return i, simulate_fn(i, path[i], u_start[i], u_fallback[i])

        def _submit(i):
            if submitted[i]:
                return
            submitted[i] = True
            executor.submit(_run, i).add_done_callback(done_q.put)
            counts["submitted"] += 1

        for i in range(len(path)):
            if _ready(i):
                _submit(i)

        while counts["done"] < counts["submitted"]:
            i, result = done_q.get().result()
            counts["done"] += 1
            write_fn(tuple(int(c) for c in path[i]), result)
            update_fn()
            for j, v in out_edges[i]:
                remaining[v][j] -= 1
                if _ready(j):
                    _submit(j)
    else:
        for i in range(len(path)):
            result = simulate_fn(i, path[i], u_start[i], u_fallback[i])
            write_fn(tuple(int(c) for c in path[i]), result)
            update_fn()
