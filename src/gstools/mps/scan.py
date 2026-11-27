"""TI search-window scanning for Direct Sampling.

Scans a candidate window for the best weighted-distance match (greedy
early-exit in DS mode, running-best argmin in DSBC mode). Candidates are
visited in the order set by ``scan_path``: ``"sequential"`` is a contiguous
wrap-around walk from a random start (Mariethoz2010 ¶19, Juda2022 §2),
``"random"`` visits distinct cells in random order (Juda2022 §2; DeeSse,
Straubhaar2020 §2.1). Depends only on distance.py; the TI arrays arrive in a
:class:`_ScanConfig` (no TrainingImage is owned here).
"""

from dataclasses import dataclass

import numpy as np
from numpy.random import default_rng

from gstools.mps.distance import compute_node_weights, variable_distance


@dataclass(frozen=True)
class _ScanConfig:
    """Domain-constant inputs to :func:`_scan_for_match`, built once per domain.

    Every field is constant over the node path of a pass. The kernel fields
    (``weights``, ``distance_power``, ``cond_weight``, ``kernel_vars``) come
    from the pass's kernel TI (the primary TI, or the post-processing TI in
    post-passes); the others describe the domain (TI data, shape, per-variable
    ``d_max``). ``scan_fraction`` differs per pass (Meerschman2013 §4 divides it
    by ``PostProcessing.factor``), so it is an argument of
    :func:`_scan_for_match`.
    """

    variables: list
    weights: dict
    ti_vars: dict
    ti_flat: dict
    ti_strides: dict
    ti_shape: object
    threshold: float  # None -> DSBC
    cond_weight: dict  # {var: float}, from the kernel TI
    distance_power: dict  # {var: float}, from the kernel TI
    kernel_vars: dict  # {var: Variable} (the metrics), from the kernel TI
    d_max: dict  # {var: float or None}, normalization range of the domain's TI
    ti_has_nan: bool
    var_has_nan: dict  # {var: bool}; NaN-free variables skip the NaN path
    scan_path: str = "sequential"  # window scan order (module docstring)
    finite_flat: object = (
        None  # flat cells defined in every variable (NaN TIs)
    )


# DS-mode scan block size.  Large enough that per-call NumPy overhead is
# negligible (essentially full vectorization speed), small enough that the
# greedy DS scan does not overcompute far past the first accepted match.
# This is a call-overhead-amortization constant, not a cache-tuned one.
_SCAN_BLOCK = 4096


def _scan_window(
    lo,
    win_shape,
    u_start_i,
    max_scan,
    threshold,
    dist_fn,
    scan_path="sequential",
):
    """Chunked vectorized TI window scan.

    Parameters
    ----------
    lo : numpy.ndarray
        Lower-left anchor of the search window in TI coordinates.
    win_shape : tuple of int
        Shape of the search window.
    u_start_i : float
        The node's pre-drawn uniform in ``[0, 1)``.  ``"sequential"``: start
        position ``int(u_start_i * win_size)``.  ``"random"``: stream seed.
    max_scan : int
        Maximum number of candidates to evaluate.
    threshold : float or None
        Distance threshold for early exit (DS mode). ``None`` → DSBC
        (no early exit).
    dist_fn : callable
        ``dist_fn(y_blk)`` → 1-D distance array for a block of candidate
        anchor coordinates ``y_blk`` of shape ``(b, dim)``.
    scan_path : str, optional
        ``"sequential"`` (default) or ``"random"``; see the module docstring.

    Returns
    -------
    y : numpy.ndarray, shape (dim,) or None
        Coordinate of the best matching TI anchor, or ``None`` when no candidate
        in the window has a finite distance (every candidate excluded — e.g. an
        all-undefined window on a masked TI). The caller treats ``None`` as the
        empty-neighbourhood case and draws a defined TI cell.
    """
    win_size = int(np.prod(win_shape))
    # "sequential": wrap-around walk from a random start (Mariethoz2010 ¶19,
    # Juda2022 §2). "random": max_scan distinct cells in random order
    # (Juda2022 §2; DeeSse, Straubhaar2020 §2.1), keyed on the node's own
    # u_start_i so no extra RNG draw is needed.
    #
    # Deliberate exception to the "use gstools.random.RNG" rule: reproducibility
    # is unaffected, since the only entropy is u_start_i, itself drawn from the
    # engine's gstools RNG. gstools RNG cannot serve this call anyway -- it
    # exposes no ``choice``, and its legacy-RandomState ``.random`` implements
    # ``choice(replace=False)`` via a full permutation, i.e. O(win_size)
    # instead of O(max_scan).
    #
    # Positions are built per ``_SCAN_BLOCK`` inside the loop below rather than
    # for all ``max_scan`` candidates upfront, since the greedy DS scan
    # typically exits long before the last block. This is unconditional
    # rather than mode-gated: the sequential walk's block is a cheap arange.
    if scan_path == "random":
        positions = default_rng(int(u_start_i * 2**53)).choice(
            win_size, max_scan, replace=False
        )
        start = None
    else:
        positions = None
        start = int(u_start_i * win_size)

    # DSBC has no threshold-based early exit, but an exact match d == 0 is
    # the best attainable candidate, so accept it immediately (this matters
    # for categorical, where d == 0 is common — Juda2022 §2). Both modes
    # therefore scan in blocks of ``_SCAN_BLOCK`` and track the running best,
    # so a full-window DSBC scan never materialises one giant distance array.
    # Block argmin + strict ``<`` keep the first-in-scan-order winner,
    # matching a single ``argmin`` over the whole window.
    dsbc = threshold is None
    accept = 0.0 if dsbc else threshold

    # DS mode (threshold > 0): strict acceptance d < t (Mariethoz2010 ¶23).
    # DSBC mode: accept the exact match d == 0, i.e. d <= 0, since distances
    # are non-negative (Juda2022 §2).
    best_d, best_y = np.inf, None
    for b0 in range(0, max_scan, _SCAN_BLOCK):
        b1 = min(b0 + _SCAN_BLOCK, max_scan)
        # The sequential walk is ``(start + k) % win_size`` and
        # ``unravel_index`` acts elementwise, so a block is a slice of the walk.
        pos_blk = (
            (start + np.arange(b0, b1)) % win_size
            if positions is None
            else positions[b0:b1]
        )
        y_blk = lo + np.column_stack(np.unravel_index(pos_blk, win_shape))
        d_blk = dist_fn(y_blk)
        # NaN would win np.argmin and hide the block's real best: rank it last.
        d_blk = np.where(np.isnan(d_blk), np.inf, d_blk)
        under = d_blk <= accept if dsbc else d_blk < accept
        if np.any(under):
            return y_blk[int(np.argmax(under))]
        k = int(np.argmin(d_blk))
        if d_blk[k] < best_d:
            best_d = float(d_blk[k])
            best_y = y_blk[k]
    # ``best_y is None`` means every candidate had a non-finite (excluded)
    # distance — only reachable on a masked TI where the whole window is
    # undefined. Signal the caller to fall back to a defined TI draw
    # (Mariethoz2010 ¶15 empty-neighbourhood handling). For a fully-defined TI the loop
    # always sets best_y, so this returns a real anchor unchanged.
    return best_y


def _scan_for_match(
    lo,
    win_shape,
    int_lags,
    de_v,
    cm_v,
    ln_v,
    u_start_i,
    scan_targets,
    scan_fraction,
    cfg,
):
    """Joint DS scan over a TI search window.

    Scans candidates in the order given by ``cfg.scan_path`` (module docstring;
    greedy DS early-exit at the first candidate with distance ``< threshold``,
    DSBC running-best argmin when ``threshold`` is ``None``).  The window is the per-variable intersection so
    every ``y + lag`` is in bounds and the gather needs no clipping; the h=0
    lag maps to ``y`` itself.  Returns the single best TI anchor coordinate
    ``y``, or ``None`` when every candidate in the window is undefined (masked
    TI — caller falls back to a random defined cell).

    Parameters
    ----------
    lo : numpy.ndarray, shape (dim,)
        Lower corner of the search window in TI coordinates.
    win_shape : tuple of int
        Shape of the search window.
    int_lags : dict of {str: numpy.ndarray}
        TI-frame integer lags per variable that has a data event.
    de_v, cm_v, ln_v : dict of {str: numpy.ndarray}
        Per variable: data-event values, conditioning mask and SG lag norms.
    u_start_i : float
        The node's pre-drawn uniform in ``[0, 1)`` (see :func:`_scan_window`).
    scan_targets : list of str
        Variables being simulated at this node.
    scan_fraction : float
        This pass's scan fraction (main path, or divided by
        ``PostProcessing.factor`` for a post-pass).
    cfg : _ScanConfig
        The scan domain.
    """
    win_size = int(np.prod(win_shape))
    ti_size = int(np.prod(cfg.ti_shape))
    # Scan fraction is of the TI (Mariethoz2010 ¶24, Juda2022 §2), capped at
    # the valid search window so we never wrap around and re-scan anchors.
    max_scan = max(1, min(win_size, int(scan_fraction * ti_size)))

    active_vars = [v for v in cfg.variables if v in int_lags]
    active_w_total = sum(cfg.weights[v] for v in active_vars)
    precomp_w = {
        v: compute_node_weights(
            len(de_v[v]),
            ln_v[v],
            cfg.distance_power[v],
            cm_v[v],
            cfg.cond_weight[v],
        )
        for v in active_vars
    }
    # Precompute flat lag offsets for each variable: flat_il[v] = int_lags[v] @ strides.
    # Combined with base = y_blk @ strides, all_de_ti = ti_flat[v][base[:,None] + flat_il[v][None,:]].
    # In-bounds invariant: every y + lag is guaranteed in-bounds by window construction
    # (_intersect_search_windows), so this flat take needs no bounds checking.
    flat_il = {v: int_lags[v] @ cfg.ti_strides[v] for v in active_vars}

    def _dist_block(y_blk):
        d = np.zeros(len(y_blk))
        for v in active_vars:
            if cfg.weights[v] == 0.0:
                continue  # carried along, not matched; also avoids 0 * inf
            base = y_blk @ cfg.ti_strides[v]  # shape (B,)
            all_de_ti = cfg.ti_flat[v][base[:, None] + flat_il[v][None, :]]
            d += cfg.weights[v] * variable_distance(
                cfg.kernel_vars[v],
                de_v[v],
                all_de_ti,
                precomp_w[v],
                cfg.d_max[v],
                has_nan=cfg.var_has_nan[v],
            )
        # Renormalize by the active weights so the joint distance keeps the
        # same scale when some variables have no data event and are excluded
        # from int_lags.
        # Without this, the threshold fires on a compressed scale and
        # accepts matches that should be rejected.
        if 0.0 < active_w_total < 1.0:
            d /= active_w_total
        if cfg.ti_has_nan:
            # Never select an anchor whose pasted value would be undefined:
            # the matched cell must be defined in every target variable.
            center_ok = np.ones(len(y_blk), dtype=bool)
            ys = tuple(y_blk.T)
            for v in scan_targets:
                cv = cfg.ti_vars[v][ys]
                if np.issubdtype(cv.dtype, np.floating):
                    center_ok &= np.isfinite(cv)
            d = np.where(center_ok, d, np.inf)
        return d

    return _scan_window(
        lo,
        win_shape,
        u_start_i,
        max_scan,
        cfg.threshold,
        _dist_block,
        cfg.scan_path,
    )
