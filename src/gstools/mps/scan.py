"""TI search-window scanning for Direct Sampling.

Scans a candidate window for the best weighted-distance match (greedy
early-exit in DS mode, running-best argmin in DSBC mode). Depends only on
distance.py; receives TI arrays and the vectorized distance callable as
parameters (does not own a TrainingImage).
"""

import numpy as np

from gstools.mps.distance import compute_node_weights

# DS-mode scan block size.  Large enough that per-call NumPy overhead is
# negligible (essentially full vectorization speed), small enough that the
# greedy DS scan does not overcompute far past the first accepted match.
# This is a call-overhead-amortization constant, not a cache-tuned one.
_SCAN_BLOCK = 4096


def _scan_window(lo, win_shape, start, max_scan, threshold, dist_fn):
    """Chunked vectorized TI window scan (pure-Python path).

    Parameters
    ----------
    lo : numpy.ndarray
        Lower-left anchor of the search window in TI coordinates.
    win_shape : tuple of int
        Shape of the search window.
    start : int
        Starting position (flat index into window) for the random scan order.
    max_scan : int
        Maximum number of candidates to evaluate.
    threshold : float
        Distance threshold for early exit (DS mode). ``<= 0`` → DSBC (no early exit).
    dist_fn : callable
        ``dist_fn(y_blk)`` → 1-D distance array for a block of candidate
        anchor coordinates ``y_blk`` of shape ``(b, dim)``.

    Returns
    -------
    y : numpy.ndarray, shape (dim,) or None
        Coordinate of the best matching TI anchor, or ``None`` when no candidate
        in the window has a finite distance (every candidate excluded — e.g. an
        all-undefined window on a masked TI). The caller treats ``None`` as the
        empty-neighbourhood case and draws a defined TI cell.
    """
    win_size = int(np.prod(win_shape))
    positions = (start + np.arange(max_scan)) % win_size
    y_all = lo + np.column_stack(np.unravel_index(positions, win_shape))

    # DSBC (threshold <= 0) has no threshold-based early exit, but an exact
    # match d == 0 is the best attainable candidate, so accept it immediately
    # (retained for categorical, where d == 0 is common — Juda2022 §2). Both
    # modes therefore scan in blocks of ``_SCAN_BLOCK`` and track the running
    # best, so a full-window DSBC scan never materialises one giant distance
    # array. Block argmin + strict ``<`` keep the first-in-scan-order winner,
    # matching a single ``argmin`` over the whole window.
    accept = threshold if threshold > 0 else 0.0

    # DS mode (threshold > 0): strict acceptance d < t (Mariethoz2010 ¶23).
    # DSBC mode (threshold <= 0, accept == 0): accept the exact match d == 0,
    # i.e. d <= 0, since distances are non-negative (Juda2022 §2).
    dsbc = threshold <= 0
    best_d, best_y = np.inf, None
    for b0 in range(0, max_scan, _SCAN_BLOCK):
        y_blk = y_all[b0 : b0 + _SCAN_BLOCK]
        d_blk = dist_fn(y_blk)
        under = d_blk <= accept if dsbc else d_blk < accept
        if np.any(under):
            return y_blk[int(np.argmax(under))]
        k = int(np.argmin(d_blk))
        if d_blk[k] < best_d:
            best_d = float(d_blk[k])
            best_y = y_blk[k]
    # ``best_y is None`` means every candidate had a non-finite (excluded)
    # distance — only reachable on a masked TI where the whole window is
    # undefined. Signal the caller to fall back to a defined TI draw (M10
    # para [15] empty-neighbourhood handling). For a fully-defined TI the loop
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
    *,
    variables,
    weights,
    ti_vars,
    ti_shape,
    scan_fraction,
    threshold,
    cond_weight,
    distance_power,
    vec_distance_var,
    ti_has_nan,
):
    """Joint multivariate DS scan over a TI search window.

    Scans candidates in randomized order (greedy DS early-exit when the first
    candidate with distance ``< threshold`` is found; DSBC running-best argmin
    when ``threshold <= 0``).  The window is the per-variable intersection so
    every ``y + lag`` is in bounds and the gather needs no clipping; the h=0
    lag maps to ``y`` itself.  Returns the single best TI anchor coordinate
    ``y``, or ``None`` when every candidate in the window is undefined (masked
    TI — caller falls back to a random defined cell).
    """
    win_size = int(np.prod(win_shape))
    ti_size = int(np.prod(ti_shape))
    # Scan fraction is of the TI (Mariethoz2010 ¶24, Juda2022 §2), capped at
    # the valid search window so we never wrap around and re-scan anchors.
    max_scan = max(1, min(win_size, int(scan_fraction * ti_size)))
    start = int(u_start_i * win_size)

    active_vars = [v for v in variables if v in int_lags]
    active_w_total = sum(weights[v] for v in active_vars)
    precomp_w = {
        v: compute_node_weights(
            len(de_v[v]), ln_v[v], distance_power, cm_v[v], cond_weight
        )
        for v in active_vars
    }

    def _dist_block(y_blk):
        d = np.zeros(len(y_blk))
        for v in active_vars:
            il = int_lags[v]
            coords = y_blk[:, None, :] + il[None, :, :]
            all_de_ti = ti_vars[v][tuple(coords.transpose(2, 0, 1))]
            d += weights[v] * vec_distance_var(
                v,
                de_v[v],
                all_de_ti,
                cm_v[v],
                cond_weight,
                ln_v[v],
                weights=precomp_w[v],
                has_nan=ti_has_nan,
            )
        # Renormalize so the joint distance stays in [0, 1] even when
        # some variables have no data event and are excluded from int_lags.
        # Without this, the threshold fires on a compressed scale and
        # accepts matches that should be rejected.
        if 0.0 < active_w_total < 1.0:
            d /= active_w_total
        if ti_has_nan:
            # Never select an anchor whose pasted value would be undefined:
            # the matched cell must be defined in every target variable.
            center_ok = np.ones(len(y_blk), dtype=bool)
            ys = tuple(y_blk.T)
            for v in scan_targets:
                cv = ti_vars[v][ys]
                if np.issubdtype(cv.dtype, np.floating):
                    center_ok &= np.isfinite(cv)
            d = np.where(center_ok, d, np.inf)
        return d

    return _scan_window(lo, win_shape, start, max_scan, threshold, _dist_block)
