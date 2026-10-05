"""
GStools subpackage providing a Gibbs sampler for inequality conditioning.

.. currentmodule:: gstools.geometry.gibbs

The following functions are provided

.. autosummary::
   gibbs_sample
"""

import warnings

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from scipy.special import ndtr, ndtri

from gstools.krige.base import P_INV

__all__ = ["gibbs_sample"]

CLIP = 1e-12
""":class:`float`: Relative eigenvalue clipping threshold of the covariance."""

TAIL = 4.0
""":class:`float`: Standardized bound beyond which tail samplers are used."""


def gibbs_sample(
    model,
    pos,
    val=None,
    low=None,
    upp=None,
    mean=0.0,
    cond_err=0.0,
    burn_in=100,
    sweeps=200,
    seed=None,
    pseudo_inv_type="pinv",
):
    """Draw Gaussian values honouring equalities and interval constraints.

    Following Freulon & de Fouquet [Freulon1993]_: conditions on the
    equality data via a Schur complement, then runs single-site Gibbs
    sweeps over the inequality-constrained coordinates using a
    precision-matrix formulation, drawing each conditional from a
    truncated normal.

    Parameters
    ----------
    model : :any:`CovModel`
        Covariance model, in the same space as the equality/inequality
        values (pure Gaussian).
    pos : :class:`numpy.ndarray`
        Positions of all data (equality + inequality), shape ``(dim, n)``.
    val : :class:`numpy.ndarray`, optional
        Equality values, NaN (or None) at inequality-only positions.
    low : :class:`numpy.ndarray`, optional
        Lower bounds at inequality positions. Default: ``-inf``
    upp : :class:`numpy.ndarray`, optional
        Upper bounds at inequality positions. Default: ``+inf``
    mean : :class:`float`, optional
        Gaussian mean (simple kriging only). Default: 0.0
    cond_err : :class:`float` or :class:`numpy.ndarray`, optional
        Measurement error variance added to the diagonal. Default: 0.0
    burn_in : :class:`int`, optional
        Number of discarded sweeps. Default: 100
    sweeps : :class:`int`, optional
        Number of kept sweeps (only the last state is returned).
        Default: 200
    seed : :class:`int`, optional
        Seed for the sampler.
    pseudo_inv_type : :class:`str` or :any:`callable`, optional
        Key into ``gstools.krige.base.P_INV`` or a callable returning
        a pseudo-inverse, as in :any:`Krige`. Default: "pinv"

    Returns
    -------
    :class:`numpy.ndarray`
        A single draw of Gaussian values at every position in ``pos``,
        shape ``(n,)``: equalities at their observed value, inequalities
        at the final sweep's sampled value.

    Raises
    ------
    ValueError
        If shapes mismatch or a lower bound exceeds its upper bound.

    References
    ----------
    .. [Freulon1993] Freulon, X. and de Fouquet, C.,
           "Conditioning a Gaussian model with inequalities",
           Geostatistics Troia'92, Springer, 1993.
    """
    pos = np.asarray(pos, dtype=np.double).reshape(model.field_dim, -1)
    n = pos.shape[1]
    val = _full(val, n, np.nan)
    low = _full(low, n, -np.inf)
    upp = _full(upp, n, np.inf)
    mean = _full(mean, n, 0.0)
    eq = np.isfinite(val)
    ineq = ~eq
    if np.any(low[ineq] > upp[ineq]):
        raise ValueError("gibbs_sample: lower bound exceeds upper bound.")
    res = val.copy()
    if not np.any(ineq):
        return res
    # covariance consistent with Krige: isometrized distances + cond_err
    iso = model.isometrize(pos)
    cov = model.covariance(cdist(iso.T, iso.T))
    cov[np.diag_indices(n)] += np.broadcast_to(cond_err, (n,))
    cov_ii = cov[np.ix_(ineq, ineq)]
    mu = mean[ineq].copy()
    # condition on the equalities once, up front (Schur complement)
    if np.any(eq):
        pinv = (
            pseudo_inv_type
            if callable(pseudo_inv_type)
            else P_INV[pseudo_inv_type]
        )
        cov_ie = cov[np.ix_(ineq, eq)]
        weights = cov_ie @ pinv(cov[np.ix_(eq, eq)])
        mu += weights @ (val[eq] - mean[eq])
        cov_ii = cov_ii - weights @ cov_ie.T
    prec = _precision(cov_ii)
    diag = np.diag(prec).copy()
    lo, up = low[ineq], upp[ineq]
    y = np.clip(mu, lo, up)
    rng = np.random.RandomState(seed)
    det = diag <= 0
    if np.any(det):
        warnings.warn(
            f"gibbs_sample: {np.sum(det)} coordinate(s) with non-positive "
            "conditional variance treated as deterministic."
        )
    sigma = np.sqrt(np.where(det, 1.0, 1.0 / np.where(det, 1.0, diag)))
    for _ in range(int(burn_in) + int(sweeps)):
        for i in range(len(y)):
            if det[i]:
                continue
            resid = y - mu
            cond_mu = mu[i] - (prec[i] @ resid - diag[i] * resid[i]) / diag[i]
            y[i] = _trunc_norm(cond_mu, sigma[i], lo[i], up[i], rng)
    res[ineq] = y
    return res


def _full(value, n, default):
    """Broadcast an optional scalar/array to a float array of size n."""
    if value is None:
        return np.full(n, default, dtype=np.double)
    value = np.asarray(value, dtype=np.double).reshape(-1)
    if value.size == 1:
        return np.full(n, value.item(), dtype=np.double)
    if value.size != n:
        raise ValueError(
            f"gibbs_sample: expected {n} values, got {value.size}"
        )
    return value.copy()


def _precision(cov):
    """Invert a covariance via its eigendecomposition with clipping."""
    lam, vec = eigh(cov)
    thresh = max(lam.max(), 0.0) * CLIP
    clipped = lam < thresh
    if np.any(clipped):
        kept = lam[~clipped]
        smallest = kept.min() if kept.size else np.nan
        warnings.warn(
            f"gibbs_sample: clipped {np.sum(clipped)} eigenvalue(s) of the "
            f"inequality covariance (smallest retained: {smallest:.3e}), "
            "check for (near) duplicate inequality positions."
        )
        lam = np.maximum(lam, thresh if thresh > 0 else 1.0)
    return (vec / lam) @ vec.T


def _trunc_norm(mu, sigma, lo, hi, rng):
    """Draw from a univariate normal truncated to ``[lo, hi]``.

    The single-site Gibbs sampler needs scalar draws with changing
    parameters, for which ``scipy.stats.truncnorm`` is about 50 times
    slower per call due to its per-call overhead. Dispatches to one of
    three numerically stable branches:

    - both bounds within ~4 sigma: inverse CDF via
      ``scipy.special.ndtr``/``ndtri``
    - a far one-sided bound: Robert (1995) exponential-proposal
      rejection sampling
    - a narrow two-sided interval in a tail: uniform-proposal rejection
      with a normal density acceptance ratio.

    Parameters
    ----------
    mu : :class:`float`
        Mean of the untruncated normal.
    sigma : :class:`float`
        Standard deviation of the untruncated normal, ``> 0``.
    lo : :class:`float`
        Lower truncation bound, ``-inf`` allowed.
    hi : :class:`float`
        Upper truncation bound, ``+inf`` allowed.
    rng : :class:`numpy.random.RandomState`
        Random state to draw from, never advances any global/master RNG.

    Returns
    -------
    :class:`float`
        A single draw.
    """
    if not lo <= hi:
        raise ValueError("_trunc_norm: lower bound exceeds upper bound.")
    if lo == hi:
        return float(lo)
    a, b = (lo - mu) / sigma, (hi - mu) / sigma
    return mu + sigma * _std_trunc_norm(a, b, rng)


def _std_trunc_norm(a, b, rng):
    """Standard normal truncated to ``[a, b]``."""
    if a >= TAIL:
        return _tail(a, b, rng)
    if b <= -TAIL:
        return -_tail(-b, -a, rng)
    # inverse CDF, evaluated on the side where ndtr is accurate
    flip = a > 0
    if flip:
        a, b = -b, -a
    p_a, p_b = ndtr(a), ndtr(b)
    if p_b - p_a <= 1e-14 * max(p_b, 1e-300):
        x_val = _uniform_rejection(a, b, rng)
    else:
        x_val = ndtri(rng.uniform(p_a, p_b))
        x_val = min(max(x_val, a), b)
    return -x_val if flip else x_val


def _tail(a, b, rng):
    """Far upper tail ``[a, b]`` with ``a >= TAIL``."""
    if b - a < 1.0 / a:
        return _uniform_rejection(a, b, rng)
    # Robert (1995): exponential proposal with optimal rate
    alpha = 0.5 * (a + np.sqrt(a * a + 4.0))
    while True:
        z_val = a + rng.exponential(1.0 / alpha)
        if z_val > b:
            continue
        if rng.uniform() <= np.exp(-0.5 * (z_val - alpha) ** 2):
            return z_val


def _uniform_rejection(a, b, rng):
    """Uniform proposal on a narrow ``[a, b]`` with normal density ratio."""
    m_val = min(max(0.0, a), b)  # density maximum on [a, b]
    while True:
        x_val = rng.uniform(a, b)
        if rng.uniform() <= np.exp(0.5 * (m_val * m_val - x_val * x_val)):
            return x_val
