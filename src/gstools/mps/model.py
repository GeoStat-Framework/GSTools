"""MPS configuration object — training image and search parameters."""

import warnings

import numpy as np

from gstools.mps.training_image import TrainingImage, _check_category_codes
from gstools.mps.zone import Zone

__all__ = ["MPSModel"]

_VALID_BOUNDARY = ("strict", "partial")


def _validate_boundary(value):
    """Validate boundary; return normalized value or raise ValueError."""
    if value not in _VALID_BOUNDARY:
        raise ValueError(
            f"MPSModel: boundary must be one of {_VALID_BOUNDARY!r}, "
            f"got {value!r}"
        )
    return value


def _validate_scan_fraction(value):
    """Validate scan_fraction in (0, 1]; return float or raise ValueError."""
    if not (0 < float(value) <= 1):
        raise ValueError(
            f"MPSModel: scan_fraction must be in (0, 1], got {value!r}"
        )
    return float(value)


def _validate_threshold(value):
    """Validate threshold >= 0, warn if > 1; return float or raise ValueError."""
    if float(value) < 0:
        raise ValueError(f"MPSModel: threshold must be >= 0, got {value!r}")
    if float(value) > 1.0:
        warnings.warn(
            "MPSModel: threshold > 1.0 guarantees the first candidate is always accepted.",
            UserWarning,
            stacklevel=3,
        )
    return float(value)


def _validate_post_processing(value):
    """Validate post-processing pass count p >= 0 (Me13 §4)."""
    v = int(value)
    if v < 0:
        raise ValueError(
            f"MPSModel: post_processing must be >= 0, got {value!r}"
        )
    return v


def _validate_post_processing_factor(value):
    """Validate p_f >= 1 (Me13 §4 divides f and n by p_f to SAVE cpu)."""
    v = float(value)
    if v < 1.0:
        raise ValueError(
            f"MPSModel: post_processing_factor must be >= 1, got {value!r}"
        )
    return v


def _validate_rotation(rotation, ndim):
    """Grid-free rotation validation; returns the spec unchanged."""
    if rotation is None:
        return None
    if ndim == 1:
        raise ValueError(
            "MPSModel: rotation must be None on 1-D grids (no rotation "
            "planes exist in 1-D); scale remains meaningful."
        )
    if callable(rotation):
        return rotation
    return np.asarray(rotation, dtype=np.float64)


def _validate_scale(scale):
    """Grid-free scale validation (positivity where checkable)."""
    if scale is None or callable(scale):
        return scale
    arr = np.asarray(scale, dtype=np.float64)
    if not np.all(arr > 0):
        raise ValueError(
            f"MPSModel: scale must be positive everywhere, got minimum "
            f"value {float(arr.min())!r}."
        )
    return arr


def _validate_zones(zones, primary_ti):
    """Every zone TI must share the primary TI's variable set and kinds."""
    if zones is None:
        return []
    zones = list(zones)
    primary_kinds = {v.name: v.categorical for v in primary_ti.variables}
    for i, z in enumerate(zones):
        if not isinstance(z, Zone):
            raise TypeError(
                f"MPSModel: zones[{i}] must be a Zone, got {type(z)!r}."
            )
        if z.ti.ndim != primary_ti.ndim:
            raise ValueError(
                f"MPSModel: zones[{i}].ti has dim {z.ti.ndim}, primary TI "
                f"has dim {primary_ti.ndim}."
            )
        zone_kinds = {v.name: v.categorical for v in z.ti.variables}
        if set(zone_kinds) != set(primary_kinds):
            raise ValueError(
                f"MPSModel: zones[{i}].ti variable set "
                f"{sorted(map(str, zone_kinds))} does not match the primary "
                f"TI's {sorted(map(str, primary_kinds))}."
            )
        for name, kind in primary_kinds.items():
            if zone_kinds[name] != kind:
                raise ValueError(
                    f"MPSModel: zones[{i}].ti variable {name!r} is "
                    f"{'categorical' if zone_kinds[name] else 'continuous'} "
                    f"but the primary TI's is "
                    f"{'categorical' if kind else 'continuous'} — the "
                    "categorical/continuous kind must match per variable."
                )
            # The scan indexes the PRIMARY TI's penalty_matrix with codes read
            # from the SELECTED domain, so a zone code >= C would raise a raw
            # IndexError mid-scan (and a negative one would wrap silently).
            penalty_matrix = primary_ti.variable(name).penalty_matrix
            if penalty_matrix is not None:
                label = "data" if name is None else f"variable {name!r} data"
                _check_category_codes(
                    z.ti.variable(name).data,
                    penalty_matrix.shape[0],
                    f"MPSModel: zones[{i}].ti {label} (checked against the "
                    "primary TI's penalty_matrix)",
                )
    return zones


_VALID_POST_PATH = ("random", "sequential", "same")


def _validate_post_processing_path(value):
    """None -> inherit main-path mode; str from _VALID_POST_PATH; else (N, dim) array."""
    if value is None or (isinstance(value, str) and value in _VALID_POST_PATH):
        return value
    if isinstance(value, str):
        raise ValueError(
            f"MPSModel: post_processing_path must be one of "
            f"{_VALID_POST_PATH!r}, an (N, dim) integer array, or None "
            f"(inherit the main path mode), got {value!r}"
        )
    arr = np.asarray(value)
    if arr.ndim != 2:
        raise ValueError(
            f"MPSModel: an explicit post_processing_path must be a 2-D "
            f"(N, dim) array, got shape {arr.shape!r}"
        )
    return arr


def _spec_repr(val):
    """Repr for a rotation/scale spec: callables by name, arrays by shape."""
    if callable(val):
        return getattr(val, "__name__", repr(val))
    arr = np.asarray(val)
    if arr.ndim == 0:
        return repr(float(arr))
    if arr.ndim == 1 and arr.size <= 4:
        return repr(arr.tolist())
    return f"array(shape={arr.shape})"


class MPSModel:
    """MPS configuration: training image and search algorithm parameters.

    Analogous to ``CovModel``: holds the training image and the search
    hyper-parameters. Carries no seed, thread count, or backend selector —
    those are call-time concerns on :class:`~gstools.mps.DirectSampling`.

    Parameters
    ----------
    ti : :any:`TrainingImage`
        Training image (univariate or multivariate).
    scan_fraction : :class:`float`, optional
        Fraction of the TI to scan per node (capped at the valid search
        window). Must be in (0, 1]. Default: 1.0.
    threshold : :class:`float`, optional
        Distance threshold for early acceptance. 0.0 → DSBC mode. Default: 0.0.
    cond_weight : :class:`float`, optional
        Weight multiplier for conditioning nodes in distance. Default: 1.0.
    boundary : :class:`str`, optional
        Search-window strategy: ``"strict"`` (default) or ``"partial"``.
    rotation : :class:`float` or array-like or callable, optional
        Geometric anisotropy: rotation angles (in radians) describing how
        simulated patterns are oriented relative to the TI. At node ``x``,
        the pattern orientation is ``θ(x)``; the engine applies the inverse
        rotation to lags. Accepted forms: scalar angle, ``(n_comp,)`` vector
        of angles, materialized array, or callable ``f(x, [y, z, …])``.
        Must be ``None`` on 1-D TI. Default: ``None`` (stationary identity).

        **Convention:** angles and scales act on *index* lags (not coordinate
        lags). Migration from prior versions: if you previously used
        ``anis = s``, use ``scale=[1, s]`` instead.
    scale : :class:`float` or array-like or callable, optional
        Dilation scales: stretch factors describing how simulated patterns
        scale relative to the TI. At node ``x``, patterns scale by ``s(x)``.
        Accepted forms: scalar, ``(n_comp,)`` vector, materialized array,
        or callable ``f(x, [y, z, …])``. All values must be strictly positive.
        No axis-0 pin; all dimensions are stretched equally if scalar.
        Default: ``None`` (no dilation).
    zones : :class:`list` of :any:`Zone`, optional
        Zonation regions for non-stationary simulation. Each zone TI must
        have the same ``ndim``, variable names, and categorical kind per
        variable as the primary TI. Default: ``[]`` (single-TI simulation).

        A zone contributes **only** its data, shape and continuous range
        (``d_max``). Every search hyper-parameter and the distance kernel —
        ``n_neighbors``, ``max_radius``, ``distance``, ``weight``,
        ``penalty_matrix`` — always come from the primary TI, so setting
        them on a zone TI has no effect. Configure them on the primary TI.
    post_processing : :class:`int`, optional
        Number of post-processing passes p (Meerschman et al. 2013, §4).
        Each pass re-simulates every non-conditioning node with a fully
        informed neighbourhood (search parameters divided by
        ``post_processing_factor``) to remove simulation noise.
        0 (default) disables post-processing. Me13 §7
        advises always adding at least one post-processing step (p=1) for
        categorical simulations, for noise removal.

        **Cost.** Each pass re-visits essentially the whole grid, so ``p``
        passes mean ``p`` full-grid sweeps. They run over the same dependency
        DAG as the main path and so honour ``num_threads``, staying
        bit-identical to serial execution for any thread count.
    post_processing_factor : :class:`float`, optional
        Factor p_f >= 1 to divide ``scan_fraction`` and ``n_neighbors``
        during post-processing passes (Me13 §4). Values > 1 reduce the
        search effort per pass (cheaper, coarser re-simulation).
        Default: 1.0 (use original parameters). Me13 finds p_f has little
        effect in general and recommends p_f=1.
    post_processing_path : :class:`str`, array-like, or None, optional
        Visit order for the post-processing passes. This is an
        implementation extension, not prescribed by the DS papers (Me13 §4
        does not specify the post-pass order). Accepted forms:

        - ``None`` (default): inherit the main-path mode — a
          ``"sequential"`` main ``path`` gives sequential post-passes,
          anything else gives a fresh random permutation per pass. This
          preserves the pre-existing behaviour exactly.
        - ``"random"``: fresh random permutation per pass, drawn from the
          same RNG stream as the main path.
        - ``"sequential"``: raster order, the same every pass; does not
          consume the path RNG.
        - ``"same"``: reuse the main pass's visit order every pass; does
          not consume the path RNG.
        - an explicit ``(N, dim)`` integer array: validated the same way as
          an explicit main ``path`` (duplicate rows and missing nodes raise
          :class:`ValueError`), but against the *post-pass node set* — every
          node with at least one non-conditioned variable — rather than the
          main path's unknown-node set. This means an explicit
          post-processing path may include already-conditioned nodes (a
          full-grid raster/spiral works unchanged); such entries are
          silently dropped.
    """

    def __init__(
        self,
        ti,
        scan_fraction=1.0,
        threshold=0.0,
        cond_weight=1.0,
        boundary="strict",
        rotation=None,
        scale=None,
        zones=None,
        post_processing=0,
        post_processing_factor=1.0,
        post_processing_path=None,
    ):
        if not isinstance(ti, TrainingImage):
            raise TypeError(
                f"MPSModel: ti must be a TrainingImage, got {type(ti)!r}"
            )
        self._ti = ti
        self._scan_fraction = _validate_scan_fraction(scan_fraction)
        self._threshold = _validate_threshold(threshold)
        self._cond_weight = float(cond_weight)
        self._boundary = _validate_boundary(boundary)
        self._rotation = _validate_rotation(rotation, ti.ndim)
        self._scale = _validate_scale(scale)
        self._zones = _validate_zones(zones, ti)
        self._post_processing = _validate_post_processing(post_processing)
        self._post_processing_factor = _validate_post_processing_factor(
            post_processing_factor
        )
        self._post_processing_path = _validate_post_processing_path(
            post_processing_path
        )

    @property
    def ti(self):
        """:any:`TrainingImage`: The training image."""
        return self._ti

    @property
    def scan_fraction(self):
        """:class:`float`: Fraction of the TI scanned per node."""
        return self._scan_fraction

    @scan_fraction.setter
    def scan_fraction(self, value):
        self._scan_fraction = _validate_scan_fraction(value)

    @property
    def threshold(self):
        """:class:`float`: Distance threshold for early acceptance."""
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        self._threshold = _validate_threshold(value)

    @property
    def cond_weight(self):
        """:class:`float`: Weight multiplier for conditioning nodes."""
        return self._cond_weight

    @cond_weight.setter
    def cond_weight(self, value):
        self._cond_weight = float(value)

    @property
    def boundary(self):
        """:class:`str`: Search-window boundary strategy (``"strict"`` or ``"partial"``)."""
        return self._boundary

    @boundary.setter
    def boundary(self, value):
        self._boundary = _validate_boundary(value)

    @property
    def post_processing(self):
        """:class:`int`: Number of post-processing passes p (Me13 §4). 0 = off."""
        return self._post_processing

    @post_processing.setter
    def post_processing(self, value):
        self._post_processing = _validate_post_processing(value)

    @property
    def post_processing_factor(self):
        """:class:`float`: p_f — scan_fraction and n_neighbors are divided by this during post-passes."""
        return self._post_processing_factor

    @post_processing_factor.setter
    def post_processing_factor(self, value):
        self._post_processing_factor = _validate_post_processing_factor(value)

    @property
    def post_processing_path(self):
        """:class:`str`, array-like, or None: post-pass visit order. ``None`` inherits the main-path mode."""
        return self._post_processing_path

    @post_processing_path.setter
    def post_processing_path(self, value):
        self._post_processing_path = _validate_post_processing_path(value)

    @property
    def rotation(self):
        """Rotation spec (scalar, vector, array, or callable); ``None`` → stationary identity."""
        return self._rotation

    @property
    def scale(self):
        """Scale spec (scalar, vector, array, or callable); ``None`` → no dilation."""
        return self._scale

    @property
    def zones(self):
        """:class:`list` of :any:`Zone`: zonation regions (empty → single-TI simulation)."""
        return list(self._zones)

    def __repr__(self):
        args = [repr(self._ti)]
        defaults = dict(
            scan_fraction=1.0,
            threshold=0.0,
            cond_weight=1.0,
            boundary="strict",
            post_processing=0,
            post_processing_factor=1.0,
        )
        for name, default in defaults.items():
            val = getattr(self, f"_{name}")
            if val != default:
                args.append(f"{name}={val!r}")
        if self._rotation is not None:
            args.append(f"rotation={_spec_repr(self._rotation)}")
        if self._scale is not None:
            args.append(f"scale={_spec_repr(self._scale)}")
        if self._zones:
            args.append(f"zones=[{len(self._zones)} zone(s)]")
        if self._post_processing_path is not None:
            ppp = self._post_processing_path
            val = repr(ppp) if isinstance(ppp, str) else _spec_repr(ppp)
            args.append(f"post_processing_path={val}")
        return f"MPSModel({', '.join(args)})"
