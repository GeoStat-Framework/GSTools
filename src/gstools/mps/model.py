"""MPS configuration object — training image and search parameters."""

import warnings

import numpy as np

from gstools.mps.training_image import TrainingImage
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
    return zones


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
        return f"MPSModel({', '.join(args)})"
