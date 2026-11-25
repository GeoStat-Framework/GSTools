"""MPS configuration object — training image and search parameters."""

import warnings
from typing import NamedTuple

import numpy as np

from gstools.mps.post_processing import PostProcessing
from gstools.mps.training_image import TrainingImage, _check_category_codes
from gstools.mps.zone import Zone

__all__ = ["MPSModel"]

_VALID_SCAN_PATH = ("sequential", "random")


def _validate_scan_path(value):
    """Validate scan_path; return normalized value or raise ValueError."""
    if value not in _VALID_SCAN_PATH:
        raise ValueError(
            f"MPSModel: scan_path must be one of {_VALID_SCAN_PATH!r}, "
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
    """Validate threshold: None (DSBC) or a float > 0 (DS). Warn if > 1."""
    if value is None:
        return None
    v = float(value)
    if not v > 0.0:  # also rejects NaN
        raise ValueError(
            f"MPSModel: threshold must be > 0 or None, got {value!r}"
        )
    if v > 1.0:
        warnings.warn(
            "MPSModel: threshold > 1.0 accepts every candidate whose distance "
            "is at most 1, so the first candidate scanned is normally accepted.",
            UserWarning,
            stacklevel=3,
        )
    return v


def _validate_ti(value):
    """Return ``value`` if it is a TrainingImage, else raise ``TypeError``."""
    if not isinstance(value, TrainingImage):
        raise TypeError(
            f"MPSModel: ti must be a TrainingImage, got {type(value)!r}"
        )
    return value


def _validate_post_processing(value):
    """None (off) or a PostProcessing; anything else raises ``TypeError``."""
    if value is not None and not isinstance(value, PostProcessing):
        raise TypeError(
            "MPSModel: post_processing must be a PostProcessing or None, "
            f"got {type(value)!r}."
        )
    return value


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
    arr = np.asarray(rotation, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("MPSModel: rotation angles must be finite.")
    return arr


def _validate_scale(scale):
    """Grid-free scale validation (positivity where checkable)."""
    if scale is None or callable(scale):
        return scale
    arr = np.asarray(scale, dtype=np.float64)
    if not np.all((arr > 0) & np.isfinite(arr)):
        raise ValueError(
            "MPSModel: scale must be finite and positive everywhere."
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
                f"{sorted(zone_kinds)} does not match the primary "
                f"TI's {sorted(primary_kinds)}."
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
                label = f"variable {name!r} data"
                _check_category_codes(
                    z.ti.variable(name).data,
                    penalty_matrix.shape[0],
                    f"MPSModel: zones[{i}].ti {label} (checked against the "
                    "primary TI's penalty_matrix)",
                )
    return zones


class _Resolved(NamedTuple):
    """What the TI-dependent checks produce: normalized zones and the post-pass TI."""

    zones: list
    post_ti: object  # TrainingImage, or None when post-processing is off
    post_n_k: object  # {variable: neighbours per post-pass}, or None


def _post_pass_n_neighbors(post_ti, p_f):
    """Post-pass ``n_neighbors`` per variable: ``round(n / p_f)``, floored at 1.

    Raises ``ValueError`` when a ``distance="variation"`` variable would get
    fewer than 2 neighbours (a one-point event has no local mean).
    """
    n_k = {}
    for v in post_ti.variables:
        n = max(1, int(round(v.n_neighbors / p_f)))
        if v.variation_p_norm is not None and n < 2:
            raise ValueError(
                f"MPSModel: post_processing: variable {v.name!r} has "
                f"distance='variation' with n_neighbors={v.n_neighbors}; "
                f"factor={p_f:g} gives {n} post-pass neighbour(s), but "
                "variation needs >= 2."
            )
        n_k[v.name] = n
    return n_k


def _resolve_post(ti, zones, post):
    """Apply the overrides of ``post`` to ``ti``; return ``(post_ti, post_n_k)``."""
    if post is None:
        return None, None
    unknown = set(post.overrides) - {v.name for v in ti.variables}
    if unknown:
        raise ValueError(
            f"MPSModel: post_processing overrides name(s) {sorted(unknown)!r} "
            f"not found in the TI's variables {[v.name for v in ti.variables]!r}."
        )
    post_vars = []
    for var in ti.variables:
        fields = post.overrides.get(var.name)
        if not fields:
            post_vars.append(var)
            continue
        label = f"post_processing overrides[{var.name!r}]"
        try:
            new = var.replace(**fields)
        except ValueError as err:
            raise ValueError(f"MPSModel: {label}: {err}") from err
        if "penalty_matrix" in fields and new.penalty_matrix is not None:
            for i, z in enumerate(zones):
                _check_category_codes(
                    z.ti.variable(var.name).data,
                    new.penalty_matrix.shape[0],
                    f"MPSModel: zones[{i}].ti data (checked against "
                    f"{label}['penalty_matrix'])",
                )
        post_vars.append(new)
    try:
        post_ti = TrainingImage(post_vars)
    except ValueError as err:
        raise ValueError(
            f"MPSModel: post_processing overrides: {err}"
        ) from err
    return post_ti, _post_pass_n_neighbors(post_ti, post.factor)


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
    Every parameter is also a property that can be set later; the new value is
    validated against the others and a failed set leaves the model unchanged.

    Parameters
    ----------
    ti : :any:`TrainingImage`
        Training image.
    scan_fraction : :class:`float`, optional
        Fraction of the TI to scan per node (capped at the valid search
        window). Must be in (0, 1]. Default: 1.0. Post-processing passes
        scan ``scan_fraction / factor`` (see :class:`PostProcessing`).
    threshold : :class:`float` or None, optional
        Distance threshold for early acceptance (Mariethoz2010 ¶23).
        ``None`` → DSBC mode (no threshold; take the best candidate found
        in the scan). Otherwise must be > 0; a value above 1 warns, since
        distances are normally at most 1 and the first candidate scanned is
        then accepted. Default: ``None``.
    rotation : :class:`float` or array-like or callable, optional
        Geometric anisotropy: rotation angles (in radians) describing how
        simulated patterns are oriented relative to the TI. At node ``x``,
        the pattern orientation is ``θ(x)``; the engine applies the inverse
        rotation to lags. Accepted forms: scalar angle, ``(n_comp,)`` vector
        of angles, materialized array, or callable ``f(x, [y, z, …])``.
        A single value per node is the first angle (x-y plane) with the
        others zero, as in :func:`gstools.tools.set_angles`.
        Must be ``None`` on 1-D TI. Default: ``None`` (stationary identity).

        **Convention:** angles and scales act on *index* lags (not coordinate
        lags).
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
        variable as the primary TI. Default: ``None`` (single-TI simulation;
        the :attr:`zones` property then returns ``[]``).

        A zone contributes **only** its data, shape and continuous range
        (``d_max``). Every other per-variable setting — ``distance``,
        ``weight``, ``n_neighbors``, ``max_radius``, ``penalty_matrix``,
        ``distance_power``, ``cond_weight`` — comes from the primary TI (or,
        in post-processing passes, from its overrides), so setting them on a
        zone TI has no effect.
    post_processing : :any:`PostProcessing` or None, optional
        Post-processing passes (Meerschman2013 §4) run after the main pass;
        see :class:`PostProcessing` for the passes, effort factor, visit
        order, per-variable overrides and node mask. Meerschman2013 §7
        advises at least one pass for categorical simulations. ``None``
        (default) disables post-processing. The overrides are checked
        against the TI when the model is built or changed.
    scan_path : :class:`str`, optional
        TI window scan order per node. ``"sequential"``: contiguous walk from
        a random start (Mariethoz2010 ¶19). ``"random"``: the same number of
        distinct cells in random order (Juda2022 §2; DeeSse, Straubhaar2020
        §2.1), which removes the contiguous run's spatial bias when
        ``scan_fraction < 1``. In DSBC mode ties between equally good
        candidates go to the first one in scan order, which is uniformly
        random among them only for ``"random"``. Default: ``"sequential"``.

    Raises
    ------
    TypeError
        If ``ti`` is not a :class:`TrainingImage`, ``post_processing`` is not
        a :class:`PostProcessing` or ``None``, or ``zones`` holds a non-:class:`Zone`.
    ValueError
        If ``scan_fraction`` is outside (0, 1], ``threshold`` is not > 0,
        ``scan_path`` is unknown, ``rotation`` is given for a
        1-D TI or has non-finite values, ``scale`` is not finite and positive,
        a zone TI differs from the primary TI in dimension, variable names or
        categorical kind, or a post-processing override is invalid (unknown
        variable, invalid value, mixed weights, or fewer than 2 neighbours
        for a ``"variation"`` variable after dividing by ``factor``). Setting
        an attribute later raises the same errors and leaves the model
        unchanged.

    Warns
    -----
    UserWarning
        If ``threshold`` is above 1, or a pass would scan fewer than 2 TI
        candidates per node.

    Examples
    --------
    >>> import numpy as np
    >>> import gstools as gs
    >>> ti = gs.mps.TrainingImage(np.indices((20, 20)).sum(axis=0) % 3.0)
    >>> model = gs.mps.MPSModel(ti, scan_fraction=0.5, threshold=0.05)
    >>> model.scan_fraction
    0.5
    >>> model.post_processing = gs.mps.PostProcessing(1)
    >>> model.post_processing.passes
    1
    """

    def __init__(
        self,
        ti,
        *,
        scan_fraction=1.0,
        threshold=None,
        rotation=None,
        scale=None,
        zones=None,
        post_processing=None,
        scan_path="sequential",
    ):
        ti = _validate_ti(ti)
        self._scan_fraction = _validate_scan_fraction(scan_fraction)
        self._threshold = _validate_threshold(threshold)
        self._rotation = _validate_rotation(rotation, ti.ndim)
        self._scale = _validate_scale(scale)
        self._scan_path = _validate_scan_path(scan_path)
        self._update(
            ti=ti,
            zones=zones,
            post_processing=_validate_post_processing(post_processing),
        )
        self._warn_tiny_scan()

    def _check_consistency(self, **candidate):
        """Check ``candidate`` (any of ``ti``, ``zones``, ``post_processing``) against the rest.

        Returns the :class:`_Resolved` of the combination; raises, without
        changing the model, if the combination is invalid.
        """

        def pick(name):
            return (
                candidate[name]
                if name in candidate
                else getattr(self, f"_{name}")
            )

        ti = pick("ti")
        zones = _validate_zones(pick("zones"), ti)
        post = pick("post_processing")
        return _Resolved(zones, *_resolve_post(ti, zones, post))

    def _update(self, **candidate):
        """Assign ``candidate`` (and the resolved zones and post-pass TI) only if consistent."""
        resolved = self._check_consistency(**candidate)
        for name, value in candidate.items():
            setattr(self, f"_{name}", value)
        self._zones = resolved.zones
        self._resolved = resolved

    def _warn_tiny_scan(self):
        """Warn when a pass scans fewer than 2 TI candidates per node."""
        tis = [self._ti] + [z.ti for z in self._zones]
        n_min = min(int(np.prod(t.shape)) for t in tis)
        fractions = [self._scan_fraction]
        if self._post_processing is not None:
            fractions.append(
                self._scan_fraction / self._post_processing.factor
            )
        for frac in fractions:
            if int(frac * n_min) < 2:
                warnings.warn(
                    "MPSModel: fewer than 2 TI candidates per node: no pattern "
                    f"matching (effective scan fraction {frac:g}, smallest TI "
                    f"size {n_min}).",
                    UserWarning,
                    stacklevel=3,
                )
                return

    @property
    def ti(self):
        """:any:`TrainingImage`: The training image."""
        return self._ti

    @ti.setter
    def ti(self, value):
        value = _validate_ti(value)
        _validate_rotation(self._rotation, value.ndim)
        self._update(ti=value)
        self._warn_tiny_scan()

    @property
    def scan_fraction(self):
        """:class:`float`: Fraction of the TI scanned per node."""
        return self._scan_fraction

    @scan_fraction.setter
    def scan_fraction(self, value):
        self._scan_fraction = _validate_scan_fraction(value)
        self._warn_tiny_scan()

    @property
    def threshold(self):
        """:class:`float` or None: Distance threshold for early acceptance; ``None`` is DSBC mode."""
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        self._threshold = _validate_threshold(value)

    @property
    def post_processing(self):
        """:any:`PostProcessing` or None: Post-processing passes; ``None`` = off."""
        return self._post_processing

    @post_processing.setter
    def post_processing(self, value):
        self._update(post_processing=_validate_post_processing(value))
        self._warn_tiny_scan()

    @property
    def scan_path(self):
        """:class:`str`: TI window scan order — ``"sequential"`` or ``"random"``."""
        return self._scan_path

    @scan_path.setter
    def scan_path(self, value):
        self._scan_path = _validate_scan_path(value)

    @property
    def rotation(self):
        """Rotation spec as given (scalar, vector, array or callable); ``None`` → stationary identity.

        It is resolved against the simulation grid when :class:`DirectSampling`
        is called.
        """
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        self._rotation = _validate_rotation(value, self._ti.ndim)

    @property
    def scale(self):
        """Scale spec (scalar, vector, array, or callable); ``None`` → no dilation."""
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = _validate_scale(value)

    @property
    def zones(self):
        """:class:`list` of :any:`Zone`: zonation regions (empty → single-TI simulation)."""
        return list(self._zones)

    @zones.setter
    def zones(self, value):
        self._update(zones=value)
        self._warn_tiny_scan()

    def __repr__(self):
        args = [repr(self._ti)]
        defaults = dict(
            scan_fraction=1.0,
            threshold=None,
            scan_path="sequential",
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
        if self._post_processing is not None:
            args.append(f"post_processing={self._post_processing!r}")
        return f"MPSModel({', '.join(args)})"
