"""MPS configuration object — training image and search parameters."""

import warnings
from copy import deepcopy

import numpy as np

from gstools.mps._validation import exact_count
from gstools.mps.training_image import (
    _REPLACE_FIELDS,
    TrainingImage,
    Variable,
    _check_category_codes,
    _post_pass_n_neighbors,
    _resolve_post_ti,
)
from gstools.mps.zone import Zone
from gstools.tools.geometric import no_of_angles

__all__ = ["MPSModel", "PostProcessing"]

_VALID_BOUNDARY = ("strict", "partial")
_VALID_SCAN_PATH = ("sequential", "random")
_VALID_POST_PATH = ("same", "random", "sequential")


def _validate_post_processing(value):
    """Validate the post-processing pass count: a non-negative integer.

    Rejects a fractional value *before* truncating it (audit DS15): the earlier
    ``int(value)`` silently turned ``0.9`` into ``0`` and ``1.9`` into ``1``
    despite the integer-count API. ``bool`` is accepted as an explicit 0/1, and
    NumPy integer types are supported without conversion through float64.
    """
    return exact_count(value, minimum=0, context="MPSModel: post_processing")


def _validate_post_processing_factor(value):
    """Validate p_f > 0 and finite (it divides the main f and n_neighbors)."""
    v = float(value)
    if not (np.isfinite(v) and v > 0.0 and np.isfinite(1.0 / v)):
        raise ValueError(
            "MPSModel: post_processing_factor must be finite and > 0, "
            f"got {value!r}"
        )
    return v


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


def _validate_post_processing_variables(
    post_processing_variables, primary_ti, zones=()
):
    """Validate a list[Variable] of post-processing overrides against primary_ti."""
    if post_processing_variables is None:
        return []
    post_processing_variables = list(post_processing_variables)
    if not all(isinstance(v, Variable) for v in post_processing_variables):
        raise TypeError(
            "MPSModel: post_processing_variables must be a list of Variable "
            "objects."
        )
    names = [v.name for v in post_processing_variables]
    if len(set(names)) != len(names):
        raise ValueError(
            f"MPSModel: post_processing_variables names must be unique, "
            f"got {names!r}."
        )
    primary_names = {v.name for v in primary_ti.variables}
    unknown = set(names) - primary_names
    if unknown:
        raise ValueError(
            f"MPSModel: post_processing_variables name(s) "
            f"{sorted(map(str, unknown))!r} not found in the primary TI's "
            f"variables {sorted(map(str, primary_names))!r}."
        )
    # An override changes search settings only: post-processing pastes from the
    # primary (and zone) data, so its data and categorical kind must match.
    for ov in post_processing_variables:
        pv = primary_ti.variable(ov.name)
        label = f"post_processing_variables[{ov.name!r}]"
        same_data = ov.data is pv.data or np.array_equal(
            ov.data, pv.data, equal_nan=True
        )
        if ov.categorical != pv.categorical or not same_data:
            raise ValueError(
                f"MPSModel: {label} must keep the primary Variable's data "
                "and categorical kind; build it with "
                "ti.variable(name).replace(...)."
            )
        if ov.penalty_matrix is not None:
            for i, z in enumerate(zones):
                _check_category_codes(
                    z.ti.variable(ov.name).data,
                    ov.penalty_matrix.shape[0],
                    f"MPSModel: zones[{i}].ti data (checked against "
                    f"{label}.penalty_matrix)",
                )
    # Early check of the post-pass TI (weights set on all variables or none).
    _resolve_post_ti(primary_ti, post_processing_variables)
    return post_processing_variables


def _validate_post_pass_effort(ti, post_processing, p_f, overrides):
    """Early check that post-pass neighbour counts are valid (variation >= 2)."""
    if post_processing > 0:
        _post_pass_n_neighbors(_resolve_post_ti(ti, overrides), p_f)


def _validate_post_processing_mask(value):
    """Light validation only; grid-dependent shape is re-checked at call time."""
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.dtype != bool:
        raise TypeError(
            f"MPSModel: post_processing_mask must be a boolean array or "
            f"None, got dtype {arr.dtype!r}."
        )
    return arr


class PostProcessing:
    """Settings for Direct Sampling cleanup passes.

    Pass this object as the post_processing argument of MPSModel. The factor
    divides the scan fraction and neighbor counts. The path determines the
    revisit order, and a boolean mask selects the active nodes to revisit.
    The overrides dictionary maps variable names (None for an anonymous TI)
    to dictionaries of search settings. Omitted settings inherit from the
    primary variable.
    """

    def __init__(
        self, passes, *, factor=1.0, path=None, overrides=None, mask=None
    ):
        self.passes = exact_count(
            passes, minimum=1, context="PostProcessing.passes"
        )
        self.factor = _validate_post_processing_factor(factor)
        self.path = deepcopy(_validate_post_processing_path(path))
        self.mask = deepcopy(_validate_post_processing_mask(mask))
        if overrides is not None and not isinstance(overrides, dict):
            raise TypeError("PostProcessing.overrides must be a dictionary")
        self.overrides = deepcopy(overrides) if overrides is not None else {}
        for name, fields in self.overrides.items():
            if not isinstance(fields, dict):
                raise TypeError(
                    f"PostProcessing.overrides[{name!r}] must be a dictionary"
                )
            unknown = set(fields) - _REPLACE_FIELDS
            if unknown:
                raise ValueError(
                    f"PostProcessing.overrides[{name!r}]: unknown search fields "
                    f"{sorted(map(str, unknown))!r}"
                )

    def _variables(self, ti):
        variables = {v.name: v for v in ti.variables}
        unknown = set(self.overrides) - variables.keys()
        if unknown:
            raise ValueError(
                "PostProcessing.overrides: unknown variables "
                f"{sorted(map(str, unknown))!r}"
            )
        return [
            variables[name].replace(**fields)
            for name, fields in self.overrides.items()
        ]


def _validate_active_mask(value):
    """Require an explicit boolean array; shape is checked against each grid."""
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.dtype != bool:
        raise TypeError(
            "MPSModel: active_mask must be a boolean array or None"
        )
    return arr.copy()


def _validate_boundary(value):
    """Validate boundary; return normalized value or raise ValueError."""
    if value not in _VALID_BOUNDARY:
        raise ValueError(
            f"MPSModel: boundary must be one of {_VALID_BOUNDARY!r}, "
            f"got {value!r}"
        )
    return value


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
    """Validate a finite threshold, with None equivalent to zero."""
    if value is None:
        return 0.0
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(f"MPSModel: threshold must be finite, got {value!r}")
    if v < 0:
        raise ValueError(f"MPSModel: threshold must be >= 0, got {value!r}")
    if v > 1.0:
        warnings.warn(
            "MPSModel: threshold > 1.0 accepts the first selectable "
            "candidate for distances bounded by one. Lp distances can "
            "exceed one with conditioning outside the training-image range.",
            UserWarning,
            stacklevel=3,
        )
    return v


def _validate_rotation(spec, dim):
    """Validate a grid-independent rotation spec.

    Grid-dependent shape resolution happens at call time in
    :func:`gstools.mps.nonstationary.resolve_spec`; here we only reject what is
    knowable without the grid: rotation on a 1-D training image (no angles),
    and non-finite scalar/array values. Callables are stored unvalidated and
    checked when resolved.
    """
    if spec is None:
        return None
    if no_of_angles(dim) == 0:
        raise ValueError(
            f"MPSModel: rotation is undefined for a {dim}-D training image "
            "(1-D has no rotation angles)."
        )
    if callable(spec):
        return spec
    arr = np.asarray(spec, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("MPSModel: all rotation values must be finite.")
    return arr


def _validate_scale(spec, dim):
    """Validate a grid-independent scale spec (finite and strictly positive).

    Supported in every dimension (uniform scale dilates all axes). Callables
    are stored unvalidated and checked when resolved.
    """
    if spec is None:
        return None
    if callable(spec):
        return spec
    arr = np.asarray(spec, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("MPSModel: all scale values must be finite.")
    if not np.all(arr > 0):
        raise ValueError(
            "MPSModel: all scale values must be > 0 "
            f"(minimum found: {float(np.min(arr))!r})."
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
        window). For a nonempty window, the budget is
        max(1, min(window size, floor(fraction * selected TI size))). Zones use
        their own TI size. Must be in (0, 1]. Default: 1.0.
    threshold : float or None, optional
        Default 0.0 (or None): select the lowest-distance selectable candidate
        within the scan budget; an exact zero match may stop early. For finite
        positive t, accept the first distance strictly below t, otherwise the
        best visited candidate. Ties retain the first minimum. This is not an
        exact-only rule or necessarily a full-TI scan. Hard data are preserved
        independently of threshold.
    boundary : :class:`str`, optional
        Search-window strategy: ``"strict"`` (default) or ``"partial"``.
    scan_path : :class:`str`, optional
        TI scan order at each node: ``"sequential"`` (default) — a wrap-around
        walk from a random start — or ``"random"`` — ``scan_fraction``'s worth
        of distinct window cells in a node-local random order. Both are
        reproducible and thread-count independent. Default: ``"sequential"``.
    zones : :class:`list` of :any:`Zone` or None, optional
        Region-bound training images (Mariethoz2010 para [40]): nodes inside a
        zone's region scan that zone's TI instead of the primary one, while the
        data event still spans all simulation-grid neighbours. Every zone TI
        must share the primary TI's variable set and kinds. ``None`` → no zones.
        Default: ``None``.
    post_processing : :any:`PostProcessing` or int, optional
        Configuration or number of cleanup passes (Meerschman et al. 2013, Sect. 4): each
        re-simulates already-simulated nodes against a fully informed
        neighbourhood *after* the main path, to reduce noise. This is the
        repeated-sweep cleanup of Meerschman 2013, not Mariethoz 2010 §7
        recursive syn-processing (which repairs neighbourhoods during the
        simulation). ``0`` → no cleanup. Default: ``0``.
    post_processing_factor : :class:`float`, optional
        Divides ``scan_fraction`` and ``n_neighbors`` for the cleanup passes
        (``> 1`` cheaper, ``< 1`` more thorough). Must be finite and > 0.
        Counts round to nearest (ties to even), floored at one; requested
        variation counts must remain at least two. Actual common support may
        contain one point. Candidate budgets remain capped by the valid window.
        Default: ``1.0``.
    post_processing_path : :class:`str`, array-like, or None, optional
        Cleanup visit order: ``None`` → inherit the main-path mode; ``"same"``
        → reuse the main path (filtered to eligible nodes); ``"random"`` /
        ``"sequential"``; or an explicit ``(N, dim)`` array. Default: ``None``.
    post_processing_variables : :class:`list` of :any:`Variable` or None, optional
        Per-variable search-setting overrides for the cleanup passes, built with
        ``ti.variable(name).replace(...)`` (data/kind must match the primary).
        ``None`` → inherit the primary settings. Default: ``None``.
    post_processing_mask : :class:`numpy.ndarray` of bool or None, optional
        Which simulation-grid nodes are eligible for cleanup update (shape
        checked at call time). ``None`` → all simulated nodes. Default: ``None``.
    active_mask : numpy.ndarray of bool or None, optional
        True cells are eligible for simulation and may contribute to neighbor
        events. False cells remain NaN and are excluded as targets and
        neighbors; hard conditions on them are rejected. Shape must equal the
        simulation grid. ``None`` keeps the full rectangular workload.
    rotation : scalar, array-like, or callable, optional
        Grid-independent non-stationary rotation spec (radians, GSTools
        Tait-Bryan ordering;
        ``no_of_angles(dim)`` components). Resolved per node at call time by
        :func:`gstools.mps.nonstationary.resolve_spec`. Undefined in 1-D.
        ``None`` → no rotation. Default: ``None``.
    scale : scalar, array-like, or callable, optional
        Grid-independent non-stationary scale spec (``dim`` components; a
        scalar dilates all axes). Positive dimensionless SG-to-TI lag factors;
        offsets/radii use grid-index units, not physical coordinate spacing.
        Resolved per node
        at call time. ``None`` → no scaling. Default: ``None``.
    """

    def __init__(
        self,
        ti,
        scan_fraction=1.0,
        threshold=0.0,
        boundary="strict",
        scan_path="sequential",
        rotation=None,
        scale=None,
        zones=None,
        post_processing=0,
        post_processing_factor=1.0,
        post_processing_path=None,
        post_processing_variables=None,
        post_processing_mask=None,
        active_mask=None,
    ):
        if not isinstance(ti, TrainingImage):
            raise TypeError(
                f"MPSModel: ti must be a TrainingImage, got {type(ti)!r}"
            )
        self._ti = ti
        self._active_mask = _validate_active_mask(active_mask)
        self._scan_fraction = _validate_scan_fraction(scan_fraction)
        self._threshold = _validate_threshold(threshold)
        self._boundary = _validate_boundary(boundary)
        self._scan_path = _validate_scan_path(scan_path)
        dim = len(ti.shape)
        self._rotation = _validate_rotation(rotation, dim)
        self._scale = _validate_scale(scale, dim)
        self._zones = _validate_zones(zones, ti)
        if isinstance(post_processing, PostProcessing):
            if (
                post_processing_factor != 1.0
                or post_processing_path is not None
                or post_processing_variables is not None
                or post_processing_mask is not None
            ):
                raise ValueError(
                    "Use the PostProcessing object or individual cleanup settings."
                )
            self.post_processing = post_processing
            return
        self._post_processing = _validate_post_processing(post_processing)
        self._post_processing_factor = _validate_post_processing_factor(
            post_processing_factor
        )
        self._post_processing_path = _validate_post_processing_path(
            post_processing_path
        )
        self._post_processing_variables = _validate_post_processing_variables(
            post_processing_variables, ti, self._zones
        )
        self._post_processing_mask = _validate_post_processing_mask(
            post_processing_mask
        )
        # Early failure for an impossible post-pass effort (variation < 2).
        _validate_post_pass_effort(
            ti,
            self._post_processing,
            self._post_processing_factor,
            self._post_processing_variables,
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
        """float: Zero/None selects the best visited candidate; positive values
        accept the first distance strictly below the threshold. Scan budget
        limits the examined candidates; exact-zero matches may stop early.
        """
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        self._threshold = _validate_threshold(value)

    @property
    def boundary(self):
        """:class:`str`: Search-window boundary strategy (``"strict"`` or ``"partial"``)."""
        return self._boundary

    @boundary.setter
    def boundary(self, value):
        self._boundary = _validate_boundary(value)

    @property
    def active_mask(self):
        """Boolean array or None: True cells are eligible for simulation.

        Active cells may contribute to events. Inactive output remains NaN;
        conditions outside the mask are rejected. The mask applies to the main
        simulation and every cleanup pass.
        """
        return self._active_mask

    @active_mask.setter
    def active_mask(self, value):
        self._active_mask = _validate_active_mask(value)

    @property
    def rotation(self):
        """scalar, array-like, callable, or None: non-stationary rotation spec."""
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        self._rotation = _validate_rotation(value, len(self._ti.shape))

    @property
    def scale(self):
        """scalar, array-like, callable, or None: non-stationary scale spec."""
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = _validate_scale(value, len(self._ti.shape))

    @property
    def scan_path(self):
        """:class:`str`: TI scan order per node, ``"sequential"`` or ``"random"``."""
        return self._scan_path

    @scan_path.setter
    def scan_path(self, value):
        self._scan_path = _validate_scan_path(value)

    @property
    def zones(self):
        """:class:`list` of :any:`Zone`: Zone TIs scanned inside their regions (``[]`` if none)."""
        return list(self._zones)

    @zones.setter
    def zones(self, value):
        new_zones = _validate_zones(value, self._ti)
        # DS14: a later assignment must honour the same cleanup-override
        # penalty-code invariant the constructor enforces, so re-screen the new
        # zones against the resolved post-processing overrides. A rejected
        # assignment leaves the previous configuration untouched.
        _validate_post_processing_variables(
            self._post_processing_variables, self._ti, new_zones
        )
        self._zones = new_zones

    @property
    def post_processing(self):
        """:class:`int`: Number of cleanup (post-processing) passes (``0`` → none)."""
        return self._post_processing

    @post_processing.setter
    def post_processing(self, value):
        if isinstance(value, PostProcessing):
            passes = exact_count(
                value.passes, minimum=1, context="PostProcessing.passes"
            )
            factor = _validate_post_processing_factor(value.factor)
            path = deepcopy(_validate_post_processing_path(value.path))
            mask = deepcopy(_validate_post_processing_mask(value.mask))
            variables = _validate_post_processing_variables(
                value._variables(self._ti), self._ti, self._zones
            )
            _validate_post_pass_effort(self._ti, passes, factor, variables)
            self._post_processing = passes
            self._post_processing_factor = factor
            self._post_processing_path = path
            self._post_processing_variables = variables
            self._post_processing_mask = mask
            return
        new_value = _validate_post_processing(value)
        _validate_post_pass_effort(
            self._ti,
            new_value,
            self._post_processing_factor,
            self._post_processing_variables,
        )
        self._post_processing = new_value

    @property
    def post_processing_factor(self):
        """:class:`float`: Divisor for cleanup ``scan_fraction``/``n_neighbors`` effort."""
        return self._post_processing_factor

    @post_processing_factor.setter
    def post_processing_factor(self, value):
        new_value = _validate_post_processing_factor(value)
        _validate_post_pass_effort(
            self._ti,
            self._post_processing,
            new_value,
            self._post_processing_variables,
        )
        self._post_processing_factor = new_value

    @property
    def post_processing_path(self):
        """:class:`str`, array, or None: Cleanup visit order (``None`` → inherit main path)."""
        return self._post_processing_path

    @post_processing_path.setter
    def post_processing_path(self, value):
        self._post_processing_path = _validate_post_processing_path(value)

    @property
    def post_processing_variables(self):
        """:class:`list` of :any:`Variable`: Per-variable cleanup overrides (``[]`` if none)."""
        return list(self._post_processing_variables)

    @post_processing_variables.setter
    def post_processing_variables(self, value):
        new_variables = _validate_post_processing_variables(
            value, self._ti, self._zones
        )
        _validate_post_pass_effort(
            self._ti,
            self._post_processing,
            self._post_processing_factor,
            new_variables,
        )
        self._post_processing_variables = new_variables

    @property
    def post_processing_mask(self):
        """:class:`numpy.ndarray` of bool or None: Which nodes are eligible for cleanup."""
        return self._post_processing_mask

    @post_processing_mask.setter
    def post_processing_mask(self, value):
        self._post_processing_mask = _validate_post_processing_mask(value)

    def __repr__(self):
        args = [repr(self._ti)]
        defaults = dict(
            scan_fraction=1.0,
            threshold=0.0,
            boundary="strict",
            scan_path="sequential",
            post_processing=0,
            post_processing_factor=1.0,
        )
        for name, default in defaults.items():
            val = getattr(self, f"_{name}")
            if val != default:
                args.append(f"{name}={val!r}")
        # Cleanup path/overrides/mask and the nonstationary maps can make two
        # materially different models look identical (audit DS16); show concise
        # non-default markers without dumping full arrays.
        if self._post_processing_path is not None:
            args.append(f"post_processing_path={self._post_processing_path!r}")
        if self._post_processing_variables:
            args.append(
                f"post_processing_variables=<{len(self._post_processing_variables)}>"
            )
        if self._active_mask is not None:
            args.append("active_mask=<set>")
        if self._post_processing_mask is not None:
            args.append("post_processing_mask=<set>")
        if self._zones:
            args.append(f"zones=<{len(self._zones)}>")
        if self._rotation is not None:
            args.append("rotation=<set>")
        if self._scale is not None:
            args.append("scale=<set>")
        return f"MPSModel({', '.join(args)})"
