"""
GStools subpackage providing the TrainingImage class for MPS simulations.

.. currentmodule:: gstools.mps

The following classes and functions are provided

.. autosummary::
   Variable
   TrainingImage
"""

import functools
import warnings

import numpy as np

# MPS block-distance kernels come from gstools.mps._core, which checks the
# installed gstools_core package and its supported MPS API revision.
from gstools.mps._core import (
    mps_dist_block_cat,
    mps_dist_block_cat_masked,
    mps_dist_block_l1,
    mps_dist_block_l1_masked,
    mps_dist_block_l2,
    mps_dist_block_l2_masked,
    mps_dist_block_lp,
    mps_dist_block_lp_masked,
    mps_dist_block_variation,
    mps_dist_block_variation_masked,
)
from gstools.mps._validation import _validate_real_data, exact_count
from gstools.mps.distance import compute_node_weights

__all__ = ["Variable", "TrainingImage"]


def _any_float_nan(*arrays):
    """Return True if any floating-dtype array in *arrays* contains NaN."""
    return any(
        np.issubdtype(arr.dtype, np.floating) and np.isnan(arr).any()
        for arr in arrays
    )


def _data_range(arr):
    """Continuous data range over *defined* cells (NaN excluded).

    Returns the ``max - min`` spread of the finite values, or ``1.0`` when the
    array is constant or entirely undefined (avoids a zero/NaN normalizer).

    Integer extrema are subtracted as Python integers before float conversion.
    Floating extrema are each widened before subtraction, so their dtype cannot
    overflow the subtraction (audit DS02): e.g. ``int8`` ``[-128, 127]`` yields
    ``255`` rather than the ``1`` that ``int8(127) - int8(-128)`` wraps to, and
    ``float32`` extrema do not saturate to ``inf``. A genuinely non-constant
    range that still exceeds float64 is a hard error rather than a silent
    ``inf`` normalizer that would collapse every distance to zero.
    """
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 1.0
    if np.issubdtype(arr.dtype, np.integer):
        dmax = float(int(np.max(finite)) - int(np.min(finite)))
    else:
        dmax = float(np.max(finite)) - float(np.min(finite))
    if dmax == 0.0:
        return 1.0
    if not np.isfinite(dmax):
        raise ValueError(
            "TrainingImage: continuous data range is not finite (the spread of "
            "the data exceeds float64); rescale the data before simulation."
        )
    return dmax


def _parse_distance(distance):
    """Parse a continuous-distance string into (p_norm, variation_p_norm).

    Exactly one of the two return values is non-``None``:
    ``"l<p>"`` -> ``(p, None)``; ``"variation"`` -> ``(None, 2.0)``;
    ``"variation<p>"`` -> ``(None, p)``.

    Returns
    -------
    tuple of (float or None, float or None)
    """
    distance_lower = str(distance).lower()
    if distance_lower.startswith("l"):
        try:
            p_val = float(distance_lower[1:])
        except ValueError:
            raise ValueError(
                f"TrainingImage: distance starting with 'l' must be followed by "
                f"a positive number (e.g. 'l1', 'l2', 'l3.5'). Got {distance!r}"
            )
        if not 0 < p_val < np.inf:  # also rejects 'lnan' / 'linf'
            raise ValueError(
                f"TrainingImage: Lp norm exponent must be finite and > 0, "
                f"got {p_val}."
            )
        return p_val, None
    if distance_lower == "variation":
        return None, 2.0
    if distance_lower.startswith("variation"):
        try:
            p_val = float(distance_lower[len("variation") :])
        except ValueError:
            raise ValueError(
                f"TrainingImage: distance starting with 'variation' must be "
                f"followed by a positive number (e.g. 'variation1', 'variation1.5'). "
                f"Got {distance!r}"
            )
        if not 0 < p_val < np.inf:
            raise ValueError(
                f"TrainingImage: variation exponent must be finite and > 0, "
                f"got {p_val}."
            )
        return None, p_val
    raise ValueError(
        f"TrainingImage: distance must be 'l<p>' (e.g. 'l1', 'l2'), "
        f"'variation', or 'variation<p>' (e.g. 'variation1'). "
        f"Got {distance!r}"
    )


def _validate_n_neighbors(value, variation_p_norm):
    """Validate n_neighbors >= 1 (and >= 2 under distance='variation')."""
    v = exact_count(value, minimum=1, context="Variable: n_neighbors")
    if variation_p_norm is not None and v < 2:
        raise ValueError(
            "Variable: distance='variation' requires n_neighbors >= 2 "
            "(a single-point data event has no local mean deviation)."
        )
    return v


def _check_category_codes(values, n_cat, context):
    """Validate that every finite value is an integer category code < ``n_cat``.

    A ``penalty_matrix`` is indexed directly by the category codes it meets, so
    any non-integer or out-of-range code turns into a raw ``IndexError`` deep in
    the scan loop (or, for a negative code, a silently wrapped lookup). Every
    array that can reach that indexing — a :class:`Variable`'s own data, the
    conditioning values, a zone TI's data — must be screened here first.
    ``context`` prefixes the error message with what was checked.
    """
    finite_vals = np.asarray(values, dtype=np.double)
    if np.isinf(finite_vals).any():
        raise ValueError(f"{context}: infinite category values are invalid.")
    finite_vals = finite_vals[np.isfinite(finite_vals)]
    if not finite_vals.size:
        return
    non_integer = finite_vals != np.floor(finite_vals)
    out_of_range = (finite_vals < 0) | (finite_vals >= n_cat)
    invalid = non_integer | out_of_range
    if invalid.any():
        bad_val = finite_vals[invalid].flat[0]
        raise ValueError(
            f"{context}: with a penalty_matrix set, values must be "
            f"non-negative integer-valued category codes < {n_cat}; found "
            f"invalid value {bad_val!r}."
        )


_SENTINEL = object()

_REPLACE_FIELDS = frozenset(
    {
        "distance",
        "weight",
        "n_neighbors",
        "max_radius",
        "penalty_matrix",
        "distance_power",
        "cond_weight",
    }
)


def _record_explicit(init):
    """Remember which search-config fields the caller passed explicitly.

    Used by post-processing overrides: ``ti.variable(name).replace(distance=…)``
    must apply only the fields the caller actually set, inheriting the rest from
    the primary variable (see :func:`_resolve_post_ti`).
    """

    @functools.wraps(init)
    def wrapper(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self._explicit = frozenset(kwargs) & _REPLACE_FIELDS

    return wrapper


def _view_variable(var, slices):
    """Variable sharing ``var``'s configuration with ``data = var.data[slices]``.

    The data is a NumPy **view** (no copy; the parent buffer is module-owned
    and frozen, so ``Variable.__init__`` is told to skip copying it).
    ``d_max``/``has_nan`` are recomputed from the sliced data as a side
    effect of delegating to ``Variable``'s own constructor.
    """
    view = var.data[slices]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return Variable(
            var._name,
            view,
            categorical=var._categorical,
            distance=var._distance,
            weight=var._weight,
            n_neighbors=var._n_neighbors,
            max_radius=var._max_radius,
            penalty_matrix=var._penalty_matrix,
            distance_power=var._distance_power,
            cond_weight=var._cond_weight,
            _trusted=True,
        )


class Variable:
    """One channel's data and per-variable distance configuration for MPS.

    Parameters
    ----------
    name : str or None
        Valid Python identifier (unique within a TrainingImage). ``None`` is
        reserved for the anonymous variable created by univariate
        ``TrainingImage`` sugar — not for direct user construction.
    data : numpy.ndarray
        n-D array. Copied defensively at construction.
    categorical : bool, optional
        Whether the variable is categorical. Default: ``True``.
    distance : str, optional
        Continuous-distance metric: ``"l1"`` (default), ``"l2"``, ``"lp"``
        (p > 0), ``"variation"``, ``"variation<p>"``. Ignored when
        ``categorical=True``.
    weight : float or None, optional
        Relative weight for the multivariate joint distance. ``None`` means
        equal share (resolved at ``TrainingImage`` level). Default: ``None``.
    n_neighbors : int, optional
        Maximum neighbours in the data event. Default: 32.
    max_radius : float or None, optional
        Exclude SG neighbours beyond this Euclidean distance from the data
        event. ``None`` → no limit (default).
    """

    @_record_explicit
    def __init__(
        self,
        name,
        data,
        *,
        categorical=True,
        distance="l1",
        weight=None,
        n_neighbors=32,
        max_radius=None,
        penalty_matrix=None,
        distance_power=0.0,
        cond_weight=1.0,
        _trusted=False,
    ):
        if name is not None and (
            not isinstance(name, str) or not name.isidentifier()
        ):
            raise ValueError(
                f"Variable: name must be a valid Python identifier, got {name!r}."
            )
        self._name = name
        # Defensive copy of external data. Only module-internal callers
        # (``replace()``, ``_view_variable()``) pass ``_trusted=True``: they
        # hand in this module's own already-frozen buffer. A read-only *view*
        # of a writable base is not safe to share, so the flag is never
        # inferred from ``flags.writeable``. has_nan/d_max are still recomputed.
        data_arr = np.asarray(data)
        _validate_real_data(data_arr, bool(categorical), f"Variable {name!r}")
        self._data = data_arr if _trusted else np.array(data_arr, copy=True)
        if (
            np.issubdtype(self._data.dtype, np.floating)
            and np.isinf(self._data).any()
        ):
            raise ValueError(
                f"Variable {name!r}: data contains +-inf; use NaN to mark "
                "undefined cells."
            )
        self._categorical = bool(categorical)
        self._distance = str(distance)
        self._weight = None if weight is None else float(weight)
        if self._weight is not None and not 0 <= self._weight < np.inf:
            raise ValueError(
                f"Variable: weight must be finite and >= 0, got {weight!r}."
            )
        self._distance_power = float(distance_power)
        if not 0 <= self._distance_power < np.inf:
            raise ValueError(
                "Variable: distance_power must be finite and >= 0, got "
                f"{distance_power!r}."
            )
        self._cond_weight = float(cond_weight)
        if not 0 <= self._cond_weight < np.inf:
            raise ValueError(
                f"Variable: cond_weight must be finite and >= 0, got "
                f"{cond_weight!r}."
            )
        if max_radius is not None and not (
            np.isfinite(float(max_radius)) and float(max_radius) > 0
        ):
            raise ValueError(
                f"Variable: max_radius must be a finite positive float or None, got {max_radius!r}."
            )
        self._max_radius = None if max_radius is None else float(max_radius)
        self._has_nan = _any_float_nan(self._data)
        if self._has_nan:
            warnings.warn(
                f"Variable {name!r}: contains NaN cell(s); treated as undefined "
                "(excluded from continuous data range and pattern distances, "
                "never pasted into the simulation).",
                UserWarning,
                stacklevel=2,
            )
        if not self._categorical:
            self._p_norm, self._variation_p_norm = _parse_distance(
                self._distance
            )
            self._d_max = _data_range(self._data)
        else:
            self._p_norm = None
            self._variation_p_norm = None
            self._d_max = None
        self._penalty_matrix = self._validate_penalty_matrix(penalty_matrix)
        self._n_neighbors = _validate_n_neighbors(
            n_neighbors, self._variation_p_norm
        )
        # Mark the data array read-only: the engine reads it directly and
        # nothing downstream should ever write through .data.
        self._data.flags.writeable = False

    def _validate_penalty_matrix(self, penalty_matrix):
        """Validate and freeze a per-category penalty matrix, or return None.

        Constraints: 2-D square, zero diagonal, off-diagonal entries in
        ``(0, 1]`` (NaN rejected); asymmetric matrices are allowed. The
        variable's own data must contain non-negative integer-valued codes
        ``< C`` at every finite cell.
        """
        if penalty_matrix is None:
            return None
        if not self._categorical:
            raise ValueError(
                f"Variable {self._name!r}: penalty_matrix is only valid for "
                "categorical variables (categorical=False was given)."
            )
        _validate_real_data(penalty_matrix, False, "penalty_matrix")
        mat = np.array(penalty_matrix, copy=True, dtype=np.float64)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError(
                f"Variable {self._name!r}: penalty_matrix must be a 2D square "
                f"array, got shape {mat.shape!r}."
            )
        n_cat = mat.shape[0]
        if n_cat == 0:
            raise ValueError(
                "Variable: penalty_matrix must have at least one category."
            )
        diag = np.diagonal(mat)
        bad_diag = np.flatnonzero(diag != 0.0)
        if bad_diag.size:
            raise ValueError(
                f"Variable {self._name!r}: penalty_matrix diagonal must be all-"
                f"zero; violated at index/indices {bad_diag.tolist()} "
                f"(values {diag[bad_diag].tolist()})."
            )
        off_diag_mask = ~np.eye(n_cat, dtype=bool)
        off_vals = mat[off_diag_mask]
        bad_off = ~((off_vals > 0.0) & (off_vals <= 1.0))  # NaN is bad too
        if bad_off.any():
            off_idx = np.argwhere(off_diag_mask)
            bad_pairs = off_idx[bad_off]
            bad_vals = off_vals[bad_off]
            raise ValueError(
                f"Variable {self._name!r}: penalty_matrix off-diagonal entries "
                "must be in (0, 1]; violated at "
                f"{[tuple(p) for p in bad_pairs.tolist()]} "
                f"(values {bad_vals.tolist()})."
            )
        _check_category_codes(
            self._data, n_cat, f"Variable {self._name!r} data"
        )
        mat.flags.writeable = False
        return mat

    # --- read-only properties ---

    @property
    def name(self):
        """:class:`str` or None: Variable name."""
        return self._name

    @property
    def data(self):
        """:class:`numpy.ndarray`: Training image data array (shared internal reference).

        The returned array is the live internal buffer and must **not** be
        mutated by the caller.  The array is marked read-only (``flags.writeable
        == False``) at construction time to enforce this at the NumPy level.
        """
        return self._data

    @property
    def categorical(self):
        """:class:`bool`: Whether the variable is categorical."""
        return self._categorical

    @property
    def distance(self):
        """:class:`str`: Distance metric string (e.g. ``"l1"``, ``"variation"``)."""
        return self._distance

    @property
    def weight(self):
        """:class:`float` or None: Raw (un-normalized) weight, or None for equal share."""
        return self._weight

    @property
    def p_norm(self):
        """:class:`float` or None: Lp exponent for continuous variables; None for categorical/variation."""
        return self._p_norm

    @property
    def variation_p_norm(self):
        """:class:`float` or None: Variation-distance exponent; None unless distance is 'variation'."""
        return self._variation_p_norm

    @property
    def d_max(self):
        """:class:`float` or None: Data range (max−min over finite cells) for normalization."""
        return self._d_max

    @property
    def has_nan(self):
        """:class:`bool`: True if the data array contains any NaN."""
        return self._has_nan

    @property
    def max_radius(self):
        """:class:`float` or None: Maximum neighbour radius, or ``None`` for no limit."""
        return self._max_radius

    @property
    def distance_power(self):
        """:class:`float`: Spatial-decay exponent δ for node weighting (Mariethoz2010 Eq. 3)."""
        return self._distance_power

    @property
    def cond_weight(self):
        """:class:`float`: Weight multiplier for conditioning nodes of this variable."""
        return self._cond_weight

    @property
    def penalty_matrix(self):
        """:class:`numpy.ndarray` or None: Frozen per-category penalty matrix (read-only)."""
        return self._penalty_matrix

    # --- mutable property ---

    @property
    def n_neighbors(self):
        """:class:`int`: Maximum neighbours in the data event."""
        return self._n_neighbors

    @n_neighbors.setter
    def n_neighbors(self, value):
        self._n_neighbors = _validate_n_neighbors(
            value, self._variation_p_norm
        )

    def replace(self, **overrides):
        """New Variable sharing this one's data buffer; only search config differs.

        No copy: the buffer is this module's own frozen array, so sharing it is
        safe (``_trusted`` skips the copy). ``data`` and ``categorical`` cannot
        be overridden -- they define what the variable *is*; construct a new
        Variable instead.
        """
        if "data" in overrides or "categorical" in overrides:
            raise ValueError(
                "Variable.replace(): 'data' and 'categorical' cannot be "
                "overridden here -- they define what the variable is; "
                "construct a new Variable instead."
            )
        unknown = set(overrides) - _REPLACE_FIELDS
        if unknown:
            raise ValueError(
                f"Variable.replace(): unknown field(s) {sorted(unknown)!r}; "
                f"valid fields are {sorted(_REPLACE_FIELDS)!r}."
            )
        kwargs = dict(
            categorical=self._categorical,
            distance=self._distance,
            weight=self._weight,
            n_neighbors=self._n_neighbors,
            max_radius=self._max_radius,
            penalty_matrix=self._penalty_matrix,
            distance_power=self._distance_power,
            cond_weight=self._cond_weight,
        )
        kwargs.update(overrides)
        # data/categorical are unchanged, so has_nan is already known --
        # suppress the NaN warning re-firing on every replace().
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            variable = Variable(
                self._name, self._data, _trusted=True, **kwargs
            )
        variable._explicit = frozenset(overrides)
        return variable

    def __repr__(self):
        parts = [
            f"Variable({self._name!r}, shape={self._data.shape}",
            f"categorical={self._categorical}",
            f"distance={self._distance!r}",
            f"n_neighbors={self._n_neighbors}",
        ]
        if self._max_radius is not None:
            parts.append(f"max_radius={self._max_radius!r}")
        if self._penalty_matrix is not None:
            parts.append(f"penalty_matrix={self._penalty_matrix.shape!r}")
        return ", ".join(parts) + ")"


class TrainingImage:
    """Training image for multiple point statistics simulation.

    The MPS analogue of :class:`gstools.CovModel`: encapsulates training
    data and the distance function for comparing data events.

    Parameters
    ----------
    data : :class:`numpy.ndarray` or list of :class:`Variable`
        Training image data. Pass a numpy array for a univariate TI; pass a
        non-empty list of :class:`Variable` objects for a multivariate TI.
    categorical : :class:`bool`, optional
        Whether the variable is categorical (univariate TIs only). Default: ``True``.
    distance : :class:`str`, optional
        Distance metric for continuous variables (univariate TIs only):
        ``"l1"`` (default), ``"l2"``, ``"lp"`` (p > 0), ``"variation"``,
        ``"variation<p>"``. Ignored when ``categorical=True``.
    n_neighbors : :class:`int`, optional
        Maximum neighbours in the data event (univariate TIs only). Default: 32.
    max_radius : :class:`float` or None, optional
        Exclude SG neighbours beyond this distance (univariate TIs only). Default: None.
    distance_power : :class:`float`, optional
        Exponent δ for spatial-decay weighting of neighbours
        (Mariethoz2010 Eq. 3). Applied to **all** distance types.
        ``0.0`` → uniform weights (oracle-compatible default).
        ``1.0`` → closer neighbours weighted more heavily. Bare-array
        (univariate) sugar only — set it on each :class:`Variable` otherwise.
    cond_weight : :class:`float`, optional
        Weight multiplier for conditioning nodes of this variable. Bare-array
        (univariate) sugar only — set it on each :class:`Variable` otherwise.
    """

    def __init__(
        self,
        data,
        *,
        categorical=True,
        distance="l1",
        n_neighbors=32,
        max_radius=None,
        distance_power=_SENTINEL,
        cond_weight=_SENTINEL,
    ):
        # distance_power/cond_weight are per-Variable settings. TrainingImage
        # only forwards them for the bare-array (anonymous-Variable) sugar; for
        # a Variable (or list) they must be configured on the Variable itself.
        if isinstance(data, (list, Variable)) and (
            distance_power is not _SENTINEL or cond_weight is not _SENTINEL
        ):
            raise ValueError(
                "TrainingImage: distance_power/cond_weight must be set on each "
                "Variable, not on TrainingImage, when passing a Variable or a "
                "list of Variables."
            )

        if isinstance(data, list):
            self._init_from_variables(data)
        elif isinstance(data, Variable):
            # Univariate sugar: caller supplies a pre-configured Variable
            self._multivariate = False
            self._variables = [data]
            self._var_map = {data.name: data}
            self._shape = data.data.shape
            self._has_nan = data.has_nan
            self._normalized_weights = None
        else:
            # Univariate sugar: bare array → anonymous Variable (name=None)
            self._multivariate = False
            var = Variable(
                None,
                data,
                categorical=categorical,
                distance=distance,
                weight=1.0,
                n_neighbors=n_neighbors,
                max_radius=max_radius,
                distance_power=(
                    0.0 if distance_power is _SENTINEL else distance_power
                ),
                cond_weight=(1.0 if cond_weight is _SENTINEL else cond_weight),
            )
            self._variables = [var]
            self._var_map = {None: var}
            self._shape = var.data.shape
            self._has_nan = var.has_nan
            self._normalized_weights = None

    def _init_from_variables(self, var_list):
        if not var_list or not all(isinstance(v, Variable) for v in var_list):
            raise TypeError(
                "TrainingImage: data must be a numpy.ndarray (univariate) or a "
                "non-empty list of Variable objects (multivariate)."
            )
        names = [v.name for v in var_list]
        if any(n is None for n in names):
            raise ValueError(
                "TrainingImage: Variable names must not be None in a Variable list."
            )
        if len(set(names)) != len(names):
            raise ValueError(
                f"TrainingImage: Variable names must be unique, got {names!r}."
            )
        shapes = {v.data.shape for v in var_list}
        if len(shapes) != 1:
            raise ValueError(
                "TrainingImage: All variables must have the same shape."
            )
        self._multivariate = True
        self._variables = list(var_list)
        self._var_map = {v.name: v for v in self._variables}
        self._shape = shapes.pop()
        self._has_nan = any(v.has_nan for v in self._variables)

        # Weight normalization: all-None → uniform 1/m; all-explicit → normalize;
        # mixed → error.
        raw = [v.weight for v in var_list]
        n = len(var_list)
        if all(w is None for w in raw):
            self._normalized_weights = {v.name: 1.0 / n for v in var_list}
        elif any(w is None for w in raw):
            raise ValueError(
                "TrainingImage: mixing None and explicit weights is not allowed. "
                "Either specify weight on every Variable or on none."
            )
        else:
            # Normalize after scaling by the largest positive weight (audit
            # DS15): summing raw weights first overflows to inf for large finite
            # inputs such as [1e308, 1e308] and then normalizes to [0, 0],
            # violating the sum-to-one joint-distance contract. Dividing by the
            # max keeps every term in [0, 1]; the result is invariant to a
            # common positive rescaling of the weights.
            scale = max(raw)
            if scale <= 0:
                raise ValueError(
                    "TrainingImage: sum of weights must be positive."
                )
            scaled = [w / scale for w in raw]
            total = sum(scaled)
            self._normalized_weights = {
                v.name: s / total for v, s in zip(var_list, scaled)
            }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def has_nan(self):
        """:class:`bool`: True if any Variable contains NaN."""
        return self._has_nan

    @property
    def _sole_variable(self):
        """The single Variable of a univariate TI, by whatever key it has.

        A univariate TI built from a named ``Variable`` is keyed by that name,
        not ``None``, so the univariate conveniences must look the variable up
        by identity rather than assuming the ``None`` key (audit DS12).
        """
        return next(iter(self._var_map.values()))

    @property
    def data(self):
        """:class:`numpy.ndarray` or None: Raw data for univariate TIs; None for MV."""
        if self._multivariate:
            return None
        return self._sole_variable.data

    @property
    def ndim(self):
        """:class:`int`: Number of spatial dimensions."""
        return len(self._shape)

    @property
    def shape(self):
        """:class:`tuple`: Shape of the training image."""
        return self._shape

    @property
    def multivariate(self):
        """:class:`bool`: True for Variable-list TIs; False for bare-array TIs."""
        return self._multivariate

    @property
    def variables(self):
        """:class:`list` of :class:`Variable`: All variables (univariate has one anonymous entry)."""
        return list(self._variables)

    @property
    def weights(self):
        """:class:`dict` or None: Normalized {name: float} weights for MV TIs; None for UV."""
        if not self._multivariate:
            return None
        return dict(self._normalized_weights)

    def variable(self, name=_SENTINEL):
        """Return the live Variable by name, or the sole variable if called with no argument.

        Parameters
        ----------
        name : str, optional
            Variable name. Omit to return the single variable of a one-variable TI.

        Returns
        -------
        Variable
        """
        if name is _SENTINEL:
            if len(self._variables) != 1:
                raise TypeError(
                    "variable() with no argument requires exactly one variable; "
                    f"this TI has {len(self._variables)}."
                )
            return self._variables[0]
        if name not in self._var_map:
            raise KeyError(f"TrainingImage: no variable named {name!r}.")
        return self._var_map[name]

    def _validate_window_slices(self, slices):
        """Normalize/validate window slices: unit step, in-bounds, non-degenerate."""
        if isinstance(slices, slice):
            slices = (slices,)
        try:
            slices = tuple(slices)
        except TypeError:
            raise ValueError(
                f"TrainingImage.window: expected a tuple of {self.ndim} "
                f"slice objects (one per dimension), got {slices!r}."
            )
        if len(slices) != self.ndim or not all(
            isinstance(s, slice) for s in slices
        ):
            raise ValueError(
                f"TrainingImage.window: expected a tuple of {self.ndim} "
                f"slice objects (one per dimension), got {slices!r}."
            )
        out = []
        for d, sl in enumerate(slices):
            if sl.step not in (None, 1):
                raise ValueError(
                    f"TrainingImage.window: axis {d} has step {sl.step!r}; "
                    "only unit-step slices are allowed — a strided slice "
                    "subsamples the TI and destroys pattern continuity."
                )
            start = 0 if sl.start is None else int(sl.start)
            stop = self._shape[d] if sl.stop is None else int(sl.stop)
            if start < 0 or stop > self._shape[d]:
                raise ValueError(
                    f"TrainingImage.window: slice {start}:{stop} on axis {d} "
                    f"is out of bounds for TI shape {self._shape!r} "
                    "(negative indices are not supported)."
                )
            if stop - start < 1:
                raise ValueError(
                    f"TrainingImage.window: slice {start}:{stop} on axis {d} "
                    "is degenerate (empty)."
                )
            out.append(slice(start, stop))
        return tuple(out)

    def window(self, slices):
        """Return a TrainingImage **view** over a sub-region of this TI.

        Supports zonated simulation from sub-regions of one large TI
        (Mariethoz et al. 2010, para [40]) without copying data. Per-variable
        ``d_max`` is recomputed from the sliced data; all other per-variable
        settings carry over unchanged.

        Parameters
        ----------
        slices : tuple of :class:`slice`
            One unit-step, in-bounds, non-degenerate slice per dimension,
            e.g. ``np.s_[:250, 300:]``. A single slice is accepted for 1-D.

        Returns
        -------
        TrainingImage
            A view sharing this TI's data buffers.
        """
        slices = self._validate_window_slices(slices)
        new_vars = [_view_variable(v, slices) for v in self._variables]
        return (
            TrainingImage(new_vars)
            if self._multivariate
            else TrainingImage(new_vars[0])
        )

    @property
    def categorical(self):
        """:class:`bool`: Whether the variable is categorical (univariate TIs only)."""
        if self._multivariate:
            raise AttributeError(
                "TrainingImage.categorical is not available for multivariate TIs. "
                "Use ti.variable(name).categorical."
            )
        return self._sole_variable.categorical

    @property
    def distance_type(self):
        """:class:`str`: Distance metric string (univariate TIs only)."""
        if self._multivariate:
            raise AttributeError(
                "TrainingImage.distance_type is not available for multivariate TIs. "
                "Use ti.variable(name).distance."
            )
        return self._sole_variable.distance

    @property
    def distance_power(self):
        """:class:`float`: Spatial-decay exponent δ (univariate TIs only).

        Per-variable for multivariate TIs; use ``ti.variable(name).distance_power``.
        """
        if self._multivariate:
            raise AttributeError(
                "TrainingImage.distance_power is not available for multivariate "
                "TIs. Use ti.variable(name).distance_power."
            )
        return self._sole_variable.distance_power

    # ------------------------------------------------------------------
    # Distance
    # ------------------------------------------------------------------

    def _dispatch_metric(
        self,
        categorical,
        p_norm,
        vp_norm,
        d_max,
        de_sim,
        all_de_ti,
        w,
        has_nan=False,
        penalty_matrix=None,
    ):
        """Select and call the right vectorized distance function.

        Parameters
        ----------
        categorical : bool
        p_norm : float or None
        vp_norm : float or None
        d_max : float or None
        de_sim : numpy.ndarray, shape (n,)
        all_de_ti : numpy.ndarray, shape (max_scan, n)
        w : numpy.ndarray, shape (n,)
        has_nan : bool, optional
            Force masked comparison; NaNs in supplied events enable it
            automatically. Comparable weights are renormalized per candidate.
            A masked candidate with zero weight support has infinite distance.

        Returns
        -------
        numpy.ndarray, shape (max_scan,)
            Per-candidate distance. With normalized weights, categorical and
            variation scores lie in [0, 1]. Lp scores can exceed one for SG
            values outside the TI range. No comparable masked weight gives inf.
        """
        # Route to the Rust block-distance kernels. The kernels normally gather
        # candidates from a flat TI via base + lag indices; here `all_de_ti` is
        # already a materialized (max_scan, n) block, so we present it as a flat
        # "TI" with row-start bases and column-index lags — kernel[i, j] then
        # reads all_de_ti[i, j]. Same maths as the former Python vec_*_dist,
        # computed in Rust. Dispatch order: categorical > l1 > l2 > lp > variation.
        de = np.asarray(de_sim, dtype=np.float64)
        block = np.ascontiguousarray(all_de_ti, dtype=np.float64)
        m, n = block.shape
        if m == 0:
            return np.zeros(0)
        ti_flat = block.ravel()
        base = np.arange(m, dtype=np.int64) * n
        lags = np.arange(n, dtype=np.int64)
        w = np.asarray(w, dtype=np.float64)
        if categorical and penalty_matrix is not None:
            _check_category_codes(de, len(penalty_matrix), "SG event")
            _check_category_codes(block, len(penalty_matrix), "TI event")
            support = np.isfinite(block)
            # NaNs have no matrix index: use zero only for the unused lookup.
            codes = np.where(support, block, 0).astype(np.intp)
            cost = penalty_matrix[de.astype(np.intp)[None, :], codes]
            total = np.sum(np.where(support, cost * w, 0.0), axis=1)
            if has_nan:
                surviving = np.sum(support * w, axis=1)
                return np.divide(
                    total,
                    surviving,
                    out=np.full(m, np.inf),
                    where=surviving > 0,
                )
            return total
        if categorical:
            kernel = (
                mps_dist_block_cat_masked if has_nan else mps_dist_block_cat
            )
            return np.asarray(kernel(de, ti_flat, base, lags, w))
        if p_norm == 1.0:
            kernel = mps_dist_block_l1_masked if has_nan else mps_dist_block_l1
            return np.asarray(kernel(de, ti_flat, base, lags, w, d_max))
        if p_norm == 2.0:
            kernel = mps_dist_block_l2_masked if has_nan else mps_dist_block_l2
            return np.asarray(kernel(de, ti_flat, base, lags, w, d_max))
        if p_norm is not None:
            kernel = mps_dist_block_lp_masked if has_nan else mps_dist_block_lp
            return np.asarray(
                kernel(de, ti_flat, base, lags, w, d_max, p_norm)
            )
        kernel = (
            mps_dist_block_variation_masked
            if has_nan
            else mps_dist_block_variation
        )
        return np.asarray(kernel(de, ti_flat, base, lags, w, d_max, vp_norm))

    def adjust_value(self, ti_val, data_event_sim, data_event_ti, var=None):
        """Adjust matched TI value before assignment to SG.

        For ``distance="variation"``, applies the mean-shift correction
        (Mariethoz2010 Eq. 9) over the common finite SG/TI support.
        Empty common support applies zero shift; nonfinite arithmetic raises.
        For all other metrics returns *ti_val* unchanged.

        Parameters
        ----------
        ti_val : float
            Raw value at the matched TI node.
        data_event_sim : array-like
            SG data event (used to compute Z̄(x_i)).
        data_event_ti : array-like
            TI data event (used to compute Z̄(y)).
        var : str or None, optional
            Variable name for multivariate TIs. ``None`` selects the anonymous
            univariate variable.

        Returns
        -------
        float
        """
        v = (
            self.variable()
            if var is None and len(self._variables) == 1
            else self.variable(var)
        )
        _validate_real_data(ti_val, v.categorical, "TI center")
        _validate_real_data(data_event_sim, v.categorical, "SG event")
        _validate_real_data(data_event_ti, v.categorical, "TI event")
        if v.variation_p_norm is None or v.categorical:
            return ti_val
        sim = np.asarray(data_event_sim, dtype=np.float64)
        ti = np.asarray(data_event_ti, dtype=np.float64)
        if sim.ndim != 1 or sim.shape != ti.shape:
            raise ValueError(
                "adjust_value: events must be equal-length 1-D arrays"
            )
        if (
            np.isinf(sim).any()
            or np.isinf(ti).any()
            or not np.isfinite(ti_val)
        ):
            raise ValueError(
                "adjust_value: infinite events or nonfinite center"
            )
        support = np.isfinite(sim) & np.isfinite(ti)
        with np.errstate(over="ignore", invalid="ignore"):
            shift = (
                np.mean(sim[support] - ti[support]) if support.any() else 0.0
            )
            result = float(ti_val + shift)
        if not np.isfinite(result):
            raise ValueError(
                "adjust_value: mean shift/result exceeds float64; rescale data"
            )
        return result

    def vec_distance_var(
        self,
        var,
        de_sim,
        all_de_ti,
        cond_mask=None,
        cond_weight=None,
        lag_norms=None,
        weights=None,
        has_nan=False,
    ):
        """Vectorized distance for one variable over all TI scan candidates.

        Candidate distances are computed by GSTools-Core Rust block kernels;
        Python prepares node weights and the candidate array.

        Parameters
        ----------
        var : str or None
            Variable name (str for multivariate TIs; None for univariate internal).
        de_sim : array-like, shape (n,)
            SG data event for this variable.
        all_de_ti : array-like, shape (max_scan, n)
            TI data events for every scan candidate.
        cond_mask : array-like of bool, optional
        cond_weight : float or None, optional
            Conditioning multiplier; None uses the Variable configuration.
        lag_norms : array-like, shape (n,), optional
        weights : numpy.ndarray, optional
            Pre-computed node weights. If given, skips the internal
            ``compute_node_weights`` call.
        has_nan : bool, optional
            Force masked comparison; NaNs in supplied events enable it
            automatically. Comparable weights are renormalized per candidate.
            A masked candidate with zero weight support has infinite distance.

        Returns
        -------
        numpy.ndarray, shape (max_scan,)
            Per-candidate distance. With normalized weights, categorical and
            variation scores lie in [0, 1]. Lp scores can exceed one for SG
            values outside the TI range. No comparable masked weight gives inf.
        """
        v = (
            self.variable()
            if var is None and len(self._variables) == 1
            else self.variable(var)
        )
        _validate_real_data(de_sim, v.categorical, "SG event")
        _validate_real_data(all_de_ti, v.categorical, "TI events")
        de_sim = np.asarray(de_sim, dtype=np.float64)
        all_de_ti = np.asarray(all_de_ti, dtype=np.float64)
        if (
            de_sim.ndim != 1
            or all_de_ti.ndim != 2
            or all_de_ti.shape[1] != de_sim.size
        ):
            raise ValueError(
                "vec_distance_var: expected (n,) event and (candidates, n) block"
            )
        if np.isinf(de_sim).any() or np.isinf(all_de_ti).any():
            raise ValueError(
                "vec_distance_var: infinite event values are invalid"
            )
        n = len(de_sim)
        if cond_weight is None:
            cond_weight = v.cond_weight
        if not np.isfinite(cond_weight) or cond_weight < 0:
            raise ValueError(
                "vec_distance_var: cond_weight must be finite and nonnegative"
            )
        for name, value in (
            ("cond_mask", cond_mask),
            ("lag_norms", lag_norms),
            ("weights", weights),
        ):
            if value is not None and np.shape(value) != (n,):
                raise ValueError(
                    f"vec_distance_var: {name} must have shape ({n},)"
                )
        if lag_norms is not None:
            _validate_real_data(lag_norms, False, "lag_norms")
        if weights is not None:
            _validate_real_data(weights, False, "weights")
        if lag_norms is not None and (
            not np.isfinite(lag_norms).all()
            or np.any(np.asarray(lag_norms) < 0)
        ):
            raise ValueError(
                "vec_distance_var: lag_norms must be finite and nonnegative"
            )
        w = (
            np.asarray(weights, dtype=float)
            if weights is not None
            else compute_node_weights(
                n, lag_norms, v.distance_power, cond_mask, cond_weight
            )
        )
        with np.errstate(over="ignore"):
            finite_total = np.isfinite(w.sum())
        if not np.isfinite(w).all() or np.any(w < 0) or not finite_total:
            raise ValueError(
                "vec_distance_var: weights must be finite and nonnegative"
            )
        defined = np.isfinite(de_sim)
        if not defined.all():
            de_sim, all_de_ti, w = (
                de_sim[defined],
                all_de_ti[:, defined],
                w[defined],
            )
            has_nan = True
        if de_sim.size == 0:
            return np.zeros(len(all_de_ti))
        # Missingness belongs to the supplied block, not only the original TI.
        return self._dispatch_metric(
            v.categorical,
            v.p_norm,
            v.variation_p_norm,
            v.d_max,
            de_sim,
            all_de_ti,
            w,
            has_nan=has_nan or np.isnan(all_de_ti).any(),
            penalty_matrix=v.penalty_matrix,
        )

    def __repr__(self):
        if self._multivariate:
            names = [v.name for v in self._variables]
            return f"TrainingImage(variables={names!r}, shape={self.shape})"
        v = self._sole_variable
        return (
            f"TrainingImage(shape={self.shape}, "
            f"categorical={v.categorical}, "
            f"distance={v.distance!r})"
        )


def _resolve_post_ti(primary_ti, post_processing_variables):
    """Build the post-processing TrainingImage: primary variables, with the
    explicitly set fields of any named override applied (unset ones inherit).

    Defensively validates names against ``primary_ti`` even though
    :class:`~gstools.mps.MPSModel` already does.
    """
    override_map = {v.name: v for v in post_processing_variables}
    unknown = set(override_map) - {v.name for v in primary_ti.variables}
    if unknown:
        raise ValueError(
            f"post_processing_variables name(s) {sorted(map(str, unknown))!r} "
            "not found in the primary TrainingImage's variables "
            f"{sorted(str(v.name) for v in primary_ti.variables)!r}."
        )
    post_vars = [
        primary.replace(**{f: getattr(ov, f) for f in ov._explicit})
        if (ov := override_map.get(primary.name)) is not None
        else primary
        for primary in primary_ti.variables
    ]
    if primary_ti.multivariate:
        raw = [v.weight for v in post_vars]
        if any(w is None for w in raw) and not all(w is None for w in raw):
            raise ValueError(
                "MPSModel: post-processing weights must be set on every "
                "variable (primary or override) or on none"
            )
    return (
        TrainingImage(post_vars)
        if primary_ti.multivariate
        else TrainingImage(post_vars[0])
    )


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
                "post-processing: "
                + (f"variable {v.name!r}" if v.name is not None else "the TI")
                + " has distance='variation' "
                f"with n_neighbors={v.n_neighbors}; post_processing_factor="
                f"{p_f:g} gives {n} post-pass neighbour(s), but variation "
                "needs >= 2."
            )
        n_k[v.name] = n
    return n_k
