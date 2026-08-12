"""
GStools subpackage providing the TrainingImage class for MPS simulations.

.. currentmodule:: gstools.mps

The following classes and functions are provided

.. autosummary::
   Variable
   TrainingImage
"""

import warnings

import numpy as np

from gstools.mps.distance import (
    compute_node_weights,
    vec_categorical_dist,
    vec_categorical_penalty_dist,
    vec_l1_dist,
    vec_l2_dist,
    vec_lp_dist,
    vec_variation_dist,
)

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
    """
    finite = arr[np.isfinite(arr)]
    dmax = float(finite.max() - finite.min()) if finite.size else 0.0
    return dmax if dmax > 0 else 1.0


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
        if p_val <= 0:
            raise ValueError(
                f"TrainingImage: Lp norm exponent must be > 0, got {p_val}."
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
        if p_val <= 0:
            raise ValueError(
                f"TrainingImage: variation exponent must be > 0, got {p_val}."
            )
        return None, p_val
    raise ValueError(
        f"TrainingImage: distance must be 'l<p>' (e.g. 'l1', 'l2'), "
        f"'variation', or 'variation<p>' (e.g. 'variation1'). "
        f"Got {distance!r}"
    )


def _validate_n_neighbors(value, variation_p_norm):
    """Validate n_neighbors >= 1 (and >= 2 under distance='variation')."""
    v = int(value)
    if v < 1:
        raise ValueError(f"Variable: n_neighbors must be >= 1, got {value!r}.")
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


def _view_variable(var, slices):
    """Variable sharing ``var``'s configuration with ``data = var.data[slices]``.

    The data is a NumPy **view** (no copy; the parent buffer is already
    read-only, so the view is too -- ``Variable.__init__`` skips copying it).
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
        )


_SENTINEL = object()

_REPLACE_FIELDS = frozenset(
    {
        "distance", "weight", "n_neighbors", "max_radius", "penalty_matrix",
        "distance_power", "cond_weight",
    }
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
    penalty_matrix : numpy.ndarray or None, optional
        ``(C, C)`` per-category mismatch penalty matrix ``T[u, v]`` for
        categorical variables (DS_Feature_Checklist §3.3). ``T[u, u] == 0``
        and ``T[u, v] ∈ (0, 1]`` for ``u != v``; asymmetric matrices are
        allowed. ``None`` (default) uses the standard binary mismatch.
        Only valid when ``categorical=True``; ``data`` must then contain
        non-negative integer-valued codes ``< C`` at every finite cell.
    distance_power : float, optional
        Exponent δ for spatial-decay weighting of this variable's
        neighbours (Mariethoz2010 Eq. 3). ``0.0`` → uniform weights
        (oracle-compatible default). ``1.0`` → closer neighbours weighted
        more heavily.
    cond_weight : float, optional
        Weight multiplier for this variable's conditioning nodes in the
        distance (Meerschman2013 §6, Mariethoz2010 para [26]). Default: 1.0.
    """

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
    ):
        if name is not None and (
            not isinstance(name, str) or not name.isidentifier()
        ):
            raise ValueError(
                f"Variable: name must be a valid Python identifier, got {name!r}."
            )
        self._name = name
        # Defensive copy, except when `data` is already a frozen (read-only)
        # array -- nothing can mutate it in place, so sharing it is safe.
        # This is what makes `replace()`/`_view_variable()` avoid the O(n)
        # data copy: they pass in this Variable's own already-frozen buffer.
        # has_nan/d_max are still recomputed from that buffer below (O(n)),
        # so replace() is a large constant-factor win over a fresh
        # construction, not literally O(1).
        data_arr = np.asarray(data)
        self._data = (
            data_arr if not data_arr.flags.writeable else np.array(data_arr, copy=True)
        )
        self._categorical = bool(categorical)
        self._distance = str(distance)
        self._weight = None if weight is None else float(weight)
        if max_radius is not None and float(max_radius) <= 0:
            raise ValueError(
                f"Variable: max_radius must be a positive float, got {max_radius!r}."
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
        self._distance_power = float(distance_power)
        if self._distance_power < 0:
            raise ValueError(
                f"Variable: distance_power must be >= 0, got {distance_power!r}."
            )
        self._cond_weight = float(cond_weight)
        self._n_neighbors = _validate_n_neighbors(
            n_neighbors, self._variation_p_norm
        )
        # Mark the data array read-only: the engine reads it directly and
        # nothing downstream should ever write through .data.
        self._data.flags.writeable = False

    def _validate_penalty_matrix(self, penalty_matrix):
        """Validate and freeze a per-category penalty matrix, or return None.

        Constraints (DS_Feature_Checklist §3.3): 2-D square, zero diagonal,
        off-diagonal entries in ``(0, 1]``; asymmetric matrices are allowed.
        The variable's own data must contain non-negative integer-valued
        codes ``< C`` at every finite cell.
        """
        if penalty_matrix is None:
            return None
        if not self._categorical:
            raise ValueError(
                f"Variable {self._name!r}: penalty_matrix is only valid for "
                "categorical variables (categorical=False was given)."
            )
        mat = np.array(penalty_matrix, copy=True, dtype=np.float64)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError(
                f"Variable {self._name!r}: penalty_matrix must be a 2D square "
                f"array, got shape {mat.shape!r}."
            )
        n_cat = mat.shape[0]
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
        bad_off = (off_vals <= 0.0) | (off_vals > 1.0)
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
    def penalty_matrix(self):
        """:class:`numpy.ndarray` of shape (C, C) or None: Per-category mismatch penalties; None uses the default binary mismatch."""
        return self._penalty_matrix

    @property
    def distance_power(self):
        """:class:`float`: Spatial-decay exponent δ (Mariethoz2010 Eq. 3)."""
        return self._distance_power

    @property
    def cond_weight(self):
        """:class:`float`: Weight multiplier for this variable's conditioning nodes."""
        return self._cond_weight

    @property
    def n_neighbors(self):
        """:class:`int`: Maximum neighbours in the data event."""
        return self._n_neighbors

    def replace(self, **overrides):
        """New Variable sharing this one's data buffer; only search config differs.

        No copy: the buffer is already frozen read-only, so sharing it is
        safe (``__init__`` skips copying a read-only array). ``data`` and
        ``categorical`` cannot be overridden -- they define what the
        variable *is*; construct a new Variable instead.
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
        # suppress the NaN warning re-firing on every replace() of
        # NaN-containing data.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return Variable(self._name, self._data, **kwargs)

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
    distance_power, cond_weight : see :class:`Variable`
        Univariate-only sugar: forwarded verbatim to the anonymous Variable.
        Ignored when ``data`` is a Variable or list of Variable.
    penalty_matrix : :class:`numpy.ndarray` or None, optional
        Per-category mismatch penalty matrix (univariate TIs only; see
        :class:`Variable`). Default: ``None`` (binary mismatch).
    """

    def __init__(
        self,
        data,
        *,
        categorical=_SENTINEL,
        distance=_SENTINEL,
        n_neighbors=_SENTINEL,
        max_radius=_SENTINEL,
        penalty_matrix=_SENTINEL,
        distance_power=_SENTINEL,
        cond_weight=_SENTINEL,
    ):
        sugar_kwargs = {
            "categorical": categorical,
            "distance": distance,
            "n_neighbors": n_neighbors,
            "max_radius": max_radius,
            "penalty_matrix": penalty_matrix,
            "distance_power": distance_power,
            "cond_weight": cond_weight,
        }
        explicitly_given = sorted(
            k for k, v in sugar_kwargs.items() if v is not _SENTINEL
        )

        if isinstance(data, list):
            if explicitly_given:
                raise ValueError(
                    f"TrainingImage: {explicitly_given!r} are univariate-"
                    "sugar keyword arguments and have no effect when data "
                    "is a list of Variable objects (each Variable already "
                    "carries its own search config). Configure them on the "
                    "Variable instead, or omit them here."
                )
            self._init_from_variables(data)
        elif isinstance(data, Variable):
            if explicitly_given:
                raise ValueError(
                    f"TrainingImage: {explicitly_given!r} are univariate-"
                    "sugar keyword arguments and have no effect when data "
                    "is a pre-built Variable (it already carries its own "
                    "search config). Configure them on the Variable "
                    "instead, or omit them here."
                )
            # Univariate sugar: caller supplies a pre-configured Variable
            self._multivariate = False
            self._variables = [data]
            self._var_map = {data.name: data}
            self._shape = data.data.shape
            self._has_nan = data.has_nan
            self._normalized_weights = None
        else:
            # Univariate sugar: bare array -> anonymous Variable (name=None).
            # Resolve sentinels to real defaults here -- the one place they
            # matter, since a bare array has no pre-existing Variable to
            # shadow.
            var = Variable(
                None,
                data,
                categorical=True if categorical is _SENTINEL else categorical,
                distance="l1" if distance is _SENTINEL else distance,
                weight=1.0,
                n_neighbors=32 if n_neighbors is _SENTINEL else n_neighbors,
                max_radius=None if max_radius is _SENTINEL else max_radius,
                penalty_matrix=(
                    None if penalty_matrix is _SENTINEL else penalty_matrix
                ),
                distance_power=(
                    0.0 if distance_power is _SENTINEL else distance_power
                ),
                cond_weight=1.0 if cond_weight is _SENTINEL else cond_weight,
            )
            self._multivariate = False
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
            total = sum(raw)
            if total <= 0:
                raise ValueError(
                    "TrainingImage: sum of weights must be positive."
                )
            self._normalized_weights = {
                v.name: v.weight / total for v in var_list
            }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def has_nan(self):
        """:class:`bool`: True if any Variable contains NaN."""
        return self._has_nan

    @property
    def data(self):
        """:class:`numpy.ndarray` or None: Raw data for univariate TIs; None for MV."""
        if self._multivariate:
            return None
        return self._var_map[None].data

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
        (Mariethoz et al. 2010, para [40]) without copying data.

        Parameters
        ----------
        slices : tuple of :class:`slice`
            One unit-step, in-bounds, non-degenerate slice per dimension,
            e.g. ``np.s_[:250, 300:]``. A single slice is accepted for 1-D.

        Returns
        -------
        TrainingImage
            A view sharing this TI's data buffers. Per-variable ``d_max`` is
            recomputed from the sliced data; all other per-variable settings
            (kind, distance, weight, ``n_neighbors``, ``max_radius``) carry
            over unchanged.

        Examples
        --------
        >>> import numpy as np
        >>> import gstools as gs
        >>> ti = gs.TrainingImage(np.zeros((100, 100)))
        >>> sub = ti.window(np.s_[:50, 25:75])
        >>> sub.shape
        (50, 50)
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
        return self._var_map[None].categorical

    @property
    def distance_type(self):
        """:class:`str`: Distance metric string (univariate TIs only)."""
        if self._multivariate:
            raise AttributeError(
                "TrainingImage.distance_type is not available for multivariate TIs. "
                "Use ti.variable(name).distance."
            )
        return self._var_map[None].distance

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
        penalty_matrix=None,
        has_nan=False,
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
        penalty_matrix : numpy.ndarray of shape (C, C) or None, optional
            Per-category mismatch penalties for categorical variables; ``None``
            uses the default binary mismatch (:func:`vec_categorical_dist`).
        has_nan : bool, optional
            Enable per-row exclusion of undefined (NaN) TI positions, with
            per-row weight renormalization. Default ``False``.

        Returns
        -------
        numpy.ndarray, shape (max_scan,)
            Distance in [0, 1] for each candidate.
        """
        # Dispatch order: categorical > l1 (p==1) > l2 (p==2) > lp > variation.
        # l1/l2 are explicit fast-paths (not folded into lp) so the specialised
        # BLAS-friendly kernels are always selected for p in {1, 2}.
        if categorical:
            if penalty_matrix is not None:
                return vec_categorical_penalty_dist(
                    de_sim, all_de_ti, w, penalty_matrix, has_nan=has_nan
                )
            return vec_categorical_dist(de_sim, all_de_ti, w, has_nan=has_nan)
        if p_norm == 1.0:
            return vec_l1_dist(de_sim, all_de_ti, w, d_max, has_nan=has_nan)
        if p_norm == 2.0:
            return vec_l2_dist(de_sim, all_de_ti, w, d_max, has_nan=has_nan)
        if p_norm is not None:
            return vec_lp_dist(
                de_sim, all_de_ti, w, d_max, p_norm, has_nan=has_nan
            )
        return vec_variation_dist(
            de_sim, all_de_ti, w, d_max, vp_norm, has_nan=has_nan
        )

    def adjust_value(self, ti_val, data_event_sim, data_event_ti, var=None):
        """Adjust matched TI value before assignment to SG.

        For ``distance="variation"``, applies the mean-shift correction
        (Mariethoz2010 Eq. 9): Z(x_i) = Z(y) − Z̄(y) + Z̄(x_i).
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
        v = self._var_map[var]
        if v.variation_p_norm is None or v.categorical:
            return ti_val
        data_event_sim = np.asarray(data_event_sim, dtype=np.float64)
        data_event_ti = np.asarray(data_event_ti, dtype=np.float64)
        if data_event_sim.size == 0 or data_event_ti.size == 0:
            return ti_val
        ti_mean = (
            np.nanmean(data_event_ti)
            if np.isfinite(data_event_ti).any()
            else 0.0
        )
        return float(ti_val - ti_mean + np.nanmean(data_event_sim))

    def vec_distance_var(
        self,
        var,
        de_sim,
        all_de_ti,
        cond_mask=None,
        cond_weight=1.0,
        lag_norms=None,
        weights=None,
        d_max=None,
        has_nan=False,
    ):
        """Vectorized distance for one variable over all TI scan candidates.

        Parameters
        ----------
        var : str or None
            Variable name (str for multivariate TIs; None for univariate internal).
        de_sim : array-like, shape (n,)
            SG data event for this variable.
        all_de_ti : array-like, shape (max_scan, n)
            TI data events for every scan candidate.
        cond_mask : array-like of bool, optional
        cond_weight : float, optional
        lag_norms : array-like, shape (n,), optional
        weights : numpy.ndarray, optional
            Pre-computed node weights. If given, skips the internal
            ``compute_node_weights`` call.
        d_max : float or None, optional
            Override for the normalization range — used by zonated
            simulation, where ``d_max`` comes from the selected zone TI;
            ``None`` -> this variable's own ``d_max``.
        has_nan : bool, optional
            Enable per-row exclusion of undefined (NaN) TI positions, with
            per-row weight renormalization. Default ``False``.

        Returns
        -------
        numpy.ndarray, shape (max_scan,)
            Distance in [0, 1] for each candidate.
        """
        v = self._var_map[var]  # var is a str or None (univariate internal)
        # Categorical distance only uses !=, which is dtype-agnostic; skip the
        # float64 allocation on the categorical path and keep it for continuous.
        if v.categorical:
            de_sim = np.asarray(de_sim)
            all_de_ti = np.asarray(all_de_ti)
        else:
            de_sim = np.asarray(de_sim, dtype=np.float64)
            all_de_ti = np.asarray(all_de_ti, dtype=np.float64)
        n = len(de_sim)
        if n == 0:
            return np.zeros(len(all_de_ti))
        w = (
            weights
            if weights is not None
            else compute_node_weights(
                n, lag_norms, v.distance_power, cond_mask, cond_weight
            )
        )
        return self._dispatch_metric(
            v.categorical,
            v.p_norm,
            v.variation_p_norm,
            v.d_max if d_max is None else d_max,
            de_sim,
            all_de_ti,
            w,
            penalty_matrix=v.penalty_matrix,
            has_nan=has_nan,
        )

    def __repr__(self):
        if self._multivariate:
            names = [v.name for v in self._variables]
            return f"TrainingImage(variables={names!r}, shape={self.shape})"
        v = self._var_map[None]
        return (
            f"TrainingImage(shape={self.shape}, "
            f"categorical={v.categorical}, "
            f"distance={v.distance!r})"
        )


def _resolve_post_ti(primary_ti, post_processing_variables):
    """Build the post-processing TrainingImage: primary variables, with any
    named override from ``post_processing_variables`` substituted in.

    Defensively validates names against ``primary_ti`` even though
    :class:`MPSModel` already does -- this function is reachable directly
    from :func:`gstools.mps.simulate.ds_simulate`, which does not go through
    MPSModel's validation.
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
        override_map.get(primary.name, primary)
        for primary in primary_ti.variables
    ]
    return (
        TrainingImage(post_vars)
        if primary_ti.multivariate
        else TrainingImage(post_vars[0])
    )
