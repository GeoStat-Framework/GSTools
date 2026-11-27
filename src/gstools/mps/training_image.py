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

from gstools.mps._validation import check_float64_integers, exact_count

__all__ = ["Variable", "TrainingImage"]


def _any_float_nan(*arrays):
    """Return True if any floating-dtype array in *arrays* contains NaN."""
    return any(
        np.issubdtype(arr.dtype, np.floating) and np.isnan(arr).any()
        for arr in arrays
    )


def _data_range(arr):
    """``max - min`` of the finite values; ``1.0`` if constant or undefined.

    Integer extrema are subtracted as Python ints; float extrema are widened to
    float64 first. Raises ``ValueError`` if the float64 spread overflows.
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
    """Variable sharing ``var``'s configuration with ``data = var.data[slices]`` (a view, no copy)."""
    return Variable._derive(var, data=var.data[slices])


_SENTINEL = object()

ANON_NAME = "field"

SEARCH_FIELDS = (
    "distance",
    "weight",
    "n_neighbors",
    "max_radius",
    "penalty_matrix",
    "distance_power",
    "cond_weight",
)


class Variable:
    """One channel's data and per-variable distance configuration for MPS.

    Parameters
    ----------
    name : str
        Valid Python identifier (unique within a TrainingImage).
    data : numpy.ndarray
        n-D array; copied at construction and then read-only.
    categorical : bool, optional
        Whether the variable is categorical. Default: ``True``.
    distance : str, optional
        Distance metric of a continuous variable: ``"l1"`` (default),
        ``"l<p>"`` with finite ``p > 0`` (e.g. ``"l2"``, ``"l3.5"``),
        ``"variation"`` or ``"variation<p>"``. A categorical variable only
        accepts ``"l1"`` (its mismatch distance does not use it); any other
        value raises. With ``"variation"``, a realised data event of one
        point (e.g. a small ``max_radius``) gives zero distance for every
        candidate.
    weight : float or None, optional
        Relative weight in the multivariate joint distance, finite and
        ``>= 0``. ``None`` means equal share; weights must be set on every
        variable of a :class:`TrainingImage` or on none. Default: ``None``.
    n_neighbors : int, optional
        Maximum neighbours in the data event, ``>= 1`` (``>= 2`` for
        ``"variation"``). Default: 32.
    max_radius : float or None, optional
        Exclude SG neighbours beyond this Euclidean distance (in grid cells,
        finite and ``>= 1``) from the data event. ``None`` → no limit
        (default).
    penalty_matrix : numpy.ndarray or None, optional
        ``(C, C)`` per-category mismatch penalty matrix ``T[u, v]`` for
        categorical variables. ``T[u, u] == 0``
        and ``T[u, v] ∈ (0, 1]`` for ``u != v``; asymmetric matrices are
        allowed. ``None`` (default) uses the standard binary mismatch.
        Only valid when ``categorical=True``; ``data`` must then contain
        non-negative integer-valued codes ``< C`` at every finite cell.
    distance_power : float, optional
        Exponent δ for spatial-decay weighting of this variable's
        neighbours (Mariethoz2010 Eq. 3), finite and ``>= 0``. ``0.0`` →
        uniform weights (default). ``1.0`` → closer neighbours weighted
        more heavily.
    cond_weight : float, optional
        Weight multiplier for this variable's conditioning nodes in the
        distance (Meerschman2013 §6, Mariethoz2010 ¶26), finite and
        ``>= 0``. Default: 1.0.

    Raises
    ------
    ValueError
        If ``name`` is not an identifier, ``data`` contains ``+-inf`` or
        integers not exact in float64, or a search setting is invalid or
        inconsistent with the data (e.g. a ``penalty_matrix`` on a
        continuous variable).

    Examples
    --------
    >>> import numpy as np
    >>> import gstools as gs
    >>> facies = np.indices((10, 10)).sum(axis=0) % 2.0
    >>> var = gs.mps.Variable("facies", facies, n_neighbors=8, max_radius=3.0)
    >>> var.n_neighbors, var.max_radius
    (8, 3.0)
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
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(
                f"Variable: name must be a valid Python identifier, got {name!r}."
            )
        self._name = name
        self._data = np.array(data, copy=True)
        if (
            np.issubdtype(self._data.dtype, np.floating)
            and np.isinf(self._data).any()
        ):
            raise ValueError(
                f"Variable {name!r}: data contains +-inf; use NaN to mark "
                "undefined cells."
            )
        self._categorical = bool(categorical)
        check_float64_integers(
            self._data, self._categorical, f"Variable {name!r}: data"
        )
        self._has_nan = _any_float_nan(self._data)
        if self._has_nan:
            warnings.warn(
                f"Variable {name!r}: contains NaN cell(s); treated as undefined "
                "(excluded from continuous data range and pattern distances, "
                "never pasted into the simulation).",
                UserWarning,
                stacklevel=2,
            )
        self._d_max = None if self._categorical else _data_range(self._data)
        self._configure(
            distance=distance,
            weight=weight,
            n_neighbors=n_neighbors,
            max_radius=max_radius,
            penalty_matrix=penalty_matrix,
            distance_power=distance_power,
            cond_weight=cond_weight,
        )
        # Mark the data array read-only: the engine reads it directly and
        # nothing downstream should ever write through .data.
        self._data.flags.writeable = False

    def _configure(self, **fields):
        """Validate and set the given ``SEARCH_FIELDS`` (the data is never touched)."""
        name = self._name
        if "distance" in fields:
            distance = str(fields["distance"])
            if self._categorical:
                if distance.lower() != "l1":
                    raise ValueError(
                        f"Variable {name!r}: distance={distance!r} only applies "
                        "to continuous variables; pass categorical=False."
                    )
                p_norm = variation_p_norm = None
            else:
                p_norm, variation_p_norm = _parse_distance(distance)
            self._distance = distance
            self._p_norm, self._variation_p_norm = p_norm, variation_p_norm
        if "weight" in fields:
            weight = fields["weight"]
            weight = None if weight is None else float(weight)
            if weight is not None and not 0 <= weight < np.inf:
                raise ValueError(
                    f"Variable: weight must be finite and >= 0, got {weight!r}."
                )
            self._weight = weight
        if "max_radius" in fields:
            max_radius = fields["max_radius"]
            if max_radius is not None and not 1 <= float(max_radius) < np.inf:
                raise ValueError(
                    "Variable: max_radius is in grid cells and must be finite "
                    f"and >= 1 (None for no limit), got {max_radius!r}."
                )
            self._max_radius = (
                None if max_radius is None else float(max_radius)
            )
        if "penalty_matrix" in fields:
            self._penalty_matrix = self._validate_penalty_matrix(
                fields["penalty_matrix"]
            )
        if "distance_power" in fields:
            power = float(fields["distance_power"])
            if not 0 <= power < np.inf:
                raise ValueError(
                    "Variable: distance_power must be finite and >= 0, got "
                    f"{fields['distance_power']!r}."
                )
            self._distance_power = power
        if "cond_weight" in fields:
            cond_weight = float(fields["cond_weight"])
            if not 0 <= cond_weight < np.inf:
                raise ValueError(
                    "Variable: cond_weight must be finite and >= 0, got "
                    f"{fields['cond_weight']!r}."
                )
            self._cond_weight = cond_weight
        if "n_neighbors" in fields or "distance" in fields:
            n = (
                fields["n_neighbors"]
                if "n_neighbors" in fields
                else self._n_neighbors
            )
            self._n_neighbors = _validate_n_neighbors(
                n, self._variation_p_norm
            )

    @classmethod
    def _derive(cls, source, data=None, **changes):
        """Copy of ``source`` sharing its frozen buffer, validating only ``changes``.

        With ``data`` (a view of the source's buffer) the NaN flag and range
        are recomputed from it; otherwise they are copied from the source.
        """
        new = object.__new__(cls)
        new.__dict__.update(source.__dict__)
        if data is not None:
            new._data = data
            new._has_nan = _any_float_nan(data)
            new._d_max = None if new._categorical else _data_range(data)
        new._configure(**changes)
        return new

    def _validate_penalty_matrix(self, penalty_matrix):
        """Validate and freeze a per-category penalty matrix, or return None.

        Constraints: 2-D square, zero diagonal,
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
        """:class:`str`: Variable name."""
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

    def replace(self, **changes):
        """New Variable sharing this one's data; only the search fields in ``changes`` differ.

        Parameters
        ----------
        **changes
            New values for any of ``distance``, ``weight``, ``n_neighbors``,
            ``max_radius``, ``penalty_matrix``, ``distance_power`` and
            ``cond_weight`` (see :class:`Variable`).

        Raises
        ------
        ValueError
            If ``changes`` names ``data``, ``categorical`` or an unknown
            field, or a value is invalid.
        """
        if "data" in changes or "categorical" in changes:
            raise ValueError(
                "Variable.replace(): 'data' and 'categorical' cannot be "
                "overridden here -- they define what the variable is; "
                "construct a new Variable instead."
            )
        unknown = set(changes) - set(SEARCH_FIELDS)
        if unknown:
            raise ValueError(
                f"Variable.replace(): unknown field(s) {sorted(unknown)!r}; "
                f"valid fields are {sorted(SEARCH_FIELDS)!r}."
            )
        return Variable._derive(self, **changes)

    def __repr__(self):
        parts = [
            f"Variable({self._name!r}, shape={self._data.shape}",
            f"categorical={self._categorical}",
            f"distance={self._distance!r}",
            f"n_neighbors={self._n_neighbors}",
        ]
        if self._weight is not None:
            parts.append(f"weight={self._weight!r}")
        if self._max_radius is not None:
            parts.append(f"max_radius={self._max_radius!r}")
        if self._penalty_matrix is not None:
            parts.append(f"penalty_matrix={self._penalty_matrix.shape!r}")
        if self._distance_power != 0.0:
            parts.append(f"distance_power={self._distance_power!r}")
        if self._cond_weight != 1.0:
            parts.append(f"cond_weight={self._cond_weight!r}")
        return ", ".join(parts) + ")"


class TrainingImage:
    """Training image for multiple point statistics simulation.

    The MPS analogue of :class:`gstools.CovModel`: holds the training
    data as one or more :class:`Variable` objects, which carry the
    per-variable distance and search settings, plus their joint weights.

    Parameters
    ----------
    data : :class:`numpy.ndarray`, :class:`Variable`, or list of :class:`Variable`
        Training image data. A numpy array becomes one variable named
        ``"field"``; a list or tuple of :class:`Variable` objects gives a named
        variable each. A list or tuple makes a simulation return a
        ``{name: array}`` dict; an array or a single :class:`Variable` makes it
        return an array.
    categorical, distance, n_neighbors, max_radius, penalty_matrix : optional
    distance_power, cond_weight : optional
        Array-input sugar, forwarded to the ``"field"`` :class:`Variable`
        (see there; defaults are those of :class:`Variable`). Not allowed when
        ``data`` is a :class:`Variable` or a list or tuple of them, and
        ``distance`` needs ``categorical=False``.

    Raises
    ------
    TypeError
        If ``data`` is an empty list or holds a non-:class:`Variable`.
    ValueError
        If sugar arguments are given with :class:`Variable` input, ``distance``
        is given without ``categorical``, variable names repeat, shapes
        differ, or weights are mixed (some ``None``) or sum to zero.

    Examples
    --------
    >>> import numpy as np
    >>> import gstools as gs
    >>> ti = gs.mps.TrainingImage(np.zeros((100, 100)))
    >>> ti.shape
    (100, 100)
    >>> ti_mv = gs.mps.TrainingImage(
    ...     [
    ...         gs.mps.Variable("a", np.zeros((10, 10))),
    ...         gs.mps.Variable("b", np.ones((10, 10)), categorical=False),
    ...     ]
    ... )
    >>> ti_mv.weights
    {'a': 0.5, 'b': 0.5}
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
        given = {k: v for k, v in sugar_kwargs.items() if v is not _SENTINEL}

        if isinstance(data, (list, tuple)):
            if given:
                raise ValueError(
                    f"TrainingImage: {sorted(given)!r} are array-input "
                    "sugar keyword arguments and have no effect when data "
                    "is a list of Variable objects (each Variable already "
                    "carries its own search config). Configure them on the "
                    "Variable instead, or omit them here."
                )
            self._init_from_variables(list(data))
        elif isinstance(data, Variable):
            if given:
                raise ValueError(
                    f"TrainingImage: {sorted(given)!r} are array-input "
                    "sugar keyword arguments and have no effect when data "
                    "is a pre-built Variable (it already carries its own "
                    "search config). Configure them on the Variable "
                    "instead, or omit them here."
                )
            self._init_from_variables([data])
        else:
            if categorical is _SENTINEL and distance is not _SENTINEL:
                raise ValueError(
                    "TrainingImage: distance is given but categorical "
                    "defaults to True (distance is then unused); pass "
                    "categorical=False for continuous data."
                )
            self._init_from_variables([Variable(ANON_NAME, data, **given)])

    def _init_from_variables(self, var_list):
        if not var_list or not all(isinstance(v, Variable) for v in var_list):
            raise TypeError(
                "TrainingImage: data must be a numpy.ndarray, a Variable, or a "
                "non-empty list or tuple of Variable objects."
            )
        names = [v.name for v in var_list]
        if len(set(names)) != len(names):
            raise ValueError(
                f"TrainingImage: Variable names must be unique, got {names!r}."
            )
        shapes = {v.data.shape for v in var_list}
        if len(shapes) != 1:
            raise ValueError(
                "TrainingImage: All variables must have the same shape."
            )
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
            if not np.isfinite(total):  # huge weights: rescale to avoid inf
                raw = [w / max(raw) for w in raw]
                total = sum(raw)
            if total <= 0:
                raise ValueError(
                    "TrainingImage: sum of weights must be positive."
                )
            self._normalized_weights = {
                v.name: w / total for v, w in zip(var_list, raw)
            }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def has_nan(self):
        """:class:`bool`: True if any Variable contains NaN."""
        return self._has_nan

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
        """:class:`bool`: True if the TI has more than one variable."""
        return len(self._variables) > 1

    @property
    def variables(self):
        """:class:`list` of :class:`Variable`: All variables (at least one, unique names)."""
        return list(self._variables)

    @property
    def weights(self):
        """:class:`dict`: Normalized ``{name: float}`` weights (sum to 1)."""
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

        Raises
        ------
        TypeError
            If ``name`` is omitted and the TI has several variables.
        KeyError
            If no variable has this name.
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
        (Mariethoz2010 ¶40) without copying data.

        Parameters
        ----------
        slices : tuple of :class:`slice`
            One unit-step, in-bounds, non-degenerate slice per dimension,
            e.g. ``np.s_[:250, 300:]``. A single slice is accepted for 1-D.

        Returns
        -------
        TrainingImage
            A view sharing this TI's data buffers. Per-variable ``d_max`` is
            recomputed from the sliced data; every other per-variable setting
            carries over unchanged.

        Raises
        ------
        ValueError
            If ``slices`` is not one slice per dimension, or a slice has a
            step other than 1, is out of bounds or is empty.

        Examples
        --------
        >>> import numpy as np
        >>> import gstools as gs
        >>> ti = gs.mps.TrainingImage(np.zeros((100, 100)))
        >>> sub = ti.window(np.s_[:50, 25:75])
        >>> sub.shape
        (50, 50)
        """
        slices = self._validate_window_slices(slices)
        new_vars = [_view_variable(v, slices) for v in self._variables]
        return TrainingImage(new_vars[0] if len(new_vars) == 1 else new_vars)

    def __repr__(self):
        variables = ", ".join(repr(v) for v in self._variables)
        if len(self._variables) > 1:
            variables = f"[{variables}]"
        return f"TrainingImage({variables})"
