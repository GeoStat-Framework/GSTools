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


def _view_variable(var, slices):
    """Variable sharing ``var``'s configuration with ``data = var.data[slices]``.

    The data is a NumPy **view** (no copy; the parent buffer is already
    read-only, so the view is too). ``d_max`` is recomputed from the sliced
    data so per-zone continuous normalization reflects the sub-region.
    """
    view = var.data[slices]
    new = Variable.__new__(Variable)
    new._name = var._name
    new._data = view
    new._categorical = var._categorical
    new._distance = var._distance
    new._weight = var._weight
    new._max_radius = var._max_radius
    new._has_nan = _any_float_nan(view)
    new._p_norm = var._p_norm
    new._variation_p_norm = var._variation_p_norm
    new._d_max = None if var._categorical else _data_range(view)
    new._n_neighbors = var._n_neighbors
    return new


_SENTINEL = object()


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
    ):
        if name is not None and (
            not isinstance(name, str) or not name.isidentifier()
        ):
            raise ValueError(
                f"Variable: name must be a valid Python identifier, got {name!r}."
            )
        self._name = name
        self._data = np.array(data, copy=True)
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
        # set via setter so the variation guard runs on construction too
        self._n_neighbors = 32
        self.n_neighbors = n_neighbors
        # Mark the data array read-only: the engine reads it directly and
        # nothing downstream should ever write through .data.
        self._data.flags.writeable = False

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

    # --- mutable property ---

    @property
    def n_neighbors(self):
        """:class:`int`: Maximum neighbours in the data event."""
        return self._n_neighbors

    @n_neighbors.setter
    def n_neighbors(self, value):
        v = int(value)
        if v < 1:
            raise ValueError(
                f"Variable: n_neighbors must be >= 1, got {value!r}."
            )
        if self._variation_p_norm is not None and v < 2:
            raise ValueError(
                "Variable: distance='variation' requires n_neighbors >= 2 "
                "(a single-point data event has no local mean deviation)."
            )
        self._n_neighbors = v

    def __repr__(self):
        parts = [
            f"Variable({self._name!r}, shape={self._data.shape}",
            f"categorical={self._categorical}",
            f"distance={self._distance!r}",
            f"n_neighbors={self._n_neighbors}",
        ]
        if self._max_radius is not None:
            parts.append(f"max_radius={self._max_radius!r}")
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
        ``1.0`` → closer neighbours weighted more heavily.
    """

    def __init__(
        self,
        data,
        *,
        categorical=True,
        distance="l1",
        n_neighbors=32,
        max_radius=None,
        distance_power=0.0,
    ):
        self._distance_power = float(distance_power)
        if self._distance_power < 0:
            raise ValueError("TrainingImage: distance_power must be >= 0")

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
        ti = TrainingImage.__new__(TrainingImage)
        ti._distance_power = self._distance_power
        ti._multivariate = self._multivariate
        ti._variables = new_vars
        ti._var_map = {v.name: v for v in new_vars}
        ti._shape = new_vars[0].data.shape
        ti._has_nan = any(v.has_nan for v in new_vars)
        ti._normalized_weights = (
            dict(self._normalized_weights) if self._multivariate else None
        )
        return ti

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

    @property
    def distance_power(self):
        """:class:`float`: Spatial-decay exponent δ for node weighting."""
        return self._distance_power

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
                n, lag_norms, self._distance_power, cond_mask, cond_weight
            )
        )
        return self._dispatch_metric(
            v.categorical,
            v.p_norm,
            v.variation_p_norm,
            v.d_max,
            de_sim,
            all_de_ti,
            w,
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
