"""
GStools subpackage providing the TrainingImage class for MPS simulations.

.. currentmodule:: gstools.mps

The following classes and functions are provided

.. autosummary::
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

__all__ = ["TrainingImage"]


def _warn_if_nan(*arrays):
    """Warn once if any (float) TI array contains undefined (NaN) cells."""
    for arr in arrays:
        if np.issubdtype(arr.dtype, np.floating) and np.isnan(arr).any():
            warnings.warn(
                "TrainingImage contains NaN cell(s); they are treated as "
                "undefined (masked): excluded from the continuous data range "
                "and from pattern distances, and never pasted into the "
                "simulation.",
                UserWarning,
                stacklevel=3,
            )
            return


def _data_range(arr):
    """Continuous data range over *defined* cells (NaN excluded).

    Returns the ``max - min`` spread of the finite values, or ``1.0`` when the
    array is constant or entirely undefined (avoids a zero/NaN normalizer).
    """
    finite = arr[np.isfinite(arr)]
    dmax = float(finite.max() - finite.min()) if finite.size else 0.0
    return dmax if dmax > 0 else 1.0


class TrainingImage:
    """Training image for multiple point statistics simulation.

    The MPS analogue of :class:`gstools.CovModel`: encapsulates training
    data and the distance function for comparing data events.

    Parameters
    ----------
    data : numpy.ndarray or dict of {str: numpy.ndarray}
        Training image data (n-d array). Pass a dict of named arrays to create
        a multivariate (co-simulation) training image; all arrays must share
        the same shape.
    categorical : bool or dict of {str: bool}, optional
        Whether the variable is categorical. For multivariate TIs, a dict gives
        one flag per variable (a scalar is broadcast to all). Default: ``True``.
    weights : dict of {str: float}, optional
        Per-variable distance weights for multivariate TIs (must sum to 1).
        Default: uniform. Ignored for univariate TIs.
    distance : str or dict of {str: str}, optional
        Distance metric for continuous variables: ``"l1"`` (Juda2022
        Eq. 7, default), ``"l2"`` (Mariethoz2010 Eq. 4–5), or
        ``"variation"`` (Mariethoz2010 Eq. 9). Ignored when categorical.
    distance_power : float, optional
        Exponent δ for spatial-decay weighting of neighbours
        (Mariethoz2010 Eq. 3). Applied to **all** distance types.
        ``0.0`` → uniform weights (oracle-compatible default).
        ``1.0`` → closer neighbours weighted more heavily.
    """

    def __init__(
        self,
        data,
        categorical=True,
        distance="l1",
        distance_power=0.0,
        weights=None,
    ):
        self._distance_power = float(distance_power)
        if self._distance_power < 0:
            raise ValueError("distance_power must be >= 0")

        if isinstance(data, dict):
            self._init_multivariate(data, categorical, weights, distance)
            return

        # ---- univariate (unchanged behaviour) ----
        self._multivariate = False
        self._variables = None
        self._weights = None
        self._data = np.array(data, copy=True)
        self._shape = self._data.shape
        self._categorical = bool(categorical)
        self._distance_type = distance
        self._p_norm = None
        self._variation_p_norm = None
        _warn_if_nan(self._data)
        if not self._categorical:
            self._p_norm, self._variation_p_norm = self._parse_distance(
                distance
            )
            self._d_max = _data_range(self._data)
        else:
            self._d_max = None

    @staticmethod
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

    def _init_multivariate(self, data, categorical, weights, distance):
        """Initialise a multivariate (dict-valued) training image.

        Parameters
        ----------
        data : dict of {str: numpy.ndarray}
            Named variable arrays (all the same shape).
        categorical : bool or dict of {str: bool}
            Categorical flag, scalar (broadcast) or per-variable.
        weights : dict of {str: float} or None
            Per-variable distance weights (must sum to 1); ``None`` → uniform.
        distance : str or dict of {str: str}
            Continuous-distance metric, scalar (broadcast) or per-variable.
        """
        self._multivariate = True
        self._data = None
        self._variables = {k: np.array(v, copy=True) for k, v in data.items()}
        if len(self._variables) == 0:
            raise ValueError("TrainingImage: multivariate data dict is empty.")
        shapes = {v.shape for v in self._variables.values()}
        if len(shapes) != 1:
            raise ValueError("All variables must have the same shape.")
        self._shape = shapes.pop()
        names = list(self._variables)
        _warn_if_nan(*self._variables.values())

        self._categorical = (
            {k: bool(categorical[k]) for k in names}
            if isinstance(categorical, dict)
            else {k: bool(categorical) for k in names}
        )
        self._distance_type = (
            {k: distance[k] for k in names}
            if isinstance(distance, dict)
            else {k: distance for k in names}
        )
        self._p_norm, self._variation_p_norm, self._d_max = {}, {}, {}
        for k in names:
            if self._categorical[k]:
                self._p_norm[k] = None
                self._variation_p_norm[k] = None
                self._d_max[k] = None
            else:
                self._p_norm[k], self._variation_p_norm[k] = (
                    self._parse_distance(self._distance_type[k])
                )
                self._d_max[k] = _data_range(self._variables[k])

        if weights is None:
            self._weights = {k: 1.0 / len(names) for k in names}
        else:
            missing = set(names) - set(weights)
            if missing:
                raise ValueError(
                    f"TrainingImage: weights missing for variables {sorted(missing)}."
                )
            extra = set(weights) - set(names)
            if extra:
                raise ValueError(
                    f"TrainingImage: weights has unknown variables {sorted(extra)}."
                )
            wsum = float(sum(weights[k] for k in names))
            if not np.isclose(wsum, 1.0):
                raise ValueError(f"weights must sum to 1.0, got {wsum}.")
            self._weights = {k: float(weights[k]) for k in names}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def data(self):
        """numpy.ndarray: Raw training image data."""
        return self._data

    @property
    def ndim(self):
        """int: Number of spatial dimensions."""
        return len(self._shape)

    @property
    def shape(self):
        """tuple: Shape of the training image (shared across variables)."""
        return self._shape

    @property
    def multivariate(self):
        """bool: Whether the TI holds multiple co-simulated variables."""
        return self._multivariate

    @property
    def variables(self):
        """list or None: Variable names (insertion order, all equal), or None if univariate."""
        return list(self._variables) if self._multivariate else None

    @property
    def weights(self):
        """dict or None: Per-variable distance weights (sum to 1), or None."""
        return dict(self._weights) if self._multivariate else None

    def variable(self, name):
        """numpy.ndarray: Data array for one variable (multivariate TIs).

        Parameters
        ----------
        name : str
            Variable name.

        Returns
        -------
        numpy.ndarray
        """
        if not self._multivariate:
            raise TypeError(
                "variable() is only available on multivariate TrainingImages."
            )
        return self._variables[name]

    @property
    def categorical(self):
        """bool or dict of {str: bool}: Whether the variable(s) are categorical."""
        return self._categorical

    @property
    def distance_type(self):
        """str or dict of {str: str}: Distance metric(s) (e.g. ``"l1"``, ``"l2"``, ``"variation"``)."""
        return self._distance_type

    @property
    def distance_power(self):
        """float: Spatial-decay exponent δ for node weighting."""
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
        var : str, optional
            Variable name for multivariate TIs. When ``None``, uses the
            univariate attributes.

        Returns
        -------
        float
        """
        if var is None:
            categorical = self._categorical
            vp_norm = self._variation_p_norm
        else:
            categorical = self._categorical[var]
            vp_norm = self._variation_p_norm[var]
        if vp_norm is None or categorical:
            return ti_val
        data_event_sim = np.asarray(data_event_sim, dtype=np.float64)
        data_event_ti = np.asarray(data_event_ti, dtype=np.float64)
        if data_event_sim.size == 0 or data_event_ti.size == 0:
            return ti_val
        # nanmean: the matched TI event may include undefined (NaN) neighbours
        # on a masked TI; de-mean over the defined positions only. For a fully
        # finite event this is identical to ``mean``. If every TI neighbour is
        # undefined, drop the TI mean-shift term (treat as 0).
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
        var : str
            Variable name.
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
            self._categorical[var],
            self._p_norm[var],
            self._variation_p_norm[var],
            self._d_max[var],
            de_sim,
            all_de_ti,
            w,
            has_nan=has_nan,
        )

    def __repr__(self):
        return (
            f"TrainingImage(shape={self.shape}, "
            f"categorical={self._categorical}, "
            f"distance={self._distance_type!r})"
        )
