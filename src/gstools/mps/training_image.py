"""
GStools subpackage providing the TrainingImage class for MPS simulations.

.. currentmodule:: gstools.mps

The following classes and functions are provided

.. autosummary::
   TrainingImage
"""

import numpy as np

from gstools.mps.distance import (
    categorical_dist,
    compute_node_weights,
    l1_dist,
    l2_dist,
    lp_dist,
    variation_dist,
    vec_categorical_dist,
    vec_l1_dist,
    vec_l2_dist,
    vec_lp_dist,
    vec_variation_dist,
)

__all__ = ["TrainingImage"]


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
        weights=None,
        distance="l1",
        distance_power=0.0,
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
        if not self._categorical:
            self._p_norm, self._variation_p_norm = self._parse_distance(
                distance
            )
            dmax = float(self._data.max() - self._data.min())
            self._d_max = dmax if dmax > 0 else 1.0
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
                dmax = float(
                    self._variables[k].max() - self._variables[k].min()
                )
                self._d_max[k] = dmax if dmax > 0 else 1.0

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

    def distance(
        self,
        data_event_sim,
        data_event_ti,
        cond_mask=None,
        cond_weight=1.0,
        lag_norms=None,
    ):
        """Distance between two data events.

        Applies spatial-decay weights (Mariethoz2010 Eq. 3) to all
        distance types when ``distance_power > 0``.

        Parameters
        ----------
        data_event_sim : array-like, shape (n,)
            Values at SG neighbourhood nodes.
        data_event_ti : array-like, shape (n,)
            Values at TI neighbourhood nodes.
        cond_mask : array-like of bool, optional
            True where the neighbour is a conditioning datum.
        cond_weight : float, optional
            Weight multiplier δ for conditioning nodes
            (Mariethoz2010 §3 ¶26). Default: ``1.0``.
        lag_norms : array-like, shape (n,), optional
            Euclidean norms ``‖h_i‖`` of each lag vector. Required for
            spatial-decay weighting (``distance_power > 0``).

        Returns
        -------
        float
            Distance in [0, 1].
        """
        if self._multivariate:
            total = 0.0
            for k in self._variables:
                lns = (
                    lag_norms.get(k)
                    if isinstance(lag_norms, dict)
                    else lag_norms
                )
                cms = (
                    cond_mask.get(k)
                    if isinstance(cond_mask, dict)
                    else cond_mask
                )
                total += self._weights[k] * self._distance_var(
                    k,
                    data_event_sim[k],
                    data_event_ti[k],
                    cms,
                    cond_weight,
                    lns,
                )
            return total

        data_event_sim = np.asarray(data_event_sim, dtype=np.float64)
        data_event_ti = np.asarray(data_event_ti, dtype=np.float64)
        n = len(data_event_sim)
        if n == 0:
            return 0.0

        w = compute_node_weights(
            n, lag_norms, self._distance_power, cond_mask, cond_weight
        )

        if self._categorical:
            return categorical_dist(data_event_sim, data_event_ti, w)
        if self._p_norm == 1.0:
            return l1_dist(data_event_sim, data_event_ti, w, self._d_max)
        if self._p_norm == 2.0:
            return l2_dist(data_event_sim, data_event_ti, w, self._d_max)
        if self._p_norm is not None:
            return lp_dist(
                data_event_sim, data_event_ti, w, self._d_max, self._p_norm
            )
        return variation_dist(
            data_event_sim,
            data_event_ti,
            w,
            self._d_max,
            self._variation_p_norm,
        )

    def vec_distance(
        self,
        data_event_sim,
        all_de_ti,
        cond_mask=None,
        cond_weight=1.0,
        lag_norms=None,
    ):
        """Vectorized distance between SG data event and all TI candidates.

        Same maths as :meth:`distance` but operates on all TI scan candidates
        at once, returning a distance per candidate instead of a scalar.

        Parameters
        ----------
        data_event_sim : array-like, shape (n,)
            Values at SG neighbourhood nodes.
        all_de_ti : array-like, shape (max_scan, n)
            TI data events for every scan candidate.
        cond_mask : array-like of bool, optional
            True where the neighbour is a conditioning datum.
        cond_weight : float, optional
            Weight multiplier δ for conditioning nodes. Default: ``1.0``.
        lag_norms : array-like, shape (n,), optional
            Euclidean norms of each lag vector.

        Returns
        -------
        numpy.ndarray, shape (max_scan,)
            Distance in [0, 1] for each candidate.
        """
        data_event_sim = np.asarray(data_event_sim, dtype=np.float64)
        all_de_ti = np.asarray(all_de_ti, dtype=np.float64)
        n = len(data_event_sim)
        if n == 0:
            return np.zeros(len(all_de_ti))
        w = compute_node_weights(
            n, lag_norms, self._distance_power, cond_mask, cond_weight
        )
        if self._categorical:
            return vec_categorical_dist(data_event_sim, all_de_ti, w)
        if self._p_norm == 1.0:
            return vec_l1_dist(data_event_sim, all_de_ti, w, self._d_max)
        if self._p_norm == 2.0:
            return vec_l2_dist(data_event_sim, all_de_ti, w, self._d_max)
        if self._p_norm is not None:
            return vec_lp_dist(
                data_event_sim, all_de_ti, w, self._d_max, self._p_norm
            )
        return vec_variation_dist(
            data_event_sim, all_de_ti, w, self._d_max, self._variation_p_norm
        )

    def adjust_value(self, ti_val, data_event_sim, data_event_ti):
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

        Returns
        -------
        float
        """
        if self._variation_p_norm is None or self._categorical:
            return ti_val
        data_event_sim = np.asarray(data_event_sim, dtype=np.float64)
        data_event_ti = np.asarray(data_event_ti, dtype=np.float64)
        if data_event_sim.size == 0 or data_event_ti.size == 0:
            return ti_val
        return float(ti_val - data_event_ti.mean() + data_event_sim.mean())

    def _distance_var(
        self, var, de_sim, de_ti, cond_mask, cond_weight, lag_norms
    ):
        """Scalar distance for one variable (multivariate component)."""
        de_sim = np.asarray(de_sim, dtype=np.float64)
        de_ti = np.asarray(de_ti, dtype=np.float64)
        n = len(de_sim)
        if n == 0:
            return 0.0
        w = compute_node_weights(
            n, lag_norms, self._distance_power, cond_mask, cond_weight
        )
        if self._categorical[var]:
            return categorical_dist(de_sim, de_ti, w)
        p, dmax = self._p_norm[var], self._d_max[var]
        if p == 1.0:
            return l1_dist(de_sim, de_ti, w, dmax)
        if p == 2.0:
            return l2_dist(de_sim, de_ti, w, dmax)
        if p is not None:
            return lp_dist(de_sim, de_ti, w, dmax, p)
        return variation_dist(
            de_sim, de_ti, w, dmax, self._variation_p_norm[var]
        )

    def vec_distance_var(
        self,
        var,
        de_sim,
        all_de_ti,
        cond_mask=None,
        cond_weight=1.0,
        lag_norms=None,
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
        w = compute_node_weights(
            n, lag_norms, self._distance_power, cond_mask, cond_weight
        )
        if self._categorical[var]:
            return vec_categorical_dist(de_sim, all_de_ti, w)
        p, dmax = self._p_norm[var], self._d_max[var]
        if p == 1.0:
            return vec_l1_dist(de_sim, all_de_ti, w, dmax)
        if p == 2.0:
            return vec_l2_dist(de_sim, all_de_ti, w, dmax)
        if p is not None:
            return vec_lp_dist(de_sim, all_de_ti, w, dmax, p)
        return vec_variation_dist(
            de_sim, all_de_ti, w, dmax, self._variation_p_norm[var]
        )

    def adjust_value_var(self, var, ti_val, de_sim, de_ti):
        """Mean-shift correction for one variable (variation distance only).

        Returns *ti_val* unchanged for categorical / Lp variables, or when the
        data event is empty (no neighbours to anchor the mean shift).

        Parameters
        ----------
        var : str
            Variable name.
        ti_val : float
            Raw value at the matched TI node for this variable.
        de_sim : array-like
            SG data event for this variable (used to compute the SG mean).
        de_ti : array-like
            TI data event for this variable (used to compute the TI mean).

        Returns
        -------
        float
        """
        if self._categorical[var] or self._variation_p_norm[var] is None:
            return ti_val
        de_sim = np.asarray(de_sim, dtype=np.float64)
        de_ti = np.asarray(de_ti, dtype=np.float64)
        if de_sim.size == 0 or de_ti.size == 0:
            return ti_val
        return float(ti_val - de_ti.mean() + de_sim.mean())

    def __repr__(self):
        return (
            f"TrainingImage(shape={self.shape}, "
            f"categorical={self._categorical}, "
            f"distance={self._distance_type!r})"
        )
