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
    data : numpy.ndarray
        Training image data (n-d array).
    categorical : bool, optional
        Whether the variable is categorical. Default: ``True``.
    distance : str, optional
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
        self, data, categorical=True, distance="l1", distance_power=0.0
    ):
        self._data = np.array(data, copy=True)
        self._categorical = bool(categorical)
        self._distance_power = float(distance_power)
        if self._distance_power < 0:
            raise ValueError("distance_power must be >= 0")
        self._distance_type = distance
        self._p_norm = None
        self._variation_p_norm = None
        if not self._categorical:
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
                self._p_norm = p_val
            elif distance_lower == "variation":
                self._variation_p_norm = 2.0
            elif distance_lower.startswith("variation"):
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
                self._variation_p_norm = p_val
            else:
                raise ValueError(
                    f"TrainingImage: distance must be 'l<p>' (e.g. 'l1', 'l2'), "
                    f"'variation', or 'variation<p>' (e.g. 'variation1'). "
                    f"Got {distance!r}"
                )

        if not self._categorical:
            dmax = float(self._data.max() - self._data.min())
            self._d_max = dmax if dmax > 0 else 1.0
        else:
            self._d_max = None

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
        return self._data.ndim

    @property
    def shape(self):
        """tuple: Shape of the training image."""
        return self._data.shape

    @property
    def categorical(self):
        """bool: Whether the variable is categorical."""
        return self._categorical

    @property
    def distance_type(self):
        """str: Distance metric (e.g. ``"l1"``, ``"l2"``, ``"variation"``, ``"variation1"``)."""
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
        return float(ti_val - data_event_ti.mean() + data_event_sim.mean())

    def __repr__(self):
        return (
            f"TrainingImage(shape={self.shape}, "
            f"categorical={self._categorical}, "
            f"distance={self._distance_type!r})"
        )
