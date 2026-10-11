from abc import ABC, abstractmethod

__all__ = ["Correlogram"]


class Correlogram(ABC):
    """
    Abstract base class for cross-covariance models in collocated cokriging.

    A correlogram encapsulates the spatial relationship between primary and
    secondary variables, including their cross-covariance structure and
    statistical parameters (means, variances).

    This design allows for different cross-covariance models (MM1, MM2, etc.)
    to be implemented as separate classes, making the cokriging framework
    extensible and future-proof.

    Parameters
    ----------
    primary_model : :any:`CovModel`
        Covariance model for the primary variable.
    cross_corr : :class:`float`
        Cross-correlation coefficient between primary and secondary variables
        at zero lag (collocated). Must be in [-1, 1].
    secondary_var : :class:`float`
        Variance of the secondary variable. Must be positive.
    primary_mean : :class:`float`, optional
        Mean value of the primary variable. Default: 0.0
    secondary_mean : :class:`float`, optional
        Mean value of the secondary variable. Default: 0.0

    Attributes
    ----------
    primary_model : :any:`CovModel`
        The primary variable's covariance model.
    cross_corr : :class:`float`
        Cross-correlation at zero lag.
    secondary_var : :class:`float`
        Secondary variable variance.
    primary_mean : :class:`float`
        Primary variable mean.
    secondary_mean : :class:`float`
        Secondary variable mean.

    Notes
    -----
    Subclasses must implement :any:`compute_covariances` and
    :any:`cross_covariance` to define the cross-covariance structure.
    """

    def __init__(
        self,
        primary_model,
        cross_corr,
        secondary_var,
        primary_mean=0.0,
        secondary_mean=0.0,
    ):
        """Initialize the correlogram with spatial and statistical parameters."""
        self.primary_model = primary_model
        self.cross_corr = float(cross_corr)
        self.secondary_var = float(secondary_var)
        self.primary_mean = float(primary_mean)
        self.secondary_mean = float(secondary_mean)

        # Validate parameters
        self._validate()

    def _validate(self):
        """
        Validate correlogram parameters.

        Raises
        ------
        ValueError
            If cross_corr is not in [-1, 1] or secondary_var is not positive.
        """
        if not -1.0 <= self.cross_corr <= 1.0:
            raise ValueError(
                f"cross_corr must be in [-1, 1], got {self.cross_corr}"
            )

        if self.secondary_var <= 0:
            raise ValueError(
                f"secondary_var must be positive, got {self.secondary_var}"
            )

    @abstractmethod
    def compute_covariances(self):
        """
        Compute covariances at zero lag.

        Returns
        -------
        C_Z0 : :class:`float`
            Primary variable variance :math:`C_Z(0)`.
        C_Y0 : :class:`float`
            Secondary variable variance :math:`C_Y(0)`.
        C_YZ0 : :class:`float`
            Cross-covariance between primary and secondary at zero lag
            :math:`C_{YZ}(0)`.

        Notes
        -----
        This method defines how the cross-covariance at zero lag is computed
        from the cross-correlation and variances. Different correlogram models
        may use different formulas.
        """

    @abstractmethod
    def cross_covariance(self, h):
        """
        Compute cross-covariance :math:`C_{YZ}(h)` at distance :math:`h`.

        Parameters
        ----------
        h : :class:`float` or :class:`numpy.ndarray`
            Distance(s) at which to compute cross-covariance.

        Returns
        -------
        C_YZ_h : :class:`float` or :class:`numpy.ndarray`
            Cross-covariance at distance :math:`h`.

        Notes
        -----
        This is the key method that differentiates correlogram models.
        For example, MM1 uses the primary variable's spatial structure
        while MM2 would use the secondary variable's structure.
        """
