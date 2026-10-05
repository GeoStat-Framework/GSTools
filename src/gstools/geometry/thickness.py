"""
GStools subpackage providing thickness links for stochastic geometry.

.. currentmodule:: gstools.geometry.thickness

The following classes and functions are provided

.. autosummary::
   ThicknessLink
   LogLink
   set_link
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.special import ndtri

__all__ = ["ThicknessLink", "LogLink", "LINK", "set_link"]


class ThicknessLink(ABC):
    """A monotone map between Gaussian values and layer thickness."""

    @abstractmethod
    def thickness(self, gauss):
        """Map Gaussian values to thicknesses.

        Parameters
        ----------
        gauss : :class:`numpy.ndarray`
            Gaussian values, shape ``(n,)``.

        Returns
        -------
        :class:`numpy.ndarray`
            Thicknesses, shape ``(n,)``, all ``>= 0``.
        """

    @abstractmethod
    def gauss_value(self, thick):
        """Map thicknesses to Gaussian values.

        Parameters
        ----------
        thick : :class:`numpy.ndarray`
            Thicknesses, shape ``(n,)``.

        Returns
        -------
        :class:`numpy.ndarray`
            Gaussian values, shape ``(n,)``. NaN where the thickness lies
            on a censored atom of the link and must be handled as an
            inequality instead of an equality.
        """

    @abstractmethod
    def gauss_bounds(self, low, upp):
        """Map a thickness interval to a Gaussian interval.

        Parameters
        ----------
        low : :class:`float`
            Lower thickness bound.
        upp : :class:`float`
            Upper thickness bound.

        Returns
        -------
        :class:`tuple` of :class:`float`
            ``(lo, up)`` Gaussian bounds, ``+/-inf`` allowed.
        """

    @property
    def name(self):
        """:class:`str`: Class name of the link."""
        return self.__class__.__name__


class LogLink(ThicknessLink):
    """Log-thickness link: ``t = exp(g)``.

    Mathematically the :class:`gstools.normalizer.LogNormal` transform, but
    deliberately not a :any:`Normalizer`: "layer absent" is a censored
    atom, so the map is not invertible there and a likelihood fit would be
    meaningless.

    Parameters
    ----------
    absent : :class:`float`, optional
        Thickness treated as "layer absent" for the purpose of
        :any:`LogLink.gauss_bounds`. Default: 1e-6

    Warnings
    --------
    "Layer absent" data become the bound ``g <= log(absent)``, which for
    small ``absent`` lies many prior standard deviations below the mean
    and can make the conditioned field overshoot elsewhere. Choose
    ``absent`` as a physically meaningful resolution thickness.
    """

    def __init__(self, absent=1e-6):
        self._absent = float(absent)

    def thickness(self, gauss):
        """Map Gaussian values to thicknesses, ``t = exp(g)``.

        See :any:`ThicknessLink.thickness`
        """
        return np.exp(np.asarray(gauss, dtype=np.double))

    def gauss_value(self, thick):
        """Map thicknesses to Gaussian values, ``g = log(t)``.

        NaN for ``t = 0`` (layer absent), see :any:`ThicknessLink.gauss_value`
        """
        thick = np.asarray(thick, dtype=np.double)
        return np.where(
            thick > 0, np.log(np.where(thick > 0, thick, 1.0)), np.nan
        )

    def gauss_bounds(self, low, upp):
        """Map a thickness interval to a Gaussian interval.

        An upper bound of ``0`` (layer absent) becomes ``log(absent)``.
        See :any:`ThicknessLink.gauss_bounds`
        """
        low = np.asarray(low, dtype=np.double)
        upp = np.asarray(upp, dtype=np.double)
        with np.errstate(divide="ignore"):
            lo = np.where(
                low > 0, np.log(np.where(low > 0, low, 1.0)), -np.inf
            )
            upp_eff = np.maximum(upp, self._absent)
            up = np.where(np.isinf(upp), np.inf, np.log(upp_eff))
        return lo, up

    @property
    def absent(self):
        """:class:`float`: Thickness treated as "layer absent"."""
        return self._absent

    @classmethod
    def from_moments(cls, mean_thickness, cv=None, std=None):
        """Derive Gaussian mean/variance from mean and spread of thickness.

        The spread is given either as coefficient of variation ``cv`` or
        as standard deviation ``std`` of the thickness, with
        ``cv = std / mean_thickness``.

        Parameters
        ----------
        mean_thickness : :class:`float`
            Mean thickness.
        cv : :class:`float`, optional
            Coefficient of variation of the thickness.
        std : :class:`float`, optional
            Standard deviation of the thickness.

        Returns
        -------
        :class:`tuple` of :class:`float`
            ``(mu_g, var_g)``.

        Raises
        ------
        ValueError
            If not exactly one of ``cv`` and ``std`` is given.
        """
        if (cv is None) == (std is None):
            raise ValueError("LogLink.from_moments: give either cv or std.")
        if cv is None:
            cv = np.asarray(std, dtype=np.double) / mean_thickness
        var_g = np.log1p(np.asarray(cv, dtype=np.double) ** 2)
        return np.log(mean_thickness) - var_g / 2.0, var_g

    @classmethod
    def from_median(cls, median, sigma):
        """Derive Gaussian mean/variance from median and log-sigma.

        Parameters
        ----------
        median : :class:`float`
            Median thickness.
        sigma : :class:`float`
            Standard deviation in Gaussian (log) space.

        Returns
        -------
        :class:`tuple` of :class:`float`
            ``(mu_g, var_g)``.
        """
        return np.log(median), np.asarray(sigma, dtype=np.double) ** 2

    @classmethod
    def from_percentiles(cls, low, high, prob=0.8):
        """Derive Gaussian mean/variance from a central thickness interval.

        ``low`` and ``high`` bound the central interval holding the
        thickness with probability ``prob``. For the default ``prob=0.8``
        they are the 10th and 90th percentiles.

        Parameters
        ----------
        low : :class:`float`
            Lower end of the interval, ``> 0``.
        high : :class:`float`
            Upper end of the interval, ``> low``.
        prob : :class:`float`, optional
            Probability of the central interval, in ``(0, 1)``.
            Default: 0.8

        Returns
        -------
        :class:`tuple` of :class:`float`
            ``(mu_g, var_g)``.

        Raises
        ------
        ValueError
            If ``not 0 < low < high`` or ``not 0 < prob < 1``.
        """
        if not 0 < low < high:
            raise ValueError(
                "LogLink.from_percentiles: need 0 < low < high, "
                f"got low={low}, high={high}."
            )
        if not 0 < prob < 1:
            raise ValueError(
                f"LogLink.from_percentiles: need 0 < prob < 1, got {prob}."
            )
        log_low, log_high = np.log(low), np.log(high)
        sigma = (log_high - log_low) / (2.0 * ndtri(0.5 + prob / 2.0))
        return (log_low + log_high) / 2.0, sigma**2

    def __repr__(self):
        return f"LogLink(absent={self._absent})"


LINK = {"log": LogLink}
"""dict: Standard thickness link classes."""


def set_link(link, **kwargs):
    """Resolve a string or class to a :any:`ThicknessLink` instance.

    Parameters
    ----------
    link : :class:`str`, :any:`ThicknessLink` or subclass
        Either a name registered in ``LINK``, a :any:`ThicknessLink`
        instance (returned unchanged), or a :any:`ThicknessLink` subclass
        to instantiate.
    **kwargs
        Passed to the link constructor if a class or name is given.

    Returns
    -------
    :any:`ThicknessLink`
        The resolved link instance.

    Raises
    ------
    ValueError
        If the link is unknown, or keyword arguments are given together
        with an instance.
    """
    if isinstance(link, ThicknessLink):
        if kwargs:
            raise ValueError("set_link: kwargs given for a link instance.")
        return link
    if isinstance(link, str):
        if link not in LINK:
            raise ValueError(
                f"set_link: unknown link '{link}', use one of {sorted(LINK)}"
            )
        return LINK[link](**kwargs)
    if isinstance(link, type) and issubclass(link, ThicknessLink):
        return link(**kwargs)
    raise ValueError(f"set_link: unknown or wrong link: {link}")
