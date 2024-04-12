"""
GStools subpackage providing different normalizer transformations.

.. currentmodule:: gstools.normalizer.methods

The following classes are provided

.. autosummary::
   LogNormal
   BoxCox
   BoxCoxShift
   YeoJohnson
   Modulus
   Manly
"""

# pylint: disable=E1101
import numpy as np

from gstools.normalizer.base import Normalizer


class LogNormal(Normalizer):
    r"""Log-normal fields.

    Notes
    -----
    This parameter-free transformation is given by:

    .. math::
       y=\log(x)
    """

    normalize_range = (0.0, np.inf)
    """Valid range for input data."""

    def _denormalize(self, data):
        return np.exp(data)

    def _normalize(self, data):
        return np.log(data)

    def _derivative(self, data):
        return np.power(data, -1)


class BoxCox(Normalizer):
    r"""Box-Cox (1964) transformed fields.

    Parameters
    ----------
    data : array_like, optional
        Input data to fit the transformation in order to gain normality.
        The default is None.
    lmbda : :class:`float`, optional
        Shape parameter. Default: 1

    Notes
    -----
    This transformation is given by [Box1964]_:

    .. math::
       y=\begin{cases}
       \frac{x^{\lambda} - 1}{\lambda} & \lambda\neq 0 \\
       \log(x) & \lambda = 0
       \end{cases}

    References
    ----------
    .. [Box1964] G.E.P. Box and D.R. Cox,
           "An Analysis of Transformations",
           Journal of the Royal Statistical Society B, 26, 211-252, (1964)
    """

    default_parameter = {"lmbda": 1}
    """:class:`dict`: Default parameter of the BoxCox-Normalizer."""
    normalize_range = (0.0, np.inf)
    """:class:`tuple`: Valid range for input data."""

    @property
    def denormalize_range(self):
        """:class:`tuple`: Valid range for output data depending on lmbda.

        `(-1/lmbda, inf)` or `(-inf, -1/lmbda)`
        """
        if np.isclose(self.lmbda, 0):
            return (-np.inf, np.inf)
        if self.lmbda < 0:
            return (-np.inf, -np.divide(1, self.lmbda))
        return (-np.divide(1, self.lmbda), np.inf)

    def _denormalize(self, data):
        if np.isclose(self.lmbda, 0):
            return np.exp(data)
        return (1 + np.multiply(data, self.lmbda)) ** (1 / self.lmbda)

    def _normalize(self, data):
        if np.isclose(self.lmbda, 0):
            return np.log(data)
        return (np.power(data, self.lmbda) - 1) / self.lmbda

    def _derivative(self, data):
        return np.power(data, self.lmbda - 1)


class BoxCoxShift(Normalizer):
    r"""Box-Cox (1964) transformed fields including shifting.

    Parameters
    ----------
    data : array_like, optional
        Input data to fit the transformation in order to gain normality.
        The default is None.
    lmbda : :class:`float`, optional
        Shape parameter. Default: 1
    shift : :class:`float`, optional
        Shift parameter. Default: 0

    Notes
    -----
    This transformation is given by [Box1964]_:

    .. math::
       y=\begin{cases}
       \frac{(x+s)^{\lambda} - 1}{\lambda} & \lambda\neq 0 \\
       \log(x+s) & \lambda = 0
       \end{cases}

    Fitting the shift parameter is rather hard. You should consider skipping
    "shift" during fitting:

    >>> data = range(5)
    >>> norm = BoxCoxShift(shift=0.5)
    >>> norm.fit(data, skip=["shift"])
    {'shift': 0.5, 'lmbda': 0.6747515267420799}

    References
    ----------
    .. [Box1964] G.E.P. Box and D.R. Cox,
           "An Analysis of Transformations",
           Journal of the Royal Statistical Society B, 26, 211-252, (1964)
    """

    default_parameter = {"shift": 0, "lmbda": 1}
    """:class:`dict`: Default parameters of the BoxCoxShift-Normalizer."""

    @property
    def normalize_range(self):
        """:class:`tuple`: Valid range for input data depending on shift.

        `(-shift, inf)`
        """
        return (-self.shift, np.inf)

    @property
    def denormalize_range(self):
        """:class:`tuple`: Valid range for output data depending on lmbda.

        `(-1/lmbda, inf)` or `(-inf, -1/lmbda)`
        """
        if np.isclose(self.lmbda, 0):
            return (-np.inf, np.inf)
        if self.lmbda < 0:
            return (-np.inf, -np.divide(1, self.lmbda))
        return (-np.divide(1, self.lmbda), np.inf)

    def _denormalize(self, data):
        if np.isclose(self.lmbda, 0):
            return np.exp(data) - self.shift
        return (1 + np.multiply(data, self.lmbda)) ** (
            1 / self.lmbda
        ) - self.shift

    def _normalize(self, data):
        if np.isclose(self.lmbda, 0):
            return np.log(np.add(data, self.shift))
        return (np.add(data, self.shift) ** self.lmbda - 1) / self.lmbda

    def _derivative(self, data):
        return np.power(np.add(data, self.shift), self.lmbda - 1)


class YeoJohnson(Normalizer):
    r"""Yeo-Johnson (2000) transformed fields.

    Parameters
    ----------
    data : array_like, optional
        Input data to fit the transformation in order to gain normality.
        The default is None.
    lmbda : :class:`float`, optional
        Shape parameter. Default: 1

    Notes
    -----
    This transformation is given by [Yeo2000]_:

    .. math::
       y=\begin{cases}
       \frac{(x+1)^{\lambda} - 1}{\lambda}
       & x\geq 0,\, \lambda\neq 0 \\
       \log(x+1)
       & x\geq 0,\, \lambda = 0 \\
       -\frac{(|x|+1)^{2-\lambda} - 1}{2-\lambda}
       & x<0,\, \lambda\neq 2 \\
       -\log(|x|+1)
       & x<0,\, \lambda = 2
       \end{cases}


    References
    ----------
    .. [Yeo2000] I.K. Yeo and R.A. Johnson,
           "A new family of power transformations to improve normality or
           symmetry." Biometrika, 87(4), pp.954-959, (2000).
    """

    default_parameter = {"lmbda": 1}
    """:class:`dict`: Default parameter of the YeoJohnson-Normalizer."""

    def _denormalize(self, data):
        data = np.asanyarray(data)
        res = np.zeros_like(data, dtype=np.double)
        pos = data >= 0
        # when data >= 0
        if np.isclose(self.lmbda, 0):
            res[pos] = np.expm1(data[pos])
        else:  # self.lmbda != 0
            res[pos] = np.power(data[pos] * self.lmbda + 1, 1 / self.lmbda) - 1
        # when data < 0
        if np.isclose(self.lmbda, 2):
            res[~pos] = -np.expm1(-data[~pos])
        else:  # self.lmbda != 2
            res[~pos] = 1 - np.power(
                -(2 - self.lmbda) * data[~pos] + 1, 1 / (2 - self.lmbda)
            )
        return res

    def _normalize(self, data):
        data = np.asanyarray(data)
        res = np.zeros_like(data, dtype=np.double)
        pos = data >= 0
        # when data >= 0
        if np.isclose(self.lmbda, 0):
            res[pos] = np.log1p(data[pos])
        else:  # self.lmbda != 0
            res[pos] = (np.power(data[pos] + 1, self.lmbda) - 1) / self.lmbda
        # when data < 0
        if np.isclose(self.lmbda, 2):
            res[~pos] = -np.log1p(-data[~pos])
        else:  # self.lmbda != 2
            res[~pos] = -(np.power(-data[~pos] + 1, 2 - self.lmbda) - 1) / (
                2 - self.lmbda
            )
        return res

    def _derivative(self, data):
        return (np.abs(data) + 1) ** (np.sign(data) * (self.lmbda - 1))


class Modulus(Normalizer):
    r"""Modulus or John-Draper (1980) transformed fields.

    Parameters
    ----------
    data : array_like, optional
        Input data to fit the transformation in order to gain normality.
        The default is None.
    lmbda : :class:`float`, optional
        Shape parameter. Default: 1

    Notes
    -----
    This transformation is given by [John1980]_:

    .. math::
       y=\begin{cases}
       \mathrm{sgn}(x)\frac{(|x|+1)^{\lambda} - 1}{\lambda} & \lambda\neq 0 \\
       \mathrm{sgn}(x)\log(|x|+1) & \lambda = 0
       \end{cases}

    References
    ----------
    .. [John1980] J. A. John, and N. R. Draper,
           "An Alternative Family of Transformations." Journal
           of the Royal Statistical Society C, 29.2, 190-197, (1980)
    """

    default_parameter = {"lmbda": 1}
    """:class:`dict`: Default parameter of the Modulus-Normalizer."""

    def _denormalize(self, data):
        if np.isclose(self.lmbda, 0):
            return np.sign(data) * np.expm1(np.abs(data))
        return np.sign(data) * (
            (1 + self.lmbda * np.abs(data)) ** (1 / self.lmbda) - 1
        )

    def _normalize(self, data):
        if np.isclose(self.lmbda, 0):
            return np.sign(data) * np.log1p(np.abs(data))
        return (
            np.sign(data) * ((np.abs(data) + 1) ** self.lmbda - 1) / self.lmbda
        )

    def _derivative(self, data):
        return np.power(np.abs(data) + 1, self.lmbda - 1)


class Manly(Normalizer):
    r"""Manly (1971) transformed fields.

    Parameters
    ----------
    data : array_like, optional
        Input data to fit the transformation in order to gain normality.
        The default is None.
    lmbda : :class:`float`, optional
        Shape parameter. Default: 1

    Notes
    -----
    This transformation is given by [Manly1976]_:

    .. math::
       y=\begin{cases}
       \frac{\exp(\lambda x) - 1}{\lambda} & \lambda\neq 0 \\
       x  & \lambda = 0
       \end{cases}

    References
    ----------
    .. [Manly1976] B. F. J. Manly, "Exponential data transformations.",
           Journal of the Royal Statistical Society D, 25.1, 37-42 (1976).
    """

    default_parameter = {"lmbda": 1}
    """:class:`dict`: Default parameter of the Manly-Normalizer."""

    @property
    def denormalize_range(self):
        """:class:`tuple`: Valid range for output data depending on lmbda.

        `(-1/lmbda, inf)` or `(-inf, -1/lmbda)`
        """
        if np.isclose(self.lmbda, 0):
            return (-np.inf, np.inf)
        if self.lmbda < 0:
            return (-np.inf, np.divide(1, self.lmbda))
        return (-np.divide(1, self.lmbda), np.inf)

    def _denormalize(self, data):
        if np.isclose(self.lmbda, 0):
            return data
        return np.log1p(np.multiply(data, self.lmbda)) / self.lmbda

    def _normalize(self, data):
        if np.isclose(self.lmbda, 0):
            return data
        return np.expm1(np.multiply(data, self.lmbda)) / self.lmbda

    def _derivative(self, data):
        return np.exp(np.multiply(data, self.lmbda))
