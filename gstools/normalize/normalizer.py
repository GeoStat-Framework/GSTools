# -*- coding: utf-8 -*-
"""
GStools subpackage providing different normalizer transformations.

.. currentmodule:: gstools.normalize.normalizer

The following classes are provided

.. autosummary::
   LogNormal
   BoxCox
   BoxCoxShift
   YeoJohnson
   Modulus
   Manly
"""
import numpy as np
from gstools.normalize.base import Normalizer


class LogNormal(Normalizer):
    r"""Log-normal fields.

    Notes
    -----
    This parameter-free transformation is given by:

    .. math::
       y=\log(x)
    """

    def denormalize(self, values):
        """Transform to log-normal distribution."""
        return np.exp(values)

    def normalize(self, values):
        """Transform to normal distribution."""
        return np.log(values)

    def derivative(self, values):
        """Factor for normal PDF to gain target PDF."""
        return np.power(values, -1)


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
    This transformation is given by  [1]_:

    .. math::
       y=\begin{cases}
       \frac{x^{\lambda} - 1}{\lambda} & \lambda\neq 0 \\
       \log(x) & \lambda = 0
       \end{cases}

    References
    ----------
    .. [1] G.E.P. Box and D.R. Cox, "An Analysis of Transformations", Journal
           of the Royal Statistical Society B, 26, 211-252 (1964).
    """

    def default_parameter(self):
        """Get default parameters."""
        return {"lmbda": 1}

    def denormalize(self, values):
        """Transform to target distribution."""
        if np.isclose(self.lmbda, 0):
            return np.exp(values)
        return (1 + np.multiply(values, self.lmbda)) ** (1 / self.lmbda)

    def normalize(self, values):
        """Transform to normal distribution."""
        if np.isclose(self.lmbda, 0):
            return np.log(values)
        return (np.power(values, self.lmbda) - 1) / self.lmbda

    def derivative(self, values):
        """Factor for normal PDF to gain target PDF."""
        return np.power(values, self.lmbda - 1)


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
    This transformation is given by [1]_:

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
    .. [1] G.E.P. Box and D.R. Cox, "An Analysis of Transformations", Journal
           of the Royal Statistical Society B, 26, 211-252 (1964).
    """

    def default_parameter(self):
        """Get default parameters."""
        return {"shift": 0, "lmbda": 1}

    def denormalize(self, values):
        """Transform to target distribution."""
        if np.isclose(self.lmbda, 0):
            return np.exp(values) - self.shift
        return (1 + np.multiply(values, self.lmbda)) ** (
            1 / self.lmbda
        ) - self.shift

    def normalize(self, values):
        """Transform to normal distribution."""
        if np.isclose(self.lmbda, 0):
            return np.log(np.add(values, self.shift))
        return (np.add(values, self.shift) ** self.lmbda - 1) / self.lmbda

    def derivative(self, values):
        """Factor for normal PDF to gain target PDF."""
        return np.power(np.add(values, self.shift), self.lmbda - 1)


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
    This transformation is given by [1]_:

    .. math::
       y=\begin{cases}
       \frac{(x+1)^{\lambda} - 1}{\lambda}
       & x\geq 0,\, \lambda\neq 0 \\
       \log(x+1)
       & x\geq 0,\, \lambda = 0 \\
       \frac{(|x|+1)^{2-\lambda} - 1}{2-\lambda}
       & x<0,\, \lambda\neq 2 \\
       -\log(|x|+1)
       & x<0,\, \lambda = 2
       \end{cases}


    References
    ----------
    .. [1] I.K. Yeo and R.A. Johnson, "A new family of power transformations to
           improve normality or symmetry." Biometrika, 87(4), pp.954-959,
           (2000).
    """

    def default_parameter(self):
        """Get default parameters."""
        return {"lmbda": 1}

    def denormalize(self, values):
        """Transform to target distribution."""
        values = np.asanyarray(values)
        res = np.zeros_like(values, dtype=np.double)
        pos = values >= 0
        # when values >= 0
        if np.isclose(self.lmbda, 0):
            res[pos] = np.expm1(values[pos])
        else:  # self.lmbda != 0
            res[pos] = (
                np.power(values[pos] * self.lmbda + 1, 1 / self.lmbda) - 1
            )
        # when values < 0
        if np.isclose(self.lmbda, 2):
            res[~pos] = -np.expm1(-values[~pos])
        else:  # self.lmbda != 2
            res[~pos] = 1 - np.power(
                -(2 - self.lmbda) * values[~pos] + 1, 1 / (2 - self.lmbda)
            )
        return res

    def normalize(self, values):
        """Transform to normal distribution."""
        values = np.asanyarray(values)
        res = np.zeros_like(values, dtype=np.double)
        pos = values >= 0
        # when values >= 0
        if np.isclose(self.lmbda, 0):
            res[pos] = np.log1p(values[pos])
        else:  # self.lmbda != 0
            res[pos] = (np.power(values[pos] + 1, self.lmbda) - 1) / self.lmbda
        # when values < 0
        if np.isclose(self.lmbda, 2):
            res[~pos] = -np.log1p(-values[~pos])
        else:  # self.lmbda != 2
            res[~pos] = -(np.power(-values[~pos] + 1, 2 - self.lmbda) - 1) / (
                2 - self.lmbda
            )
        return res

    def derivative(self, values):
        """Factor for normal PDF to gain target PDF."""
        return (np.abs(values) + 1) ** (np.sign(values) * (self.lmbda - 1))


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
    This transformation is given by [1]_:

    .. math::
       y=\begin{cases}
       \mathrm{sgn}(x)\frac{(|x|+1)^{\lambda} - 1}{\lambda} & \lambda\neq 0 \\
       \mathrm{sgn}(x)\log(|x|+1) & \lambda = 0
       \end{cases}

    References
    ----------
    .. [1] J. A. John, and N. R. Draper,
           “An Alternative Family of Transformations.” Journal
           of the Royal Statistical Society C, 29.2, 190-197, (1980)
    """

    def default_parameter(self):
        """Get default parameters."""
        return {"lmbda": 1}

    def denormalize(self, values):
        """Transform to target distribution."""
        if np.isclose(self.lmbda, 0):
            return np.sign(values) * np.expm1(np.abs(values))
        return np.sign(values) * (
            (1 + self.lmbda * np.abs(values)) ** (1 / self.lmbda) - 1
        )

    def normalize(self, values):
        """Transform to normal distribution."""
        if np.isclose(self.lmbda, 0):
            return np.sign(values) * np.log1p(np.abs(values))
        return (
            np.sign(values)
            * ((np.abs(values) + 1) ** self.lmbda - 1)
            / self.lmbda
        )

    def derivative(self, values):
        """Factor for normal PDF to gain target PDF."""
        return np.power(np.abs(values) + 1, self.lmbda - 1)


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
    This transformation is given by [1]_:

    .. math::
       y=\begin{cases}
       \frac{\exp(\lambda x) - 1}{\lambda} & \lambda\neq 0 \\
       x  & \lambda = 0
       \end{cases}

    References
    ----------
    .. [1] B. F. J. Manly, "Exponential data transformations.",
           Journal of the Royal Statistical Society D, 25.1, 37-42 (1976).
    """

    def default_parameter(self):
        """Get default parameters."""
        return {"lmbda": 1}

    def denormalize(self, values):
        """Transform to target distribution."""
        if np.isclose(self.lmbda, 0):
            return values
        return np.log1p(np.multiply(values, self.lmbda)) / self.lmbda

    def normalize(self, values):
        """Transform to normal distribution."""
        if np.isclose(self.lmbda, 0):
            return values
        return np.expm1(np.multiply(values, self.lmbda)) / self.lmbda

    def derivative(self, values):
        """Factor for normal PDF to gain target PDF."""
        return np.exp(np.multiply(values, self.lmbda))
