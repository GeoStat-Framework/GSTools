# -*- coding: utf-8 -*-
"""
GStools subpackage providing the base class for normalizers.

.. currentmodule:: gstools.normalize.base

The following classes are provided

.. autosummary::
   Normalizer
"""
import warnings
import numpy as np
import scipy.misc as spm
import scipy.optimize as spo


class Normalizer:
    """Normalizer class.

    Parameters
    ----------
    data : array_like, optional
        Input data to fit the transformation to in order to gain normality.
        The default is None.
    **parameter
        Specified parameters given by name. If not given, default values
        will be applied.
    """

    def __init__(self, data=None, **parameter):
        # only use parameter, that have a provided default value
        for key, value in self.default_parameter().items():
            setattr(self, key, parameter.get(key, value))
        # fit parameters if data is given
        if data is not None:
            self.fit(data)
        # optimization results
        self._opti = None
        # precision for printing
        self._prec = 3

    def default_parameter(self):
        """Get default parameters for the transformation.

        Returns
        -------
        :class:`dict`
            Default parameters.
        """
        return {}

    def denormalize(self, values):
        """Transform to input distribution.

        Parameters
        ----------
        values : array_like
            Input values (normal distributed).

        Returns
        -------
        :class:`numpy.ndarray`
            Denormalized values.
        """
        return values

    def normalize(self, values):
        """Transform to normal distribution.

        Parameters
        ----------
        values : array_like
            Input values (not normal distributed).

        Returns
        -------
        :class:`numpy.ndarray`
            Normalized values.
        """
        return values

    def derivative(self, values):
        """Factor for normal PDF to gain target PDF.

        Parameters
        ----------
        values : array_like
            Input values.

        Returns
        -------
        :class:`numpy.ndarray`
            Derivative of the normalization transformation function.
        """
        return spm.derivative(self.normalize, np.asanyarray(values), dx=1e-6)

    def likelihood(self, data):
        """Likelihood for given data with current parameters.

        Parameters
        ----------
        data : array_like
            Input data to fit the transformation to in order to gain normality.

        Returns
        -------
        :class:`float`
            Likelihood of the given data.
        """
        return np.exp(self.loglikelihood(data))

    def loglikelihood(self, data):
        """Log-Likelihood for given data with current parameters.

        Parameters
        ----------
        data : array_like
            Input data to fit the transformation to in order to gain normality.

        Returns
        -------
        :class:`float`
            Log-Likelihood of the given data.
        """
        add = -0.5 * np.size(data) * (np.log(2 * np.pi) + 1)
        return self.kernel_loglikelihood(data) + add

    def kernel_loglikelihood(self, data):
        """Kernel Log-Likelihood for given data with current parameters.

        Parameters
        ----------
        data : array_like
            Input data to fit the transformation to in order to gain normality.

        Returns
        -------
        :class:`float`
            Kernel Log-Likelihood of the given data.

        Notes
        -----
        This loglikelihood function is neglecting additive constants,
        that are not needed for optimization.
        """
        res = -0.5 * np.size(data) * np.log(np.var(self.normalize(data)))
        return res + np.sum(np.log(np.maximum(1e-16, self.derivative(data))))

    def fit(self, data, skip=None, **kwargs):
        """Fitting the transformation to data by maximizing Log-Likelihood.

        Parameters
        ----------
        data : array_like
            Input data to fit the transformation to in order to gain normality.
        skip : :class:`list` of :class:`str` or :any:`None`, optional
            Names of parameters to be skiped in fitting.
            The default is None.
        **kwargs
            Keyword arguments passed to :any:`scipy.optimize.minimize_scalar`
            when only one parameter present or :any:`scipy.optimize.minimize`.

        Returns
        -------
        :class:`dict`
            Optimal paramters given by names.
        """
        skip = [] if skip is None else skip
        all_names = sorted(self.default_parameter())
        para_names = [name for name in all_names if name not in skip]

        def _neg_kllf(par, dat):
            for name, val in zip(para_names, np.atleast_1d(par)):
                setattr(self, name, val)
            return -self.kernel_loglikelihood(dat)

        if len(para_names) == 0:  # transformations without para. (no opti.)
            warnings.warn(self.__class__.__name__ + ".fit: no parameters!")
            return {}
        if len(para_names) == 1:  # one-para. transformations (simple opti.)
            # default bracket like in scipy's boxcox (if not given)
            kwargs.setdefault("bracket", (-2, 2))
            out = spo.minimize_scalar(_neg_kllf, args=(data,), **kwargs)
        else:  # general case
            # init guess from current values (if x0 not given)
            kwargs.setdefault("x0", [getattr(self, p) for p in para_names])
            out = spo.minimize(_neg_kllf, args=(data,), **kwargs)
        # save optimization results
        self._opti = out
        for name, val in zip(para_names, np.atleast_1d(out.x)):
            setattr(self, name, val)
        return {name: getattr(self, name) for name in all_names}

    def __repr__(self):
        """Return String representation."""
        para_strs = [
            "{0}={1:.{2}}".format(p, float(getattr(self, p)), self._prec)
            for p in sorted(self.default_parameter())
        ]
        return self.__class__.__name__ + "(" + ", ".join(para_strs) + ")"
