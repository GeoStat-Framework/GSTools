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
        Input data to fit the transformation in order to gain normality.
        The default is None.
    **parameter
        Specified parameters given by name. If not given, default values
        will be applied.
    """

    def __init__(self, data=None, **parameter):
        # only use values, that have a provided default value
        for key, value in self.default_parameter().items():
            setattr(self, key, parameter.get(key, value))
        if data is not None:
            self.fit(data)
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

    def loglikelihood(self, data):
        """Log-Likelihood for given data with current parameters.

        Parameters
        ----------
        data : array_like
            Input data to fit the transformation in order to gain normality.

        Returns
        -------
        :class:`float`
            Log-Likelihood of the given data.

        Notes
        -----
        This loglikelihood function could be missing additive constants,
        that are not needed for optimization.
        """
        res = -0.5 * np.size(data) * np.log(np.var(self.normalize(data)))
        res += np.sum(np.log(np.maximum(1e-16, self.derivative(data))))
        return res

    def fit(self, data, skip=None, **kwargs):
        """Fitting the transformation to data by maximizing Log-Likelihood.

        Parameters
        ----------
        data : array_like
            Input data to fit the transformation in order to gain normality.
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
        all_names = list(self.default_parameter().keys())
        para_names = [p for p in all_names if p not in skip]

        def _neg_llf(par, dat):
            for name, val in zip(para_names, np.atleast_1d(par)):
                setattr(self, name, val)
            return -self.loglikelihood(dat)

        if len(para_names) == 0:  # transformations without para. (no opti.)
            warnings.warn("Normalizer.fit: no parameters for fitting.")
            return {}
        if len(para_names) == 1:  # one-para. transformations (simple opti.)
            # default bracket like in scipy's boxcox (if not given)
            kwargs.setdefault("bracket", (-2, 2))
            out = spo.minimize_scalar(_neg_llf, args=(data,), **kwargs)
        else:  # general case
            # init guess from current values (if x0 not given)
            kwargs.setdefault("x0", [getattr(self, p) for p in para_names])
            out = spo.minimize(_neg_llf, args=(data,), **kwargs)
        # save optimization results
        self._opti = out
        for name, val in zip(para_names, np.atleast_1d(out.x)):
            setattr(self, name, val)
        return {name: getattr(self, name) for name in all_names}

    def __str__(self):
        """Return String representation."""
        return self.__repr__()

    def __repr__(self):
        """Return String representation."""
        out_str = self.__class__.__name__
        para_strs = []
        for p in self.default_parameter():
            para_strs.append(
                "{0}={1:.{2}}".format(p, float(getattr(self, p)), self._prec)
            )
        return out_str + "(" + ", ".join(para_strs) + ")"
