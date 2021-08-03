# -*- coding: utf-8 -*-
"""
GStools subpackage providing the base class for normalizers.

.. currentmodule:: gstools.normalizer.base

The following classes are provided

.. autosummary::
   Normalizer
"""
# pylint: disable=R0201
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
        Specified parameters given by name. If not given, default parameters
        will be used.
    """

    default_parameter = {}
    """:class:`dict`: Default parameters of the Normalizer."""
    normalize_range = (-np.inf, np.inf)
    """:class:`tuple`: Valid range for input data."""
    denormalize_range = (-np.inf, np.inf)
    """:class:`tuple`: Valid range for output/normal data."""
    _dx = 1e-6  # dx for numerical derivative

    def __init__(self, data=None, **parameter):
        # only use parameter, that have a provided default value
        for key, value in self.default_parameter.items():
            setattr(self, key, parameter.get(key, value))
        # fit parameters if data is given
        if data is not None:
            self.fit(data)
        # optimization results
        self._opti = None
        # precision for printing
        self._prec = 3

    def _denormalize(self, data):
        return data

    def _normalize(self, data):
        return data

    def _derivative(self, data):
        return spm.derivative(self._normalize, data, dx=self._dx)

    def _loglikelihood(self, data):
        add = -0.5 * np.size(data) * (np.log(2 * np.pi) + 1)
        return self._kernel_loglikelihood(data) + add

    def _kernel_loglikelihood(self, data):
        res = -0.5 * np.size(data) * np.log(np.var(self._normalize(data)))
        return res + np.sum(np.log(np.maximum(1e-16, self._derivative(data))))

    def _check_input(self, data, data_range=None, return_output_template=True):
        is_data = np.logical_not(np.isnan(data))
        if return_output_template:
            out = np.full_like(data, np.nan, dtype=np.double)
        data = np.asarray(data, dtype=np.double)[is_data]
        if data_range is not None and np.min(np.abs(data_range)) < np.inf:
            dat_in = np.logical_and(data > data_range[0], data < data_range[1])
            if not np.all(dat_in):
                warnings.warn(
                    "{0}: data (min: {1}, max: {2}) out of range: {3}. "
                    "Affected values will be treated as NaN.".format(
                        self.name, np.min(data), np.max(data), data_range
                    )
                )
                is_data[is_data] &= dat_in
                data = data[dat_in]
        if return_output_template:
            return data, is_data, out
        return data

    def denormalize(self, data):
        """Transform to input distribution.

        Parameters
        ----------
        data : array_like
            Input data (normal distributed).

        Returns
        -------
        :class:`numpy.ndarray`
            Denormalized data.
        """
        data, is_data, out = self._check_input(data, self.denormalize_range)
        out[is_data] = self._denormalize(data)
        return out

    def normalize(self, data):
        """Transform to normal distribution.

        Parameters
        ----------
        data : array_like
            Input data (not normal distributed).

        Returns
        -------
        :class:`numpy.ndarray`
            Normalized data.
        """
        data, is_data, out = self._check_input(data, self.normalize_range)
        out[is_data] = self._normalize(data)
        return out

    def derivative(self, data):
        """Factor for normal PDF to gain target PDF.

        Parameters
        ----------
        data : array_like
            Input data (not normal distributed).

        Returns
        -------
        :class:`numpy.ndarray`
            Derivative of the normalization transformation function.
        """
        data, is_data, out = self._check_input(data, self.normalize_range)
        out[is_data] = self._derivative(data)
        return out

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
        data = self._check_input(data, self.normalize_range, False)
        return self._loglikelihood(data)

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
        data = self._check_input(data, self.normalize_range, False)
        return self._kernel_loglikelihood(data)

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
        all_names = sorted(self.default_parameter)
        para_names = [name for name in all_names if name not in skip]

        def _neg_kllf(par, dat):
            for name, val in zip(para_names, np.atleast_1d(par)):
                setattr(self, name, val)
            return -self.kernel_loglikelihood(dat)

        if len(para_names) == 0:  # transformations without para. (no opti.)
            warnings.warn(f"{self.name}.fit: no parameters!")
            return {}
        if len(para_names) == 1:  # one-para. transformations (simple opti.)
            # default bracket like in scipy's boxcox (if not given)
            kwargs.setdefault("bracket", (-2, 2))
            out = spo.minimize_scalar(_neg_kllf, args=(data,), **kwargs)
        else:  # general case
            # init guess from current parameters (if x0 not given)
            kwargs.setdefault("x0", [getattr(self, p) for p in para_names])
            out = spo.minimize(_neg_kllf, args=(data,), **kwargs)
        # save optimization results
        self._opti = out
        for name, val in zip(para_names, np.atleast_1d(out.x)):
            setattr(self, name, val)
        return {name: getattr(self, name) for name in all_names}

    def __eq__(self, other):
        """Compare Normalizers."""
        # check for correct base class
        if type(self) is not type(other):
            return False
        # if base class is same, this is safe
        for val in self.default_parameter:
            if not np.isclose(getattr(self, val), getattr(other, val)):
                return False
        return True

    @property
    def name(self):
        """:class:`str`: The name of the normalizer class."""
        return self.__class__.__name__

    def __repr__(self):
        """Return String representation."""
        para_strs = [
            "{0}={1:.{2}}".format(p, float(getattr(self, p)), self._prec)
            for p in sorted(self.default_parameter)
        ]
        return f"{self.name}({', '.join(para_strs)})"
