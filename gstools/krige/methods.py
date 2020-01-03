# -*- coding: utf-8 -*-
"""
GStools subpackage providing a class for simple kriging.

.. currentmodule:: gstools.krige.methods

The following classes are provided

.. autosummary::
   Simple
   Ordinary
   Universal
   ExtDrift
   Detrended
"""
# pylint: disable=C0103
import numpy as np
from scipy.linalg import inv
from gstools.krige.base import Krige
from gstools.krige.tools import eval_func

__all__ = ["Simple", "Ordinary", "Universal", "ExtDrift", "Detrended"]


class Simple(Krige):
    """
    A class for simple kriging.

    Simple kriging is used to interpolate data with a given mean.

    Parameters
    ----------
    model : :any:`CovModel`
        Covariance Model used for kriging.
    cond_pos : :class:`list`
        tuple, containing the given condition positions (x, [y, z])
    cond_val : :class:`numpy.ndarray`
        the values of the conditions
    mean : :class:`float`, optional
        mean value of the kriging field
    """

    def __init__(self, model, cond_pos, cond_val, mean=0.0):
        super().__init__(model, cond_pos, cond_val, mean=mean)
        self._unbiased = False

    def get_krige_mat(self):
        """Update the kriging model settings."""
        return inv(self.model.cov_nugget(self.get_dists(self.krige_pos)))

    def get_krige_vecs(self, pos, chunk_slice=(0, None), ext_drift=None):
        """Calculate the RHS of the kriging equation."""
        return self.model.cov_nugget(
            self.get_dists(self.krige_pos, pos, chunk_slice)
        )

    def post_field(self, field, krige_var):
        """
        Postprocessing and saving of kriging field and error variance.

        Parameters
        ----------
        field : :class:`numpy.ndarray`
            Raw kriging field.
        krige_var : :class:`numpy.ndarray`
            Raw kriging error variance.
        """
        # add the given mean
        self.field = field + self.mean
        self.krige_var = self.model.sill - krige_var

    @property
    def krige_cond(self):
        """:class:`numpy.ndarray`: The prepared kriging conditions."""
        return self.cond_val - self.mean

    def __repr__(self):
        """Return String representation."""
        return "Simple(model={0}, mean={1}, cond_pos={2}, cond_val={3}".format(
            self.model, self.mean, self.cond_pos, self.cond_val
        )


class Ordinary(Krige):
    """
    A class for ordinary kriging.

    Ordinary kriging is used to estimate a constant mean from the given data.

    Parameters
    ----------
    model : :any:`CovModel`
        Covariance Model used for kriging.
    cond_pos : :class:`list`
        tuple, containing the given condition positions (x, [y, z])
    cond_val : :class:`numpy.ndarray`
        the values of the conditions
    """

    def __init__(self, model, cond_pos, cond_val):
        super().__init__(model, cond_pos, cond_val)

    def get_krige_mat(self):
        """Update the kriging model settings."""
        size = self.cond_no + int(self.unbiased)
        res = np.empty((size, size), dtype=np.double)
        res[: self.cond_no, : self.cond_no] = self.model.vario_nugget(
            self.get_dists(self.krige_pos)
        )
        if self.unbiased:
            res[self.cond_no, :] = 1
            res[:, self.cond_no] = 1
            res[self.cond_no, self.cond_no] = 0
        return inv(res)

    def get_krige_vecs(self, pos, chunk_slice=(0, None), ext_drift=None):
        """Calculate the RHS of the kriging equation."""
        chunk_size = len(pos[0]) if chunk_slice[1] is None else chunk_slice[1]
        chunk_size -= chunk_slice[0]
        size = self.cond_no + int(self.unbiased)
        res = np.empty((size, chunk_size), dtype=np.double)
        res[: self.cond_no, :] = self.model.vario_nugget(
            self.get_dists(self.krige_pos, pos, chunk_slice)
        )
        if self.unbiased:
            res[self.cond_no, :] = 1
        return res

    def get_mean(self):
        """Calculate the estimated mean."""
        mean_est = np.concatenate(
            (np.full_like(self.cond_val, self.model.sill), [1])
        )
        return np.einsum("i,ij,j", self.krige_cond, self.krige_mat, mean_est)

    def __repr__(self):
        """Return String representation."""
        return "Ordinary(model={0}, cond_pos={1}, cond_val={2}".format(
            self.model, self.cond_pos, self.cond_val
        )


class Universal(Krige):
    """
    A class for universal kriging.

    Universal kriging is used to interpolate given data with a variable mean,
    that is determined by a functional drift.

    Parameters
    ----------
    model : :any:`CovModel`
        Covariance Model used for kriging.
    cond_pos : :class:`list`
        tuple, containing the given condition positions (x, [y, z])
    cond_val : :class:`numpy.ndarray`
        the values of the conditions
    drift_functions : :class:`list` of :any:`callable` or :class:`str`
        Either a list of callable functions or one of the following strings:

            * "linear" : regional linear drift
            * "quadratic" : regional quadratic drift
    """

    def __init__(self, model, cond_pos, cond_val, drift_functions):
        super().__init__(
            model, cond_pos, cond_val, drift_functions=drift_functions
        )

    def get_krige_mat(self):
        """Update the kriging model settings."""
        size = self.cond_no + int(self.unbiased) + self.drift_no
        res = np.empty((size, size), dtype=np.double)
        res[: self.cond_no, : self.cond_no] = self.model.vario_nugget(
            self.get_dists(self.krige_pos)
        )
        if self.unbiased:
            res[self.cond_no, : self.cond_no] = 1
            res[: self.cond_no, self.cond_no] = 1
        for i, f in enumerate(self.drift_functions):
            drift_tmp = f(*self.cond_pos)
            res[-self.drift_no + i, : self.cond_no] = drift_tmp
            res[: self.cond_no, -self.drift_no + i] = drift_tmp
        res[self.cond_no :, self.cond_no :] = 0
        return inv(res)

    def get_krige_vecs(self, pos, chunk_slice=(0, None), ext_drift=None):
        """Calculate the RHS of the kriging equation."""
        chunk_size = len(pos[0]) if chunk_slice[1] is None else chunk_slice[1]
        chunk_size -= chunk_slice[0]
        size = self.cond_no + int(self.unbiased) + self.drift_no
        res = np.empty((size, chunk_size), dtype=np.double)
        res[: self.cond_no, :] = self.model.vario_nugget(
            self.get_dists(self.krige_pos, pos, chunk_slice)
        )
        if self.unbiased:
            res[self.cond_no, :] = 1
        chunk_pos = list(pos[: self.model.dim])
        for i in range(self.model.dim):
            chunk_pos[i] = chunk_pos[i][slice(*chunk_slice)]
        for i, f in enumerate(self.drift_functions):
            res[-self.drift_no + i, :] = f(*chunk_pos)
        return res

    def __repr__(self):
        """Return String representation."""
        return "Universal(model={0}, cond_pos={1}, cond_val={2}".format(
            self.model, self.cond_pos, self.cond_val
        )


class ExtDrift(Krige):
    """
    A class for external drift kriging (EDK).

    Universal kriging is used to interpolate given data with a variable mean,
    that is determined by an external drift.

    Parameters
    ----------
    model : :any:`CovModel`
        Covariance Model used for kriging.
    cond_pos : :class:`list`
        tuple, containing the given condition positions (x, [y, z])
    cond_val : :class:`numpy.ndarray`
        the values of the conditions
    ext_drift : :class:`numpy.ndarray`
        the external drift values at the given cond. positions (only for EDK)
    """

    def __init__(self, model, cond_pos, cond_val, ext_drift):
        super().__init__(model, cond_pos, cond_val, ext_drift=ext_drift)

    def get_krige_mat(self):
        """Update the kriging model settings."""
        size = self.cond_no + int(self.unbiased) + self.drift_no
        res = np.empty((size, size), dtype=np.double)
        res[: self.cond_no, : self.cond_no] = self.model.vario_nugget(
            self.get_dists(self.krige_pos)
        )
        if self.unbiased:
            res[self.cond_no, : self.cond_no] = 1
            res[: self.cond_no, self.cond_no] = 1
        res[-self.drift_no :, : self.cond_no] = self.krige_ext_drift
        res[: self.cond_no, -self.drift_no :] = self.krige_ext_drift.T
        res[self.cond_no :, self.cond_no :] = 0
        return inv(res)

    def get_krige_vecs(self, pos, chunk_slice=(0, None), ext_drift=None):
        """Calculate the RHS of the kriging equation."""
        chunk_size = len(pos[0]) if chunk_slice[1] is None else chunk_slice[1]
        chunk_size -= chunk_slice[0]
        size = self.cond_no + int(self.unbiased) + self.drift_no
        res = np.empty((size, chunk_size), dtype=np.double)
        res[: self.cond_no, :] = self.model.vario_nugget(
            self.get_dists(self.krige_pos, pos, chunk_slice)
        )
        if self.unbiased:
            res[self.cond_no, :] = 1
        res[-self.drift_no :, :] = ext_drift[:, slice(*chunk_slice)]
        return res

    def __repr__(self):
        """Return String representation."""
        return "ExtDrift(model={0}, cond_pos={1}, cond_val={2}".format(
            self.model, self.cond_pos, self.cond_val
        )


class Detrended(Simple):
    """
    A class for detrended kriging.

    In detrended kriging, the data is detrended before interpolation.
    The trend needs to be a callable function the user has to provide.
    This can be used for regression kriging.

    Parameters
    ----------
    model : :any:`CovModel`
        Covariance Model used for kriging.
    cond_pos : :class:`list`
        tuple, containing the given condition positions (x, [y, z])
    cond_val : :class:`numpy.ndarray`
        the values of the conditions
    func : :any:`callable`
        The callable trend function. Should have the signiture: f(x, [y, z])
    """

    def __init__(self, model, cond_pos, cond_val, trend_function):
        self._krige_trend = None
        self._trend_function = None
        self.trend_function = trend_function
        super().__init__(model, cond_pos, cond_val, mean=0.0)

    def update_model(self):
        """Update the kriging model settings."""
        x, y, z, __, __, __, __ = self.pre_pos(self.cond_pos)
        self._krige_pos = (x, y, z)[: self.model.dim]
        self._krige_mat = self.get_krige_mat()
        self._krige_trend = self.trend_function(*self._krige_pos)

    def post_field(self, field, krige_var):
        """
        Postprocessing and saving of kriging field and error variance.

        Parameters
        ----------
        field : :class:`numpy.ndarray`
            Raw kriging field.
        krige_var : :class:`numpy.ndarray`
            Raw kriging error variance.
        """
        # add the given mean
        self.field = field + eval_func(
            self.trend_function, self.pos, self.mesh_type
        )
        self.krige_var = self.model.sill - krige_var

    @property
    def krige_cond(self):
        """:class:`numpy.ndarray`: The prepared kriging conditions."""
        return self.cond_val - self.krige_trend

    @property
    def krige_trend(self):
        """:class:`numpy.ndarray`: Trend at the conditions."""
        return self._krige_trend

    @property
    def trend_function(self):
        """:any:`callable`: The trend function."""
        return self._trend_function

    @trend_function.setter
    def trend_function(self, trend_function):
        if not callable(trend_function):
            raise ValueError("Detrended kriging: trend function not callable.")
        self._trend_function = trend_function

    def __repr__(self):
        """Return String representation."""
        return "Detrended(model={0} cond_pos={1}, cond_val={2}".format(
            self.model, self.cond_pos, self.cond_val
        )


if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod()
