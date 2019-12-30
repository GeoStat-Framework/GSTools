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
"""
# pylint: disable=C0103
import numpy as np
from scipy.linalg import inv
from gstools.krige.base import Krige

__all__ = ["Simple", "Ordinary", "Universal", "ExtDrift"]


class Simple(Krige):
    """
    A class for simple kriging.

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

    def __repr__(self):
        """Return String representation."""
        return "Ordinary(model={0}, cond_pos={1}, cond_val={2}".format(
            self.model, self.cond_pos, self.cond_val
        )


class Universal(Krige):
    """
    A class for universal kriging.

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
        res[:size, :size] = self.model.vario_nugget(
            self.get_dists(self.krige_pos)
        )
        if self.unbiased:
            res[size, :] = 1
            res[:, size] = 1
            res[size, size] = 0
        for i, f in enumerate(self.drift_functions):
            drift_tmp = f(*self.cond_pos)
            res[-self.drift_no + i, : self.cond_no] = drift_tmp
            res[: self.cond_no, -self.drift_no + i] = drift_tmp
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
        chunk_pos = pos[: self.model.dim]
        for i in range(self.model.dim):
            chunk_pos[i] = chunk_pos[i][slice(*chunk_slice)]
        for i, f in enumerate(self.drift_functions):
            res[-self.drift_no + i, :] = f(*chunk_pos)
        return res

    @property
    def drift_no(self):
        """:class:`int`: Number of drift functions."""
        return len(self.drift_functions)

    def __repr__(self):
        """Return String representation."""
        return "Universal(model={0}, cond_pos={1}, cond_val={2}".format(
            self.model, self.cond_pos, self.cond_val
        )


class ExtDrift(Krige):
    """
    A class for external drift kriging (EDK).

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
        res[:size, :size] = self.model.vario_nugget(
            self.get_dists(self.krige_pos)
        )
        if self.unbiased:
            res[size, :] = 1
            res[:, size] = 1
            res[size, size] = 0
        res[-self.drift_no :, : self.cond_no] = self.krige_ext_drift
        res[: self.cond_no, -self.drift_no :] = self.krige_ext_drift.T
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
        res[-self.drift_no :, :] = ext_drift[: slice(*chunk_slice)]
        return res

    @property
    def drift_no(self):
        """:class:`int`: Number of drift values per point."""
        return self.krige_ext_drift.shape[0]

    def __repr__(self):
        """Return String representation."""
        return "ExtDrift(model={0}, cond_pos={2}, cond_val={3}".format(
            self.model, self.cond_pos, self.cond_val
        )


if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod()
