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
from gstools.field.tools import make_anisotropic, rotate_mesh
from gstools.tools.geometric import pos2xyz, xyz2pos
from gstools.krige.base import Krige
from gstools.krige.tools import eval_func, no_trend

__all__ = ["Simple", "Ordinary", "Universal", "ExtDrift", "Detrended"]


class Simple(Krige):
    """
    Simple kriging.

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
    trend_function : :any:`callable`, optional
        A callable trend function. Should have the signiture: f(x, [y, z])
        This is used for detrended kriging, where the trended is subtracted
        from the conditions before kriging is applied.
        This can be used for regression kriging, where the trend function
        is determined by an external regression algorithm.
    """

    def __init__(
        self, model, cond_pos, cond_val, mean=0.0, trend_function=None
    ):
        super().__init__(
            model, cond_pos, cond_val, mean=mean, trend_function=trend_function
        )
        self._unbiased = False

    def _get_krige_mat(self):
        """Calculate the inverse matrix of the kriging equation."""
        return inv(self.model.cov_nugget(self._get_dists(self._krige_pos)))

    def _get_krige_vecs(self, pos, chunk_slice=(0, None), ext_drift=None):
        """Calculate the RHS of the kriging equation."""
        return self.model.cov_nugget(
            self._get_dists(self._krige_pos, pos, chunk_slice)
        )

    def _post_field(self, field, krige_var):
        """
        Postprocessing and saving of kriging field and error variance.

        Parameters
        ----------
        field : :class:`numpy.ndarray`
            Raw kriging field.
        krige_var : :class:`numpy.ndarray`
            Raw kriging error variance.
        """
        if self.trend_function is no_trend:
            self.field = field + self.mean
        else:
            self.field = (
                field
                + self.mean
                + eval_func(self.trend_function, self.pos, self.mesh_type)
            )
        # add the given mean
        self.krige_var = self.model.sill - krige_var

    @property
    def _krige_cond(self):
        """:class:`numpy.ndarray`: The prepared kriging conditions."""
        return self.cond_val - self.mean - self.cond_trend

    def __repr__(self):
        """Return String representation."""
        return "Simple(model={0}, cond_pos={1}, cond_val={2}, mean={3})".format(
            self.model, self.cond_pos, self.cond_val, self.mean
        )


class Ordinary(Krige):
    """
    Ordinary kriging.

    Ordinary kriging is used to interpolate data and estimate a proper mean.

    Parameters
    ----------
    model : :any:`CovModel`
        Covariance Model used for kriging.
    cond_pos : :class:`list`
        tuple, containing the given condition positions (x, [y, z])
    cond_val : :class:`numpy.ndarray`
        the values of the conditions
    trend_function : :any:`callable`, optional
        A callable trend function. Should have the signiture: f(x, [y, z])
        This is used for detrended kriging, where the trended is subtracted
        from the conditions before kriging is applied.
        This can be used for regression kriging, where the trend function
        is determined by an external regression algorithm.
    """

    def __init__(self, model, cond_pos, cond_val, trend_function=None):
        super().__init__(
            model, cond_pos, cond_val, trend_function=trend_function
        )

    def _get_krige_mat(self):
        """Calculate the inverse matrix of the kriging equation."""
        size = self.cond_no + int(self.unbiased)
        res = np.empty((size, size), dtype=np.double)
        res[: self.cond_no, : self.cond_no] = self.model.vario_nugget(
            self._get_dists(self._krige_pos)
        )
        if self.unbiased:
            res[self.cond_no, :] = 1
            res[:, self.cond_no] = 1
            res[self.cond_no, self.cond_no] = 0
        return inv(res)

    def _get_krige_vecs(self, pos, chunk_slice=(0, None), ext_drift=None):
        """Calculate the RHS of the kriging equation."""
        chunk_size = len(pos[0]) if chunk_slice[1] is None else chunk_slice[1]
        chunk_size -= chunk_slice[0]
        size = self.cond_no + int(self.unbiased)
        res = np.empty((size, chunk_size), dtype=np.double)
        res[: self.cond_no, :] = self.model.vario_nugget(
            self._get_dists(self._krige_pos, pos, chunk_slice)
        )
        if self.unbiased:
            res[self.cond_no, :] = 1
        return res

    def get_mean(self):
        """Calculate the estimated mean."""
        mean_est = np.concatenate(
            (np.full_like(self.cond_val, self.model.sill), [1])
        )
        return np.einsum("i,ij,j", self._krige_cond, self._krige_mat, mean_est)

    def __repr__(self):
        """Return String representation."""
        return "Ordinary(model={0}, cond_pos={1}, cond_val={2}".format(
            self.model, self.cond_pos, self.cond_val
        )


class Universal(Krige):
    """
    Universal kriging.

    Universal kriging is used to interpolate given data with a variable mean,
    that is determined by a functional drift.

    This estimator is set to be unbiased by default.
    This means, that the weights in the kriging equation sum up to 1.
    Consequently no constant function needs to be given for a constant drift,
    since the unbiased condition is applied to all given drift functions.

    Parameters
    ----------
    model : :any:`CovModel`
        Covariance Model used for kriging.
    cond_pos : :class:`list`
        tuple, containing the given condition positions (x, [y, z])
    cond_val : :class:`numpy.ndarray`
        the values of the conditions
    drift_functions : :class:`list` of :any:`callable`, :class:`str` or :class:`int`
        Either a list of callable functions, an integer representing
        the polynomial order of the drift or one of the following strings:

            * "linear" : regional linear drift (equals order=1)
            * "quadratic" : regional quadratic drift (equals order=2)

    trend_function : :any:`callable`, optional
        A callable trend function. Should have the signiture: f(x, [y, z])
        This is used for detrended kriging, where the trended is subtracted
        from the conditions before kriging is applied.
        This can be used for regression kriging, where the trend function
        is determined by an external regression algorithm.
    """

    def __init__(
        self, model, cond_pos, cond_val, drift_functions, trend_function=None
    ):
        super().__init__(
            model,
            cond_pos,
            cond_val,
            drift_functions=drift_functions,
            trend_function=trend_function,
        )

    def _get_krige_mat(self):
        """Calculate the inverse matrix of the kriging equation."""
        size = self.cond_no + int(self.unbiased) + self.drift_no
        res = np.empty((size, size), dtype=np.double)
        res[: self.cond_no, : self.cond_no] = self.model.vario_nugget(
            self._get_dists(self._krige_pos)
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

    def _get_krige_vecs(self, pos, chunk_slice=(0, None), ext_drift=None):
        """Calculate the RHS of the kriging equation."""
        chunk_size = len(pos[0]) if chunk_slice[1] is None else chunk_slice[1]
        chunk_size -= chunk_slice[0]
        size = self.cond_no + int(self.unbiased) + self.drift_no
        res = np.empty((size, chunk_size), dtype=np.double)
        res[: self.cond_no, :] = self.model.vario_nugget(
            self._get_dists(self._krige_pos, pos, chunk_slice)
        )
        if self.unbiased:
            res[self.cond_no, :] = 1
        # trend function need the anisotropic and rotated positions
        if not self.model.is_isotropic:
            x, y, z = pos2xyz(pos, max_dim=self.model.dim)
            y, z = make_anisotropic(self.model.dim, self.model.anis, y, z)
            if self.model.do_rotation:
                x, y, z = rotate_mesh(
                    self.model.dim, self.model.angles, x, y, z
                )
            pos = xyz2pos(x, y, z, max_dim=self.model.dim)
        chunk_pos = list(pos[: self.model.dim])
        for i in range(self.model.dim):
            chunk_pos[i] = chunk_pos[i][slice(*chunk_slice)]
        for i, f in enumerate(self.drift_functions):
            res[-self.drift_no + i, :] = f(*chunk_pos)
        return res

    def __repr__(self):
        """Return String representation."""
        return "Universal(model={0}, cond_pos={1}, cond_val={2})".format(
            self.model, self.cond_pos, self.cond_val
        )


class ExtDrift(Krige):
    """
    External drift kriging (EDK).

    External drift kriging is used to interpolate given data
    with a variable mean, that is determined by an external drift.

    This estimator is set to be unbiased by default.
    This means, that the weights in the kriging equation sum up to 1.
    Consequently no constant external drift needs to be given to estimate
    a proper mean.

    Parameters
    ----------
    model : :any:`CovModel`
        Covariance Model used for kriging.
    cond_pos : :class:`list`
        tuple, containing the given condition positions (x, [y, z])
    cond_val : :class:`numpy.ndarray`
        the values of the conditions
    ext_drift : :class:`numpy.ndarray`
        the external drift values at the given condition positions.
    trend_function : :any:`callable`, optional
        A callable trend function. Should have the signiture: f(x, [y, z])
        This is used for detrended kriging, where the trended is subtracted
        from the conditions before kriging is applied.
        This can be used for regression kriging, where the trend function
        is determined by an external regression algorithm.
    """

    def __init__(
        self, model, cond_pos, cond_val, ext_drift, trend_function=None
    ):
        super().__init__(
            model,
            cond_pos,
            cond_val,
            ext_drift=ext_drift,
            trend_function=trend_function,
        )

    def _get_krige_mat(self):
        """Calculate the inverse matrix of the kriging equation."""
        size = self.cond_no + int(self.unbiased) + self.drift_no
        res = np.empty((size, size), dtype=np.double)
        res[: self.cond_no, : self.cond_no] = self.model.vario_nugget(
            self._get_dists(self._krige_pos)
        )
        if self.unbiased:
            res[self.cond_no, : self.cond_no] = 1
            res[: self.cond_no, self.cond_no] = 1
        res[-self.drift_no :, : self.cond_no] = self.cond_ext_drift
        res[: self.cond_no, -self.drift_no :] = self.cond_ext_drift.T
        res[self.cond_no :, self.cond_no :] = 0
        return inv(res)

    def _get_krige_vecs(self, pos, chunk_slice=(0, None), ext_drift=None):
        """Calculate the RHS of the kriging equation."""
        chunk_size = len(pos[0]) if chunk_slice[1] is None else chunk_slice[1]
        chunk_size -= chunk_slice[0]
        size = self.cond_no + int(self.unbiased) + self.drift_no
        res = np.empty((size, chunk_size), dtype=np.double)
        res[: self.cond_no, :] = self.model.vario_nugget(
            self._get_dists(self._krige_pos, pos, chunk_slice)
        )
        if self.unbiased:
            res[self.cond_no, :] = 1
        res[-self.drift_no :, :] = ext_drift[:, slice(*chunk_slice)]
        return res

    def __repr__(self):
        """Return String representation."""
        return "ExtDrift(model={0}, cond_pos={1}, cond_val={2})".format(
            self.model, self.cond_pos, self.cond_val
        )


class Detrended(Simple):
    """
    Detrended simple kriging.

    In detrended kriging, the data is detrended before interpolation by
    simple kriging with zero mean.

    The trend needs to be a callable function the user has to provide.
    This can be used for regression kriging, where the trend function
    is determined by an external regression algorithm.

    This is just a shortcut for simple kriging with a given trend function
    and zero mean. A trend can be given with EVERY provided kriging routine.

    Parameters
    ----------
    model : :any:`CovModel`
        Covariance Model used for kriging.
    cond_pos : :class:`list`
        tuple, containing the given condition positions (x, [y, z])
    cond_val : :class:`numpy.ndarray`
        the values of the conditions
    trend_function : :any:`callable`
        The callable trend function. Should have the signiture: f(x, [y, z])
    """

    def __init__(self, model, cond_pos, cond_val, trend_function):
        super().__init__(
            model, cond_pos, cond_val, trend_function=trend_function
        )

    def __repr__(self):
        """Return String representation."""
        return "Detrended(model={0} cond_pos={1}, cond_val={2})".format(
            self.model, self.cond_pos, self.cond_val
        )


if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod()
