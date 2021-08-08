# -*- coding: utf-8 -*-
"""
GStools subpackage providing a base class for kriging.

.. currentmodule:: gstools.krige.base

The following classes are provided

.. autosummary::
   Krige
"""
# pylint: disable=C0103, W0221, E1102, R0201
import collections
import numpy as np

from scipy.spatial.distance import cdist
import scipy.linalg as spl
from gstools.field.base import Field
from gstools.krige.krigesum import (
    calc_field_krige_and_variance,
    calc_field_krige,
)
from gstools.krige.tools import set_condition, get_drift_functions
from gstools.tools.misc import eval_func
from gstools.tools.geometric import rotated_main_axes
from gstools.variogram import vario_estimate

__all__ = ["Krige"]


P_INV = {"pinv": spl.pinv, "pinv2": spl.pinv2, "pinvh": spl.pinvh}
"""dict: Standard pseudo-inverse routines"""


class Krige(Field):
    """
    A Swiss Army knife for kriging.

    A Kriging class enabling the basic kriging routines:
    Simple-, Ordinary-, Univseral-, External Drift-
    and detrended/regression-Kriging as well as
    Kriging the Mean [Wackernagel2003]_.

    Parameters
    ----------
    model : :any:`CovModel`
        Covariance Model used for kriging.
    cond_pos : :class:`list`
        tuple, containing the given condition positions (x, [y, z])
    cond_val : :class:`numpy.ndarray`
        the values of the conditions (nan values will be ignored)
    drift_functions : :class:`list` of :any:`callable`, :class:`str` or :class:`int`
        Either a list of callable functions, an integer representing
        the polynomial order of the drift or one of the following strings:

            * "linear" : regional linear drift (equals order=1)
            * "quadratic" : regional quadratic drift (equals order=2)

    ext_drift : :class:`numpy.ndarray` or :any:`None`, optional
        the external drift values at the given cond. positions.
    mean : :class:`float`, optional
        mean value used to shift normalized conditioning data.
        Could also be a callable. The default is None.
    normalizer : :any:`None` or :any:`Normalizer`, optional
        Normalizer to be applied to the input data to gain normality.
        The default is None.
    trend : :any:`None` or :class:`float` or :any:`callable`, optional
        A callable trend function. Should have the signiture: f(x, [y, z, ...])
        This is used for detrended kriging, where the trended is subtracted
        from the conditions before kriging is applied.
        This can be used for regression kriging, where the trend function
        is determined by an external regression algorithm.
        If no normalizer is applied, this behaves equal to 'mean'.
        The default is None.
    unbiased : :class:`bool`, optional
        Whether the kriging weights should sum up to 1, so the estimator
        is unbiased. If unbiased is `False` and no drifts are given,
        this results in simple kriging.
        Default: True
    exact : :class:`bool`, optional
        Whether the interpolator should reproduce the exact input values.
        If `False`, `cond_err` is interpreted as measurement error
        at the conditioning points and the result will be more smooth.
        Default: False
    cond_err : :class:`str`, :class :class:`float` or :class:`list`, optional
        The measurement error at the conditioning points.
        Either "nugget" to apply the model-nugget, a single value applied to
        all points or an array with individual values for each point.
        The "exact=True" variant only works with "cond_err='nugget'".
        Default: "nugget"
    pseudo_inv : :class:`bool`, optional
        Whether the kriging system is solved with the pseudo inverted
        kriging matrix. If `True`, this leads to more numerical stability
        and redundant points are averaged. But it can take more time.
        Default: True
    pseudo_inv_type : :class:`str` or :any:`callable`, optional
        Here you can select the algorithm to compute the pseudo-inverse matrix:

            * `"pinv"`: use `pinv` from `scipy` which uses `lstsq`
            * `"pinv2"`: use `pinv2` from `scipy` which uses `SVD`
            * `"pinvh"`: use `pinvh` from `scipy` which uses eigen-values

        If you want to use another routine to invert the kriging matrix,
        you can pass a callable which takes a matrix and returns the inverse.
        Default: `"pinv"`
    fit_normalizer : :class:`bool`, optional
        Wheater to fit the data-normalizer to the given conditioning data.
        Default: False
    fit_variogram : :class:`bool`, optional
        Wheater to fit the given variogram model to the data.
        This is done by using isotropy settings of the given model,
        assuming the sill to be the data variance and with the
        standard bins provided by the :any:`standard_bins` routine.
        Default: False

    Notes
    -----
    If you have changed any properties in the class, you can update the kriging
    setup by calling :any:`Krige.set_condition` without any arguments.

    References
    ----------
    .. [Wackernagel2003] Wackernagel, H.,
           "Multivariate geostatistics",
           Springer, Berlin, Heidelberg (2003)
    """

    default_field_names = ["field", "krige_var", "mean_field"]
    """:class:`list`: Default field names."""

    def __init__(
        self,
        model,
        cond_pos,
        cond_val,
        drift_functions=None,
        ext_drift=None,
        mean=None,
        normalizer=None,
        trend=None,
        unbiased=True,
        exact=False,
        cond_err="nugget",
        pseudo_inv=True,
        pseudo_inv_type="pinv",
        fit_normalizer=False,
        fit_variogram=False,
    ):
        super().__init__(model, mean=mean, normalizer=normalizer, trend=trend)
        self._unbiased = bool(unbiased)
        self._exact = bool(exact)
        self._pseudo_inv = bool(pseudo_inv)
        self._pseudo_inv_type = None
        self.pseudo_inv_type = pseudo_inv_type
        # initialize private attributes
        self._cond_pos = None
        self._cond_val = None
        self._cond_err = None
        self._krige_mat = None
        self._krige_pos = None
        self._cond_trend = None
        self._cond_ext_drift = np.array([])
        self._drift_functions = None
        self.set_drift_functions(drift_functions)
        self.set_condition(
            cond_pos,
            cond_val,
            ext_drift,
            cond_err,
            fit_normalizer,
            fit_variogram,
        )

    def __call__(
        self,
        pos=None,
        mesh_type="unstructured",
        ext_drift=None,
        chunk_size=None,
        only_mean=False,
        return_var=True,
        post_process=True,
        store=True,
    ):
        """
        Generate the kriging field.

        The field is saved as `self.field` and is also returned.
        The error variance is saved as `self.krige_var` and is also returned.

        Parameters
        ----------
        pos : :class:`list`, optional
            the position tuple, containing main direction and transversal
            directions (x, [y, z])
        mesh_type : :class:`str`, optional
            'structured' / 'unstructured'
        ext_drift : :class:`numpy.ndarray` or :any:`None`, optional
            the external drift values at the given positions (only for EDK)
        chunk_size : :class:`int`, optional
            Chunk size to cut down the size of the kriging system to prevent
            memory errors.
            Default: None
        only_mean : :class:`bool`, optional
            Whether to only calculate the mean of the kriging field.
            Default: `False`
        return_var : :class:`bool`, optional
            Whether to return the variance along with the field.
            Default: `True`
        post_process : :class:`bool`, optional
            Whether to apply mean, normalizer and trend to the field.
            Default: `True`
        store : :class:`str` or :class:`bool` or :class:`list`, optional
            Whether to store kriging fields (True/False) with default name
            or with specified names.
            The default is :any:`True` for default names
            ["field", "krige_var"] or "mean_field" if `only_mean=True`.

        Returns
        -------
        field : :class:`numpy.ndarray`
            the kriged field or mean_field
        krige_var : :class:`numpy.ndarray`, optional
            the kriging error variance
            (if return_var is True and only_mean is False)
        """
        return_var &= not only_mean  # don't return variance when calc. mean
        fld_cnt = 2 if return_var else 1
        default = self.default_field_names[2] if only_mean else None
        name, save = self.get_store_config(store, default, fld_cnt)

        iso_pos, shape = self.pre_pos(pos, mesh_type)
        pnt_cnt = len(iso_pos[0])

        field = np.empty(pnt_cnt, dtype=np.double)
        krige_var = np.empty(pnt_cnt, dtype=np.double) if return_var else None
        # set constant mean if present and wanted
        if only_mean and self.drift_no == 0:
            field[...] = self.get_mean(post_process=False)
        # execute the kriging routine
        else:
            # set chunk size
            chunk_size = pnt_cnt if chunk_size is None else int(chunk_size)
            chunk_no = int(np.ceil(pnt_cnt / chunk_size))
            ext_drift = self._pre_ext_drift(pnt_cnt, ext_drift)
            # iterate chunks
            for i in range(chunk_no):
                # get chunk slice for actual chunk
                chunk_slice = (
                    i * chunk_size,
                    min(pnt_cnt, (i + 1) * chunk_size),
                )
                c_slice = slice(*chunk_slice)
                # get RHS of the kriging system
                k_vec = self._get_krige_vecs(
                    iso_pos, chunk_slice, ext_drift, only_mean
                )
                # generate the raw kriging field and error variance
                self._summate(field, krige_var, c_slice, k_vec, return_var)
        # reshape field if we got a structured mesh
        field = np.reshape(field, shape)
        # save field to class
        field = self.post_field(field, name[0], post_process, save[0])
        if return_var:  # care about the estimated error variance
            krige_var = np.reshape(
                np.maximum(self.model.sill - krige_var, 0), shape
            )
            krige_var = self.post_field(krige_var, name[1], False, save[1])
            return field, krige_var
        return field

    def _summate(self, field, krige_var, c_slice, k_vec, return_var):
        if return_var:  # estimate error variance
            field[c_slice], krige_var[c_slice] = calc_field_krige_and_variance(
                self._krige_mat, k_vec, self._krige_cond
            )
        else:  # solely calculate the interpolated field
            field[c_slice] = calc_field_krige(
                self._krige_mat, k_vec, self._krige_cond
            )

    def _inv(self, mat):
        # return pseudo-inverted matrix if wanted (numerically more stable)
        if self.pseudo_inv:
            # if the given type is a callable, call it
            if callable(self.pseudo_inv_type):
                return self.pseudo_inv_type(mat)
            # use the selected method to compute the pseudo-inverse matrix
            return P_INV[self.pseudo_inv_type](mat)
        # if no pseudo-inverse is wanted, calculate the real inverse
        return spl.inv(mat)

    def _get_krige_mat(self):
        """Calculate the inverse matrix of the kriging equation."""
        res = np.empty((self.krige_size, self.krige_size), dtype=np.double)
        # fill the kriging matrix with the covariance
        res[: self.cond_no, : self.cond_no] = self.model.covariance(
            self._get_dists(self._krige_pos)
        )
        # apply the measurement error (nugget by default)
        res[np.diag_indices(self.cond_no)] += self.cond_err
        # set unbias condition (weights have to sum up to 1)
        if self.unbiased:
            res[self.cond_no, : self.cond_no] = 1
            res[: self.cond_no, self.cond_no] = 1
        # set functional drift terms
        for i, f in enumerate(self.drift_functions):
            drift_tmp = f(*self.cond_pos)
            res[-self.drift_no + i, : self.cond_no] = drift_tmp
            res[: self.cond_no, -self.drift_no + i] = drift_tmp
        # set external drift terms
        if self.ext_drift_no > 0:
            ext_size = self.krige_size - self.ext_drift_no
            res[ext_size:, : self.cond_no] = self.cond_ext_drift
            res[: self.cond_no, ext_size:] = self.cond_ext_drift.T
        # set lower right part of the matrix to 0
        res[self.cond_no :, self.cond_no :] = 0
        return self._inv(res)

    def _get_krige_vecs(
        self, pos, chunk_slice=(0, None), ext_drift=None, only_mean=False
    ):
        """Calculate the RHS of the kriging equation."""
        # determine the chunk size
        chunk_size = len(pos[0]) if chunk_slice[1] is None else chunk_slice[1]
        chunk_size -= chunk_slice[0]
        res = np.empty((self.krige_size, chunk_size), dtype=np.double)
        if only_mean:
            # set points to limit of the covariance to only get the mean
            res[: self.cond_no, :] = 0
        else:
            # get correct covarinace functions (depending on exact values)
            cf = self.model.cov_nugget if self.exact else self.model.covariance
            res[: self.cond_no, :] = cf(
                self._get_dists(self._krige_pos, pos, chunk_slice)
            )
        # apply the unbiased condition
        if self.unbiased:
            res[self.cond_no, :] = 1
        # drift function need the anisotropic and rotated positions
        if self.int_drift_no > 0:
            chunk_pos = self.model.anisometrize(pos)[:, slice(*chunk_slice)]
        # apply functional drift
        for i, f in enumerate(self.drift_functions):
            res[-self.drift_no + i, :] = f(*chunk_pos)
        # apply external drift
        if self.ext_drift_no > 0:
            ext_size = self.krige_size - self.ext_drift_no
            res[ext_size:, :] = ext_drift[:, slice(*chunk_slice)]
        return res

    def _pre_ext_drift(self, pnt_cnt, ext_drift=None, set_cond=False):
        """
        Preprocessor for external drifts.

        Parameters
        ----------
        pnt_cnt : :class:`numpy.ndarray`
            Number of points of the mesh.
        ext_drift : :class:`numpy.ndarray` or :any:`None`, optional
            the external drift values at the given positions (only for EDK)
            For multiple external drifts, the first dimension
            should be the index of the drift term.
        set_cond : :class:`bool`, optional
            State if the given external drift is set for the conditioning
            points. Default: False

        Returns
        -------
        ext_drift : :class:`numpy.ndarray` or :any:`None`
            the drift values at the given positions
        """
        if ext_drift is not None:
            ext_drift = np.array(
                ext_drift, dtype=np.double, ndmin=2, copy=False
            )
            if ext_drift.size == 0:  # treat empty array as no ext_drift
                return np.array([])
            if set_cond:
                if len(ext_drift.shape) > 2 or ext_drift.shape[1] != pnt_cnt:
                    raise ValueError("Krige: wrong number of ext. drifts.")
                return ext_drift
            ext_shape = np.shape(ext_drift)
            shape = (self.ext_drift_no, pnt_cnt)
            if self.drift_no > 1 and ext_shape[0] != self.ext_drift_no:
                raise ValueError("Krige: wrong number of external drifts.")
            if np.prod(ext_shape) != np.prod(shape):
                raise ValueError("Krige: wrong number of ext. drift values.")
            return np.asarray(ext_drift, dtype=np.double).reshape(shape)
        if not set_cond and self._cond_ext_drift.size > 0:
            raise ValueError("Krige: wrong number of ext. drift values.")
        return np.array([])

    def _get_dists(self, pos1, pos2=None, pos2_slice=(0, None)):
        """
        Calculate pairwise distances.

        Parameters
        ----------
        pos1 : :class:`tuple` of :class:`numpy.ndarray`
            the first position tuple
        pos2 : :class:`tuple` of :class:`numpy.ndarray`, optional
            the second position tuple. If none, the first one is taken.
        pos2_slice : :class:`tuple` of :class:`int`, optional
            Start and stop of slice for the pos2 array. Default: all values.

        Returns
        -------
        :class:`numpy.ndarray`
            Matrix containing the pairwise distances.
        """
        if pos2 is None:
            return cdist(pos1.T, pos1.T)
        return cdist(pos1.T, pos2.T[slice(*pos2_slice), ...])

    def get_mean(self, post_process=True):
        """Calculate the estimated mean of the detrended field.

        Parameters
        ----------
        post_process : :class:`bool`, optional
            Whether to apply field-mean and normalizer.
            Default: `True`

        Returns
        -------
        mean : :class:`float` or :any:`None`
            Mean of the Kriging System.

        Notes
        -----
        Only not ``None`` if the Kriging System has a constant mean.
        This means, no drift is given and the given field-mean is constant.
        The result is neglecting a potential given trend.
        """
        # if there are drift-terms, no constant mean can be calculated -> None
        # if mean should not be post-processed, it exists when no drift given
        if not self.has_const_mean and (post_process or self.drift_no > 0):
            return None
        res = 0.0  # for simple kriging return the given mean
        # correctly setting given mean
        mean = 0.0 if self.mean is None else self.mean
        # for ordinary kriging return the estimated mean
        if self.unbiased:
            # set the right side of the kriging system to the limit of cov.
            mean_est = np.concatenate((np.full_like(self.cond_val, 0.0), [1]))
            # execute the kriging routine with einsum
            res = np.einsum(
                "i,ij,j", self._krige_cond, self._krige_mat, mean_est
            )
        return self.normalizer.denormalize(res + mean) if post_process else res

    def set_condition(
        self,
        cond_pos=None,
        cond_val=None,
        ext_drift=None,
        cond_err=None,
        fit_normalizer=False,
        fit_variogram=False,
    ):
        """Set the conditions for kriging.

        This method could also be used to update the kriging setup, when
        properties were changed. Then you can call it without arguments.

        Parameters
        ----------
        cond_pos : :class:`list`, optional
            the position tuple of the conditions (x, [y, z]). Default: current.
        cond_val : :class:`numpy.ndarray`, optional
            the values of the conditions (nan values will be ignored).
            Default: current.
        ext_drift : :class:`numpy.ndarray` or :any:`None`, optional
            the external drift values at the given conditions (only for EDK)
            For multiple external drifts, the first dimension
            should be the index of the drift term. When passing `None`, the
            extisting external drift will be used.
        cond_err : :class:`str`, :class :class:`float`, :class:`list`, optional
            The measurement error at the conditioning points.
            Either "nugget" to apply the model-nugget, a single value applied
            to all points or an array with individual values for each point.
            The measurement error has to be <= nugget.
            The "exact=True" variant only works with "cond_err='nugget'".
            Default: "nugget"
        fit_normalizer : :class:`bool`, optional
            Wheater to fit the data-normalizer to the given conditioning data.
            Default: False
        fit_variogram : :class:`bool`, optional
            Wheater to fit the given variogram model to the data.
            This is done by using isotropy settings of the given model,
            assuming the sill to be the data variance and with the
            standard bins provided by the :any:`standard_bins` routine.
            Default: False
        """
        # only use existing external drift, if no new positions are given
        ext_drift = (
            self._cond_ext_drift
            if (ext_drift is None and cond_pos is None)
            else ext_drift
        )
        # use existing values or set default
        cond_pos = self._cond_pos if cond_pos is None else cond_pos
        cond_val = self._cond_val if cond_val is None else cond_val
        cond_err = self._cond_err if cond_err is None else cond_err
        cond_err = "nugget" if cond_err is None else cond_err  # default
        if cond_pos is None or cond_val is None:
            raise ValueError("Krige.set_condition: missing cond_pos/cond_val.")
        # correctly format cond_pos and cond_val
        self._cond_pos, self._cond_val = set_condition(
            cond_pos, cond_val, self.dim
        )
        if fit_normalizer:  # fit normalizer to detrended data
            self.normalizer.fit(self.cond_val - self.cond_trend)
        if fit_variogram:  # fitting model to empirical variogram of data
            # normalize field
            field = self.normalizer.normalize(self.cond_val - self.cond_trend)
            field -= self.cond_mean
            sill = np.var(field)
            if self.model.is_isotropic:
                emp_vario = vario_estimate(
                    self.cond_pos, field, latlon=self.model.latlon
                )
            else:
                axes = rotated_main_axes(self.model.dim, self.model.angles)
                emp_vario = vario_estimate(
                    self.cond_pos, field, direction=axes
                )
            # set the sill to the field variance
            self.model.fit_variogram(*emp_vario, sill=sill)
        # set the measurement errors
        self.cond_err = cond_err
        # set the external drift values and the conditioning points
        self._cond_ext_drift = self._pre_ext_drift(
            self.cond_no, ext_drift, set_cond=True
        )
        # upate the internal kriging settings
        self._krige_pos = self.model.isometrize(self.cond_pos)
        # krige pos are the unrotated and isotropic condition positions
        self._krige_mat = self._get_krige_mat()

    def set_drift_functions(self, drift_functions=None):
        """
        Set the drift functions for universal kriging.

        Parameters
        ----------
        drift_functions : :class:`list` of :any:`callable`, :class:`str` or :class:`int`
            Either a list of callable functions, an integer representing
            the polynomial order of the drift or one of the following strings:

                * "linear" : regional linear drift (equals order=1)
                * "quadratic" : regional quadratic drift (equals order=2)

        Raises
        ------
        ValueError
            If the given drift functions are not callable.
        """
        if drift_functions is None:
            self._drift_functions = []
        elif isinstance(drift_functions, (str, int)):
            self._drift_functions = get_drift_functions(
                self.dim, drift_functions
            )
        else:
            if isinstance(drift_functions, collections.abc.Iterator):
                drift_functions = list(drift_functions)
            # check for a single content thats not a string
            try:
                iter(drift_functions)
            except TypeError:
                drift_functions = [drift_functions]
            for f in drift_functions:
                if not callable(f):
                    raise ValueError("Krige: Drift functions not callable")
            self._drift_functions = drift_functions

    @property
    def _krige_cond(self):
        """:class:`numpy.ndarray`: The prepared kriging conditions."""
        pad_size = self.drift_no + int(self.unbiased)
        # detrend data and normalize
        val = self.normalizer.normalize(self.cond_val - self.cond_trend)
        # set to zero mean
        val -= self.cond_mean
        return np.pad(val, (0, pad_size), mode="constant", constant_values=0)

    @property
    def cond_pos(self):
        """:class:`list`: The position tuple of the conditions."""
        return self._cond_pos

    @property
    def cond_val(self):
        """:class:`list`: The values of the conditions."""
        return self._cond_val

    @property
    def cond_err(self):
        """:class:`list`: The measurement errors at the condition points."""
        if isinstance(self._cond_err, str) and self._cond_err == "nugget":
            return self.model.nugget
        return self._cond_err

    @cond_err.setter
    def cond_err(self, value):
        if isinstance(value, str) and value == "nugget":
            self._cond_err = value
        else:
            if self.exact:
                raise ValueError(
                    "krige.cond_err: measurement errors can't be given, "
                    "when interpolator should be exact."
                )
            value = np.asarray(value, dtype=np.double).reshape(-1)
            if value.size == 1:
                self._cond_err = value.item()
            else:
                if value.size != self.cond_no:
                    raise ValueError(
                        "krige.cond_err: wrong number of measurement errors."
                    )
                self._cond_err = value

    @property
    def cond_no(self):
        """:class:`int`: The number of the conditions."""
        return len(self._cond_val)

    @property
    def cond_ext_drift(self):
        """:class:`numpy.ndarray`: The ext. drift at the conditions."""
        return self._cond_ext_drift

    @property
    def cond_mean(self):
        """:class:`numpy.ndarray`: Trend at the conditions."""
        return eval_func(self.mean, self.cond_pos, self.dim, broadcast=True)

    @property
    def cond_trend(self):
        """:class:`numpy.ndarray`: Trend at the conditions."""
        return eval_func(self.trend, self.cond_pos, self.dim, broadcast=True)

    @property
    def unbiased(self):
        """:class:`bool`: Whether the kriging is unbiased or not."""
        return self._unbiased

    @property
    def exact(self):
        """:class:`bool`: Whether the interpolator is exact."""
        return self._exact

    @property
    def pseudo_inv(self):
        """:class:`bool`: Whether pseudo inverse matrix is used."""
        return self._pseudo_inv

    @property
    def pseudo_inv_type(self):
        """:class:`str`: Method selector for pseudo inverse calculation."""
        return self._pseudo_inv_type

    @pseudo_inv_type.setter
    def pseudo_inv_type(self, val):
        if val not in P_INV and not callable(val):
            raise ValueError(f"Krige: pseudo_inv_type not in {sorted(P_INV)}")
        self._pseudo_inv_type = val

    @property
    def drift_functions(self):
        """:class:`list` of :any:`callable`: The drift functions."""
        return self._drift_functions

    @property
    def has_const_mean(self):
        """:class:`bool`: Whether the field has a constant mean or not."""
        return self.drift_no == 0 and not callable(self.mean)

    @property
    def krige_size(self):
        """:class:`int`: Size of the kriging system."""
        return self.cond_no + self.drift_no + int(self.unbiased)

    @property
    def drift_no(self):
        """:class:`int`: Number of drift values per point."""
        return self.int_drift_no + self.ext_drift_no

    @property
    def int_drift_no(self):
        """:class:`int`: Number of internal drift values per point."""
        return len(self.drift_functions)

    @property
    def ext_drift_no(self):
        """:class:`int`: Number of external drift values per point."""
        return self.cond_ext_drift.shape[0]

    def __repr__(self):
        """Return String representation."""
        return "{0}(model={1}, cond_no={2}{3})".format(
            self.name,
            self.model.name,
            self.cond_no,
            self._fmt_mean_norm_trend(),
        )
