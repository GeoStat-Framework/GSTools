# -*- coding: utf-8 -*-
"""
GStools subpackage providing a base class for kriging.

.. currentmodule:: gstools.krige.base

The following classes are provided

.. autosummary::
   Krige
"""
# pylint: disable=C0103
import collections
import numpy as np

# from scipy.linalg import inv
from scipy.spatial.distance import cdist
from gstools.field.tools import reshape_field_from_unstruct_to_struct
from gstools.field.base import Field
from gstools.krige.krigesum import krigesum
from gstools.krige.tools import (
    set_condition,
    get_drift_functions,
    no_trend,
    eval_func,
)

__all__ = ["Krige"]


class Krige(Field):
    """
    A base class for kriging.

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
    ext_drift : :class:`numpy.ndarray` or :any:`None`, optional
        the external drift values at the given cond. positions (only for EDK)
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
        self,
        model,
        cond_pos,
        cond_val,
        mean=0.0,
        ext_drift=None,
        drift_functions=None,
        trend_function=None,
    ):
        super().__init__(model, mean)
        self.krige_var = None
        # initialize private attributes
        self._unbiased = True
        self._value_type = "scalar"
        self._cond_pos = None
        self._cond_val = None
        self._krige_mat = None
        self._krige_pos = None
        self._cond_trend = None
        self._trend_function = None
        self.trend_function = trend_function
        self._cond_ext_drift = np.array([])
        self._drift_functions = []
        if drift_functions is not None:
            self.set_drift_functions(drift_functions)
        self.set_condition(cond_pos, cond_val, ext_drift)

    def __call__(
        self, pos, mesh_type="unstructured", ext_drift=None, chunk_size=None
    ):
        """
        Generate the kriging field.

        The field is saved as `self.field` and is also returned.
        The error variance is saved as `self.krige_var` and is also returned.

        Parameters
        ----------
        pos : :class:`list`
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

        Returns
        -------
        field : :class:`numpy.ndarray`
            the kriged field
        krige_var : :class:`numpy.ndarray`
            the kriging error variance
        """
        self.mesh_type = mesh_type
        # internal conversation
        x, y, z, self.pos, __, mt_changed, axis_lens = self._pre_pos(
            pos, mesh_type, make_unstruct=True
        )
        point_no = len(x)
        # set chunk size
        chunk_size = point_no if chunk_size is None else int(chunk_size)
        chunk_no = int(np.ceil(point_no / chunk_size))
        field = np.empty_like(x)
        krige_var = np.empty_like(x)
        ext_drift = self._pre_ext_drift(point_no, ext_drift)
        # iterate of chunks
        for i in range(chunk_no):
            # get chunk slice for actual chunk
            chunk_slice = (i * chunk_size, min(point_no, (i + 1) * chunk_size))
            c_slice = slice(*chunk_slice)
            # get RHS of the kriging system
            k_vec = self._get_krige_vecs((x, y, z), chunk_slice, ext_drift)
            # generate the raw kriging field and error variance
            field[c_slice], krige_var[c_slice] = krigesum(
                self._krige_mat, k_vec, self._krige_cond
            )
        # reshape field if we got a structured mesh
        if mt_changed:
            field = reshape_field_from_unstruct_to_struct(
                self.model.dim, field, axis_lens
            )
            krige_var = reshape_field_from_unstruct_to_struct(
                self.model.dim, krige_var, axis_lens
            )
        self._post_field(field, krige_var)
        return self.field, self.krige_var

    def _get_krige_mat(self):  # pragma: no cover
        """Calculate the inverse matrix of the kriging equation."""
        return None

    def _get_krige_vecs(
        self, pos, chunk_slice=(0, None), ext_drift=None
    ):  # pragma: no cover
        """Calculate the RHS of the kriging equation."""
        return None

    def _pre_ext_drift(self, point_no, ext_drift=None, set_cond=False):
        """
        Preprocessor for external drifts.

        Parameters
        ----------
        point_no : :class:`numpy.ndarray`
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
            if set_cond:
                ext_drift = np.array(ext_drift, dtype=np.double, ndmin=2)
                if len(ext_drift.shape) > 2 or ext_drift.shape[1] != point_no:
                    raise ValueError("Krige: wrong number of cond. drifts.")
                return ext_drift
            ext_drift = np.array(ext_drift, dtype=np.double, ndmin=2)
            ext_shape = np.shape(ext_drift)
            shape = (self.drift_no, point_no)
            if self.drift_no > 1 and ext_shape[0] != self.drift_no:
                raise ValueError("Krige: wrong number of external drifts.")
            if np.prod(ext_shape) != np.prod(shape):
                raise ValueError("Krige: wrong number of ext. drift values.")
            return np.array(ext_drift, dtype=np.double).reshape(shape)
        elif not set_cond and self._cond_ext_drift.size > 0:
            raise ValueError("Krige: wrong number of ext. drift values.")
        return np.array([])

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
            self.field = field
        else:
            self.field = field + eval_func(
                self.trend_function, self.pos, self.mesh_type
            )
        self.krige_var = krige_var

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
        pos1_stack = np.column_stack(pos1[: self.model.dim])
        if pos2 is None:
            return cdist(pos1_stack, pos1_stack)
        p2s = slice(*pos2_slice)
        pos2_stack = np.column_stack(pos2[: self.model.dim])[p2s, ...]
        return cdist(pos1_stack, pos2_stack)

    def get_mean(self):
        """Calculate the estimated mean."""
        return self._mean

    def set_condition(self, cond_pos, cond_val, ext_drift=None):
        """Set the conditions for kriging.

        Parameters
        ----------
        cond_pos : :class:`list`
            the position tuple of the conditions (x, [y, z])
        cond_val : :class:`numpy.ndarray`
            the values of the conditions
        ext_drift : :class:`numpy.ndarray` or :any:`None`, optional
            the external drift values at the given conditions (only for EDK)
            For multiple external drifts, the first dimension
            should be the index of the drift term.
        """
        self._cond_pos, self._cond_val = set_condition(
            cond_pos, cond_val, self.model.dim
        )
        self._cond_ext_drift = self._pre_ext_drift(
            self.cond_no, ext_drift, set_cond=True
        )
        self.update()

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
                self.model.dim, drift_functions
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
                    raise ValueError("Universal: Drift functions not callable")
            self._drift_functions = drift_functions

    def update(self):
        """Update the kriging settings."""
        x, y, z, __, __, __, __ = self._pre_pos(self.cond_pos)
        # krige pos are the unrotated and isotropic condition positions
        self._krige_pos = (x, y, z)[: self.model.dim]
        self._krige_mat = self._get_krige_mat()
        if self.trend_function is no_trend:
            self._cond_trend = 0.0
        else:
            self._cond_trend = self.trend_function(*self.cond_pos)
        self._mean = self.get_mean()

    @property
    def _krige_cond(self):
        """:class:`numpy.ndarray`: The prepared kriging conditions."""
        pad_size = self.drift_no + int(self.unbiased)
        return np.pad(
            self.cond_val - self.cond_trend,
            (0, pad_size),
            mode="constant",
            constant_values=0,
        )

    @property
    def cond_pos(self):
        """:class:`list`: The position tuple of the conditions."""
        return self._cond_pos

    @property
    def cond_val(self):
        """:class:`list`: The values of the conditions."""
        return self._cond_val

    @property
    def cond_no(self):
        """:class:`int`: The number of the conditions."""
        return len(self._cond_val)

    @property
    def cond_ext_drift(self):
        """:class:`numpy.ndarray`: The ext. drift at the conditions."""
        return self._cond_ext_drift

    @property
    def drift_functions(self):
        """:class:`list` of :any:`callable`: The drift functions."""
        return self._drift_functions

    @property
    def drift_no(self):
        """:class:`int`: Number of drift values per point."""
        return len(self.drift_functions) + self.cond_ext_drift.shape[0]

    @property
    def cond_trend(self):
        """:class:`numpy.ndarray`: Trend at the conditions."""
        return self._cond_trend

    @property
    def trend_function(self):
        """:any:`callable`: The trend function."""
        return self._trend_function

    @trend_function.setter
    def trend_function(self, trend_function):
        if trend_function is None:
            trend_function = no_trend
        if not callable(trend_function):
            raise ValueError("Detrended kriging: trend function not callable.")
        self._trend_function = trend_function

    @property
    def unbiased(self):
        """:class:`bool`: Whether the kriging is unbiased or not."""
        return self._unbiased


if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod()
