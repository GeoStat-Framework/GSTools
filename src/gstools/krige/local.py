"""
GStools subpackage providing a prototype class for local kriging.

.. currentmodule:: gstools.krige.local

The following classes are provided

.. autosummary::
   LocalKrige
"""

import json

import numpy as np
from gstools_core import calc_field_krige_local as calc_field_krige_local_gsc

from gstools.krige.base import Krige
from gstools.krige.tools import set_condition
from gstools.tools.geometric import rotated_main_axes
from gstools.variogram import vario_estimate

__all__ = [
    "LocalKrige",
    "LocalSimple",
    "LocalOrdinary",
    "LocalUniversal",
    "LocalExtDrift",
    "LocalDetrended",
]


# model classes GSTools-Core's CovModelSpec (covmodel_spec.rs) currently
# implements; the JSON "type" tag is just the lowercased class name for
# all four
_CORE_MODEL_TYPES = ("Gaussian", "Exponential", "Matern", "Spherical")


def _cov_model_to_json(model):
    """
    Serialize a :any:`gstools.CovModel` to the JSON wire format the
    GSTools-Core Rust backend expects (``CovModelSpec`` in
    ``covmodel_spec.rs``: a ``{"type": ..., ...}``-tagged enum).

    Only ``var``/``len_scale``/``nugget`` (and, for Matern, ``nu``) travel
    over the wire.
    """
    if model.name not in _CORE_MODEL_TYPES:
        raise NotImplementedError(
            f"LocalKrige: GSTools-Core does not (yet) support "
            f"the '{model.name}' covariance model; supported: "
            f"{_CORE_MODEL_TYPES}."
        )
    spec = {
        "type": model.name.lower(),
        "var": model.var,
        "len_scale": model.len_scale,
        "nugget": model.nugget,
    }
    if model.name == "Matern":
        spec["nu"] = model.nu
    return json.dumps(spec)


def _calc_field_krige_local(
    cond_pos,
    cond_val,
    target_pos,
    model,
    cond_err,
    drift_cond,
    drift_target,
    unbiased,
    exact,
    local_radius,
    num_threads=None,
):
    """A wrapper function for calling the local krige algorithm (Rust)."""
    return calc_field_krige_local_gsc(
        cond_pos,
        cond_val,
        target_pos,
        _cov_model_to_json(model),
        cond_err,
        drift_cond,
        drift_target,
        unbiased,
        exact,
        local_radius,
        num_threads=num_threads,
    )


class LocalKrige(Krige):
    """
    Prototype class for local (moving-neighborhood) kriging.


    Parameters
    ----------
    model : :any:`CovModel`
        Covariance Model used for kriging.
    cond_pos : :class:`list`
        tuple, containing the given condition positions (x, [y, z])
    cond_val : :class:`numpy.ndarray`
        the values of the conditions (nan values will be ignored)
    local_radius : :class:`float`
        Search radius (in the model's isometrized/isotropic distance):
        every conditioning point within this distance of a target
        point enters its local kriging system.
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
        A callable trend function. Should have the signature: f(x, [y, z, ...])
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
    fit_normalizer : :class:`bool`, optional
        Whether to fit the data-normalizer to the given conditioning data.
        Default: False
    fit_variogram : :class:`bool`, optional
        Whether to fit the given variogram model to the data.
        Directional variogram fitting is triggered by setting
        any anisotropy factor of the model to anything unequal 1
        but the main axes of correlation are taken from the model
        rotation angles. If the model is a spatio-temporal latlon
        model, this will raise an error.
        This assumes the sill to be the data variance and with
        standard bins provided by the :any:`standard_bins` routine.
        Default: False
    """

    def __init__(
        self,
        model,
        cond_pos,
        cond_val,
        local_radius,
        drift_functions=None,
        ext_drift=None,
        mean=None,
        normalizer=None,
        trend=None,
        unbiased=True,
        exact=False,
        cond_err="nugget",
        fit_normalizer=False,
        fit_variogram=False,
    ):
        self.local_radius = local_radius
        self._cond_drift = None
        self._target_drift = None
        # super().__init__ calls set_condition and set_drift_functions
        super().__init__(
            model,
            cond_pos,
            cond_val,
            drift_functions=drift_functions,
            ext_drift=ext_drift,
            mean=mean,
            normalizer=normalizer,
            trend=trend,
            unbiased=unbiased,
            exact=exact,
            cond_err=cond_err,
            fit_normalizer=fit_normalizer,
            fit_variogram=fit_variogram,
        )

    def __call__(
        self,
        pos=None,
        mesh_type="unstructured",
        ext_drift=None,
        only_mean=False,
        return_var=True,
        post_process=True,
        store=True,
    ):
        """
        Generate the local kriging field.

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
        only_mean : :class:`bool`, optional
            Whether to only calculate the mean of the kriging field.
            Not implemented yet for local kriging -- accepted here (and
            has to be `False`) only so :any:`LocalKrige` is a drop-in
            replacement for :any:`Krige` in places like :any:`CondSRF`,
            which always call with ``only_mean=False``.
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
            ["field", "krige_var"].

        Returns
        -------
        field : :class:`numpy.ndarray`
            the kriged field
        krige_var : :class:`numpy.ndarray`, optional
            the kriging error variance (if return_var is True)
        """
        if only_mean:
            raise NotImplementedError(
                "LocalKrige: only_mean=True is not implemented yet."
            )
        fld_cnt = 2 if return_var else 1
        name, save = self.get_store_config(store, None, fld_cnt)

        iso_targ, shape = self.pre_pos(pos, mesh_type)
        pnt_cnt = iso_targ.shape[1]

        # cond values already normalized/detrended/zero-mean (unpadded)
        cond_val = self._krige_cond[: self.cond_no]
        cond_err = self.cond_err
        cond_err_arr = (
            np.full(self.cond_no, cond_err, dtype=np.double)
            if np.isscalar(cond_err)
            else np.asarray(cond_err, dtype=np.double)
        )

        ext_drift = self._pre_ext_drift(pnt_cnt, ext_drift)
        drift_target = self._calc_target_drift(iso_targ, ext_drift, pnt_cnt)

        # call the local kriging routine (Rust or NumPy fallback)
        field, error = _calc_field_krige_local(
            cond_pos=self._krige_pos,
            cond_val=cond_val,
            target_pos=iso_targ,
            model=self.model,
            cond_err=cond_err_arr,
            drift_cond=self._cond_drift,
            drift_target=drift_target,
            unbiased=self.unbiased,
            exact=self.exact,
            local_radius=self.local_radius,
        )

        field = np.reshape(field, shape)
        field = self.post_field(field, name[0], post_process, save[0])
        if return_var:  # care about the estimated error variance
            krige_var = np.reshape(
                np.maximum(self.model.sill - error, 0), shape
            )
            krige_var = self.post_field(krige_var, name[1], False, save[1])
            return field, krige_var
        return field

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
        
        This function is the same as :any:`Krige.set_condition`, but it also precomputes
        the drift terms at the conditioning points for the local kriging solve. 
        (last line of this function)
        TODO: This function should be refactored to avoid code duplication with the base class.

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
            Whether to fit the data-normalizer to the given conditioning data.
            Default: False
        fit_variogram : :class:`bool`, optional
            Whether to fit the given variogram model to the data.
            Directional variogram fitting is triggered by setting
            any anisotropy factor of the model to anything unequal 1
            but the main axes of correlation are taken from the model
            rotation angles. If the model is a spatio-temporal latlon
            model, this will raise an error.
            This assumes the sill to be the data variance and with
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
            if self.model.latlon and self.model.temporal:
                msg = "Krige: can't fit variogram for spatio-temporal latlon data."
                raise ValueError(msg)
            field = self.normalizer.normalize(self.cond_val - self.cond_trend)
            field -= self.cond_mean
            sill = np.var(field)
            if self.model.is_isotropic:
                emp_vario = vario_estimate(
                    self.cond_pos,
                    field,
                    latlon=self.model.latlon,
                    geo_scale=self.model.geo_scale,
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
        # local kriging: no global kriging matrix -- pre-compute the drift
        # terms at the conditioning points instead, for the local solve
        self._cond_drift = self._calc_cond_drift()

    def _calc_cond_drift(self):
        """
        Evaluate all drift terms at the conditioning points once.

        Internal (functional) drift and external drift are unified into
        a single ``(drift_no, cond_no)`` array: external drift is
        already array-shaped, so it is simply stacked below the
        evaluated functional drift rows.

        Returns
        -------
        :class:`numpy.ndarray`
            Shape ``(drift_no, cond_no)``.
        """
        cond_pos = np.asarray(self.cond_pos)
        int_functions = self.drift_functions
        int_no = len(int_functions)
        ext_no = self.ext_drift_no
        drift_no = int_no + ext_no

        drift_cond = np.empty((drift_no, self.cond_no), dtype=np.double)
        for i, f in enumerate(int_functions):
            drift_cond[i] = f(*cond_pos)
        if ext_no > 0:
            drift_cond[int_no:] = self.cond_ext_drift
        return drift_cond

    def _calc_target_drift(self, iso_targ, ext_drift, pnt_cnt):
        """
        Evaluate all drift terms at the target positions once.

        Mirrors :any:`_calc_cond_drift`, but for the (isometrized) target
        positions of the current evaluation call instead of the
        conditioning points. Internal (functional) drift and external
        drift are unified into a single ``(drift_no, pnt_cnt)`` array,
        the same layout :any:`_calc_cond_drift` uses for the
        conditioning points.

        Parameters
        ----------
        iso_targ : :class:`numpy.ndarray`
            Isometrized target positions, shape ``(dim, pnt_cnt)``.
        ext_drift : :class:`numpy.ndarray`
            Preprocessed external drift values at the target positions
            (output of :any:`Krige._pre_ext_drift`), shape
            ``(ext_drift_no, pnt_cnt)``.
        pnt_cnt : :class:`int`
            Number of target points.

        Returns
        -------
        :class:`numpy.ndarray`
            Shape ``(drift_no, pnt_cnt)``.
        """
        int_functions = self.drift_functions
        int_no = len(int_functions)
        ext_no = self.ext_drift_no
        drift_no = int_no + ext_no

        drift_target = np.empty((drift_no, pnt_cnt), dtype=np.double)
        if int_no > 0:
            # drift functions live in the original (anisotropic) frame
            targ_pos = self.model.anisometrize(iso_targ)
            for i, f in enumerate(int_functions):
                drift_target[i] = f(*targ_pos)
        if ext_no > 0:
            drift_target[int_no:] = ext_drift
        self._target_drift = drift_target
        return drift_target

    @property
    def local_radius(self):
        """:class:`float`: The search radius for local kriging."""
        return self._local_radius

    @local_radius.setter
    def local_radius(self, value):
        value = float(value)
        if value <= 0:
            raise ValueError("LocalKrige: local_radius must be > 0.")
        self._local_radius = value


class LocalSimple(LocalKrige):
    """
    Local simple kriging.

    Local simple kriging is used to interpolate data with a given mean,
    using only the conditioning points within ``local_radius`` of each
    target point. See :any:`LocalKrige` and :any:`gstools.krige.Simple`.

    Parameters
    ----------
    model : :any:`CovModel`
        Covariance Model used for kriging.
    cond_pos : :class:`list`
        tuple, containing the given condition positions (x, [y, z])
    cond_val : :class:`numpy.ndarray`
        the values of the conditions (nan values will be ignored)
    local_radius : :class:`float`
        Search radius (in the model's isometrized/isotropic distance):
        every conditioning point within this distance of a target
        point enters its local kriging system.
    mean : :class:`float`, optional
        mean value used to shift normalized conditioning data.
        Could also be a callable. The default is None.
    normalizer : :any:`None` or :any:`Normalizer`, optional
        Normalizer to be applied to the input data to gain normality.
        The default is None.
    trend : :any:`None` or :class:`float` or :any:`callable`, optional
        A callable trend function. Should have the signature: f(x, [y, z, ...])
        This is used for detrended kriging, where the trended is subtracted
        from the conditions before kriging is applied.
        This can be used for regression kriging, where the trend function
        is determined by an external regression algorithm.
        If no normalizer is applied, this behaves equal to 'mean'.
        The default is None.
    exact : :class:`bool`, optional
        Whether the interpolator should reproduce the exact input values.
        If `False`, `cond_err` is interpreted as measurement error
        at the conditioning points and the result will be more smooth.
        Default: False
    cond_err : :class:`str`, :class :class:`float` or :class:`list`, optional
        The measurement error at the conditioning points.
        Either "nugget" to apply the model-nugget, a single value applied to
        all points or an array with individual values for each point.
        The measurement error has to be <= nugget.
        The "exact=True" variant only works with "cond_err='nugget'".
        Default: "nugget"
    fit_normalizer : :class:`bool`, optional
        Whether to fit the data-normalizer to the given conditioning data.
        Default: False
    fit_variogram : :class:`bool`, optional
        Whether to fit the given variogram model to the data.
        Default: False
    """

    def __init__(
        self,
        model,
        cond_pos,
        cond_val,
        local_radius,
        mean=0.0,
        normalizer=None,
        trend=None,
        exact=False,
        cond_err="nugget",
        fit_normalizer=False,
        fit_variogram=False,
    ):
        super().__init__(
            model,
            cond_pos,
            cond_val,
            local_radius,
            mean=mean,
            normalizer=normalizer,
            trend=trend,
            unbiased=False,
            exact=exact,
            cond_err=cond_err,
            fit_normalizer=fit_normalizer,
            fit_variogram=fit_variogram,
        )


class LocalOrdinary(LocalKrige):
    """
    Local ordinary kriging.

    Local ordinary kriging is used to interpolate data and estimate a
    proper mean, using only the conditioning points within
    ``local_radius`` of each target point. See :any:`LocalKrige` and
    :any:`gstools.krige.Ordinary`.

    Parameters
    ----------
    model : :any:`CovModel`
        Covariance Model used for kriging.
    cond_pos : :class:`list`
        tuple, containing the given condition positions (x, [y, z])
    cond_val : :class:`numpy.ndarray`
        the values of the conditions (nan values will be ignored)
    local_radius : :class:`float`
        Search radius (in the model's isometrized/isotropic distance):
        every conditioning point within this distance of a target
        point enters its local kriging system.
    normalizer : :any:`None` or :any:`Normalizer`, optional
        Normalizer to be applied to the input data to gain normality.
        The default is None.
    trend : :any:`None` or :class:`float` or :any:`callable`, optional
        A callable trend function. Should have the signature: f(x, [y, z, ...])
        This is used for detrended kriging, where the trended is subtracted
        from the conditions before kriging is applied.
        This can be used for regression kriging, where the trend function
        is determined by an external regression algorithm.
        If no normalizer is applied, this behaves equal to 'mean'.
        The default is None.
    exact : :class:`bool`, optional
        Whether the interpolator should reproduce the exact input values.
        If `False`, `cond_err` is interpreted as measurement error
        at the conditioning points and the result will be more smooth.
        Default: False
    cond_err : :class:`str`, :class :class:`float` or :class:`list`, optional
        The measurement error at the conditioning points.
        Either "nugget" to apply the model-nugget, a single value applied to
        all points or an array with individual values for each point.
        The measurement error has to be <= nugget.
        The "exact=True" variant only works with "cond_err='nugget'".
        Default: "nugget"
    fit_normalizer : :class:`bool`, optional
        Whether to fit the data-normalizer to the given conditioning data.
        Default: False
    fit_variogram : :class:`bool`, optional
        Whether to fit the given variogram model to the data.
        Default: False
    """

    def __init__(
        self,
        model,
        cond_pos,
        cond_val,
        local_radius,
        normalizer=None,
        trend=None,
        exact=False,
        cond_err="nugget",
        fit_normalizer=False,
        fit_variogram=False,
    ):
        super().__init__(
            model,
            cond_pos,
            cond_val,
            local_radius,
            normalizer=normalizer,
            trend=trend,
            exact=exact,
            cond_err=cond_err,
            fit_normalizer=fit_normalizer,
            fit_variogram=fit_variogram,
        )


class LocalUniversal(LocalKrige):
    """
    Local universal kriging.

    Local universal kriging is used to interpolate given data with a
    variable mean determined by a functional drift, using only the
    conditioning points within ``local_radius`` of each target point.
    See :any:`LocalKrige` and :any:`gstools.krige.Universal`.

    This estimator is set to be unbiased by default.

    Parameters
    ----------
    model : :any:`CovModel`
        Covariance Model used for kriging.
    cond_pos : :class:`list`
        tuple, containing the given condition positions (x, [y, z])
    cond_val : :class:`numpy.ndarray`
        the values of the conditions (nan values will be ignored)
    local_radius : :class:`float`
        Search radius (in the model's isometrized/isotropic distance):
        every conditioning point within this distance of a target
        point enters its local kriging system.
    drift_functions : :class:`list` of :any:`callable`, :class:`str` or :class:`int`
        Either a list of callable functions, an integer representing
        the polynomial order of the drift or one of the following strings:

            * "linear" : regional linear drift (equals order=1)
            * "quadratic" : regional quadratic drift (equals order=2)

    normalizer : :any:`None` or :any:`Normalizer`, optional
        Normalizer to be applied to the input data to gain normality.
        The default is None.
    trend : :any:`None` or :class:`float` or :any:`callable`, optional
        A callable trend function. Should have the signature: f(x, [y, z, ...])
        This is used for detrended kriging, where the trended is subtracted
        from the conditions before kriging is applied.
        This can be used for regression kriging, where the trend function
        is determined by an external regression algorithm.
        If no normalizer is applied, this behaves equal to 'mean'.
        The default is None.
    exact : :class:`bool`, optional
        Whether the interpolator should reproduce the exact input values.
        If `False`, `cond_err` is interpreted as measurement error
        at the conditioning points and the result will be more smooth.
        Default: False
    cond_err : :class:`str`, :class :class:`float` or :class:`list`, optional
        The measurement error at the conditioning points.
        Either "nugget" to apply the model-nugget, a single value applied to
        all points or an array with individual values for each point.
        The measurement error has to be <= nugget.
        The "exact=True" variant only works with "cond_err='nugget'".
        Default: "nugget"
    fit_normalizer : :class:`bool`, optional
        Whether to fit the data-normalizer to the given conditioning data.
        Default: False
    fit_variogram : :class:`bool`, optional
        Whether to fit the given variogram model to the data.
        Default: False
    """

    def __init__(
        self,
        model,
        cond_pos,
        cond_val,
        local_radius,
        drift_functions,
        normalizer=None,
        trend=None,
        exact=False,
        cond_err="nugget",
        fit_normalizer=False,
        fit_variogram=False,
    ):
        super().__init__(
            model,
            cond_pos,
            cond_val,
            local_radius,
            drift_functions=drift_functions,
            normalizer=normalizer,
            trend=trend,
            exact=exact,
            cond_err=cond_err,
            fit_normalizer=fit_normalizer,
            fit_variogram=fit_variogram,
        )


class LocalExtDrift(LocalKrige):
    """
    Local external drift kriging (local EDK).

    Local external drift kriging is used to interpolate given data with
    a variable mean determined by an external drift, using only the
    conditioning points within ``local_radius`` of each target point.
    See :any:`LocalKrige` and :any:`gstools.krige.ExtDrift`.

    This estimator is set to be unbiased by default.

    Parameters
    ----------
    model : :any:`CovModel`
        Covariance Model used for kriging.
    cond_pos : :class:`list`
        tuple, containing the given condition positions (x, [y, z])
    cond_val : :class:`numpy.ndarray`
        the values of the conditions (nan values will be ignored)
    local_radius : :class:`float`
        Search radius (in the model's isometrized/isotropic distance):
        every conditioning point within this distance of a target
        point enters its local kriging system.
    ext_drift : :class:`numpy.ndarray`
        the external drift values at the given condition positions.
    normalizer : :any:`None` or :any:`Normalizer`, optional
        Normalizer to be applied to the input data to gain normality.
        The default is None.
    trend : :any:`None` or :class:`float` or :any:`callable`, optional
        A callable trend function. Should have the signature: f(x, [y, z, ...])
        This is used for detrended kriging, where the trended is subtracted
        from the conditions before kriging is applied.
        This can be used for regression kriging, where the trend function
        is determined by an external regression algorithm.
        If no normalizer is applied, this behaves equal to 'mean'.
        The default is None.
    exact : :class:`bool`, optional
        Whether the interpolator should reproduce the exact input values.
        If `False`, `cond_err` is interpreted as measurement error
        at the conditioning points and the result will be more smooth.
        Default: False
    cond_err : :class:`str`, :class :class:`float` or :class:`list`, optional
        The measurement error at the conditioning points.
        Either "nugget" to apply the model-nugget, a single value applied to
        all points or an array with individual values for each point.
        The measurement error has to be <= nugget.
        The "exact=True" variant only works with "cond_err='nugget'".
        Default: "nugget"
    fit_normalizer : :class:`bool`, optional
        Whether to fit the data-normalizer to the given conditioning data.
        Default: False
    fit_variogram : :class:`bool`, optional
        Whether to fit the given variogram model to the data.
        Default: False
    """

    def __init__(
        self,
        model,
        cond_pos,
        cond_val,
        local_radius,
        ext_drift,
        normalizer=None,
        trend=None,
        exact=False,
        cond_err="nugget",
        fit_normalizer=False,
        fit_variogram=False,
    ):
        super().__init__(
            model,
            cond_pos,
            cond_val,
            local_radius,
            ext_drift=ext_drift,
            normalizer=normalizer,
            trend=trend,
            exact=exact,
            cond_err=cond_err,
            fit_normalizer=fit_normalizer,
            fit_variogram=fit_variogram,
        )


class LocalDetrended(LocalKrige):
    """
    Local detrended simple kriging.

    In local detrended kriging, the data is detrended before
    interpolation by local simple kriging with zero mean, using only
    the conditioning points within ``local_radius`` of each target
    point. See :any:`LocalKrige` and :any:`gstools.krige.Detrended`.

    This is just a shortcut for local simple kriging with a given trend
    function, zero mean and no normalizer.

    Parameters
    ----------
    model : :any:`CovModel`
        Covariance Model used for kriging.
    cond_pos : :class:`list`
        tuple, containing the given condition positions (x, [y, z])
    cond_val : :class:`numpy.ndarray`
        the values of the conditions (nan values will be ignored)
    local_radius : :class:`float`
        Search radius (in the model's isometrized/isotropic distance):
        every conditioning point within this distance of a target
        point enters its local kriging system.
    trend_function : :any:`callable`
        The callable trend function. Should have the signature: f(x, [y, z])
    exact : :class:`bool`, optional
        Whether the interpolator should reproduce the exact input values.
        If `False`, `cond_err` is interpreted as measurement error
        at the conditioning points and the result will be more smooth.
        Default: False
    cond_err : :class:`str`, :class :class:`float` or :class:`list`, optional
        The measurement error at the conditioning points.
        Either "nugget" to apply the model-nugget, a single value applied to
        all points or an array with individual values for each point.
        The measurement error has to be <= nugget.
        The "exact=True" variant only works with "cond_err='nugget'".
        Default: "nugget"
    fit_variogram : :class:`bool`, optional
        Whether to fit the given variogram model to the data.
        Default: False
    """

    def __init__(
        self,
        model,
        cond_pos,
        cond_val,
        local_radius,
        trend_function,
        exact=False,
        cond_err="nugget",
        fit_variogram=False,
    ):
        super().__init__(
            model,
            cond_pos,
            cond_val,
            local_radius,
            trend=trend_function,
            unbiased=False,
            exact=exact,
            cond_err=cond_err,
            fit_variogram=fit_variogram,
        )
