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
from gstools.krige.base import Krige

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
    pseudo_inv : :class:`bool`, optional
        Whether the kriging system is solved with the pseudo inverted
        kriging matrix. If `True`, this leads to more numerical stability
        and redundant points are averaged. But it can take more time.
        Default: True
    pseudo_inv_type : :class:`int` or :any:`callable`, optional
        Here you can select the algorithm to compute the pseudo-inverse matrix:

            * `1`: use `pinv` from `scipy` which uses `lstsq`
            * `2`: use `pinv2` from `scipy` which uses `SVD`
            * `3`: use `pinvh` from `scipy` which uses eigen-values

        If you want to use another routine to invert the kriging matrix,
        you can pass a callable which takes a matrix and returns the inverse.
        Default: `1`
    """

    def __init__(
        self,
        model,
        cond_pos,
        cond_val,
        mean=0.0,
        trend_function=None,
        exact=False,
        cond_err="nugget",
        pseudo_inv=True,
        pseudo_inv_type=1,
    ):
        super().__init__(
            model,
            cond_pos,
            cond_val,
            mean=mean,
            trend_function=trend_function,
            unbiased=False,
            exact=exact,
            cond_err=cond_err,
            pseudo_inv=pseudo_inv,
            pseudo_inv_type=pseudo_inv_type,
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
    pseudo_inv : :class:`bool`, optional
        Whether the kriging system is solved with the pseudo inverted
        kriging matrix. If `True`, this leads to more numerical stability
        and redundant points are averaged. But it can take more time.
        Default: True
    pseudo_inv_type : :class:`int` or :any:`callable`, optional
        Here you can select the algorithm to compute the pseudo-inverse matrix:

            * `1`: use `pinv` from `scipy` which uses `lstsq`
            * `2`: use `pinv2` from `scipy` which uses `SVD`
            * `3`: use `pinvh` from `scipy` which uses eigen-values

        If you want to use another routine to invert the kriging matrix,
        you can pass a callable which takes a matrix and returns the inverse.
        Default: `1`
    """

    def __init__(
        self,
        model,
        cond_pos,
        cond_val,
        trend_function=None,
        exact=False,
        cond_err="nugget",
        pseudo_inv=True,
        pseudo_inv_type=1,
    ):
        super().__init__(
            model,
            cond_pos,
            cond_val,
            trend_function=trend_function,
            exact=exact,
            cond_err=cond_err,
            pseudo_inv=pseudo_inv,
            pseudo_inv_type=pseudo_inv_type,
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
    pseudo_inv : :class:`bool`, optional
        Whether the kriging system is solved with the pseudo inverted
        kriging matrix. If `True`, this leads to more numerical stability
        and redundant points are averaged. But it can take more time.
        Default: True
    pseudo_inv_type : :class:`int` or :any:`callable`, optional
        Here you can select the algorithm to compute the pseudo-inverse matrix:

            * `1`: use `pinv` from `scipy` which uses `lstsq`
            * `2`: use `pinv2` from `scipy` which uses `SVD`
            * `3`: use `pinvh` from `scipy` which uses eigen-values

        If you want to use another routine to invert the kriging matrix,
        you can pass a callable which takes a matrix and returns the inverse.
        Default: `1`
    """

    def __init__(
        self,
        model,
        cond_pos,
        cond_val,
        drift_functions,
        trend_function=None,
        exact=False,
        cond_err="nugget",
        pseudo_inv=True,
        pseudo_inv_type=1,
    ):
        super().__init__(
            model,
            cond_pos,
            cond_val,
            drift_functions=drift_functions,
            trend_function=trend_function,
            exact=exact,
            cond_err=cond_err,
            pseudo_inv=pseudo_inv,
            pseudo_inv_type=pseudo_inv_type,
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
    pseudo_inv : :class:`bool`, optional
        Whether the kriging system is solved with the pseudo inverted
        kriging matrix. If `True`, this leads to more numerical stability
        and redundant points are averaged. But it can take more time.
        Default: True
    pseudo_inv_type : :class:`int` or :any:`callable`, optional
        Here you can select the algorithm to compute the pseudo-inverse matrix:

            * `1`: use `pinv` from `scipy` which uses `lstsq`
            * `2`: use `pinv2` from `scipy` which uses `SVD`
            * `3`: use `pinvh` from `scipy` which uses eigen-values

        If you want to use another routine to invert the kriging matrix,
        you can pass a callable which takes a matrix and returns the inverse.
        Default: `1`
    """

    def __init__(
        self,
        model,
        cond_pos,
        cond_val,
        ext_drift,
        trend_function=None,
        exact=False,
        cond_err="nugget",
        pseudo_inv=True,
        pseudo_inv_type=1,
    ):
        super().__init__(
            model,
            cond_pos,
            cond_val,
            ext_drift=ext_drift,
            trend_function=trend_function,
            exact=exact,
            cond_err=cond_err,
            pseudo_inv=pseudo_inv,
            pseudo_inv_type=pseudo_inv_type,
        )


class Detrended(Krige):
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
    pseudo_inv : :class:`bool`, optional
        Whether the kriging system is solved with the pseudo inverted
        kriging matrix. If `True`, this leads to more numerical stability
        and redundant points are averaged. But it can take more time.
        Default: True
    pseudo_inv_type : :class:`int` or :any:`callable`, optional
        Here you can select the algorithm to compute the pseudo-inverse matrix:

            * `1`: use `pinv` from `scipy` which uses `lstsq`
            * `2`: use `pinv2` from `scipy` which uses `SVD`
            * `3`: use `pinvh` from `scipy` which uses eigen-values

        If you want to use another routine to invert the kriging matrix,
        you can pass a callable which takes a matrix and returns the inverse.
        Default: `1`
    """

    def __init__(
        self,
        model,
        cond_pos,
        cond_val,
        trend_function,
        exact=False,
        cond_err="nugget",
        pseudo_inv=True,
        pseudo_inv_type=1,
    ):
        super().__init__(
            model,
            cond_pos,
            cond_val,
            trend_function=trend_function,
            unbiased=False,
            exact=exact,
            cond_err=cond_err,
            pseudo_inv=pseudo_inv,
            pseudo_inv_type=pseudo_inv_type,
        )
