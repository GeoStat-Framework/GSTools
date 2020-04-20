# -*- coding: utf-8 -*-
"""
GStools subpackage providing tools for the covariance-model.

.. currentmodule:: gstools.covmodel.fit

The following classes and functions are provided

.. autosummary::
   fit_variogram
"""

# pylint: disable=C0103
import numpy as np
from scipy.optimize import curve_fit
from gstools.covmodel.tools import check_arg_in_bounds, default_arg_from_bounds


__all__ = ["fit_variogram"]


DEFAULT_PARA = ["var", "len_scale", "nugget"]


def fit_variogram(
    model,
    x_data,
    y_data,
    sill=None,
    init_guess="default",
    weights=None,
    method="trf",
    loss="soft_l1",
    max_eval=None,
    return_r2=False,
    curve_fit_kwargs=None,
    **para_select
):
    """
    Fiting the isotropic variogram-model to given data.

    Parameters
    ----------
    model : :any:`CovModel`
        Covariance Model to fit.
    x_data : :class:`numpy.ndarray`
        The radii of the meassured variogram.
    y_data : :class:`numpy.ndarray`
        The messured variogram
    sill : :class:`float` or :class:`bool`, optional
        Here you can provide a fixed sill for the variogram.
        It needs to be in a fitting range for the var and nugget bounds.
        If variance or nugget are not selected for estimation,
        the nugget will be recalculated to fulfill:

            * sill = var + nugget
            * if the variance is bigger than the sill,
              nugget will bet set to its lower bound
              and the variance will be set to the fitting partial sill.

        If variance is deselected, it needs to be less than the sill,
        otherwise a ValueError comes up. Same for nugget.
        If sill=False, it will be deslected from estimation
        and set to the current sill of the model.
        Then, the procedure above is applied.
        Default: None
    init_guess : :class:`str` or :class:`dict`, optional
        Initial guess for the estimation. Either:

            * "default": using the default values of the covariance model
            * "current": using the current values of the covariance model
            * dict(name: val): specified value for each parameter by name

        Default: "default"
    weights : :class:`str`, :class:`numpy.ndarray`, :class:`callable`, optional
        Weights applied to each point in the estimation. Either:

            * 'inv': inverse distance ``1 / (x_data + 1)``
            * list: weights given per bin
            * callable: function applied to x_data

        If callable, it must take a 1-d ndarray. Then ``weights = f(x_data)``.
        Default: None
    method : {'trf', 'dogbox'}, optional
        Algorithm to perform minimization.

            * 'trf' : Trust Region Reflective algorithm, particularly suitable
              for large sparse problems with bounds. Generally robust method.
            * 'dogbox' : dogleg algorithm with rectangular trust regions,
              typical use case is small problems with bounds. Not recommended
              for problems with rank-deficient Jacobian.

        Default: 'trf'
    loss : :class:`str` or :class:`callable`, optional
        Determines the loss function in scipys curve_fit.
        The following keyword values are allowed:

            * 'linear' (default) : ``rho(z) = z``. Gives a standard
              least-squares problem.
            * 'soft_l1' : ``rho(z) = 2 * ((1 + z)**0.5 - 1)``. The smooth
              approximation of l1 (absolute value) loss. Usually a good
              choice for robust least squares.
            * 'huber' : ``rho(z) = z if z <= 1 else 2*z**0.5 - 1``. Works
              similarly to 'soft_l1'.
            * 'cauchy' : ``rho(z) = ln(1 + z)``. Severely weakens outliers
              influence, but may cause difficulties in optimization process.
            * 'arctan' : ``rho(z) = arctan(z)``. Limits a maximum loss on
              a single residual, has properties similar to 'cauchy'.

        If callable, it must take a 1-d ndarray ``z=f**2`` and return an
        array_like with shape (3, m) where row 0 contains function values,
        row 1 contains first derivatives and row 2 contains second
        derivatives. Default: 'soft_l1'
    max_eval : :class:`int` or :any:`None`, optional
        Maximum number of function evaluations before the termination.
        If None (default), the value is chosen automatically: 100 * n.
    return_r2 : :class:`bool`, optional
        Whether to return the r2 score of the estimation.
        Default: False
    curve_fit_kwargs : :class:`dict`, optional
        Other keyword arguments passed to scipys curve_fit. Default: None
    **para_select
        You can deselect parameters from fitting, by setting
        them "False" using their names as keywords.
        You could also pass fixed values for each parameter.
        Then these values will be applied and the involved parameters wont
        be fitted.
        By default, all parameters are fitted.

    Returns
    -------
    fit_para : :class:`dict`
        Dictonary with the fitted parameter values
    pcov : :class:`numpy.ndarray`
        The estimated covariance of `popt` from
        :any:`scipy.optimize.curve_fit`.
        To compute one standard deviation errors
        on the parameters use ``perr = np.sqrt(np.diag(pcov))``.
    r2_score : :class:`float`, optional
        r2 score of the curve fitting results. Only if return_r2 is True.

    Notes
    -----
    You can set the bounds for each parameter by accessing
    :any:`CovModel.set_arg_bounds`.

    The fitted parameters will be instantly set in the model.
    """
    # preprocess selected parameters
    var_last = False
    for par in para_select:
        if par not in model.arg_bounds:
            raise ValueError(
                "fit: unknow parameter in selection: {}".format(par)
            )
        if not isinstance(para_select[par], bool):
            if par == "var":
                var_last = True
                var_tmp = float(para_select[par])
            else:
                setattr(model, par, float(para_select[par]))
            para_select[par] = False
    # set variance last due to possible recalculations
    if var_last:
        model.var = var_tmp
    # remove those that were set to True
    para_select = {k: v for k, v in para_select.items() if not v}

    # handling the sill
    sill = None if (isinstance(sill, bool) and sill) else sill
    if sill is not None:
        sill = model.sill if isinstance(sill, bool) else float(sill)
        constrain_sill = True
        sill_low = model.arg_bounds["var"][0] + model.arg_bounds["nugget"][0]
        sill_up = model.arg_bounds["var"][1] + model.arg_bounds["nugget"][1]
        if not (sill_low <= sill <= sill_up):
            raise ValueError("fit: sill out of bounds.")
        if "var" in para_select and "nugget" in para_select:
            if model.var > sill:
                model.nugget = model.arg_bounds["nugget"][0]
                model.var = sill - model.nugget
            else:
                model.nugget = sill - model.var
        elif "var" in para_select:
            if model.var > sill:
                raise ValueError(
                    "fit: if sill is fixed and variance deselected, "
                    + "the set variance should be less than the given sill."
                )
            else:
                para_select["nugget"] = False
                model.nugget = sill - model.var
        elif "nugget" in para_select:
            if model.nugget > sill:
                raise ValueError(
                    "fit: if sill is fixed and nugget deselected, "
                    + "the set nugget should be less than the given sill."
                )
            else:
                para_select["var"] = False
                model.var = sill - model.nugget
        else:
            # deselect the nugget, to recalculate it accordingly
            # nugget = sill - var
            para_select["nugget"] = False
    else:
        constrain_sill = False

    # select all parameters to be fitted
    para = {par: True for par in DEFAULT_PARA}
    para.update({opt: True for opt in model.opt_arg})
    # now deselect unwanted parameters
    para.update(para_select)

    # check curve_fit kwargs
    if curve_fit_kwargs is None:
        curve_fit_kwargs = {}
    curve_fit_kwargs["loss"] = loss
    curve_fit_kwargs["max_nfev"] = max_eval

    # check method
    if method not in ["trf", "dogbox"]:
        raise ValueError("fit: method needs to be either 'trf' or 'dogbox'")
    curve_fit_kwargs["method"] = method

    # set weights
    if weights is not None:
        if callable(weights):
            weights = 1.0 / weights(np.array(x_data))
        elif weights == "inv":
            weights = 1.0 + np.array(x_data)
        else:
            weights = 1.0 / np.array(weights)
        curve_fit_kwargs["sigma"] = weights
        curve_fit_kwargs["absolute_sigma"] = True

    # we need arg1, otherwise curve_fit throws an error (bug?!)
    def curve(x, arg1, *args):
        """Adapted Variogram function."""
        args = (arg1,) + args
        para_skip = 0
        opt_skip = 0
        if para["var"]:
            var_tmp = args[para_skip]
            if constrain_sill:
                nugget_tmp = sill - var_tmp
                # punishment, if resulting nugget out of range for fixed sill
                if check_arg_in_bounds(model, "nugget", nugget_tmp) > 0:
                    return np.full_like(x, np.inf)
                # nugget estimation deselected in this case
                model.nugget = nugget_tmp
            para_skip += 1
        if para["len_scale"]:
            model.len_scale = args[para_skip]
            para_skip += 1
        if para["nugget"]:
            model.nugget = args[para_skip]
            para_skip += 1
        for opt in model.opt_arg:
            if para[opt]:
                setattr(model, opt, args[para_skip + opt_skip])
                opt_skip += 1
        # set var at last because of var_factor (other parameter needed)
        if para["var"]:
            model.var = var_tmp
        return model.variogram(x)

    # set the lower/upper boundaries for the variogram-parameters
    low_bounds = []
    top_bounds = []
    init_guess_list = []
    for par in DEFAULT_PARA:
        if para[par]:
            low_bounds.append(model.arg_bounds[par][0])
            if par == "var" and constrain_sill:  # var <= sill in this case
                top_bounds.append(sill)
            else:
                top_bounds.append(model.arg_bounds[par][1])
            init_guess_list.append(
                _init_guess(
                    bounds=[low_bounds[-1], top_bounds[-1]],
                    current=getattr(model, par),
                    default=1.0,
                    typ=init_guess,
                )
            )
    for opt in model.opt_arg:
        if para[opt]:
            low_bounds.append(model.arg_bounds[opt][0])
            top_bounds.append(model.arg_bounds[opt][1])
            init_guess_list.append(
                _init_guess(
                    bounds=[low_bounds[-1], top_bounds[-1]],
                    current=getattr(model, opt),
                    default=model.default_opt_arg()[opt],
                    typ=init_guess,
                )
            )

    # set the remaining kwargs for curve_fit
    curve_fit_kwargs["bounds"] = (low_bounds, top_bounds)
    curve_fit_kwargs["p0"] = init_guess_list
    curve_fit_kwargs["f"] = curve
    curve_fit_kwargs["xdata"] = np.array(x_data)
    curve_fit_kwargs["ydata"] = np.array(y_data)

    # fit the variogram
    popt, pcov = curve_fit(**curve_fit_kwargs)
    # convert the results
    fit_para = {}
    para_skip = 0
    opt_skip = 0
    for par in DEFAULT_PARA:
        if para[par]:
            if par == "var":  # set variance last
                var_tmp = popt[para_skip]
            else:
                setattr(model, par, popt[para_skip])
            fit_para[par] = popt[para_skip]
            para_skip += 1
        else:
            fit_para[par] = getattr(model, par)
    for opt in model.opt_arg:
        if para[opt]:
            setattr(model, opt, popt[para_skip + opt_skip])
            fit_para[opt] = popt[para_skip + opt_skip]
            opt_skip += 1
        else:
            fit_para[opt] = getattr(model, opt)
    # set var at last because of var_factor (other parameter needed)
    if para["var"]:
        model.var = var_tmp
    # calculate the r2 score if wanted
    if return_r2:
        residuals = y_data - model.variogram(x_data)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r2_score = 1.0 - (ss_res / ss_tot)
        return fit_para, pcov, r2_score
    return fit_para, pcov


def _init_guess(bounds, current, default, typ):
    if typ == "default":
        if bounds[0] < default < bounds[1]:
            return default
        return default_arg_from_bounds(bounds)
    elif typ == "current":
        return current
    raise ValueError("CovModel.fit: unkwon init_guess type: '{}'".format(typ))


def logistic_weights(p=0.1, mean=0.7):  # pragma: no cover
    """
    Return a logistic weights function.

    Parameters
    ----------
    p : :class:`float`, optional
        Parameter for the growth rate.
        Within this percentage of the data range, the function will
        be in the upper resp. lower percentile p. The default is 0.1.
    mean : :class:`float`, optional
        Percentage of the data range, where this function has its
        sigmoid's midpoint. The default is 0.7.

    Returns
    -------
    callable
        Weighting function.
    """
    # define the callable weights function
    def func(x_data):
        """Callable function for the weights."""
        x_range = np.amax(x_data) - np.amin(x_data)
        # logit function for growth rate
        growth = np.log(p / (1 - p)) / (p * x_range)
        x_mean = mean * x_range + np.amin(x_data)
        return 1.0 / (1.0 + np.exp(growth * (x_mean - x_data)))

    return func
