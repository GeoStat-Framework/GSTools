"""
GStools subpackage providing tools for the covariance-model.

.. currentmodule:: gstools.covmodel.fit

The following classes and functions are provided

.. autosummary::
   fit_variogram
"""

# pylint: disable=C0103, W0632
import numpy as np
from scipy.optimize import curve_fit

from gstools.covmodel.tools import check_arg_in_bounds, default_arg_from_bounds
from gstools.tools.geometric import great_circle_to_chordal, set_anis

__all__ = ["fit_variogram"]


DEFAULT_PARA = ["var", "len_scale", "nugget"]


def fit_variogram(
    model,
    x_data,
    y_data,
    anis=True,
    sill=None,
    init_guess="default",
    weights=None,
    method="trf",
    loss="soft_l1",
    max_eval=None,
    return_r2=False,
    curve_fit_kwargs=None,
    **para_select,
):
    """
    Fitting a variogram-model to an empirical variogram.

    Parameters
    ----------
    model : :any:`CovModel`
        Covariance Model to fit.
    x_data : :class:`numpy.ndarray`
        The bin-centers of the empirical variogram.
    y_data : :class:`numpy.ndarray`
        The measured variogram
        If multiple are given, they are interpreted as the directional
        variograms along the main axis of the associated rotated
        coordinate system.
        Anisotropy ratios will be estimated in that case.
    anis : :class:`bool`, optional
        In case of a directional variogram, you can control anisotropy
        by this argument. Deselect the parameter from fitting, by setting
        it "False".
        You could also pass a fixed value to be set in the model.
        Then the anisotropy ratios won't be altered during fitting.
        Default: True
    sill : :class:`float` or :class:`bool` or :any:`None`, optional
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
        If sill=False, it will be deselected from estimation
        and set to the current sill of the model.
        Then, the procedure above is applied.
        Default: None
    init_guess : :class:`str` or :class:`dict`, optional
        Initial guess for the estimation. Either:

            * "default": using the default values of the covariance model
              ("len_scale" will be mean of given bin centers;
              "var" and "nugget" will be mean of given variogram values
              (if in given bounds))
            * "current": using the current values of the covariance model
            * dict: dictionary with parameter names and given value
              (separate "default" can bet set to "default" or "current" for
              unspecified values to get same behavior as given above
              ("default" by default))
              Example: ``{"len_scale": 10, "default": "current"}``

        Default: "default"
    weights : :class:`str`, :class:`numpy.ndarray`, :class:`callable`optional
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
        Dictionary with the fitted parameter values
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
    para, sill, constrain_sill, anis = _pre_para(
        model, para_select, sill, anis
    )
    # check curve_fit kwargs
    curve_fit_kwargs = {} if curve_fit_kwargs is None else curve_fit_kwargs
    # check method
    if method not in ["trf", "dogbox"]:
        raise ValueError("fit: method needs to be either 'trf' or 'dogbox'")
    # prepare variogram data
    # => concatenate directional variograms to have a 1D array for x and y
    x_data, y_data, is_dir_vario = _check_vario(model, x_data, y_data)
    # prepare init guess dictionary
    init_guess = _pre_init_guess(
        model, init_guess, np.mean(x_data), np.mean(y_data)
    )
    # only fit anisotropy if a directional variogram was given
    anis &= is_dir_vario
    # set weights
    _set_weights(model, weights, x_data, curve_fit_kwargs, is_dir_vario)
    # set the lower/upper boundaries for the variogram-parameters
    bounds, init_guess_list = _init_curve_fit_para(
        model, para, init_guess, constrain_sill, sill, anis
    )
    # create the fitting curve
    curve_fit_kwargs["f"] = _get_curve(
        model, para, constrain_sill, sill, anis, is_dir_vario
    )
    # set the remaining kwargs for curve_fit
    curve_fit_kwargs["bounds"] = bounds
    curve_fit_kwargs["p0"] = init_guess_list
    curve_fit_kwargs["xdata"] = x_data
    curve_fit_kwargs["ydata"] = y_data
    curve_fit_kwargs["loss"] = loss
    curve_fit_kwargs["max_nfev"] = max_eval
    curve_fit_kwargs["method"] = method
    # fit the variogram
    popt, pcov = curve_fit(**curve_fit_kwargs)
    # convert the results
    fit_para = _post_fitting(model, para, popt, anis, is_dir_vario)
    # calculate the r2 score if wanted
    if return_r2:
        return fit_para, pcov, _r2_score(model, x_data, y_data, is_dir_vario)
    return fit_para, pcov


def _pre_para(model, para_select, sill, anis):
    """Preprocess selected parameters."""
    var_last = False
    var_tmp = 0.0  # init value
    for par in para_select:
        if par not in model.arg_bounds:
            raise ValueError(f"fit: unknown parameter in selection: {par}")
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
        if not sill_low <= sill <= sill_up:
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
                    "the set variance should be less than the given sill."
                )
            para_select["nugget"] = False
            model.nugget = sill - model.var
        elif "nugget" in para_select:
            if model.nugget > sill:
                raise ValueError(
                    "fit: if sill is fixed and nugget deselected, "
                    "the set nugget should be less than the given sill."
                )
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
    # check if anisotropy should be fitted or set
    if not isinstance(anis, bool):
        model.anis = anis
        anis = False
    return para, sill, constrain_sill, anis


def _pre_init_guess(model, init_guess, mean_x=1.0, mean_y=1.0):
    # init guess should be a dict
    if not isinstance(init_guess, dict):
        init_guess = {"default": init_guess}
    # "default" init guess is the respective default value
    default_guess = init_guess.pop("default", "default")
    if default_guess not in ["default", "current"]:
        raise ValueError(f"fit_variogram: unknown def. guess: {default_guess}")
    default = default_guess == "default"
    # check invalid names for given init guesses
    invalid_para = set(init_guess) - set(model.iso_arg + ["anis"])
    if invalid_para:
        raise ValueError(f"fit_variogram: unknown init guess: {invalid_para}")
    bnd = model.arg_bounds
    # default length scale is mean of given bin centers (respecting "rescale")
    init_guess.setdefault(
        "len_scale", mean_x * model.rescale if default else model.len_scale
    )
    # init guess for variance and nugget is mean of given variogram
    for par in ["var", "nugget"]:
        init_guess.setdefault(par, mean_y if default else getattr(model, par))
    # anis setting
    init_guess.setdefault(
        "anis", default_arg_from_bounds(bnd["anis"]) if default else model.anis
    )
    # correctly handle given values for anis (need a list of values)
    init_guess["anis"] = list(set_anis(model.dim, init_guess["anis"]))
    # set optional arguments
    for opt in model.opt_arg:
        init_guess.setdefault(
            opt,
            (
                default_arg_from_bounds(bnd[opt])
                if default
                else getattr(model, opt)
            ),
        )
    # convert all init guesses to float (except "anis")
    for arg in model.iso_arg:
        init_guess[arg] = float(init_guess[arg])
    return init_guess


def _check_vario(model, x_data, y_data):
    # prepare variogram data
    x_data = np.asarray(x_data).reshape(-1)
    y_data = np.asarray(y_data).reshape(-1)
    # if multiple variograms are given, they will be interpreted
    # as directional variograms along the main rotated axes of the model
    is_dir_vario = False
    if model.dim > 1 and x_data.size * model.dim == y_data.size:
        is_dir_vario = True
        # concatenate multiple variograms
        x_data = np.tile(x_data, model.dim)
    elif x_data.size != y_data.size:
        raise ValueError(
            "CovModel.fit_variogram: Wrong number of empirical variograms! "
            "Either provide only one variogram to fit an isotropic model, "
            "or directional ones for all main axes to fit anisotropy."
        )
    if is_dir_vario and model.latlon:
        raise ValueError(
            "CovModel.fit_variogram: lat-lon models don't support anisotropy."
        )
    if model.latlon:
        # convert to yadrenko model
        x_data = great_circle_to_chordal(x_data, model.geo_scale)
    return x_data, y_data, is_dir_vario


def _set_weights(model, weights, x_data, curve_fit_kwargs, is_dir_vario):
    if weights is not None:
        if callable(weights):
            weights = 1.0 / weights(x_data)
        elif isinstance(weights, str) and weights == "inv":
            weights = 1.0 + x_data
        else:
            if is_dir_vario and weights.size * model.dim == x_data.size:
                weights = np.tile(weights, model.dim)
            weights = 1.0 / np.asarray(weights).reshape(-1)
        curve_fit_kwargs["sigma"] = weights
        curve_fit_kwargs["absolute_sigma"] = True


def _init_curve_fit_para(model, para, init_guess, constrain_sill, sill, anis):
    """Create initial guess and bounds for fitting."""
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
                    default=init_guess[par],
                )
            )
    for opt in model.opt_arg:
        if para[opt]:
            low_bounds.append(model.arg_bounds[opt][0])
            top_bounds.append(model.arg_bounds[opt][1])
            init_guess_list.append(
                _init_guess(
                    bounds=[low_bounds[-1], top_bounds[-1]],
                    default=init_guess[opt],
                )
            )
    if anis:
        for i in range(model.dim - 1):
            low_bounds.append(model.anis_bounds[0])
            top_bounds.append(model.anis_bounds[1])
            init_guess_list.append(
                _init_guess(
                    bounds=[low_bounds[-1], top_bounds[-1]],
                    default=init_guess["anis"][i],
                )
            )
    return (low_bounds, top_bounds), init_guess_list


def _init_guess(bounds, default):
    """Proper determination of initial guess."""
    if bounds[0] < default < bounds[1]:
        return default
    return default_arg_from_bounds(bounds)


def _get_curve(model, para, constrain_sill, sill, anis, is_dir_vario):
    """Create the curve for scipys curve_fit."""
    var_save = model.var

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
        # needs to be reset for TPL models when len_scale was changed
        else:
            model.var = var_save
        if is_dir_vario:
            if anis:
                model.anis = args[1 - model.dim :]
            xs = x[: x.size // model.dim]
            out = np.array([], dtype=np.double)
            for i in range(model.dim):
                out = np.concatenate((out, model.vario_axis(xs, axis=i)))
            return out
        return model.variogram(x)

    return curve


def _post_fitting(model, para, popt, anis, is_dir_vario):
    """Postprocess fitting results and application to model."""
    fit_para = {}
    para_skip = 0
    opt_skip = 0
    var_tmp = 0.0  # init value
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
    if is_dir_vario:
        if anis:
            model.anis = popt[1 - model.dim :]
        fit_para["anis"] = model.anis
    # set var at last because of var_factor (other parameter needed)
    if para["var"]:
        model.var = var_tmp
    return fit_para


def _r2_score(model, x_data, y_data, is_dir_vario):
    """Calculate the R2 score."""
    if is_dir_vario:
        xs = x_data[: x_data.size // model.dim]
        vario = np.array([], dtype=np.double)
        for i in range(model.dim):
            vario = np.concatenate((vario, model.vario_axis(xs, axis=i)))
    else:
        vario = model.variogram(x_data)
    residuals = y_data - vario
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    return 1.0 - (ss_res / ss_tot)


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
