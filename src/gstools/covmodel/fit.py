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
    para, sill, anis, sum_cfg = _pre_para(model, para_select, sill, anis)
    # check curve_fit kwargs
    curve_fit_kwargs = curve_fit_kwargs or {}
    # check method
    if method not in ["trf", "dogbox"]:
        raise ValueError("fit: method needs to be either 'trf' or 'dogbox'")
    # prepare variogram data
    # => concatenate directional variograms to have a 1D array for x and y
    x_data, y_data, is_dir_vario = _check_vario(model, x_data, y_data)
    # only fit anisotropy if a directional variogram was given
    anis &= is_dir_vario
    sub_fitting = sum_cfg.get("var_size", 0) + sum_cfg.get("len_size", 0) > 0
    if not (any(para.values()) or anis or sub_fitting):
        raise ValueError("fit: no parameters selected for fitting.")
    # prepare init guess dictionary
    init_guess = _pre_init_guess(
        model, init_guess, np.mean(x_data), np.mean(y_data)
    )
    # set weights
    _set_weights(model, weights, x_data, curve_fit_kwargs, is_dir_vario)
    # set the lower/upper boundaries for the variogram-parameters
    bounds, init_guess_list = _init_curve_fit_para(
        model, para, init_guess, sill, anis, sum_cfg
    )
    # create the fitting curve
    curve_fit_kwargs["f"] = _get_curve(
        model, para, sill, anis, is_dir_vario, sum_cfg
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
    fit_para = _post_fitting(
        model, para, popt, sill, anis, is_dir_vario, sum_cfg
    )
    # calculate the r2 score if wanted
    if return_r2:
        return fit_para, pcov, _r2_score(model, x_data, y_data, is_dir_vario)
    return fit_para, pcov


def _pre_para(model, para_select, sill, anis):
    """Preprocess selected parameters."""
    sum_cfg = {"fix": {}}
    is_sum = hasattr(model, "sub_arg")
    sub_args = getattr(model, "sub_arg", [])
    valid_args = model.iso_arg + sub_args
    bnd = model.arg_bounds
    # if values given, set them in the model, afterwards all entries are bool
    for par, val in para_select.items():
        if par not in valid_args:
            raise ValueError(f"fit: unknown parameter in selection: {par}")
        # if parameters given with value, set it and deselect from fitting
        if not isinstance(val, bool):
            # don't set sub-args, var or len_scale in sum model yet
            if is_sum and par in sub_args + ["var", "len_scale"]:
                sum_cfg["fix"][par] = float(val)
            else:
                setattr(model, par, float(val))
            para_select[par] = False
    # remove those that were set to True
    para_select = {k: v for k, v in para_select.items() if not v}
    # handling sum models
    if is_sum:
        _check_sum(model, para_select, sum_cfg, bnd)
    # handling the sill
    sill = None if (isinstance(sill, bool) and sill) else sill
    if sill is not None:
        sill = model.sill if isinstance(sill, bool) else float(sill)
        _check_sill(model, para_select, sill, bnd, sum_cfg)
    # select all parameters to be fitted (if bounds do not indicate fixed parameter)
    para = {par: bnd[par][0] < bnd[par][1] for par in valid_args}
    # now deselect unwanted parameters
    para.update(para_select)
    # check if anisotropy should be fitted or set
    if not isinstance(anis, bool):
        model.anis = anis
        anis = False
    return para, sill, anis, sum_cfg


def _check_sill(model, para_select, sill, bnd, sum_cfg):
    """
    This functions checks if the selected values for
    variance, nugget and sill are valid.
    """
    is_sum = hasattr(model, "sub_arg")
    sill_low = bnd["var"][0] + bnd["nugget"][0]
    sill_up = bnd["var"][1] + bnd["nugget"][1]
    if not sill_low <= sill <= sill_up:
        raise ValueError("fit: sill out of bounds.")
    if is_sum:
        var_fixed = sum_cfg["var_fix"]
    else:
        var_fixed = "var" in para_select
    if var_fixed and "nugget" in para_select:
        if not np.isclose(model.var + model.nugget, sill):
            msg = "fit: if sill, var and nugget are fixed, var + nugget should match the given sill"
            raise ValueError(msg)
    elif var_fixed:
        if model.var > sill:
            raise ValueError(
                "fit: if sill is fixed and variance deselected, "
                "the set variance should be less than the given sill."
            )
        para_select["nugget"] = False
        # this also works for a pure nugget model
        model.nugget = sill - model.var
    elif "nugget" in para_select:
        if model.nugget > sill:
            raise ValueError(
                "fit: if sill is fixed and nugget deselected, "
                "the set nugget should be less than the given sill."
            )
        para_select["var"] = False
        var = sill - model.nugget
        if is_sum:
            model.set_var_weights(
                np.ones_like(sum_cfg["var_fit"]), sum_cfg["var_skip"], var
            )
            # if only 1 sub-var was to fit, this is now also fixed
            if len(sum_cfg["var_fit"]) == 1:
                i = sum_cfg["var_fit"][0]
                sum_cfg["fix"].setdefault(f"var_{i}", model.vars[i])
                sum_cfg["var_fit"] = []
                sum_cfg["var_skip"] = list(range(model.size))
            sum_cfg["fix"].setdefault("var", var)
            sum_cfg["var_fix"] = True
            # number or sub-var parameters
            sum_cfg["var_size"] = max(
                model.size - len(sum_cfg["var_skip"]) - 1, 0
            )
        else:
            # in case of a nugget model, this should raise an error
            model.var = var
    else:
        # deselect the nugget, to recalculate it accordingly
        # nugget = sill - var
        para_select["nugget"] = False


def _check_sum(model, para_select, sum_cfg, bnd):
    """Check for consistent parameter selection in case of a SumModel."""
    # check len_scale
    if "len_scale" in para_select:
        for par in para_select:
            if par.startswith("len_scale_"):
                msg = (
                    "fit: for sum-models you can only fix "
                    "'len_scale' or the sub-arguments 'len_scale_<i>', not both."
                )
                raise ValueError(msg)
        sum_cfg["fix"].setdefault("len_scale", model.len_scale)
    # use len_scale_<i> for fitting if len_scale not fixed
    # use weights for fitting if len_scale fixed
    # either way: len_scale not used for fitting in sum_model
    para_select["len_scale"] = False
    # check variance
    if "var" in para_select:
        sum_cfg["fix"].setdefault("var", model.var)
    # use var_<i> for fitting if var not fixed
    # use weights for fitting if var fixed
    # either way: var not used for fitting in sum_model
    para_select["var"] = False
    # whether var and len_scale are fixed
    var_fix = "var" in sum_cfg["fix"]
    len_fix = "len_scale" in sum_cfg["fix"]
    size = model.size
    # check for fixed bounds
    for i in range(size):
        for par in [f"var_{i}", f"len_scale_{i}"]:
            if not bnd[par][0] < bnd[par][1]:
                para_select[par] = False
    # check sub arguments (var_<i> and len_scale_<i>)
    var_skip = []
    len_skip = []
    remove = []
    for par in para_select:
        if par.startswith("var_"):
            var_skip.append(int(par[4:]))
            # for fixed var, fit by weights
            if var_fix:
                remove.append(par)
        if par.startswith("len_scale_"):
            len_skip.append(int(par[10:]))
    for par in remove:
        para_select.pop(par)
    var_skip.sort()
    len_skip.sort()
    var_fit = sorted(set(range(size)) - set(var_skip))
    len_fit = sorted(set(range(size)) - set(len_skip))
    # if all sub-vars fixed, total variance is fixed
    if not var_fit:
        para_select["var"] = False
    # set values related to var and len_scale in sum-model
    for i in var_skip:
        sum_cfg["fix"].setdefault(f"var_{i}", model.vars[i])
        setattr(model, f"var_{i}", sum_cfg["fix"][f"var_{i}"])
    for i in len_skip:
        sum_cfg["fix"].setdefault(f"len_scale_{i}", model.len_scales[i])
        setattr(model, f"len_scale_{i}", sum_cfg["fix"][f"len_scale_{i}"])
    if var_fix:
        var_min = sum(sum_cfg["fix"][f"var_{i}"] for i in var_skip)
        if var_min > sum_cfg["fix"]["var"]:
            msg = "fit: fixed sub-variances greater than fixed total variance."
            raise ValueError(msg)
        model.set_var_weights(
            np.ones_like(var_fit), var_skip, sum_cfg["fix"]["var"]
        )
        # if all sub_vars except one is fixed and total var as well, all are fixed
        if len(var_fit) == 1:
            var_fit = []
            var_skip = list(range(size))
    if len_fix:
        model.len_scale = sum_cfg["fix"]["len_scale"]
    # update config for the sum-model fitting
    sum_cfg["var_skip"] = var_skip  # sub vars to skip
    sum_cfg["len_skip"] = len_skip  # sub lens to skip
    sum_cfg["var_fit"] = var_fit  # sub vars to fit
    sum_cfg["len_fit"] = len_fit  # sub lens to fit
    sum_cfg["var_fix"] = var_fix  # total variance fixed
    sum_cfg["len_fix"] = len_fix  # total len-scale fixed
    # number or sub-var parameters
    sum_cfg["var_size"] = max(size - len(var_skip) - int(var_fix), 0)
    # number or sub-len parameters
    sum_cfg["len_size"] = max(size - len(len_skip) - int(len_fix), 0)


def _pre_init_guess(model, init_guess, mean_x=1.0, mean_y=1.0):
    # init guess should be a dict
    if not isinstance(init_guess, dict):
        init_guess = {"default": init_guess}
    # "default" init guess is the respective default value
    default_guess = init_guess.pop("default", "default")
    if default_guess not in ["default", "current"]:
        raise ValueError(f"fit_variogram: unknown def. guess: {default_guess}")
    default = default_guess == "default"
    is_sum = hasattr(model, "sub_arg")
    sub_args = getattr(model, "sub_arg", [])
    valid_args = model.iso_arg + sub_args + ["anis"]
    # check invalid names for given init guesses
    invalid_para = set(init_guess) - set(valid_args)
    if invalid_para:
        raise ValueError(f"fit_variogram: unknown init guess: {invalid_para}")
    bnd = model.arg_bounds
    # default length scale is mean of given bin centers (respecting "rescale")
    len_def = mean_x * model.rescale
    init_guess.setdefault("len_scale", len_def if default else model.len_scale)
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
    # SumModel: check for var_<i> and len_scale_<i> and set defaults
    if is_sum:
        for i in range(model.size):
            init_guess.setdefault(
                f"len_scale_{i}", len_def if default else model.len_scales[i]
            )
            init_guess.setdefault(
                f"var_{i}", mean_y / model.size if default else model.vars[i]
            )
    # convert all init guesses to float (except "anis")
    for arg in model.iso_arg + sub_args:
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


def _init_curve_fit_para(model, para, init_guess, sill, anis, sum_cfg):
    """Create initial guess and bounds for fitting."""
    is_sum = hasattr(model, "sub_arg")
    low_bounds = []
    top_bounds = []
    init_guess_list = []
    bnd = model.arg_bounds
    for par in model.iso_arg:
        if para[par]:
            low_bounds.append(bnd[par][0])
            if par == "var" and sill is not None:  # var <= sill in this case
                top_bounds.append(sill)
            else:
                top_bounds.append(bnd[par][1])
            init_guess_list.append(
                _init_guess(
                    bounds=[low_bounds[-1], top_bounds[-1]],
                    default=init_guess[par],
                )
            )
    if is_sum:
        if sum_cfg["var_fix"]:
            for _ in range(sum_cfg["var_size"]):
                low_bounds.append(0.0)
                top_bounds.append(1.0)
                init_guess_list.append(0.5)
        else:
            for i in sum_cfg["var_fit"]:
                par = f"var_{i}"
                low_bounds.append(bnd[par][0])
                top_bounds.append(bnd[par][1])
                init_guess_list.append(
                    _init_guess(
                        bounds=[low_bounds[-1], top_bounds[-1]],
                        default=init_guess[par],
                    )
                )
        if sum_cfg["len_fix"]:
            for _ in range(sum_cfg["len_size"]):
                low_bounds.append(0.0)
                top_bounds.append(1.0)
                init_guess_list.append(0.5)
        else:
            for i in sum_cfg["len_fit"]:
                par = f"len_scale_{i}"
                low_bounds.append(bnd[par][0])
                top_bounds.append(bnd[par][1])
                init_guess_list.append(
                    _init_guess(
                        bounds=[low_bounds[-1], top_bounds[-1]],
                        default=init_guess[par],
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


def _get_curve(model, para, sill, anis, is_dir_vario, sum_cfg):
    """Create the curve for scipys curve_fit."""
    is_sum = hasattr(model, "sub_arg")

    # we need arg1, otherwise curve_fit throws an error (bug?!)
    def curve(x, arg1, *args):
        """Adapted Variogram function."""
        args = (arg1,) + args
        para_skip = 0
        for par in model.iso_arg:
            if para[par]:
                setattr(model, par, args[para_skip])
                para_skip += 1
        # set var and len-scale ratios in sum-models
        if is_sum:
            if sum_cfg["var_size"] > 0:
                var_vals = args[para_skip : para_skip + sum_cfg["var_size"]]
                para_skip += sum_cfg["var_size"]
                if sum_cfg["var_fix"]:
                    model.set_var_weights(
                        stick_breaking_uniform(var_vals),
                        sum_cfg["var_skip"],
                        sum_cfg["fix"]["var"],
                    )
                else:
                    for i, val in zip(sum_cfg["var_fit"], var_vals):
                        setattr(model, f"var_{i}", val)
            if sum_cfg["len_size"] > 0:
                len_vals = args[para_skip : para_skip + sum_cfg["len_size"]]
                para_skip += sum_cfg["len_size"]
                if sum_cfg["len_fix"]:
                    model.set_len_weights(
                        stick_breaking_uniform(len_vals),
                        sum_cfg["len_skip"],
                        sum_cfg["fix"]["len_scale"],
                    )
                else:
                    for i, val in zip(sum_cfg["len_fit"], len_vals):
                        setattr(model, f"len_scale_{i}", val)
        # handle sill
        if sill is not None and para["var"]:
            nugget_tmp = sill - model.var
            # punishment, if resulting nugget out of range for fixed sill
            if check_arg_in_bounds(model, "nugget", nugget_tmp) > 0:
                return np.full_like(x, np.inf)
            # nugget estimation deselected in this case
            model.nugget = nugget_tmp
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


def _post_fitting(model, para, popt, sill, anis, is_dir_vario, sum_cfg):
    """Postprocess fitting results and application to model."""
    is_sum = hasattr(model, "sub_arg")
    fit_para = {}
    para_skip = 0
    for par in model.iso_arg:
        if para[par]:
            setattr(model, par, popt[para_skip])
            fit_para[par] = popt[para_skip]
            para_skip += 1
        else:
            fit_para[par] = getattr(model, par)
    # set var and len-scale ratios in sum-models
    if is_sum:
        if sum_cfg["var_size"] > 0:
            var_vals = popt[para_skip : para_skip + sum_cfg["var_size"]]
            para_skip += sum_cfg["var_size"]
            if sum_cfg["var_fix"]:
                model.set_var_weights(
                    stick_breaking_uniform(var_vals),
                    sum_cfg["var_skip"],
                    sum_cfg["fix"]["var"],
                )
            else:
                for i, val in zip(sum_cfg["var_fit"], var_vals):
                    setattr(model, f"var_{i}", val)
        if sum_cfg["len_size"] > 0:
            len_vals = popt[para_skip : para_skip + sum_cfg["len_size"]]
            para_skip += sum_cfg["len_size"]
            if sum_cfg["len_fix"]:
                model.set_len_weights(
                    stick_breaking_uniform(len_vals),
                    sum_cfg["len_skip"],
                    sum_cfg["fix"]["len_scale"],
                )
            else:
                for i, val in zip(sum_cfg["len_fit"], len_vals):
                    setattr(model, f"len_scale_{i}", val)
        for i in range(model.size):
            fit_para[f"var_{i}"] = model.vars[i]
            fit_para[f"len_scale_{i}"] = model.len_scales[i]
    # handle sill
    if sill is not None and para["var"]:
        nugget = sill - model.var
        fit_para["nugget"] = nugget
        model.nugget = nugget
    if is_dir_vario:
        if anis:
            model.anis = popt[1 - model.dim :]
        fit_para["anis"] = model.anis
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


def stick_breaking_uniform(u):
    """
    Generate a single sample (x_1, ..., x_n) uniformly from the (n-1)-simplex.

    This is using Beta transforms of uniform samples. The x_i will sum to 1.

    Parameters
    ----------
    u : array-like of shape (n-1,)
        Uniform(0,1) random values

    Returns
    -------
    x : ndarray of shape (n,)
        A random sample in the (n-1)-simplex.
    """
    n = len(u) + 1
    x = np.zeros(n)
    leftover = 1.0
    for i in range(n - 1):
        # 2) Compute the inverse CDF of Beta(1, b) = Beta(1, n-1-i)
        fraction = 1.0 - (1.0 - u[i]) ** (1.0 / (n - 1 - i))
        # 3) Break that fraction of the current leftover
        x[i] = leftover * fraction
        # 4) Subtract that from leftover
        leftover -= x[i]
    # Last coordinate
    x[-1] = leftover
    return x
