"""
GStools subpackage providing tools for sum-models.

.. currentmodule:: gstools.covmodel.sum_tools

The following classes and functions are provided

.. autosummary::
   ARG_DEF
   default_mod_kwargs
   sum_check
   sum_compare
   sum_default_arg_bounds
   sum_default_opt_arg_bounds
   sum_set_norm_var_ratios
   sum_set_norm_len_ratios
   sum_model_repr
"""

# pylint: disable=W0212
import numpy as np

from gstools.tools import RADIAN_SCALE
from gstools.tools.misc import list_format

__all__ = [
    "ARG_DEF",
    "default_mod_kwargs",
    "sum_check",
    "sum_compare",
    "sum_default_arg_bounds",
    "sum_default_opt_arg_bounds",
    "sum_set_var_weights",
    "sum_set_len_weights",
    "sum_model_repr",
]


ARG_DEF = {
    "dim": 3,
    "latlon": False,
    "temporal": False,
    "geo_scale": RADIAN_SCALE,
    "spatial_dim": None,
    "hankel_kw": None,
}
"""dict: default model arguments"""


def default_mod_kwargs(kwargs):
    """Generate default model keyword arguments."""
    mod_kw = {}
    for arg, default in ARG_DEF.items():
        mod_kw[arg] = kwargs.get(arg, default)
    return mod_kw


def sum_check(summod):
    """Check consistency of contained models."""
    # prevent dim error in anis and angles
    if any(mod.dim != summod.dim for mod in summod):
        msg = "SumModel: models need to have same dimension."
        raise ValueError(msg)
    if any(mod.latlon != summod.latlon for mod in summod):
        msg = "SumModel: models need to have same latlon config."
        raise ValueError(msg)
    if any(mod.temporal != summod.temporal for mod in summod):
        msg = "SumModel: models need to have same temporal config."
        raise ValueError(msg)
    if not all(np.isclose(mod.nugget, 0) for mod in summod):
        msg = "SumModel: models need to have 0 nugget."
        raise ValueError(msg)
    if not np.allclose([mod.geo_scale for mod in summod], summod.geo_scale):
        msg = "SumModel: models need to have same geo_scale."
        raise ValueError(msg)
    if not all(np.allclose(mod.anis, summod.anis) for mod in summod):
        msg = "SumModel: models need to have same anisotropy ratios."
        raise ValueError(msg)
    if not all(np.allclose(mod.angles, summod.angles) for mod in summod):
        msg = "SumModel: models need to have same rotation angles."
        raise ValueError(msg)


def sum_default_arg_bounds(summod):
    """Default boundaries for arguments as dict."""
    var_bnds = [mod.var_bounds for mod in summod.models]
    len_bnds = [mod.len_scale_bounds for mod in summod.models]
    var_lo = sum((bnd[0] for bnd in var_bnds), start=0.0)
    var_hi = sum((bnd[1] for bnd in var_bnds), start=0.0)
    len_lo = min((bnd[0] for bnd in len_bnds), default=0.0)
    len_hi = max((bnd[1] for bnd in len_bnds), default=0.0)
    res = {
        "var": (var_lo, var_hi),
        "len_scale": (len_lo, len_hi),
        "nugget": (0.0, np.inf, "co"),
        "anis": (0.0, np.inf, "oo"),
    }
    return res


def sum_default_opt_arg_bounds(summod):
    """Defaults boundaries for optional arguments as dict."""
    bounds = {}
    for i, mod in enumerate(summod.models):
        bounds.update(
            {f"{opt}_{i}": bnd for opt, bnd in mod.opt_arg_bounds.items()}
        )
    return bounds


def sum_set_var_weights(summod, weights, skip=None, var=None):
    """
    Set variances of contained models by weights.

    Parameters
    ----------
    weights : iterable
        Weights to set. Should have a length of len(models) - len(skip)
    skip : iterable, optional
        Model indices to skip. Should have compatible length, by default None
    var : float, optional
        Desired variance, by default current variance

    Raises
    ------
    ValueError
        If number of weights is not matching.
    """
    skip = set() if skip is None else set(skip)
    if len(summod) != len(weights) + len(skip):
        msg = "SumModel.set_var_weights: number of ratios not matching."
        raise ValueError(msg)
    ids = range(len(summod))
    if fail := set(skip) - set(ids):
        msg = (
            f"SumModel.set_var_weights: ids given by 'skip' not valid: {fail}"
        )
        raise ValueError(msg)
    var = summod.var if var is None else float(var)
    var_sum = sum(summod.models[i].var for i in skip)
    var_diff = var - var_sum
    if var_diff < 0:
        msg = (
            "SumModel.set_var_weights: summed variances selected "
            "with 'skip' already too big to keep total variance."
        )
        raise ValueError(msg)
    weights_sum = sum(weights)
    var_list = summod.vars
    j = 0
    for i in ids:
        if i in skip:
            continue
        var_list[i] = var_diff * weights[j] / weights_sum
        j += 1
    summod.vars = var_list


def sum_set_len_weights(summod, weights, skip=None, len_scale=None):
    """
    Set length scales of contained models by weights.

    Parameters
    ----------
    weights : iterable
        Weights to set. Should have a length of len(models) - len(skip)
    skip : iterable, optional
        Model indices to skip. Should have compatible length, by default None
    len_scale : float, optional
        Desired len_scale, by default current len_scale

    Raises
    ------
    ValueError
        If number of weights is not matching.
    """
    skip = set() if skip is None else set(skip)
    if len(summod) != len(weights) + len(skip):
        msg = "SumModel.set_len_weights: number of weights not matching."
        raise ValueError(msg)
    ids = range(len(summod))
    if fail := set(skip) - set(ids):
        msg = (
            f"SumModel.set_len_weights: ids given by 'skip' not valid: {fail}"
        )
        raise ValueError(msg)
    len_scale = summod.len_scale if len_scale is None else float(len_scale)
    # also skip models with no variance (not contributing to total len scale)
    j = 0
    wei = []
    for i in ids:
        if i in skip:
            continue
        if np.isclose(summod.ratios[i], 0):
            skip.add(i)
        else:
            wei.append(weights[j])
        j += 1
    weights = wei
    len_sum = sum(summod[i].len_scale * summod.ratios[i] for i in skip)
    len_diff = len_scale - len_sum
    if len_diff < 0:
        msg = (
            "SumModel.set_len_weights: summed length scales "
            "selected with 'skip' already too big to keep total length scale."
        )
        raise ValueError(msg)
    weights_sum = sum(weights)
    len_scales = summod.len_scales
    j = 0
    for i in ids:
        if i in skip:
            continue
        len_scales[i] = len_diff * weights[j] / weights_sum / summod.ratios[j]
        j += 1
    summod.len_scales = len_scales


def sum_compare(this, that):
    """
    Compare SumModels.

    Parameters
    ----------
    this / that : :any:`SumModel`
        The sum models to compare.
    """
    if len(this) != len(that):
        return False
    if not np.isclose(this.nugget, that.nugget):
        return False
    return all(mod1 == mod2 for (mod1, mod2) in zip(this, that))


def sum_model_repr(summod):  # pragma: no cover
    """
    Generate the sum-model string representation.

    Parameters
    ----------
    model : :any:`SumModel`
        The sum-model in use.
    """
    m, p = summod, summod._prec
    ani_str, ang_str, o_str, r_str, p_str = "", "", "", "", ""
    m_str = ", ".join([mod.name for mod in m.models])
    t_str = ", temporal=True" if m.temporal else ""
    d_str = f"latlon={m.latlon}" if m.latlon else f"dim={m.spatial_dim}"
    if len(m) > 0:
        m_str += ", "
        p_str += f", vars={list_format(m.vars, p)}"
        p_str += f", len_scales={list_format(m.len_scales, p)}"
    p_str += "" if np.isclose(m.nugget, 0) else f", nugget={m.nugget:.{p}}"
    for opt in m.opt_arg:
        o_str += f", {opt}={getattr(m, opt):.{p}}"
    if m.latlon:
        if not m.is_isotropic and m.temporal:
            ani_str = f", anis={m.anis[-1]:.{p}}"
        if not np.isclose(m.geo_scale, 1):
            r_str = f", geo_scale={m.geo_scale:.{p}}"
    else:
        if not m.is_isotropic:
            ani_str = f", anis={list_format(m.anis, p)}"
        if m.do_rotation:
            ang_str = f", angles={list_format(m.angles, p)}"
    return f"{m.name}({m_str}{d_str}{t_str}{p_str}{ani_str}{ang_str}{r_str}{o_str})"
