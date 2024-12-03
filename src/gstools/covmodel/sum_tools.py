"""
GStools subpackage providing tools for sum-models.

.. currentmodule:: gstools.covmodel.sum_tools

The following classes and functions are provided

.. autosummary::
   RatioError
   ARG_DEF
   default_mod_kwargs
   sum_check
   sum_default_arg_bounds
   sum_default_opt_arg_bounds
   sum_set_norm_var_ratios
   sum_set_norm_len_ratios
   sum_model_repr
"""

# pylint: disable=W0212
import numpy as np

from gstools.covmodel.tools import check_arg_in_bounds
from gstools.tools import RADIAN_SCALE
from gstools.tools.misc import list_format

__all__ = [
    "RatioError",
    "ARG_DEF",
    "default_mod_kwargs",
    "sum_check",
    "sum_default_arg_bounds",
    "sum_default_opt_arg_bounds",
    "sum_set_norm_var_ratios",
    "sum_set_norm_len_ratios",
    "sum_model_repr",
]


class RatioError(Exception):
    """Error for invalid ratios in SumModel."""


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


def sum_set_norm_var_ratios(summod, ratios, skip=None, var=None):
    """
    Set variances of contained models by normalized ratios in [0, 1].

    Ratios are given as normalized ratios in [0, 1] as relative ratio of
    variance to remaining difference to total variance of the Sum-Model.

    Parameters
    ----------
    ratios : iterable
        Ratios to set. Should have a length of len(models) - len(exclude) - 1
    skip : iterable, optional
        Model indices to skip. Should have compatible lenth, by default None
    var : float, optional
        Desired variance, by default current variance

    Raises
    ------
    ValueError
        If number of ratios is not matching.
    """
    skip = skip or set()
    if len(summod) != len(ratios) + len(skip) + 1:
        msg = "SumModel.set_norm_ratios: number of ratios not matching."
        raise ValueError(msg)
    ids = range(len(summod))
    if fail := set(skip) - set(ids):
        msg = f"SumModel.set_norm_var_ratios: skip ids not valid: {fail}"
        raise ValueError(msg)
    var = summod.var if var is None else float(var)
    check_arg_in_bounds(summod, "var", var, error=True)
    var_sum = sum(summod.models[i].var for i in skip)
    if var_sum > var:
        msg = "SumModel.set_norm_var_ratios: skiped variances to big."
        raise RatioError(msg)
    j = 0
    for i in ids:
        if i in skip:
            continue
        var_diff = var - var_sum
        # last model gets remaining diff
        var_set = var_diff * ratios[j] if j < len(ratios) else var_diff
        summod[i].var = var_set
        var_sum += var_set
        j += 1


def sum_set_norm_len_ratios(summod, ratios, skip=None, len_scale=None):
    """
    Set length scales of contained models by normalized ratios in [0, 1].

    Ratios are given as normalized ratios in [0, 1] as relative ratio of
    len_scale * var / total_var to remaining difference to
    total len_scale of the Sum-Model.

    Parameters
    ----------
    ratios : iterable
        Ratios to set. Should have a length of len(models) - len(exclude) - 1
    skip : iterable, optional
        Model indices to skip. Should have compatible lenth, by default None
    len_scale : float, optional
        Desired len_scale, by default current len_scale

    Raises
    ------
    ValueError
        If number of ratios is not matching.
    """
    skip = skip or set()
    if len(summod) != len(ratios) + len(skip) + 1:
        msg = "SumModel.set_norm_len_ratios: number of ratios not matching."
        raise ValueError(msg)
    ids = range(len(summod))
    if fail := set(skip) - set(ids):
        msg = f"SumModel.set_norm_len_ratios: skip ids not valid: {fail}"
        raise ValueError(msg)
    len_scale = summod.len_scale if len_scale is None else float(len_scale)
    check_arg_in_bounds(summod, "len_scale", len_scale, error=True)
    len_sum = sum(summod[i].len_scale * summod.ratios[i] for i in skip)
    if len_sum > len_scale:
        msg = "SumModel.set_norm_len_ratios: skiped length scales to big."
        raise RatioError(msg)
    j = 0
    for i in ids:
        if i in skip:
            continue
        len_diff = len_scale - len_sum
        # last model gets remaining diff
        len_set = len_diff * ratios[j] if j < len(ratios) else len_diff
        summod[i].len_scale = (
            0.0
            if np.isclose(summod.ratios[j], 0)
            else len_set / summod.ratios[j]
        )
        len_sum += len_set
        j += 1


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
