"""
GStools subpackage providing tools for the covariance-model.

.. currentmodule:: gstools.covmodel.tools

The following classes and functions are provided

.. autosummary::
   AttributeWarning
   rad_fac
   set_opt_args
   set_len_anis
   check_bounds
   check_arg_in_bounds
   default_arg_from_bounds
   spectral_rad_pdf
   percentile_scale
   set_arg_bounds
   check_arg_bounds
   set_dim
   compare
   model_repr
"""

import warnings

import numpy as np
from hankel import SymmetricFourierTransform as SFT
from scipy import special as sps
from scipy.optimize import root

from gstools.tools.geometric import no_of_angles, set_angles, set_anis
from gstools.tools.misc import list_format

__all__ = [
    "AttributeWarning",
    "rad_fac",
    "set_opt_args",
    "set_len_anis",
    "set_model_angles",
    "check_bounds",
    "check_arg_in_bounds",
    "default_arg_from_bounds",
    "spectral_rad_pdf",
    "percentile_scale",
    "set_arg_bounds",
    "check_arg_bounds",
    "set_dim",
    "compare",
    "model_repr",
]


class AttributeWarning(UserWarning):
    """Attribute warning for CovModel class."""


def _init_subclass(cls):
    """Initialize gstools covariance model."""

    def variogram(self, r):
        """Isotropic variogram of the model."""
        return self.var - self.covariance(r) + self.nugget

    def covariance(self, r):
        """Covariance of the model."""
        return self.var * self.correlation(r)

    def correlation(self, r):
        """Correlation function of the model."""
        return 1.0 - (self.variogram(r) - self.nugget) / self.var

    def correlation_from_cor(self, r):
        """Correlation function of the model."""
        r = np.asarray(np.abs(r), dtype=np.double)
        return self.cor(r / self.len_rescaled)

    def cor_from_correlation(self, h):
        """Correlation taking a non-dimensional range."""
        h = np.asarray(np.abs(h), dtype=np.double)
        return self.correlation(h * self.len_rescaled)

    abstract = True
    if hasattr(cls, "cor"):
        if not hasattr(cls, "correlation"):
            cls.correlation = correlation_from_cor
        abstract = False
    else:
        cls.cor = cor_from_correlation
    if not hasattr(cls, "variogram"):
        cls.variogram = variogram
    else:
        abstract = False
    if not hasattr(cls, "covariance"):
        cls.covariance = covariance
    else:
        abstract = False
    if not hasattr(cls, "correlation"):
        cls.correlation = correlation
    else:
        abstract = False
    if abstract:
        raise TypeError(
            f"Can't instantiate class '{cls.__name__}', "
            "without providing at least one of the methods "
            "'cor', 'variogram', 'covariance' or 'correlation'."
        )


# Helping functions ###########################################################


def rad_fac(dim, r):
    """Volume element of the n-dimensional spherical coordinates.

    Given as a factor for integration of a radial-symmetric function.

    Parameters
    ----------
    dim : :class:`int`
        spatial dimension
    r : :class:`numpy.ndarray`
        Given radii.
    """
    if dim == 1:
        fac = 2.0
    elif dim == 2:
        fac = 2 * np.pi * r
    elif dim == 3:
        fac = 4 * np.pi * r**2
    else:  # pragma: no cover
        fac = (
            dim
            * r ** (dim - 1)
            * np.sqrt(np.pi) ** dim
            / sps.gamma(dim / 2 + 1)
        )
    return fac


def set_opt_args(model, opt_arg):
    """
    Set optional arguments in the model class.

    Parameters
    ----------
    model : :any:`CovModel`
        The covariance model in use.
    opt_arg : :class:`dict`
        Dictionary with optional arguments.

    Raises
    ------
    ValueError
        When an optional argument has an already taken name.
    """
    model._opt_arg = []
    # look up the defaults for the optional arguments (defined by the user)
    default = model.default_opt_arg()
    for opt_name in opt_arg:
        if opt_name not in default:
            warnings.warn(
                f"The given optional argument '{opt_name}' "
                "is unknown or has at least no defined standard value. "
                "Or you made a Typo... hehe.",
                AttributeWarning,
            )
    # add the default values if not specified
    for def_arg in default:
        if def_arg not in opt_arg:
            opt_arg[def_arg] = default[def_arg]
    # save names of the optional arguments (sort them by name)
    model._opt_arg = sorted(opt_arg)
    # add the optional arguments as attributes to the class
    for opt_name in opt_arg:
        if opt_name in dir(model):  # "dir" also respects properties
            raise ValueError(
                f"parameter '{opt_name}' has a 'bad' name, "
                "since it is already present in "
                "the class. It could not be added to the model."
            )
        # Magic happens here
        setattr(model, opt_name, float(opt_arg[opt_name]))


def set_len_anis(dim, len_scale, anis, latlon=False):
    """Set the length scale and anisotropy factors for the given dimension.

    Parameters
    ----------
    dim : :class:`int`
        spatial dimension
    len_scale : :class:`float` or :class:`list`
        the length scale of the SRF in x direction or in x- (y-, ...) direction
    anis : :class:`float` or :class:`list`
        the anisotropy of length scales along the transversal axes
    latlon : :class:`bool`, optional
        Whether the model is describing 2D fields on earths surface described
        by latitude and longitude. In this case there is no spatial anisotropy.
        Default: False

    Returns
    -------
    len_scale : :class:`float`
        the main length scale of the SRF in x direction
    anis : :class:`list`, optional
        the anisotropy of length scales along the transversal axes

    Notes
    -----
    If ``len_scale`` is given by at least two values,
    ``anis`` will be recalculated.

    If ``len_scale`` is given as list with to few values, the latter value will
    be used for the remaining dimensions. (e.g. [l_1, l_2] in 3D is equal to
    [l_1, l_2, l_2])

    If to few ``anis`` values are given, the first dimensions will be filled
    up with 1. (eg. anis=[e] in 3D is equal to anis=[1, e])
    """
    ls_tmp = np.array(len_scale, dtype=np.double)
    ls_tmp = np.atleast_1d(ls_tmp)[:dim]
    # use just one length scale (x-direction)
    out_len_scale = ls_tmp[0]
    # set the anisotropies in y- and z-direction according to the input
    if len(ls_tmp) == 1:
        out_anis = set_anis(dim, anis)
    else:
        # fill up length-scales with the latter len_scale, such that len()==dim
        if len(ls_tmp) < dim:
            ls_tmp = np.pad(ls_tmp, (0, dim - len(ls_tmp)), "edge")
        # if multiple length-scales are given, calculate the anisotropies
        out_anis = np.zeros(dim - 1, dtype=np.double)
        for i in range(1, dim):
            out_anis[i - 1] = ls_tmp[i] / ls_tmp[0]
    # sanity check
    for ani in out_anis:
        if not ani > 0.0:
            raise ValueError(
                f"anisotropy-ratios needs to be > 0, got: {out_anis}"
            )
    # no spatial anisotropy for latlon
    if latlon:
        out_anis[:2] = 1.0
    return out_len_scale, out_anis


def set_model_angles(dim, angles, latlon=False, temporal=False):
    """Set the model angles for the given dimension.

    Parameters
    ----------
    dim : :class:`int`
        spatial dimension
    angles : :class:`float` or :class:`list`
        the angles of the SRF
    latlon : :class:`bool`, optional
        Whether the model is describing 2D fields on earths surface described
        by latitude and longitude.
        Default: False
    temporal : :class:`bool`, optional
        Whether a time-dimension is appended.
        Default: False

    Returns
    -------
    angles : :class:`float`
        the angles fitting to the dimension

    Notes
    -----
        If too few angles are given, they are filled up with `0`.
    """
    if latlon:
        return np.array(no_of_angles(dim) * [0], dtype=np.double)
    out_angles = set_angles(dim, angles)
    if temporal:
        # no rotation between spatial dimensions and temporal dimension
        out_angles[no_of_angles(dim - 1) :] = 0.0
    return out_angles


def check_bounds(bounds):
    """
    Check if given bounds are valid.

    Parameters
    ----------
    bounds : list
        bound can contain 2 to 3 values:
            1. lower bound
                float
            2. upper bound
                float
            3. Interval type (optional)
                * "oo" : open - open
                * "oc" : open - close
                * "co" : close - open
                * "cc" : close - close (default)
    """
    typ = bounds[2] if len(bounds) == 3 else "cc"
    if len(bounds) not in (2, 3):
        return False
    if (typ == "cc" and bounds[1] < bounds[0]) or (
        typ != "cc" and bounds[1] <= bounds[0]
    ):
        return False
    if len(bounds) == 3 and bounds[2] not in ("oo", "oc", "co", "cc"):
        return False
    return True


def check_arg_in_bounds(model, arg, val=None, error=False):
    """Check if given argument value is in bounds of the given model."""
    if arg not in model.arg_bounds:
        raise ValueError(f"check bounds: unknown argument: {arg}")
    bnd = list(model.arg_bounds[arg])
    val = getattr(model, arg) if val is None else val
    val = np.asarray(val)
    error_case = 0
    if len(bnd) == 2:
        bnd = list(bnd)
        bnd.append("cc")  # use closed intervals by default
    if bnd[2][0] == "c":
        if np.any(val < bnd[0]):
            error_case = 1
    else:
        if np.any(val <= bnd[0]):
            error_case = 2
    if bnd[2][1] == "c":
        if np.any(val > bnd[1]):
            error_case = 3
    else:
        if np.any(val >= bnd[1]):
            error_case = 4
    if error:
        if error_case == 1:
            raise ValueError(f"{arg} needs to be >= {bnd[0]}, got: {val}")
        if error_case == 2:
            raise ValueError(f"{arg} needs to be > {bnd[0]}, got: {val}")
        if error_case == 3:
            raise ValueError(f"{arg} needs to be <= {bnd[1]}, got: {val}")
        if error_case == 4:
            raise ValueError(f"{arg} needs to be < {bnd[1]}, got: {val}")
    return error_case


def default_arg_from_bounds(bounds):
    """
    Determine a default value from given bounds.

    Parameters
    ----------
    bounds : list
        bounds for the value.

    Returns
    -------
    float
        Default value in the given bounds.
    """
    if bounds[0] > -np.inf and bounds[1] < np.inf:
        return (bounds[0] + bounds[1]) / 2.0
    if bounds[0] > -np.inf:
        return bounds[0] + 1.0
    if bounds[1] < np.inf:
        return bounds[1] - 1.0
    return 0.0  # pragma: no cover


# outsourced routines


def spectral_rad_pdf(model, r):
    """
    Spectral radians PDF of a model.

    Parameters
    ----------
    model : :any:`CovModel`
        The covariance model in use.
    r : :class:`numpy.ndarray`
        Given radii.

    Returns
    -------
    :class:`numpy.ndarray`
        PDF values.

    """
    r = np.asarray(np.abs(r), dtype=np.double)
    if model.dim > 1:
        r_gz = np.logical_not(np.isclose(r, 0))
        # to prevent numerical errors, we just calculate where r>0
        res = np.zeros_like(r, dtype=np.double)
        res[r_gz] = rad_fac(model.dim, r[r_gz]) * np.abs(
            model.spectral_density(r[r_gz])
        )
    else:
        res = rad_fac(model.dim, r) * np.abs(model.spectral_density(r))
    # prevent numerical errors in hankel for small r values (set 0)
    res[np.logical_not(np.isfinite(res))] = 0.0
    # prevent numerical errors in hankel for big r (set non-negative)
    res = np.maximum(res, 0.0)
    return res


def percentile_scale(model, per=0.9):
    """
    Calculate the percentile scale of the isotrope model.

    This is the distance, where the given percentile of the variance
    is reached by the variogram


    Parameters
    ----------
    model : :any:`CovModel`
        The covariance model in use.
    per : float, optional
        Percentile to use. The default is 0.9.

    Raises
    ------
    ValueError
        When percentile is not in (0, 1).

    Returns
    -------
    float
        Percentile scale.

    """
    # check the given percentile
    if not 0.0 < per < 1.0:
        raise ValueError(f"percentile needs to be within (0, 1), got: {per}")

    # define a curve, that has its root at the wanted point
    def curve(x):
        return 1.0 - model.correlation(x) - per

    # take 'per * len_rescaled' as initial guess
    return root(curve, per * model.len_rescaled)["x"][0]


def set_arg_bounds(model, check_args=True, **kwargs):
    r"""Set bounds for the parameters of the model.

    Parameters
    ----------
    model : :any:`CovModel`
        The covariance model in use.
    check_args : bool, optional
        Whether to check if the arguments are in their valid bounds.
        In case not, a proper default value will be determined.
        Default: True
    **kwargs
        Parameter name as keyword ("var", "len_scale", "nugget", <opt_arg>)
        and a list of 2 or 3 values as value:

            * ``[a, b]`` or
            * ``[a, b, <type>]``

        <type> is one of ``"oo"``, ``"cc"``, ``"oc"`` or ``"co"``
        to define if the bounds are open ("o") or closed ("c").
    """
    for arg, bounds in kwargs.items():
        if not check_bounds(bounds):
            msg = f"Given bounds for '{arg}' are not valid, got: {bounds}"
            raise ValueError(msg)
        if arg in getattr(model, "sub_arg", []):
            # var_<i> and len_scale_<i>
            name, i = _split_sub(arg)
            setattr(model[i], f"_{name}_bounds", bounds)
        elif arg in model.opt_arg:
            model._opt_arg_bounds[arg] = bounds
        elif arg in model.arg_bounds:
            setattr(model, f"_{arg}_bounds", bounds)
        else:
            raise ValueError(f"set_arg_bounds: unknown argument '{arg}'")
        if check_args and check_arg_in_bounds(model, arg) > 0:
            def_arg = default_arg_from_bounds(bounds)
            if arg == "anis":
                setattr(model, arg, [def_arg] * (model.dim - 1))
            else:
                setattr(model, arg, def_arg)


def _split_sub(name):
    if name.startswith("var_"):
        return "var", int(name[4:])
    if name.startswith("len_scale_"):
        return "len_scale", int(name[10:])
    msg = f"Unknown sub variable: {name}"
    raise ValueError(msg)


def check_arg_bounds(model):
    """
    Check arguments to be within their given bounds.

    Parameters
    ----------
    model : :any:`CovModel`
        The covariance model in use.

    Raises
    ------
    ValueError
        When an argument is not in its valid bounds.
    """
    # check var, len_scale, nugget and optional-arguments
    for arg in model.arg_bounds:
        check_arg_in_bounds(model, arg, error=True)


def set_dim(model, dim):
    """
    Set the dimension in the given model.

    Parameters
    ----------
    model : :any:`CovModel`
        The covariance model in use.
    dim : :class:`int`
        dimension of the model.

    Raises
    ------
    ValueError
        When dimension is < 1.
    """
    # check if a fixed dimension should be used
    if model.fix_dim() is not None and model.fix_dim() != dim:
        warnings.warn(
            f"{model.name}: using fixed dimension {model.fix_dim()}",
            AttributeWarning,
        )
        dim = model.fix_dim()
        if model.latlon and dim != (3 + int(model.temporal)):
            raise ValueError(
                f"{model.name}: using fixed dimension {model.fix_dim()}, "
                f"which is not compatible with a latlon model (with temporal={model.temporal})."
            )
    # force dim=3 (or 4 when temporal=True) for latlon models
    dim = (3 + int(model.temporal)) if model.latlon else dim
    # set the dimension
    if dim < 1:
        raise ValueError("Only dimensions of d >= 1 are supported.")
    if not model.check_dim(dim):
        warnings.warn(
            f"Dimension {dim} is not appropriate for this model.",
            AttributeWarning,
        )
    model._dim = int(dim)
    # create fourier transform just once (recreate for dim change)
    if model.needs_fourier_transform:
        model._sft = SFT(ndim=model.dim, **model.hankel_kw)
    # recalculate dimension related parameters (if model initialized)
    if model._init:
        model.len_scale, model.anis = set_len_anis(
            model.dim, model.len_scale, model.anis
        )
        model.angles = set_model_angles(
            model.dim, model.angles, model.latlon, model.temporal
        )
        model.check_arg_bounds()


def compare(this, that):
    """
    Compare CovModels.

    Parameters
    ----------
    this / that : :any:`CovModel`
        The covariance models to compare.
    """
    # prevent attribute error in opt_arg if the are not equal
    if set(this.opt_arg) != set(that.opt_arg):
        return False
    # prevent dim error in anis and angles
    if this.dim != that.dim:
        return False
    equal = True
    equal &= this.name == that.name
    equal &= np.isclose(this.var, that.var)
    equal &= np.isclose(this.nugget, that.nugget)
    equal &= np.isclose(this.len_scale, that.len_scale)
    equal &= np.all(np.isclose(this.anis, that.anis))
    equal &= np.all(np.isclose(this.angles, that.angles))
    equal &= np.isclose(this.rescale, that.rescale)
    equal &= np.isclose(this.geo_scale, that.geo_scale)
    equal &= this.latlon == that.latlon
    equal &= this.temporal == that.temporal
    for opt in this.opt_arg:
        equal &= np.isclose(getattr(this, opt), getattr(that, opt))
    return equal


def model_repr(model):  # pragma: no cover
    """
    Generate the model string representation.

    Parameters
    ----------
    model : :any:`CovModel`
        The covariance model in use.
    """
    m, p = model, model._prec
    ani_str, ang_str, o_str, r_str = "", "", "", ""
    t_str = ", temporal=True" if m.temporal else ""
    d_str = f"latlon={m.latlon}" if m.latlon else f"dim={m.spatial_dim}"
    p_str = f", var={m.var:.{p}}, len_scale={m.len_scale:.{p}}"
    p_str += "" if np.isclose(m.nugget, 0) else f", nugget={m.nugget:.{p}}"
    if not np.isclose(m.rescale, m.default_rescale()):
        o_str += f", rescale={m.rescale:.{p}}"
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
    return f"{m.name}({d_str}{t_str}{p_str}{ani_str}{ang_str}{r_str}{o_str})"
