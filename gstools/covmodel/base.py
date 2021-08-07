# -*- coding: utf-8 -*-
"""
GStools subpackage providing the base class for covariance models.

.. currentmodule:: gstools.covmodel.base

The following classes are provided

.. autosummary::
   CovModel
"""
# pylint: disable=C0103, R0201, E1101, C0302, W0613
import copy
import numpy as np
from scipy.integrate import quad as integral
from hankel import SymmetricFourierTransform as SFT
from gstools.tools.geometric import (
    set_angles,
    matrix_anisometrize,
    matrix_isometrize,
    rotated_main_axes,
    latlon2pos,
    pos2latlon,
)
from gstools.covmodel.tools import (
    _init_subclass,
    set_opt_args,
    set_len_anis,
    check_bounds,
    spectral_rad_pdf,
    percentile_scale,
    set_arg_bounds,
    check_arg_bounds,
    set_dim,
    compare,
    model_repr,
    default_arg_from_bounds,
)
from gstools.covmodel import plot
from gstools.covmodel.fit import fit_variogram

__all__ = ["CovModel"]

# default arguments for hankel.SymmetricFourierTransform
HANKEL_DEFAULT = {"a": -1, "b": 1, "N": 200, "h": 0.001, "alt": True}


class CovModel:
    r"""Base class for the GSTools covariance models.

    Parameters
    ----------
    dim : :class:`int`, optional
        dimension of the model. Default: ``3``
    var : :class:`float`, optional
        variance of the model (the nugget is not included in "this" variance)
        Default: ``1.0``
    len_scale : :class:`float` or :class:`list`, optional
        length scale of the model.
        If a single value is given, the same length-scale will be used for
        every direction. If multiple values (for main and transversal
        directions) are given, `anis` will be
        recalculated accordingly. If only two values are given in 3D,
        the latter one will be used for both transversal directions.
        Default: ``1.0``
    nugget : :class:`float`, optional
        nugget of the model. Default: ``0.0``
    anis : :class:`float` or :class:`list`, optional
        anisotropy ratios in the transversal directions [e_y, e_z].

            * e_y = l_y / l_x
            * e_z = l_z / l_x

        If only one value is given in 3D, e_y will be set to 1.
        This value will be ignored, if multiple len_scales are given.
        Default: ``1.0``
    angles : :class:`float` or :class:`list`, optional
        angles of rotation (given in rad):

            * in 2D: given as rotation around z-axis
            * in 3D: given by yaw, pitch, and roll (known as Tait–Bryan angles)

         Default: ``0.0``
    integral_scale : :class:`float` or :class:`list` or :any:`None`, optional
        If given, ``len_scale`` will be ignored and recalculated,
        so that the integral scale of the model matches the given one.
        Default: :any:`None`
    rescale : :class:`float` or :any:`None`, optional
        Optional rescaling factor to divide the length scale with.
        This could be used for unit convertion or rescaling the length scale
        to coincide with e.g. the integral scale.
        Will be set by each model individually.
        Default: :any:`None`
    latlon : :class:`bool`, optional
        Whether the model is describing 2D fields on earths surface described
        by latitude and longitude. When using this, the model will internally
        use the associated 'Yadrenko' model to represent a valid model.
        This means, the spatial distance :math:`r` will be replaced by
        :math:`2\sin(\alpha/2)`, where :math:`\alpha` is the great-circle
        distance, which is equal to the spatial distance of two points in 3D.
        As a consequence, `dim` will be set to `3` and anisotropy will be
        disabled. `rescale` can be set to e.g. earth's radius,
        to have a meaningful `len_scale` parameter.
        Default: False
    var_raw : :class:`float` or :any:`None`, optional
        raw variance of the model which will be multiplied with
        :any:`CovModel.var_factor` to result in the actual variance.
        If given, ``var`` will be ignored.
        (This is just for models that override :any:`CovModel.var_factor`)
        Default: :any:`None`
    hankel_kw: :class:`dict` or :any:`None`, optional
        Modify the init-arguments of
        :any:`hankel.SymmetricFourierTransform`
        used for the spectrum calculation. Use with caution (Better: Don't!).
        ``None`` is equivalent to ``{"a": -1, "b": 1, "N": 1000, "h": 0.001}``.
        Default: :any:`None`
    **opt_arg
        Optional arguments are covered by these keyword arguments.
        If present, they are described in the section `Other Parameters`.
    """

    def __init__(
        self,
        dim=3,
        var=1.0,
        len_scale=1.0,
        nugget=0.0,
        anis=1.0,
        angles=0.0,
        integral_scale=None,
        rescale=None,
        latlon=False,
        var_raw=None,
        hankel_kw=None,
        **opt_arg,
    ):
        # assert, that we use a subclass
        # this is the case, if __init_subclass__ is called, which creates
        # the "variogram"... so we check for that
        if not hasattr(self, "variogram"):
            raise TypeError("Don't instantiate 'CovModel' directly!")

        # prepare dim setting
        self._dim = None
        self._hankel_kw = None
        self._sft = None
        # prepare parameters (they are checked in dim setting)
        self._rescale = None
        self._len_scale = None
        self._anis = None
        self._angles = None
        # prepare parameters boundaries
        self._var_bounds = None
        self._len_scale_bounds = None
        self._nugget_bounds = None
        self._anis_bounds = None
        self._opt_arg_bounds = {}
        # Set latlon first
        self._latlon = bool(latlon)
        # SFT class will be created within dim.setter but needs hankel_kw
        self.hankel_kw = hankel_kw
        self.dim = dim

        # optional arguments for the variogram-model
        set_opt_args(self, opt_arg)

        # set standard boundaries for variance, len_scale, nugget and opt_arg
        bounds = self.default_arg_bounds()
        bounds.update(self.default_opt_arg_bounds())
        self.set_arg_bounds(check_args=False, **bounds)

        # set parameters
        self.rescale = rescale
        self._nugget = float(nugget)
        # set anisotropy and len_scale, disable anisotropy for latlon models
        self._len_scale, anis = set_len_anis(self.dim, len_scale, anis)
        if self.latlon:
            self._anis = np.array((self.dim - 1) * [1], dtype=np.double)
            self._angles = np.array(self.dim * [0], dtype=np.double)
        else:
            self._anis = anis
            self._angles = set_angles(self.dim, angles)
        # set var at last, because of the var_factor (to be right initialized)
        if var_raw is None:
            self._var = None
            self.var = var
        else:
            self._var = float(var_raw)
        self._integral_scale = None
        self.integral_scale = integral_scale
        # set var again, if int_scale affects var_factor
        if var_raw is None:
            self._var = None
            self.var = var
        else:
            self._var = float(var_raw)
        # final check for parameter bounds
        self.check_arg_bounds()
        # additional checks for the optional arguments (provided by user)
        self.check_opt_arg()
        # precision for printing
        self._prec = 3

    # one of these functions needs to be overridden
    def __init_subclass__(cls):
        """Initialize gstools covariance model."""
        _init_subclass(cls)

        # modify the docstrings: class docstring gets attributes added
        if cls.__doc__ is None:
            cls.__doc__ = "User defined GSTools Covariance-Model."
        cls.__doc__ += CovModel.__doc__[45:]
        # overridden functions get standard doc if no new doc was created
        ign = ["__", "variogram", "covariance", "cor"]
        for att, attr_cls in cls.__dict__.items():
            if any(att.startswith(i) for i in ign) or att not in dir(CovModel):
                continue
            attr_doc = getattr(CovModel, att).__doc__
            if attr_cls.__doc__ is None:
                attr_cls.__doc__ = attr_doc

    # special variogram functions

    def vario_axis(self, r, axis=0):
        r"""Variogram along axis of anisotropy."""
        if axis == 0:
            return self.variogram(r)
        return self.variogram(np.abs(r) / self.anis[axis - 1])

    def cov_axis(self, r, axis=0):
        r"""Covariance along axis of anisotropy."""
        if axis == 0:
            return self.covariance(r)
        return self.covariance(np.abs(r) / self.anis[axis - 1])

    def cor_axis(self, r, axis=0):
        r"""Correlation along axis of anisotropy."""
        if axis == 0:
            return self.correlation(r)
        return self.correlation(np.abs(r) / self.anis[axis - 1])

    def vario_yadrenko(self, zeta):
        r"""Yadrenko variogram for great-circle distance from latlon-pos."""
        return self.variogram(2 * np.sin(zeta / 2))

    def cov_yadrenko(self, zeta):
        r"""Yadrenko covariance for great-circle distance from latlon-pos."""
        return self.covariance(2 * np.sin(zeta / 2))

    def cor_yadrenko(self, zeta):
        r"""Yadrenko correlation for great-circle distance from latlon-pos."""
        return self.correlation(2 * np.sin(zeta / 2))

    def vario_spatial(self, pos):
        r"""Spatial variogram respecting anisotropy and rotation."""
        return self.variogram(self._get_iso_rad(pos))

    def cov_spatial(self, pos):
        r"""Spatial covariance respecting anisotropy and rotation."""
        return self.covariance(self._get_iso_rad(pos))

    def cor_spatial(self, pos):
        r"""Spatial correlation respecting anisotropy and rotation."""
        return self.correlation(self._get_iso_rad(pos))

    def vario_nugget(self, r):
        """Isotropic variogram of the model respecting the nugget at r=0."""
        r = np.asarray(np.abs(r), dtype=np.double)
        r_gz = np.logical_not(np.isclose(r, 0))
        res = np.empty_like(r, dtype=np.double)
        res[r_gz] = self.variogram(r[r_gz])
        res[np.logical_not(r_gz)] = 0.0
        return res

    def cov_nugget(self, r):
        """Isotropic covariance of the model respecting the nugget at r=0."""
        r = np.asarray(np.abs(r), dtype=np.double)
        r_gz = np.logical_not(np.isclose(r, 0))
        res = np.empty_like(r, dtype=np.double)
        res[r_gz] = self.covariance(r[r_gz])
        res[np.logical_not(r_gz)] = self.sill
        return res

    def plot(self, func="variogram", **kwargs):  # pragma: no cover
        """
        Plot a function of a the CovModel.

        Parameters
        ----------
        func : :class:`str`, optional
            Function to be plotted. Could be one of:

                * "variogram"
                * "covariance"
                * "correlation"
                * "vario_spatial"
                * "cov_spatial"
                * "cor_spatial"
                * "vario_yadrenko"
                * "cov_yadrenko"
                * "cor_yadrenko"
                * "vario_axis"
                * "cov_axis"
                * "cor_axis"
                * "spectrum"
                * "spectral_density"
                * "spectral_rad_pdf"

        **kwargs
            Keyword arguments forwarded to the plotting function
            `"plot_" + func` in :any:`gstools.covmodel.plot`.

        See Also
        --------
        gstools.covmodel.plot
        """
        routine = getattr(plot, "plot_" + func)
        return routine(self, **kwargs)

    # pykrige functions

    def pykrige_vario(self, args=None, r=0):
        """Isotropic variogram of the model for pykrige."""
        return self.variogram(r)  # pragma: no cover

    @property
    def pykrige_anis(self):
        """2D anisotropy ratio for pykrige."""
        if self.dim == 2:
            return 1 / self.anis[0]
        return 1.0  # pragma: no cover

    @property
    def pykrige_anis_y(self):
        """3D anisotropy ratio in y direction for pykrige."""
        if self.dim >= 2:
            return 1 / self.anis[0]
        return 1.0  # pragma: no cover

    @property
    def pykrige_anis_z(self):
        """3D anisotropy ratio in z direction for pykrige."""
        if self.dim == 3:
            return 1 / self.anis[1]
        return 1.0  # pragma: no cover

    @property
    def pykrige_angle(self):
        """2D rotation angle for pykrige."""
        if self.dim == 2:
            return self.angles[0] / np.pi * 180
        return 0.0  # pragma: no cover

    @property
    def pykrige_angle_z(self):
        """3D rotation angle around z for pykrige."""
        if self.dim >= 2:
            return self.angles[0] / np.pi * 180
        return 0.0  # pragma: no cover

    @property
    def pykrige_angle_y(self):
        """3D rotation angle around y for pykrige."""
        if self.dim == 3:
            return self.angles[1] / np.pi * 180
        return 0.0  # pragma: no cover

    @property
    def pykrige_angle_x(self):
        """3D rotation angle around x for pykrige."""
        if self.dim == 3:
            return self.angles[2] / np.pi * 180
        return 0.0  # pragma: no cover

    @property
    def pykrige_kwargs(self):
        """Keyword arguments for pykrige routines."""
        kwargs = {
            "variogram_model": "custom",
            "variogram_parameters": [],
            "variogram_function": self.pykrige_vario,
        }
        if self.dim == 1:
            add_kwargs = {}
        elif self.dim == 2:
            add_kwargs = {
                "anisotropy_scaling": self.pykrige_anis,
                "anisotropy_angle": self.pykrige_angle,
            }
        else:
            add_kwargs = {
                "anisotropy_scaling_y": self.pykrige_anis_y,
                "anisotropy_scaling_z": self.pykrige_anis_z,
                "anisotropy_angle_x": self.pykrige_angle_x,
                "anisotropy_angle_y": self.pykrige_angle_y,
                "anisotropy_angle_z": self.pykrige_angle_z,
            }
        kwargs.update(add_kwargs)
        return kwargs

    # methods for optional/default arguments (can be overridden)

    def default_opt_arg(self):
        """Provide default optional arguments by the user.

        Should be given as a dictionary when overridden.
        """
        return {
            opt: default_arg_from_bounds(bnd)
            for (opt, bnd) in self.default_opt_arg_bounds().items()
        }

    def default_opt_arg_bounds(self):
        """Provide default boundaries for optional arguments."""
        res = {}
        for opt in self.opt_arg:
            res[opt] = [-np.inf, np.inf]
        return res

    def check_opt_arg(self):
        """Run checks for the optional arguments.

        This is in addition to the bound-checks

        Notes
        -----
        * You can use this to raise a ValueError/warning
        * Any return value will be ignored
        * This method will only be run once, when the class is initialized
        """

    def check_dim(self, dim):
        """Check the given dimension."""
        return True

    def fix_dim(self):
        """Set a fix dimension for the model."""
        return None

    def var_factor(self):
        """Factor for the variance."""
        return 1.0

    def default_rescale(self):
        """Provide default rescaling factor."""
        return 1.0

    # calculation of different scales

    def calc_integral_scale(self):
        """Calculate the integral scale of the isotrope model."""
        self._integral_scale = integral(self.correlation, 0, np.inf)[0]
        return self._integral_scale

    def percentile_scale(self, per=0.9):
        """Calculate the percentile scale of the isotrope model.

        This is the distance, where the given percentile of the variance
        is reached by the variogram
        """
        return percentile_scale(self, per)

    # spectrum methods (can be overridden for speedup)

    def spectrum(self, k):
        r"""
        Spectrum of the covariance model.

        This is given by:

        .. math:: S(\mathbf{k}) = \left(\frac{1}{2\pi}\right)^n
           \int C(r) e^{i \mathbf{k}\cdot\mathbf{r}} d^n\mathbf{r}

        Internally, this is calculated by the hankel transformation:

        .. math:: S(k) = \left(\frac{1}{2\pi}\right)^n \cdot
           \frac{(2\pi)^{n/2}}{k^{n/2-1}}
           \int_0^\infty r^{n/2} C(r) J_{n/2-1}(kr) dr

        Where :math:`C(r)` is the covariance function of the model.

        Parameters
        ----------
        k : :class:`float`
            Radius of the phase: :math:`k=\left\Vert\mathbf{k}\right\Vert`
        """
        return self.spectral_density(k) * self.var

    def spectral_density(self, k):
        r"""
        Spectral density of the covariance model.

        This is given by:

        .. math:: \tilde{S}(k) = \frac{S(k)}{\sigma^2}

        Where :math:`S(k)` is the spectrum of the covariance model.

        Parameters
        ----------
        k : :class:`float`
            Radius of the phase: :math:`k=\left\Vert\mathbf{k}\right\Vert`
        """
        k = np.asarray(np.abs(k), dtype=np.double)
        return self._sft.transform(self.correlation, k, ret_err=False)

    def spectral_rad_pdf(self, r):
        """Radial spectral density of the model."""
        return spectral_rad_pdf(self, r)

    def ln_spectral_rad_pdf(self, r):
        """Log radial spectral density of the model."""
        with np.errstate(divide="ignore"):
            return np.log(self.spectral_rad_pdf(r))

    def _has_cdf(self):
        """State if a cdf is defined with 'spectral_rad_cdf'."""
        return hasattr(self, "spectral_rad_cdf")

    def _has_ppf(self):
        """State if a ppf is defined with 'spectral_rad_ppf'."""
        return hasattr(self, "spectral_rad_ppf")

    # spatial routines

    def isometrize(self, pos):
        """Make a position tuple ready for isotropic operations."""
        pos = np.asarray(pos, dtype=np.double).reshape((self.field_dim, -1))
        if self.latlon:
            return latlon2pos(pos)
        return np.dot(matrix_isometrize(self.dim, self.angles, self.anis), pos)

    def anisometrize(self, pos):
        """Bring a position tuple into the anisotropic coordinate-system."""
        pos = np.asarray(pos, dtype=np.double).reshape((self.dim, -1))
        if self.latlon:
            return pos2latlon(pos)
        return np.dot(
            matrix_anisometrize(self.dim, self.angles, self.anis), pos
        )

    def main_axes(self):
        """Axes of the rotated coordinate-system."""
        return rotated_main_axes(self.dim, self.angles)

    def _get_iso_rad(self, pos):
        """Isometrized radians."""
        return np.linalg.norm(self.isometrize(pos), axis=0)

    # fitting routine

    def fit_variogram(
        self,
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
        Fiting the variogram-model to an empirical variogram.

        Parameters
        ----------
        x_data : :class:`numpy.ndarray`
            The bin-centers of the empirical variogram.
        y_data : :class:`numpy.ndarray`
            The messured variogram
            If multiple are given, they are interpreted as the directional
            variograms along the main axis of the associated rotated
            coordinate system.
            Anisotropy ratios will be estimated in that case.
        anis : :class:`bool`, optional
            In case of a directional variogram, you can control anisotropy
            by this argument. Deselect the parameter from fitting, by setting
            it "False".
            You could also pass a fixed value to be set in the model.
            Then the anisotropy ratios wont be altered during fitting.
            Default: True
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
        weights : :class:`str`, :class:`numpy.ndarray`, :class:`callable`, optional
            Weights applied to each point in the estimation. Either:

                * 'inv': inverse distance ``1 / (x_data + 1)``
                * list: weights given per bin
                * callable: function applied to x_data

            If callable, it must take a 1-d ndarray.
            Then ``weights = f(x_data)``.
            Default: None
        method : {'trf', 'dogbox'}, optional
            Algorithm to perform minimization.

                * 'trf' : Trust Region Reflective algorithm,
                  particularly suitable for large sparse problems with bounds.
                  Generally robust method.
                * 'dogbox' : dogleg algorithm with rectangular trust regions,
                  typical use case is small problems with bounds.
                  Not recommended for problems with rank-deficient Jacobian.

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
        return fit_variogram(
            model=self,
            x_data=x_data,
            y_data=y_data,
            anis=anis,
            sill=sill,
            init_guess=init_guess,
            weights=weights,
            method=method,
            loss=loss,
            max_eval=max_eval,
            return_r2=return_r2,
            curve_fit_kwargs=curve_fit_kwargs,
            **para_select,
        )

    # bounds setting and checks

    def default_arg_bounds(self):
        """Provide default boundaries for arguments.

        Given as a dictionary.
        """
        res = {
            "var": (0.0, np.inf, "oo"),
            "len_scale": (0.0, np.inf, "oo"),
            "nugget": (0.0, np.inf, "co"),
            "anis": (0.0, np.inf, "oo"),
        }
        return res

    def set_arg_bounds(self, check_args=True, **kwargs):
        r"""Set bounds for the parameters of the model.

        Parameters
        ----------
        check_args : bool, optional
            Whether to check if the arguments are in their valid bounds.
            In case not, a propper default value will be determined.
            Default: True
        **kwargs
            Parameter name as keyword ("var", "len_scale", "nugget", <opt_arg>)
            and a list of 2 or 3 values: ``[a, b]`` or ``[a, b, <type>]`` where
            <type> is one of ``"oo"``, ``"cc"``, ``"oc"`` or ``"co"``
            to define if the bounds are open ("o") or closed ("c").
        """
        return set_arg_bounds(self, check_args, **kwargs)

    def check_arg_bounds(self):
        """Check arguments to be within their given bounds."""
        return check_arg_bounds(self)

    # bounds properties

    @property
    def var_bounds(self):
        """:class:`list`: Bounds for the variance.

        Notes
        -----
        Is a list of 2 or 3 values: ``[a, b]`` or ``[a, b, <type>]`` where
        <type> is one of ``"oo"``, ``"cc"``, ``"oc"`` or ``"co"``
        to define if the bounds are open ("o") or closed ("c").
        """
        return self._var_bounds

    @var_bounds.setter
    def var_bounds(self, bounds):
        if not check_bounds(bounds):
            raise ValueError(
                f"Given bounds for 'var' are not valid, got: {bounds}"
            )
        self._var_bounds = bounds

    @property
    def len_scale_bounds(self):
        """:class:`list`: Bounds for the lenght scale.

        Notes
        -----
        Is a list of 2 or 3 values: ``[a, b]`` or ``[a, b, <type>]`` where
        <type> is one of ``"oo"``, ``"cc"``, ``"oc"`` or ``"co"``
        to define if the bounds are open ("o") or closed ("c").
        """
        return self._len_scale_bounds

    @len_scale_bounds.setter
    def len_scale_bounds(self, bounds):
        if not check_bounds(bounds):
            raise ValueError(
                f"Given bounds for 'len_scale' are not valid, got: {bounds}"
            )
        self._len_scale_bounds = bounds

    @property
    def nugget_bounds(self):
        """:class:`list`: Bounds for the nugget.

        Notes
        -----
        Is a list of 2 or 3 values: ``[a, b]`` or ``[a, b, <type>]`` where
        <type> is one of ``"oo"``, ``"cc"``, ``"oc"`` or ``"co"``
        to define if the bounds are open ("o") or closed ("c").
        """
        return self._nugget_bounds

    @nugget_bounds.setter
    def nugget_bounds(self, bounds):
        if not check_bounds(bounds):
            raise ValueError(
                f"Given bounds for 'nugget' are not valid, got: {bounds}"
            )
        self._nugget_bounds = bounds

    @property
    def anis_bounds(self):
        """:class:`list`: Bounds for the nugget.

        Notes
        -----
        Is a list of 2 or 3 values: ``[a, b]`` or ``[a, b, <type>]`` where
        <type> is one of ``"oo"``, ``"cc"``, ``"oc"`` or ``"co"``
        to define if the bounds are open ("o") or closed ("c").
        """
        return self._anis_bounds

    @anis_bounds.setter
    def anis_bounds(self, bounds):
        if not check_bounds(bounds):
            raise ValueError(
                f"Given bounds for 'anis' are not valid, got: {bounds}"
            )
        self._anis_bounds = bounds

    @property
    def opt_arg_bounds(self):
        """:class:`dict`: Bounds for the optional arguments.

        Notes
        -----
        Keys are the opt-arg names and values are lists of 2 or 3 values:
        ``[a, b]`` or ``[a, b, <type>]`` where
        <type> is one of ``"oo"``, ``"cc"``, ``"oc"`` or ``"co"``
        to define if the bounds are open ("o") or closed ("c").
        """
        return self._opt_arg_bounds

    @property
    def arg_bounds(self):
        """:class:`dict`: Bounds for all parameters.

        Notes
        -----
        Keys are the arg names and values are lists of 2 or 3 values:
        ``[a, b]`` or ``[a, b, <type>]`` where
        <type> is one of ``"oo"``, ``"cc"``, ``"oc"`` or ``"co"``
        to define if the bounds are open ("o") or closed ("c").
        """
        res = {
            "var": self.var_bounds,
            "len_scale": self.len_scale_bounds,
            "nugget": self.nugget_bounds,
            "anis": self.anis_bounds,
        }
        res.update(self.opt_arg_bounds)
        return res

    # geographical coordinates related

    @property
    def latlon(self):
        """:class:`bool`: Whether the model depends on geographical coords."""
        return self._latlon

    @property
    def field_dim(self):
        """:class:`int`: The field dimension of the model."""
        return 2 if self.latlon else self.dim

    # standard parameters

    @property
    def dim(self):
        """:class:`int`: The dimension of the model."""
        return self._dim

    @dim.setter
    def dim(self, dim):
        set_dim(self, dim)

    @property
    def var(self):
        """:class:`float`: The variance of the model."""
        return self._var * self.var_factor()

    @var.setter
    def var(self, var):
        self._var = float(var) / self.var_factor()
        self.check_arg_bounds()

    @property
    def var_raw(self):
        """:class:`float`: The raw variance of the model without factor.

        (See. CovModel.var_factor)
        """
        return self._var

    @var_raw.setter
    def var_raw(self, var_raw):
        self._var = float(var_raw)
        self.check_arg_bounds()

    @property
    def nugget(self):
        """:class:`float`: The nugget of the model."""
        return self._nugget

    @nugget.setter
    def nugget(self, nugget):
        self._nugget = float(nugget)
        self.check_arg_bounds()

    @property
    def len_scale(self):
        """:class:`float`: The main length scale of the model."""
        return self._len_scale

    @len_scale.setter
    def len_scale(self, len_scale):
        self._len_scale, anis = set_len_anis(self.dim, len_scale, self.anis)
        if self.latlon:
            self._anis = np.array((self.dim - 1) * [1], dtype=np.double)
        else:
            self._anis = anis
        self.check_arg_bounds()

    @property
    def rescale(self):
        """:class:`float`: Rescale factor for the length scale of the model."""
        return self._rescale

    @rescale.setter
    def rescale(self, rescale):
        rescale = self.default_rescale() if rescale is None else rescale
        self._rescale = abs(float(rescale))

    @property
    def len_rescaled(self):
        """:class:`float`: The rescaled main length scale of the model."""
        return self._len_scale / self._rescale

    @property
    def anis(self):
        """:class:`numpy.ndarray`: The anisotropy factors of the model."""
        return self._anis

    @anis.setter
    def anis(self, anis):
        if self.latlon:
            self._anis = np.array((self.dim - 1) * [1], dtype=np.double)
        else:
            self._len_scale, self._anis = set_len_anis(
                self.dim, self.len_scale, anis
            )
        self.check_arg_bounds()

    @property
    def angles(self):
        """:class:`numpy.ndarray`: Rotation angles (in rad) of the model."""
        return self._angles

    @angles.setter
    def angles(self, angles):
        if self.latlon:
            self._angles = np.array(self.dim * [0], dtype=np.double)
        else:
            self._angles = set_angles(self.dim, angles)
        self.check_arg_bounds()

    @property
    def integral_scale(self):
        """:class:`float`: The main integral scale of the model.

        Raises
        ------
        ValueError
            If integral scale is not setable.
        """
        self._integral_scale = self.calc_integral_scale()
        return self._integral_scale

    @integral_scale.setter
    def integral_scale(self, integral_scale):
        if integral_scale is not None:
            # format int-scale right
            self.len_scale = integral_scale
            integral_scale = self.len_scale
            # reset len_scale
            self.len_scale = 1.0
            int_tmp = self.calc_integral_scale()
            self.len_scale = integral_scale / int_tmp
            if not np.isclose(self.integral_scale, integral_scale, rtol=1e-3):
                raise ValueError(
                    f"{self.name}: Integral scale could not be set correctly! "
                    "Please just provide a 'len_scale'!"
                )

    @property
    def hankel_kw(self):
        """:class:`dict`: :any:`hankel.SymmetricFourierTransform` kwargs."""
        return self._hankel_kw

    @hankel_kw.setter
    def hankel_kw(self, hankel_kw):
        if self._hankel_kw is None or hankel_kw is None:
            self._hankel_kw = copy.copy(HANKEL_DEFAULT)
        if hankel_kw is not None:
            self._hankel_kw.update(hankel_kw)
        if self.dim is not None:
            self._sft = SFT(ndim=self.dim, **self.hankel_kw)

    @property
    def dist_func(self):
        """:class:`tuple` of :any:`callable`: pdf, cdf and ppf.

        Spectral distribution info from the model.
        """
        pdf = self.spectral_rad_pdf
        cdf = None
        ppf = None
        if self.has_cdf:
            cdf = self.spectral_rad_cdf
        if self.has_ppf:
            ppf = self.spectral_rad_ppf
        return pdf, cdf, ppf

    @property
    def has_cdf(self):
        """:class:`bool`: State if a cdf is defined by the user."""
        return self._has_cdf()

    @property
    def has_ppf(self):
        """:class:`bool`: State if a ppf is defined by the user."""
        return self._has_ppf()

    @property
    def sill(self):
        """:class:`float`: The sill of the variogram.

        Notes
        -----
        This is calculated by:
            * ``sill = variance + nugget``
        """
        return self.var + self.nugget

    @property
    def arg(self):
        """:class:`list` of :class:`str`: Names of all arguments."""
        return ["var", "len_scale", "nugget", "anis", "angles"] + self._opt_arg

    @property
    def arg_list(self):
        """:class:`list` of :class:`float`: Values of all arguments."""
        alist = [self.var, self.len_scale, self.nugget, self.anis, self.angles]
        for opt in self.opt_arg:
            alist.append(getattr(self, opt))
        return alist

    @property
    def iso_arg(self):
        """:class:`list` of :class:`str`: Names of isotropic arguments."""
        return ["var", "len_scale", "nugget"] + self._opt_arg

    @property
    def iso_arg_list(self):
        """:class:`list` of :class:`float`: Values of isotropic arguments."""
        alist = [self.var, self.len_scale, self.nugget]
        for opt in self.opt_arg:
            alist.append(getattr(self, opt))
        return alist

    @property
    def opt_arg(self):
        """:class:`list` of :class:`str`: Names of the optional arguments."""
        return self._opt_arg

    @property
    def len_scale_vec(self):
        """:class:`numpy.ndarray`: The length scales in each direction.

        Notes
        -----
        This is calculated by:
            * ``len_scale_vec[0] = len_scale``
            * ``len_scale_vec[1] = len_scale*anis[0]``
            * ``len_scale_vec[2] = len_scale*anis[1]``
        """
        res = np.zeros(self.dim, dtype=np.double)
        res[0] = self.len_scale
        for i in range(1, self._dim):
            res[i] = self.len_scale * self.anis[i - 1]
        return res

    @property
    def integral_scale_vec(self):
        """:class:`numpy.ndarray`: The integral scales in each direction.

        Notes
        -----
        This is calculated by:
            * ``integral_scale_vec[0] = integral_scale``
            * ``integral_scale_vec[1] = integral_scale*anis[0]``
            * ``integral_scale_vec[2] = integral_scale*anis[1]``
        """
        res = np.zeros(self.dim, dtype=np.double)
        res[0] = self.integral_scale
        for i in range(1, self.dim):
            res[i] = self.integral_scale * self.anis[i - 1]
        return res

    @property
    def name(self):
        """:class:`str`: The name of the CovModel class."""
        return self.__class__.__name__

    @property
    def do_rotation(self):
        """:any:`bool`: State if a rotation is performed."""
        return not np.all(np.isclose(self.angles, 0.0))

    @property
    def is_isotropic(self):
        """:any:`bool`: State if a model is isotropic."""
        return np.all(np.isclose(self.anis, 1.0))

    def __eq__(self, other):
        """Compare CovModels."""
        if not isinstance(other, CovModel):
            return False
        return compare(self, other)

    def __repr__(self):
        """Return String representation."""
        return model_repr(self)
