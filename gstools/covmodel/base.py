# -*- coding: utf-8 -*-
"""
GStools subpackage providing the base class for covariance models.

.. currentmodule:: gstools.covmodel.base

The following classes are provided

.. autosummary::
   CovModel
"""
# pylint: disable=C0103, R0201
from __future__ import print_function, division, absolute_import

import six
import numpy as np
from scipy.integrate import quad as integral
from scipy.optimize import curve_fit, root
from hankel import SymmetricFourierTransform as SFT
from gstools.covmodel.tools import (
    InitSubclassMeta,
    rad_fac,
    set_len_anis,
    set_angles,
    check_bounds,
)

__all__ = ["CovModel"]

# default arguments for hankel.SymmetricFourierTransform
HANKEL_DEFAULT = {
    "a": -1,  # should only be changed, if you know exactly what
    "b": 1,  # you do or if you are crazy
    "N": 1000,
    "h": 0.001,
}

# The CovModel Base-Class #####################################################


class CovModel(six.with_metaclass(InitSubclassMeta)):
    """Base class for the GSTools covariance models

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
        recalculated accordingly.
        Default: ``1.0``
    nugget : :class:`float`, optional
        nugget of the model. Default: ``0.0``
    anis : :class:`float` or :class:`list`, optional
        anisotropy ratios in the transversal directions [y, z].
        Default: ``1.0``
    angles : :class:`float` or :class:`list`, optional
        angles of rotation:

            * in 2D: given as rotation around z-axis
            * in 3D: given by yaw, pitch, and roll (known as Tait–Bryan angles)

         Default: ``0.0``
    integral_scale : :class:`float` or :class:`list` or :any:`None`, optional
        If given, ``len_scale`` will be ignored and recalculated,
        so that the integral scale of the model matches the given one.
        Default: ``None``
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

    Examples
    --------
    >>> from gstools import CovModel
    >>> import numpy as np
    >>> class Gau(CovModel):
    ...     def correlation(self, r):
    ...         return np.exp(-(r/self.len_scale)**2)
    ...
    >>> model = Gau()
    >>> model.spectrum(2)
    0.00825830126008459
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
        var_raw=None,
        hankel_kw=None,
        **opt_arg
    ):
        # assert, that we use a subclass
        # this is the case, if __init_subclass__ is called, which creates
        # the "variogram"... so we check for that
        if not hasattr(self, "variogram"):
            raise TypeError("Don't instantiate 'CovModel' directly!")

        # optional arguments for the variogram-model
        # look up the defaults for the optional arguments (defined by the user)
        default = self.default_opt_arg()
        # add the default vaules if not specified
        for def_arg in default:
            if def_arg not in opt_arg:
                opt_arg[def_arg] = default[def_arg]
        # save names of the optional arguments
        self._opt_arg = list(opt_arg.keys())
        # add the optional arguments as attributes to the class
        for opt_name in opt_arg:
            if opt_name in dir(self):  # "dir" also respects properties
                raise ValueError(
                    "parameter '"
                    + opt_name
                    + "' has a 'bad' name, since it is already present in "
                    + "the class. It could not be added to the model"
                )
            # Magic happens here
            setattr(self, opt_name, opt_arg[opt_name])

        # set standard boundaries for variance, len_scale, nugget and opt_arg
        self._var_bounds = None
        self._len_scale_bounds = None
        self._nugget_bounds = None
        self._opt_arg_bounds = {}
        bounds = self.default_arg_bounds()
        bounds.update(self.default_opt_arg_bounds())
        self.set_arg_bounds(**bounds)

        # prepare dim setting
        self._dim = None
        self._len_scale = None
        self._anis = None
        self._angles = None
        # SFT class will be created within dim.setter but needs hankel_kw
        self._hankel_kw = None
        self._sft = None
        self.hankel_kw = hankel_kw
        self.dim = dim
        # set parameters
        self._nugget = nugget
        self._angles = set_angles(self.dim, angles)
        self._len_scale, self._anis = set_len_anis(self.dim, len_scale, anis)
        # set var at last, because of the var_factor (to be right initialized)
        if var_raw is None:
            self._var = None
            self.var = var
        else:
            self._var = var_raw
        self._integral_scale = None
        self.integral_scale = integral_scale
        # set var again, if int_scale affects var_factor
        if var_raw is None:
            self._var = None
            self.var = var
        else:
            self._var = var_raw
        # final check for parameter bounds
        self.check_arg_bounds()
        # additional checks for the optional arguments (provided by user)
        self.check_opt_arg()

    ###########################################################################
    # one of these functions needs to be overridden ###########################
    ###########################################################################

    def __init_subclass__(cls):
        r"""Initialize gstools covariance model

        Warnings
        --------
        Don't instantiate ``CovModel`` directly. You need to inherit a
        child class which overrides one of the following methods:

            * ``model.variogram(r)``
                :math:`\gamma\left(r\right)=
                \sigma^2\cdot\left(1-\mathrm{cor}\left(r\right)\right)+n`
            * ``model.variogram_normed(r)``
                :math:`\tilde{\gamma}\left(r\right)=
                1-\mathrm{cor}\left(r\right)`
            * ``model.covariance(r)``
                :math:`C\left(r\right)=
                \sigma^2\cdot\mathrm{cor}\left(r\right)`
            * ``model.correlation(r)``
                :math:`\mathrm{cor}\left(r\right)`

        Best practice is to use the ``correlation`` function!
        """
        # overrid one of these ################################################
        def variogram(self, r):
            r"""Isotropic variogram of the model

            Given by: :math:`\gamma\left(r\right)=
            \sigma^2\cdot\left(1-\mathrm{cor}\left(r\right)\right)+n`

            Where :math:`\mathrm{cor}(r)` is the correlation function.
            """
            return self.var - self.covariance(r) + self.nugget

        def covariance(self, r):
            r"""Covariance of the model

            Given by: :math:`C\left(r\right)=
            \sigma^2\cdot\mathrm{cor}\left(r\right)`

            Where :math:`\mathrm{cor}(r)` is the correlation function.
            """
            return self.var * self.correlation(r)

        def correlation(self, r):
            r"""correlation function (or normalized covariance) of the model

            Given by: :math:`\mathrm{cor}\left(r\right)`

            It has to be a monotonic decreasing function with
            :math:`\mathrm{cor}(0)=1` and :math:`\mathrm{cor}(\infty)=0`.
            """
            return 1.0 - self.variogram_normed(r)

        def variogram_normed(self, r):
            r"""Normalized variogram of the model

            Given by: :math:`\tilde{\gamma}\left(r\right)=
            1-\mathrm{cor}\left(r\right)`

            Where :math:`\mathrm{cor}(r)` is the correlation function.
            """
            return (self.variogram(r) - self.nugget) / self.var

        #######################################################################

        abstract = True
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
        if not hasattr(cls, "variogram_normed"):
            cls.variogram_normed = variogram_normed
        else:
            abstract = False
        if abstract:
            raise TypeError(
                "Can't instantiate class '"
                + cls.__name__
                + "', "
                + "without overriding at least on of the methods "
                + "'variogram', 'covariance', "
                + "'correlation', or 'variogram_normed'."
            )

        # modify the docstrings ###############################################

        # class docstring gets attributes added
        if cls.__doc__ is None:
            cls.__doc__ = (
                "User defined GSTools Covariance-Model "
                + CovModel.__doc__[44:-296]
            )
        else:
            cls.__doc__ += CovModel.__doc__[44:-296]
        # overridden functions get standard doc if no new doc was created
        ignore = ["__", "variogram", "covariance", "correlation"]
        for attr in cls.__dict__:
            if any(
                [attr.startswith(ign) for ign in ignore]
            ) or attr not in dir(CovModel):
                continue
            attr_doc = getattr(CovModel, attr).__doc__
            attr_cls = cls.__dict__[attr]
            if attr_cls.__doc__ is None:
                attr_cls.__doc__ = attr_doc

    ###########################################################################
    # methods for optional arguments (can be overridden) ######################
    ###########################################################################

    def default_opt_arg(self):
        """Here you can provide a dictionary with default values for
        the optional arguments."""
        return {}

    def default_opt_arg_bounds(self):
        """Here you can provide a dictionary with default boundaries for
        the optional arguments."""
        res = {}
        for opt in self.opt_arg:
            res[opt] = [0.0, 1000.0]
        return res

    def check_opt_arg(self):
        """Here you can run checks for the optional arguments

        This is in addition to the bound-checks

        Notes
        -----
        * You can use this to raise a ValueError/warning
        * Any return value will be ignored
        * This method will only be run once, when the class is initialized
        """
        pass

    def fix_dim(self):
        """Set a fix dimension for the model"""
        return None

    def var_factor(self):
        """Optional factor for the variance"""
        return 1.0

    # calculation of different scales #########################################

    def _calc_integral_scale(self):
        """calculate the integral scale of the isotrope model"""
        self._integral_scale = integral(self.correlation, 0, np.inf)[0]
        return self._integral_scale

    def percentile_scale(self, per=0.9):
        """calculate the distance, where the given percentile of the variance
        is reached by the variogram"""
        # check the given percentile
        if not 0.0 < per < 1.0:
            raise ValueError(
                "percentile needs to be within (0, 1), got: " + str(per)
            )

        # define a curve, that has its root at the wanted point
        def curve(x):
            return self.variogram_normed(x) - per

        # take 'per * len_scale' as initial guess
        return root(curve, per * self.len_scale)["x"][0]

    ###########################################################################
    # spectrum methods (can be overridden for speedup) ########################
    ###########################################################################

    def spectrum(self, k):
        r"""
        The spectrum of the covariance model.

        This is given by:

        .. math:: S(k) = \left(\frac{1}{2\pi}\right)^n
           \int C(r) e^{i b\mathbf{k}\cdot\mathbf{r}} d^n\mathbf{r}

        Internally, this is calculated by the hankel transformation:

        .. math:: S(k) = \left(\frac{1}{2\pi}\right)^n \cdot
           \frac{(2\pi)^{n/2}}{(bk)^{n/2-1}}
           \int_0^\infty r^{n/2-1} f(r) J_{n/2-1}(bkr) r dr

        Where :math:`C(r)` is the covariance function of the model.

        Parameters
        ----------
        k : :class:`float`
            Radius of the phase: :math:`k=\left\Vert\mathbf{k}\right\Vert`
        """
        k = np.array(np.abs(k), dtype=float)
        return self._sft.transform(self.covariance, k, ret_err=False)

    def spectral_density(self, k):
        r"""
        The spectral density of the covariance model.

        This is given by:

        .. math:: \tilde{S}(k) = \frac{S(k)}{\sigma^2}

        Where :math:`S(k)` is the spectrum of the covariance model.

        Parameters
        ----------
        k : :class:`float`
            Radius of the phase: :math:`k=\left\Vert\mathbf{k}\right\Vert`
        """
        return self.spectrum(k) / self.var

    def spectral_rad_pdf(self, r):
        """
        The radial spectral density of the model depending on the dimension
        """
        r = np.array(np.abs(r), dtype=float)
        if self.dim > 1:
            r_gz = r[r > 0.0]
            # to prevent numerical errors, we just calculate where r>0
            res = np.zeros_like(r, dtype=float)
            res[r > 0.0] = rad_fac(self.dim, r_gz) * self.spectral_density(
                r_gz
            )
            # prevent numerical errors in hankel for small r values (set 0)
            res[np.logical_not(np.isfinite(res))] = 0.0
            # prevent numerical errors in hankel for big r (set non-negative)
            res = np.maximum(res, 0.0)
            return res
        # TODO: this is totally hacky (but working :D)
        # prevent num error in hankel at r=0 in 1D
        if self.dim == 1:
            r[r == 0.0] = 0.03 / self.len_scale
        res = rad_fac(self.dim, r) * self.spectral_density(r)
        # prevent numerical errors in hankel for big r (set non-negativ)
        res = np.maximum(res, 0.0)
        return res

    def ln_spectral_rad_pdf(self, r):
        """
        The log radial spectral density of the model depending on the dimension
        """
        spec = np.array(self.spectral_rad_pdf(r))
        res = np.full_like(spec, -np.inf, dtype=float)
        res[spec > 0.0] = np.log(spec[spec > 0.0])
        return res

    def _has_cdf(self):
        """State if a cdf is defined"""
        return hasattr(self, "spectral_rad_cdf")

    def _has_ppf(self):
        """State if a ppf is defined"""
        return hasattr(self, "spectral_rad_ppf")

    # fitting routine #########################################################

    def fit_variogram(self, x_data, y_data, **para_deselect):
        """
        fit the isotropic variogram-model to given data

        Parameters
        ----------
        x_data : :class:`numpy.ndarray`
            The radii of the meassured variogram.
        y_data : :class:`numpy.ndarray`
            The messured variogram
        **para_deselect
            You can deselect the parameters to be fitted, by setting
            them "False" as keywords. By default, all parameters are
            fitted.

        Returns
        -------
        fit_para : :class:`dict`
            Dictonary with the fitted parameter values
        pcov : :class:`numpy.ndarray`
            The estimated covariance of `popt` from
            :any:`scipy.optimize.curve_fit`

        Notes
        -----
        You can set the bounds for each parameter by accessing
        :any:`CovModel.set_arg_bounds`.

        The fitted parameters will be instantly set in the model.
        """
        # select all parameters to be fitted
        para = {"var": True, "len_scale": True, "nugget": True}
        for opt in self.opt_arg:
            para[opt] = True
        # deselect unwanted parameters
        para.update(para_deselect)

        # we need arg1, otherwise curve_fit throws an error (bug?!)
        def curve(x, arg1, *args):
            """dummy function for the variogram"""
            args = (arg1,) + args
            para_skip = 0
            opt_skip = 0
            if para["var"]:
                var_tmp = args[para_skip]
                para_skip += 1
            if para["len_scale"]:
                self.len_scale = args[para_skip]
                para_skip += 1
            if para["nugget"]:
                self.nugget = args[para_skip]
                para_skip += 1
            for opt in self.opt_arg:
                if para[opt]:
                    setattr(self, opt, args[para_skip + opt_skip])
                    opt_skip += 1
            # set var at last because of var_factor (other parameter needed)
            if para["var"]:
                self.var = var_tmp
            return self.variogram(x)

        # set the lower/upper boundaries for the variogram-parameters
        low_bounds = []
        top_bounds = []
        if para["var"]:
            low_bounds.append(self.var_bounds[0])
            top_bounds.append(self.var_bounds[1])
        if para["len_scale"]:
            low_bounds.append(self.len_scale_bounds[0])
            top_bounds.append(self.len_scale_bounds[1])
        if para["nugget"]:
            low_bounds.append(self.nugget_bounds[0])
            top_bounds.append(self.nugget_bounds[1])
        for opt in self.opt_arg:
            if para[opt]:
                low_bounds.append(self.opt_arg_bounds[opt][0])
                top_bounds.append(self.opt_arg_bounds[opt][1])
        # fit the variogram
        popt, pcov = curve_fit(
            curve, x_data, y_data, bounds=(low_bounds, top_bounds)
        )
        fit_para = {}
        para_skip = 0
        opt_skip = 0
        if para["var"]:
            var_tmp = popt[para_skip]
            fit_para["var"] = popt[para_skip]
            para_skip += 1
        else:
            fit_para["var"] = self.var
        if para["len_scale"]:
            self.len_scale = popt[para_skip]
            fit_para["len_scale"] = popt[para_skip]
            para_skip += 1
        else:
            fit_para["len_scale"] = self.len_scale
        if para["nugget"]:
            self.nugget = popt[para_skip]
            fit_para["nugget"] = popt[para_skip]
            para_skip += 1
        else:
            fit_para["nugget"] = self.nugget
        for opt in self.opt_arg:
            if para[opt]:
                setattr(self, opt, popt[para_skip + opt_skip])
                fit_para[opt] = popt[para_skip + opt_skip]
                opt_skip += 1
            else:
                fit_para[opt] = getattr(self, opt)
        # set var at last because of var_factor (other parameter needed)
        if para["var"]:
            self.var = var_tmp
        return fit_para, pcov

    # bounds setting and checks ###############################################

    def default_arg_bounds(self):
        """Here you can provide a dictionary with default boundaries for
        the standard arguments."""
        res = {
            "var": (0.0, 100.0, "oc"),
            "len_scale": (0.0, 1000.0, "oo"),
            "nugget": (0.0, 100.0, "cc"),
        }
        return res

    def set_arg_bounds(self, **kwargs):
        r"""Set bounds for the parameters of the model

        Parameters
        ----------
        **kwargs
            Parameter name as keyword ("var", "len_scale", "nugget", <opt_arg>)
            and a list of 2 or 3 values as value:

                * ``[a, b]`` or
                * ``[a, b, <type>]``

            <type> is one of ``"oo"``, ``"cc"``, ``"oc"`` or ``"co"``
            to define if the bounds are open ("o") or closed ("c").
        """
        for opt in kwargs:
            if opt in self.opt_arg:
                if not check_bounds(kwargs[opt]):
                    raise ValueError(
                        "Given bounds for '"
                        + opt
                        + "' are not valid, got: "
                        + str(kwargs[opt])
                    )
                self._opt_arg_bounds[opt] = kwargs[opt]
            if opt == "var":
                self.var_bounds = kwargs[opt]
            if opt == "len_scale":
                self.len_scale_bounds = kwargs[opt]
            if opt == "nugget":
                self.nugget_bounds = kwargs[opt]

    def check_arg_bounds(self):
        """Here the arguments are checked to be within the given bounds"""
        # check var, len_scale, nugget and optional-arguments
        for arg in self.arg_bounds:
            bnd = list(self.arg_bounds[arg])
            val = getattr(self, arg)
            if len(bnd) == 2:
                bnd.append("cc")  # use closed intervals by default
            if bnd[2][0] == "c":
                if val < bnd[0]:
                    raise ValueError(
                        str(arg)
                        + " needs to be >= "
                        + str(bnd[0])
                        + ", got: "
                        + str(val)
                    )
            else:
                if val <= bnd[0]:
                    raise ValueError(
                        str(arg)
                        + " needs to be > "
                        + str(bnd[0])
                        + ", got: "
                        + str(val)
                    )
            if bnd[2][1] == "c":
                if val > bnd[1]:
                    raise ValueError(
                        str(arg)
                        + " needs to be <= "
                        + str(bnd[1])
                        + ", got: "
                        + str(val)
                    )
            else:
                if val >= bnd[1]:
                    raise ValueError(
                        str(arg)
                        + " needs to be < "
                        + str(bnd[1])
                        + ", got: "
                        + str(val)
                    )

    # bounds  properties ######################################################

    @property
    def var_bounds(self):
        """:class:`list`: Bounds for the variance

        Notes
        -----
        Is a list of 2 or 3 values:

            * ``[a, b]`` or
            * ``[a, b, <type>]``

        <type> is one of ``"oo"``, ``"cc"``, ``"oc"`` or ``"co"``
        to define if the bounds are open ("o") or closed ("c").
        """
        return self._var_bounds

    @var_bounds.setter
    def var_bounds(self, bounds):
        if not check_bounds(bounds):
            raise ValueError(
                "Given bounds for 'var' are not "
                + "valid, got: "
                + str(bounds)
            )
        self._var_bounds = bounds

    @property
    def len_scale_bounds(self):
        """:class:`list`: Bounds for the lenght scale

        Notes
        -----
        Is a list of 2 or 3 values:

            * ``[a, b]`` or
            * ``[a, b, <type>]``

        <type> is one of ``"oo"``, ``"cc"``, ``"oc"`` or ``"co"``
        to define if the bounds are open ("o") or closed ("c").
        """
        return self._len_scale_bounds

    @len_scale_bounds.setter
    def len_scale_bounds(self, bounds):
        if not check_bounds(bounds):
            raise ValueError(
                "Given bounds for 'len_scale' are not "
                + "valid, got: "
                + str(bounds)
            )
        self._len_scale_bounds = bounds

    @property
    def nugget_bounds(self):
        """:class:`list`: Bounds for the nugget

        Notes
        -----
        Is a list of 2 or 3 values:

            * ``[a, b]`` or
            * ``[a, b, <type>]``

        <type> is one of ``"oo"``, ``"cc"``, ``"oc"`` or ``"co"``
        to define if the bounds are open ("o") or closed ("c").
        """
        return self._nugget_bounds

    @nugget_bounds.setter
    def nugget_bounds(self, bounds):
        if not check_bounds(bounds):
            raise ValueError(
                "Given bounds for 'nugget' are not "
                + "valid, got: "
                + str(bounds)
            )
        self._nugget_bounds = bounds

    @property
    def opt_arg_bounds(self):
        """:class:`dict`: Bounds for the optional arguments

        Notes
        -----
        Keys are the opt-arg names and values are lists of 2 or 3 values:

            * ``[a, b]`` or
            * ``[a, b, <type>]``

        <type> is one of ``"oo"``, ``"cc"``, ``"oc"`` or ``"co"``
        to define if the bounds are open ("o") or closed ("c").
        """
        return self._opt_arg_bounds

    @property
    def arg_bounds(self):
        """:class:`dict`: Bounds for all parameters

        Notes
        -----
        Keys are the opt-arg names and values are lists of 2 or 3 values:

            * ``[a, b]`` or
            * ``[a, b, <type>]``

        <type> is one of ``"oo"``, ``"cc"``, ``"oc"`` or ``"co"``
        to define if the bounds are open ("o") or closed ("c").
        """
        res = {
            "var": self.var_bounds,
            "len_scale": self.len_scale_bounds,
            "nugget": self.nugget_bounds,
        }
        res.update(self.opt_arg_bounds)
        return res

    # standard parameters #####################################################

    @property
    def dim(self):
        """:class:`int`: The dimension of the model."""
        return self._dim

    @dim.setter
    def dim(self, dim):
        # check if a fixed dimension should be used
        if self.fix_dim() is not None:
            print(self.name + ": using fixed dimension " + str(self.fix_dim()))
            dim = self.fix_dim()
        # set the dimension
        if dim < 1 or dim > 3:
            raise ValueError("Only dimensions of 1 <= d <= 3 are supported.")
        self._dim = int(dim)
        # create fourier transform just once (recreate for dim change)
        self._sft = SFT(ndim=self.dim, **self.hankel_kw)
        # recalculate dimension related parameters
        if self._anis is not None:
            self._len_scale, self._anis = set_len_anis(
                self.dim, self._len_scale, self._anis
            )
        if self._angles is not None:
            self._angles = set_angles(self.dim, self._angles)

    @property
    def var(self):
        """:class:`float`: The variance of the model."""
        return self._var * self.var_factor()

    @var.setter
    def var(self, var):
        self._var = var / self.var_factor()
        self.check_arg_bounds()

    @property
    def var_raw(self):
        """:class:`float`: The raw variance of the model without factor

        (See. CovModel.var_factor)"""
        return self._var

    @var_raw.setter
    def var_raw(self, var_raw):
        self._var = var_raw
        self.check_arg_bounds()

    @property
    def nugget(self):
        """:class:`float`: The nugget of the model."""
        return self._nugget

    @nugget.setter
    def nugget(self, nugget):
        self._nugget = nugget
        self.check_arg_bounds()

    @property
    def len_scale(self):
        """:class:`float`: The main length scale of the model."""
        return self._len_scale

    @len_scale.setter
    def len_scale(self, len_scale):
        self._len_scale, self._anis = set_len_anis(
            self.dim, len_scale, self.anis
        )
        self.check_arg_bounds()

    @property
    def anis(self):
        """:class:`numpy.ndarray`: The anisotropy factors of the model."""
        return self._anis

    @anis.setter
    def anis(self, anis):
        self._len_scale, self._anis = set_len_anis(
            self.dim, self.len_scale, anis
        )
        self.check_arg_bounds()

    @property
    def angles(self):
        """:class:`numpy.ndarray`: The rotation angles (in rad) of the model."""
        return self._angles

    @angles.setter
    def angles(self, angles):
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
        self._integral_scale = self._calc_integral_scale()
        return self._integral_scale

    @integral_scale.setter
    def integral_scale(self, integral_scale):
        if integral_scale is not None:
            # format int-scale right
            self.len_scale = integral_scale
            integral_scale = self.len_scale
            # reset len_scale
            self.len_scale = 1.0
            int_tmp = self._calc_integral_scale()
            self.len_scale = integral_scale / int_tmp
            if not np.isclose(self.integral_scale, integral_scale, rtol=1e-3):
                raise ValueError(
                    self.name
                    + ": Integral scale could not be set correctly!"
                    + " Please just give a len_scale!"
                )

    @property
    def hankel_kw(self):
        """:class:`dict`: Keywords for :any:`hankel.SymmetricFourierTransform`
        """
        return self._hankel_kw

    @hankel_kw.setter
    def hankel_kw(self, hankel_kw):
        self._hankel_kw = HANKEL_DEFAULT if hankel_kw is None else hankel_kw
        if self.dim is not None:
            self._sft = SFT(ndim=self.dim, **self.hankel_kw)

    # properties ##############################################################

    @property
    def dist_func(self):
        """:class:`tuple` of :any:`callable`:
        pdf, cdf and ppf from the radial spectral density"""
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
        """:class:`bool`: State if a cdf is defined by the user"""
        return self._has_cdf()

    @property
    def has_ppf(self):
        """:class:`bool`: State if a ppf is defined by the user"""
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
        """:class:`list` of :class:`str`: Names of all arguments"""
        return ["var", "len_scale", "nugget", "anis", "angles"] + self._opt_arg

    @property
    def opt_arg(self):
        """:class:`list` of :class:`str`: Names of the optional arguments"""
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
        res = np.zeros(self.dim, dtype=float)
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
        res = np.zeros(self.dim, dtype=float)
        res[0] = self.integral_scale
        for i in range(1, self.dim):
            res[i] = self.integral_scale * self.anis[i - 1]
        return res

    @property
    def name(self):
        """:class:`str`: The name of the CovModel class"""
        return self.__class__.__name__

    # magic methods ###########################################################

    def __eq__(self, other):
        if not isinstance(other, CovModel):
            return False
        # prevent attribute error in opt_arg if the are not equal
        if set(self.opt_arg) != set(other.opt_arg):
            return False
        # prevent dim error in anis and angles
        if self.dim != other.dim:
            return False
        equal = True
        equal &= self.name == other.name
        equal &= np.isclose(self.var, other.var)
        equal &= np.isclose(self.var_raw, other.var_raw)  # ?! needless?
        equal &= np.isclose(self.nugget, other.nugget)
        equal &= np.isclose(self.len_scale, other.len_scale)
        equal &= np.all(np.isclose(self.anis, other.anis))
        equal &= np.all(np.isclose(self.angles, other.angles))
        for opt in self.opt_arg:
            equal &= np.isclose(getattr(self, opt), getattr(other, opt))
        return equal

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        opt_str = ""
        for opt in self.opt_arg:
            opt_str += ", " + opt + "={}".format(getattr(self, opt))
        return (
            "{0}(dim={1}, var={2}, len_scale={3}, "
            "nugget={4}, anis={5}, angles={6}".format(
                self.name,
                self.dim,
                self.var,
                self.len_scale,
                self.nugget,
                self.anis,
                self.angles,
            )
            + opt_str
            + ")"
        )


if __name__ == "__main__": # pragma: no cover
    import doctest

    doctest.testmod()
