# -*- coding: utf-8 -*-
"""
GStools subpackage providing the base class for covariance models.

.. currentmodule:: gstools.covmodel.base

The following classes are provided

.. autosummary::
   CovModel
"""
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
    dim : int
        dimension of the model
    var : float
        variance of the model (the nugget is not included in "this" variance)
    nugget : float
        nugget of the model
    sill : float
        sill (limit) of the variogram given by: nugget + var
    len_scale : float
        length scale of the model in the x-direction
    len_scale_vec : array
        length scales of the model in the all directions
    integral_scale : float
        integral scale of the model in x-direction
    integral_scale_vec : array
        integral scales of the model in the all directions
    anis : array
        anisotropy ratios in the transversal directions [y, z]
    angles : array
        angles of rotation:

            * in 2D: given as rotation around z-axis
            * in 3D: given by yaw, pitch, and roll (known as Tait–Bryan angles)

    arg : list
        list of all argument names (var, len_scale, nugget, [opt_arg])
    opt_arg : list
        list of the optional-argument names
    var_bounds : list
        bounds for the variance of the model
    len_scale_bounds : list
        bounds for the length scale of the model in the x-direction
    nugget_bounds : list
        bounds for the nugget of the model
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
        var_bounds=(0.0, 100.0, "cc"),
        len_scale_bounds=(0.0, 1000.0, "oo"),
        nugget_bounds=(0.0, 100.0, "cc"),
        hankel_kw=None,
        **opt_arg
    ):
        r"""The gstools covariance model

        Parameters
        ----------
        dim : int, optional
            dimension of the model. Dafault: 3
        var : float, optional
            variance of the model
            (the nugget is not included in "this" variance). Dafault: 1.0
        len_scale : float or array, optional
            length scale of the model. Either given as single value for
            isotropic models, or as a list for anisotropic models.
            Dafault: 1.0
        nugget : float, optional
            nugget of the model. The nugget is just added to the standard
            variogram and will not be included in any scale calculations.
            Dafault: 0.0
        anis : array, optional
            anisotropy ratios in the transversal directions [y, z].
            Either a single value, or a list. If len_scale is given as list,
            these values are ignored. Dafault: 1.0
        angles : array, optional
            angles of rotation:

                * in 2D: single value given as rotation around z-axis
                * in 3D: given by yaw, pitch, and roll
                  (known as Tait–Bryan angles)

            Dafault: 0.0
        integral_scale : float or None, optional
            can be given like len_scale. Will calculate an appropriate length
            scale. Default: None
        var_bounds : list, optional
            bounds for the variance of the model. Default: (0.0, 100.0)
        len_scale_bounds : list, optional
            bounds for the length scale of the model in the main-direction.
            Default: (0.0, 1000.0)
        nugget_bounds : list, optional
            bounds for the nugget of the model. Default: (0.0, 100.0)
        hankel_kw : :class:`dict` or :class:`None`, optional
            keywords for :class:`hankel.SymmetricFourierTransform`.
            Only edit if you really know what you are doing. Default: None
        **opt_arg
            Placeholder for optional argument of derived classes.

        Caution
        -------
        Don't instantiate ``CovModel`` directly. You need to inherit a
        child class which overrides one of the following methods:

            * ``model.variogram(r)``
                :math:`\gamma\left(r\right)=
                \sigma^2\cdot\left(1-\tilde{C}\left(r\right)\right)+n`
            * ``model.variogram_normed(r)``
                :math:`\tilde{\gamma}\left(r\right)=
                1-\tilde{C}\left(r\right)`
            * ``model.covariance(r)``
                :math:`C\left(r\right)=
                \sigma^2\cdot\tilde{C}\left(r\right)`
            * ``model.covariance_normed(r)``
                :math:`\tilde{C}\left(r\right)`
        """
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

        # set standard boundaries for variance, len_scale and nugget
        self._var_bounds = None
        self.var_bounds = var_bounds
        self._len_scale_bounds = None
        self.len_scale_bounds = len_scale_bounds
        self._nugget_bounds = None
        self.nugget_bounds = nugget_bounds
        # set standard boundaries for the optional arguments
        self._opt_arg_bounds = self.default_opt_arg_bounds()

        # tuning arguments for the hankel-/fourier-transformation
        # set before "dim" set, because SFT needs the dimension
        # SFT class will be created within dim.setter
        self.hankel_kw = HANKEL_DEFAULT if hankel_kw is None else hankel_kw
        self._ft = None

        # prepare dim setting
        self._dim = None
        self._len_scale = None
        self._anis = None
        self._angles = None
        self.dim = dim
        # set parameters
        self._nugget = nugget
        self._angles = set_angles(self.dim, angles)
        self._len_scale, self._anis = set_len_anis(self.dim, len_scale, anis)
        # set var at last, because of the var_factor (to be right initialized)
        self._var = None
        self.var = var
        self._integral_scale = None
        self.integral_scale = integral_scale
        self.var = var  # set var again, if int_scale affects var_factor
        self.check_arg_bounds()
        # additional checks for the optional arguments (provided by user)
        self.check_opt_arg()

    ###########################################################################
    # one of these functions needs to be overridden ###########################
    ###########################################################################

    def __init_subclass__(cls):

        # overrid one of these ################################################
        def variogram(self, r):
            r"""Isotropic variogram of the model

            Given by: :math:`\gamma\left(r\right)=
            \sigma^2\cdot\left(1-\tilde{C}\left(r\right)\right)+n`

            Where :math:`\tilde{C}(r)` is the normalized covariance.
            """
            return self.var - self.covariance(r) + self.nugget

        def covariance(self, r):
            r"""Covariance of the model

            Given by: :math:`C\left(r\right)=
            \sigma^2\cdot\tilde{C}\left(r\right)`

            Where :math:`\tilde{C}(r)` is the normalized covariance.
            """
            return self.var * self.covariance_normed(r)

        def covariance_normed(self, r):
            r"""Normalized covariance of the model

            Given by: :math:`\tilde{C}\left(r\right)`

            It has to be a monotonic decreasing function with
            :math:`\tilde{C}(0)=1` and :math:`\tilde{C}(\infty)=0`.
            """
            return 1.0 - self.variogram_normed(r)

        def variogram_normed(self, r):
            r"""Normalized variogram of the model

            Given by: :math:`\tilde{\gamma}\left(r\right)=
            1-\tilde{C}\left(r\right)`

            Where :math:`\tilde{C}(r)` is the normalized covariance.
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
        if not hasattr(cls, "covariance_normed"):
            cls.covariance_normed = covariance_normed
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
                + "'covariance_normed', or 'variogram_normed'."
            )

        # modify the docstrings ###############################################

        # class docstring gets attributes added
        cls.__doc__ += CovModel.__doc__[44:]
        # overridden functions get standard doc added
        ignore = ["__", "variogram", "covariance"]
        for attr in cls.__dict__:
            if any(
                [attr.startswith(ign) for ign in ignore]
            ) or attr not in dir(CovModel):
                continue
            attr_doc = getattr(CovModel, attr).__doc__
            attr_cls = cls.__dict__[attr]
            if attr_cls.__doc__ is None:
                attr_cls.__doc__ = attr_doc
            else:
                attr_cls.__doc__ += "\n\n" + attr_doc

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

        Note
        ----
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

    def calc_integral_scale(self):
        """calculate the integral scale of the isotrope model"""
        self._integral_scale = integral(self.covariance_normed, 0, np.inf)[0]
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
        k : float
            Radius of the phase: :math:`k=\left\Vert\mathbf{k}\right\Vert`
        """
        k = np.array(np.abs(k), dtype=float)
        return self._ft.transform(self.covariance, k, ret_err=False)

    def spectral_density(self, k):
        r"""
        The spectral density of the covariance model.

        This is given by:

        .. math:: \tilde{S}(k) = \frac{S(k)}{\sigma^2}

        Where :math:`S(k)` is the spectrum of the covariance model.

        Parameters
        ----------
        k : float
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
        x_data : array
            The radii of the meassured variogram.
        y_data : array
            The messured variogram
        **para_deselect
            You can deselect the parameters to be fitted, by setting
            them "False" as keywords. By default, all parameters are
            fitted.

        Note
        ----
        You can set the bounds for each parameter by accessing
        ``model.set_arg_bounds(...)``
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
        out = {}
        para_skip = 0
        opt_skip = 0
        if para["var"]:
            var_tmp = popt[para_skip]
            out["var"] = popt[para_skip]
            para_skip += 1
        else:
            out["var"] = self.var
        if para["len_scale"]:
            self.len_scale = popt[para_skip]
            out["len_scale"] = popt[para_skip]
            para_skip += 1
        else:
            out["len_scale"] = self.len_scale
        if para["nugget"]:
            self.nugget = popt[para_skip]
            out["nugget"] = popt[para_skip]
            para_skip += 1
        else:
            out["nugget"] = self.nugget
        for opt in self.opt_arg:
            if para[opt]:
                setattr(self, opt, popt[para_skip + opt_skip])
                out[opt] = popt[para_skip + opt_skip]
                opt_skip += 1
            else:
                out[opt] = getattr(self, opt)
        # set var at last because of var_factor (other parameter needed)
        if para["var"]:
            self.var = var_tmp
        # recalculate the integral scale
        self._integral_scale = self.calc_integral_scale()
        out["integral_scale"] = self._integral_scale
        return out, pcov

    # bounds setting and checks ###############################################

    def set_arg_bounds(self, **kwargs):
        r"""Set bounds for the parameters of the model

        Parameters
        ----------
        **kwargs
            Parameter name as keyword ("var", "len_scale", "nugget", <opt_arg>)
            and a list of 2 or 3 values as value:

                * [a, b]
                * [a, b, <type>]

            <type> is one of "oo", "cc", "oc" or "co" to define if the bounds
            are open ("o") or closed ("c")
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
                bnd.append("cc")
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
        """Bounds for the variance

        Note
        ----
        Is a list of 2 or 3 values:

            * [a, b]
            * [a, b, <type>]

        <type> is one of "oo", "cc", "oc" or "co" to define if the bounds
        are open ("o") or closed ("c")
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
        """Bounds for the lenght scale

        Note
        ----
        Is a list of 2 or 3 values:

            * [a, b]
            * [a, b, <type>]

        <type> is one of "oo", "cc", "oc" or "co" to define if the bounds
        are open ("o") or closed ("c")
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
        """Bounds for the nugget

        Note
        ----
        Is a list of 2 or 3 values:

            * [a, b]
            * [a, b, <type>]

        <type> is one of "oo", "cc", "oc" or "co" to define if the bounds
        are open ("o") or closed ("c")
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
        """Bounds for the optional arguments

        Note
        ----
        Is a list of 2 or 3 values:

            * [a, b]
            * [a, b, <type>]

        <type> is one of "oo", "cc", "oc" or "co" to define if the bounds
        are open ("o") or closed ("c")
        """
        return self._opt_arg_bounds

    @property
    def arg_bounds(self):
        """Bounds for all parameters

        Note
        ----
        Is a list of 2 or 3 values:

            * [a, b]
            * [a, b, <type>]

        <type> is one of "oo", "cc", "oc" or "co" to define if the bounds
        are open ("o") or closed ("c")
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
        """The dimension of the model."""
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
        self._ft = SFT(ndim=self.dim, **self.hankel_kw)
        # recalculate dimension related parameters
        if self._anis is not None:
            self._len_scale, self._anis = set_len_anis(
                self.dim, self._len_scale, self._anis
            )
        if self._angles is not None:
            self._angles = set_angles(self.dim, self._angles)

    @property
    def var(self):
        """The variance of the model."""
        return self._var * self.var_factor()

    @var.setter
    def var(self, var):
        self._var = var / self.var_factor()
        self.check_arg_bounds()

    @property
    def var_raw(self):
        """The raw variance of the model without factor

        (See. CovModel.var_factor)"""
        return self._var

    @var_raw.setter
    def var_raw(self, var_raw):
        self._var = var_raw
        self.check_arg_bounds()

    @property
    def nugget(self):
        """The nugget of the model."""
        return self._nugget

    @nugget.setter
    def nugget(self, nugget):
        self._nugget = nugget
        self.check_arg_bounds()

    @property
    def len_scale(self):
        """The main length scale of the model."""
        return self._len_scale

    @len_scale.setter
    def len_scale(self, len_scale):
        self._len_scale, self._anis = set_len_anis(
            self.dim, len_scale, self.anis
        )
        self.check_arg_bounds()

    @property
    def anis(self):
        """The anisotropy factors of the model."""
        return self._anis

    @anis.setter
    def anis(self, anis):
        self._len_scale, self._anis = set_len_anis(
            self.dim, self.len_scale, anis
        )
        self.check_arg_bounds()

    @property
    def angles(self):
        """The rotation angles (in rad) of the model."""
        return self._angles

    @angles.setter
    def angles(self, angles):
        self._angles = set_angles(self.dim, angles)
        self.check_arg_bounds()

    @property
    def integral_scale(self):
        """The main integral scale of the model."""
        self._integral_scale = self.calc_integral_scale()
        return self._integral_scale

    @integral_scale.setter
    def integral_scale(self, integral_scale):
        if integral_scale is not None:
            self.len_scale = 1.0
            int_tmp = self.calc_integral_scale()
            self.len_scale = integral_scale / int_tmp
            self.check_arg_bounds()

    # properties ##############################################################

    @property
    def dist_func(self):
        """give pdf, cdf and ppf from the radial spectral density"""
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
        """State if a cdf is defined by the user"""
        return self._has_cdf()

    @property
    def has_ppf(self):
        """State if a ppf is defined by the user"""
        return self._has_ppf()

    @property
    def sill(self):
        """The sill of the variogram."""
        return self.var + self.nugget

    @property
    def arg(self):
        """Names of all arguments"""
        return ["var", "len_scale", "nugget", "anis", "angles"] + self._opt_arg

    @property
    def opt_arg(self):
        """Names of the optional arguments"""
        return self._opt_arg

    @property
    def len_scale_vec(self):
        """The length scales in each direction of the model.

        Note
        ----
        * len_scale_x = len_scale
        * len_scale_y = len_scale*anis_y
        * len_scale_z = len_scale*anis_z
        """
        res = np.zeros(self.dim, dtype=float)
        res[0] = self.len_scale
        for i in range(1, self._dim):
            res[i] = self.len_scale * self.anis[i - 1]
        return res

    @property
    def integral_scale_vec(self):
        """
        The integral scales in each direction of the model.

        Note
        ----
        This is calculated by:
            * integral_scale_x = integral_scale
            * integral_scale_y = integral_scale*anis_y
            * integral_scale_z = integral_scale*anis_z
        """
        res = np.zeros(self.dim, dtype=float)
        res[0] = self.integral_scale
        for i in range(1, self.dim):
            res[i] = self.integral_scale * self.anis[i - 1]
        return res

    @property
    def name(self):
        """
        The name of the CovModel class
        """
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
