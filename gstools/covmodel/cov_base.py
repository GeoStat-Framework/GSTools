#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 22:16:51 2018

@author: muellese
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
    check_bounds,
)

HANKEL_DEFAULT = {
    "a": -1,  # should only be changed, if you know exactly what
    "b": 1,  # you do or if you are crazy
    "N": 1000,
    "h": 0.001,
}

# The CovModel Base-Class #####################################################


class CovModel(six.with_metaclass(InitSubclassMeta)):
    """
    Base class for the GSTools covariance models

    Notes
    -----
    Don't instantiate this class directly. You need to inherit a child class
    which overrides one of the following methods:

        * ``CovModel.variogram(r)``
            :math:`\\gamma\\left(r\\right)=\\sigma^2\\cdot\\left(1-\\tilde{C}\\left(r\\right)\\right)+n`
        * ``CovModel.variogram_normed(r)``
            :math:`\\tilde{\\gamma}\\left(r\\right)=1-\\tilde{C}\\left(r\\right)`
        * ``CovModel.covariance(r)``
            :math:`C\\left(r\\right)=\\sigma^2\\cdot\\tilde{C}\\left(r\\right)`
        * ``CovModel.covariance_normed(r)``
            :math:`\\tilde{C}\\left(r\\right)`

    Attributes
    ----------
    dim : int
        dimension of the model
    var : float
        variance of the model (the nugget is not included in "this" variance)
    len_scale : float
        length scale of the model in the x-direction
    len_scale_vec : array
        length scales of the model in the all directions
    integral_scale : float
        integral scale of the model in x-direction
    integral_scale_vec : array
        integral scales of the model in the all directions
    nugget : float
        nugget of the model
    anis : array
        anisotropy ratios in the transversal directions [y, z]
    angles : array
        angles of the transversal directions [y, z]
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
        var_bounds=(0.0, 100.0),
        len_scale_bounds=(0.0, 1000.0),
        nugget_bounds=(0.0, 100.0),
        hankel_kw=None,
        **kwargs
    ):
        """Instantiate a covariance model"""
        # assert, that we use a subclass
        # this is the case, if __init_subclass__ is called, which creates
        # the "variogram"... so we check for that
        if not hasattr(self, "variogram"):
            raise TypeError("Don't instantiate 'CovModel' directly!")

        # optional arguments for the variogram-model
        # look up the defaults of the optional arguments (defined by the user)
        default = self.default_opt_arg()
        # add the default vaules if not specified
        for def_arg in default:
            if def_arg not in kwargs:
                kwargs[def_arg] = default[def_arg]
        # save names of the optional arguments
        self._opt_arg = list(kwargs.keys())
        # add the optional arguments as attributes to the class
        for kws in kwargs:
            if kws in dir(self):  # "dir" also respects properties
                raise ValueError(
                    "parameter '"
                    + kws
                    + "' has a 'bad' name, "
                    + "since it is already present in the class. "
                    + "It could not be added to the model"
                )
            # Magic happens here
            setattr(self, kws, kwargs[kws])

        # set standard boundaries for the optional arguments
        self._opt_arg_bounds = self.default_opt_arg_bounds()
        # set standard boundaries for variance, len_scale and nugget
        self._var_bounds = None
        self.var_bounds = var_bounds
        self._len_scale_bounds = None
        self.len_scale_bounds = len_scale_bounds
        self._nugget_bounds = None
        self.nugget_bounds = nugget_bounds

        # set dimension
        # check if a fixed dimension should be used
        if self.fix_dim() is not None:
            dim = self.fix_dim()
        # set the dimension
        if dim < 1 or dim > 3:
            raise ValueError("Only dimensions of 1 <= d <= 3 are supported.")
        self._dim = int(dim)
        # set the variance of the field
        self._var = var
        # set the nugget of the field
        self._nugget = nugget

        # set the rotation angles
        self._angles = np.atleast_1d(angles)
        # fill up the rotation angle array with zeros, such that len() == dim-1
        self._angles = np.pad(
            self._angles,
            (0, self._dim - len(self._angles)),
            "constant",
            constant_values=0.0,
        )

        # if integral scale is given, the length-scale is overwritten
        if integral_scale is not None:
            # first set len_scale to 1, than calculate the scaling factor
            len_scale = 1.0

        # set the length scales and the anisotropy factors
        self._len_scale, self._anis = set_len_anis(dim, len_scale, anis)

        # initialize the integral scale
        self._integral_scale = None

        # recalculate the length scale, to adopt the given integral scale
        if integral_scale is not None:
            self._integral_scale = self.calc_integral_scale()
            int_tmp, self._anis = set_len_anis(dim, integral_scale, anis)
            self._len_scale = int_tmp / self._integral_scale
            # recalculate the internal integral scale
            self._integral_scale = int_tmp

        # tuning arguments for the hankel-/fourier-transformation
        if hankel_kw is None:
            self.hankel_kw = HANKEL_DEFAULT
        else:
            self.hankel_kw = hankel_kw

        # check the arguments
        self.check_arg_bounds()
        # additional checks for the optional arguments (provided by user)
        self.check_opt_arg()

    ###########################################################################
    # one of these functions needs to be overridden ###########################
    ###########################################################################

    def __init_subclass__(cls):

        # overrid one of these ################################################
        def variogram(self, r):
            return self.var - self.covariance(r) + self.nugget

        def covariance(self, r):
            return self.var * self.covariance_normed(r)

        def covariance_normed(self, r):
            return 1.0 - self.variogram_normed(r)

        def variogram_normed(self, r):
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

    ###########################################################################
    # methods for optional arguments (can be overridden) ######################
    ###########################################################################

    def default_opt_arg(self):
        """Here you can provide a dictionary with default values for
        the optional arguments, see one of the CovModel implementations."""
        return {}

    def default_opt_arg_bounds(self):
        """Here you can provide a dictionary with default boundaries for
        the optional arguments, see one of the CovModel implementations."""
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
        """This method can be overridden to set a fixed dimension for
        your model"""
        return None

    # calculation of different scales #########################################

    def calc_integral_scale(self):
        """calculate the integral scale of the
        isotrope model (can be overwritten)"""
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

    # bounds setting and checks ###############################################

    def set_arg_bounds(self, **kwargs):
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

    ###########################################################################
    # spectrum methods (can be overridden for speedup) ########################
    ###########################################################################

    def spectrum(self, k):
        """
        The spectrum of the covariance model.
        """
        k = np.abs(np.array(k, dtype=float))
        f_t = SFT(ndim=self.dim, **self.hankel_kw)
        return f_t.transform(self.covariance, k, ret_err=False)

    def spectral_density(self, k):
        """
        The spectral density of the covariance model.
        """
        return self.spectrum(k) / self.var

    def spectral_rad_pdf(self, r):
        """
        The radial spectral density of the model depending on the dimension
        """
        r = np.abs(np.array(r, dtype=float))
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
        # TODO: this is totally hacky
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
        """State if a cdf is defined by the user (can be overwritten)"""
        return hasattr(self, "spectral_rad_cdf")

    def _has_ppf(self):
        """State if a ppf is defined by the user (can be overwritten)"""
        return hasattr(self, "spectral_rad_ppf")

    # fitting routine #########################################################

    def fit_variogram(self, x_data, y_data, **para_deselect):
        """
        fit the variogram-model to given data

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
        """

        para = {"var": True, "len_scale": True, "nugget": True}
        for opt in self.opt_arg:
            para[opt] = True
        para.update(para_deselect)

        # we need arg1, otherwise curve_fit throws an error (bug?!)
        def curve(x, arg1, *args):
            """dummy function for the variogram"""
            args = (arg1,) + args
            para_skip = 0
            opt_skip = 0
            if para["var"]:
                self._var = args[para_skip]
                para_skip += 1
            if para["len_scale"]:
                self._len_scale = args[para_skip]
                para_skip += 1
            if para["nugget"]:
                self._nugget = args[para_skip]
                para_skip += 1
            for opt in self.opt_arg:
                if para[opt]:
                    setattr(self, opt, args[para_skip + opt_skip])
                    opt_skip += 1
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

        print(low_bounds)
        print(top_bounds)

        # fit the variogram
        popt, pcov = curve_fit(
            curve, x_data, y_data, bounds=(low_bounds, top_bounds)
        )
        out = {}
        para_skip = 0
        opt_skip = 0
        if para["var"]:
            self._var = popt[para_skip]
            out["var"] = popt[para_skip]
            para_skip += 1
        else:
            out["var"] = self._var
        if para["len_scale"]:
            self._len_scale = popt[para_skip]
            out["len_scale"] = popt[para_skip]
            para_skip += 1
        else:
            out["len_scale"] = self._len_scale
        if para["nugget"]:
            self._nugget = popt[para_skip]
            out["nugget"] = popt[para_skip]
            para_skip += 1
        else:
            out["nugget"] = self._nugget
        for opt in self.opt_arg:
            if para[opt]:
                setattr(self, opt, popt[para_skip + opt_skip])
                out[opt] = popt[para_skip + opt_skip]
                opt_skip += 1
            else:
                out[opt] = getattr(self, opt)
        # recalculate the integral scale
        self._integral_scale = self.calc_integral_scale()
        out["integral_scale"] = self._integral_scale
        return out, pcov

    # bounds ##################################################################

    @property
    def var_bounds(self):
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
        return self._opt_arg_bounds

    @property
    def arg_bounds(self):
        res = {
            "var": self.var_bounds,
            "len_scale": self.len_scale_bounds,
            "nugget": self.nugget_bounds,
        }
        res.update(self.opt_arg_bounds)
        return res

    # properties ##############################################################

    @property
    def has_cdf(self):
        """State if a cdf is defined by the user"""
        return self._has_cdf()

    @property
    def has_ppf(self):
        """State if a ppf is defined by the user"""
        return self._has_ppf()

    @property
    def dim(self):
        """ The dimension of the spatial random field."""
        return self._dim

    @property
    def var(self):
        """ The variance of the spatial random field."""
        return self._var

    @property
    def len_scale(self):
        """ The main length scale of the spatial random field."""
        return self._len_scale

    @property
    def nugget(self):
        """ The nugget of the spatial random field."""
        return self._nugget

    @property
    def len_scale_vec(self):
        """The length scales in each direction of the spatial random field.

        Notes
        -----
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
    def anis(self):
        """ The anisotropy factors of the spatial random field."""
        return self._anis

    @property
    def angles(self):
        """ The rotation angles (in rad) of the spatial random field."""
        return self._angles

    @property
    def arg(self):
        """Names of all arguments"""
        return ["var", "len_scale", "nugget"] + self._opt_arg

    @property
    def opt_arg(self):
        """Names of the optional arguments"""
        return self._opt_arg

    @property
    def integral_scale(self):
        """The main integral scale of the spatial random field."""
        # just calculate it once (otherwise call 'calc_integral_scale')
        if self._integral_scale is None:
            self._integral_scale = self.calc_integral_scale()
        return self._integral_scale

    @property
    def integral_scale_vec(self):
        """
        The integral scales in each direction of the spatial random field.

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

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return (
            "{0}(dim={1}, var={2}, len_scale={3}, "
            "nugget={4}, anis={5}, angles={6})".format(
                self.name,
                self.dim,
                self.var,
                self.len_scale,
                self.nugget,
                self.anis,
                self.angles,
            )
        )
