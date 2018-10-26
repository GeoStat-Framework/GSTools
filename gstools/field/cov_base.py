#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 22:16:51 2018

@author: muellese
"""
from __future__ import print_function, division, absolute_import

import warnings
import six
import numpy as np
from scipy.integrate import quad as integral
from scipy.optimize import curve_fit, root
from scipy import special as sps
from hankel import SymmetricFourierTransform as SFT
from matplotlib import pyplot as plt


# __init_subclass__ hack ######################################################

if hasattr(object, "__init_subclass__"):
    InitSubclassMeta = type
else:
    class InitSubclassMeta(type):
        """Metaclass that implements PEP 487 protocol

        Notes
        -----
        See :
            https://www.python.org/dev/peps/pep-0487

        taken from :
            https://github.com/graphql-python/graphene/blob/master/graphene/pyutils/init_subclass.py
        """
        def __new__(cls, name, bases, ns, **kwargs):
            __init_subclass__ = ns.pop("__init_subclass__", None)
            if __init_subclass__:
                __init_subclass__ = classmethod(__init_subclass__)
                ns["__init_subclass__"] = __init_subclass__
            return super(InitSubclassMeta, cls).__new__(cls, name, bases, ns,
                                                        **kwargs)

        def __init__(cls, name, bases, ns, **kwargs):
            super(InitSubclassMeta, cls).__init__(name, bases, ns)
            super_class = super(cls, cls)
            if hasattr(super_class, "__init_subclass__"):
                super_class.__init_subclass__.__func__(cls, **kwargs)


# Helping functions ###########################################################

def rad_fac(dim, r):
    '''The volume element of the n-dimensional spherical coordinates.

    As a factor for integration of a radial-symmetric function.

    Parameters
    ----------
    dim : :class:`int`
        spatial dimension
    r : :class:`numpy.ndarray`
        Given radii.
    '''
    if dim == 1:
        fac = 2.0
    elif dim == 2:
        fac = 2*np.pi*r
    elif dim == 3:
        fac = 4*np.pi*r**2
    else:  # general solution ( for the record :D )
        fac = dim*r**(dim-1)*np.sqrt(np.pi)**dim/sps.gamma(dim/2. + 1.)
    return fac


def set_len_anis(dim, len_scale, anis):
    """Setting the length scale and anisotropy factors for the given dimension.

    Parameters
    ----------
    dim : :class:`int`
        spatial dimension
    len_scale : :class:`float`/list
        the length scale of the SRF in x direction or in x- (y-, z-) direction
    anis : :class:`float`/list
        the anisotropy of length scales along the y- and z-directions

    Returns
    -------
    len_scale : :class:`float`
        the main length scale of the SRF in x direction
    anis : :class:`float`/list, optional
        the anisotropy of length scales along the y- and z-directions

    Notes
    -----
    If ``len_scale`` is given as list, ``anis`` will be recalculated.
    """
    ls_tmp = np.atleast_1d(len_scale)[:dim]
    # use just one length scale (x-direction)
    out_len_scale = ls_tmp[0]
    # set the anisotropies in y- and z-direction according to the input
    if len(ls_tmp) == 1:
        out_anis = np.atleast_1d(anis)[:dim-1]
        if len(out_anis) < dim-1:
            # fill up the anisotropies with ones, such that len()==dim-1
            out_anis = np.pad(out_anis, (0, dim-len(out_anis)-1),
                              'constant', constant_values=1.)
    elif dim == 1:
        # there is no anisotropy in 1 dimension
        out_anis = np.empty(0)
    else:
        # fill up length-scales with main len_scale, such that len()==dim
        if len(ls_tmp) < dim:
            ls_tmp = np.pad(ls_tmp, (0, dim-len(ls_tmp)),
                            'constant', constant_values=out_len_scale)
        # if multiple length-scales are given, calculate the anisotropies
        out_anis = np.zeros(dim - 1, dtype=float)
        for i in range(1, dim):
            out_anis[i-1] = ls_tmp[i]/ls_tmp[0]

    for ani in out_anis:
        if not ani > 0.:
            raise ValueError("anisotropy-ratios needs to be > 0, " +
                             "got: "+str(out_anis))
    return out_len_scale, out_anis


def check_bounds(bounds):
    if len(bounds) not in (2, 3):
        return False
    if bounds[1] <= bounds[0]:
        return False
    if len(bounds) == 3 and bounds[2] not in ("oo", "oc", "co", "cc"):
        return False
    return True


# The CovModel Base-Class #####################################################

class CovModel(six.with_metaclass(InitSubclassMeta)):
    # TODO: docs ... jaja
    def __init__(
        self,
        dim=3,
        var=1.,
        len_scale=1.,
        anis=1.,
        angles=0.,
        integral_scale=None,
        var_bounds=(0., 100.),
        len_scale_bounds=(0., 1000.),
        hankel_kw=None,
        **kwargs
    ):
        # assert, that we use a subclass
        # this is the case, if __init_subclass__ is called, which creates
        # the "variogram"... so we check for that
        if not hasattr(self, 'variogram'):
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
                raise ValueError("parameter '"+kws+"' has a 'bad' name " +
                                 "and could not be added to the model")
            # Magic happens here
            setattr(self, kws, kwargs[kws])

        # set standard boundaries for the optional arguments
        self._opt_arg_bounds = self.default_opt_arg_bounds()
        # set standard boundaries for len_scale and variance
        self._len_scale_bounds = None
        self.len_scale_bounds = len_scale_bounds
        self._var_bounds = None
        self.var_bounds = var_bounds

        # check if a fixed dimension should be used
        if self.fix_dim() is not None:
            dim = self.fix_dim
        # set the dimension
        if dim < 1 or dim > 3:
            raise ValueError('Only dimensions of 1 <= d <= 3 are supported.')
        self._dim = dim
        # set the variance of the field
        self._var = var

        # set the rotation angles
        self._angles = np.atleast_1d(angles)
        # fill up the rotation angle array with zeros, such that len() == dim-1
        self._angles = np.pad(self._angles, (0, self._dim-len(self._angles)),
                              'constant', constant_values=0.)

        # if integral scale is given, the length-scale is overwritten
        if integral_scale is not None:
            # first set len_scale to 1, than calculate the scaling factor
            len_scale = 1.

        # set the length scales and the anisotropy factors
        self._len_scale, self._anis = set_len_anis(dim, len_scale, anis)

        # initialize the integral scale
        self._integral_scale = self.calc_integral_scale()

        # recalculate the length scale, to adopt the given integral scale
        if integral_scale is not None:
            int_tmp, self._anis = set_len_anis(dim, integral_scale, anis)
            self._len_scale = int_tmp/self._integral_scale
            # recalculate the internal integral scale
            self._integral_scale = int_tmp

        # tuning arguments for the hankel-/fourier-transformation
        if hankel_kw is None:
            self.hankel_kw = {
                "a": -1,    # should only be changed, if you know exactly what
                "b": 1,     # you do or if you are crazy
                "N": 1000,
                "h": 0.001,
            }
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
            return self.var - self.covariance(r)

        def covariance(self, r):
            return self.var*self.covariance_normed(r)

        def covariance_normed(self, r):
            return 1. - self.variogram_normed(r)

        def variogram_normed(self, r):
            return self.variogram(r)/self.var
        #######################################################################

        abstract = True
        if not hasattr(cls, 'variogram'):
            cls.variogram = variogram
        else:
            abstract = False
        if not hasattr(cls, 'covariance'):
            cls.covariance = covariance
        else:
            abstract = False
        if not hasattr(cls, 'covariance_normed'):
            cls.covariance_normed = covariance_normed
        else:
            abstract = False
        if not hasattr(cls, 'variogram_normed'):
            cls.variogram_normed = variogram_normed
        else:
            abstract = False
        if abstract:
            raise TypeError("Can't instantiate class '"+cls.__name__+"', " +
                            "without overriding at least on of the methods " +
                            "'variogram', 'covariance', " +
                            "'covariance_normed', or 'variogram_normed'.")

    ###########################################################################
    # methods for optional arguments (can be overridden) ######################
    ###########################################################################

    def default_opt_arg(self):
        '''Here you can provide a dictionary with default values for
        the optional arguments, see one of the CovModel implementations.'''
        return {}

    def default_opt_arg_bounds(self):
        '''Here you can provide a dictionary with default boundaries for
        the optional arguments, see one of the CovModel implementations.'''
        res = {}
        for opt in self.opt_arg:
            res[opt] = [0., 1000.]
        return res

    def check_opt_arg(self):
        '''Here you can run checks for the optional arguments

        This is in addition to the bound-checks

        Notes
        -----
        * You can use this to raise a ValueError/warning
        * Any return value will be ignored
        * This method will only be run once, when the class is initialized
        '''
        pass

    def fix_dim(self):
        '''This method can be overwritten to set a fixed dimension for
        your model'''
        return None

    # calculation of different scales #########################################

    def calc_integral_scale(self):
        '''calculate the integral scale of the
        isotrope model (can be overwritten)'''
        self._integral_scale = integral(self.covariance_normed, 0, np.inf)[0]
        return self._integral_scale

    def percentile_scale(self, per=0.9):
        '''calculate the distance, where the given percentile of the variance
        is reached by the variogram'''
        # check the given percentile
        if not 0. < per < 1.:
            raise ValueError("percentile needs to be within (0, 1), " +
                             "got: "+str(per))

        # define a curve, that has its root at the wanted point
        def curve(x):
            return self.variogram_normed(x) - per

        # take 'per * integral_scale' as initial guess
        return root(curve, per*self.integral_scale)["x"][0]

    # bounds setting and checks ###############################################

    def set_arg_bounds(self, **kwargs):
        for opt in kwargs:
            if opt in self.opt_arg:
                if not check_bounds(kwargs[opt]):
                    raise ValueError("Given bounds for '"+opt+"' are not " +
                                     "valid, got: "+str(kwargs[opt]))
                self._opt_arg_bounds[opt] = kwargs[opt]
            if opt == "len_scale":
                self.len_scale_bounds = kwargs[opt]
            if opt == "var":
                self.var_bounds = kwargs[opt]

    def check_arg_bounds(self):
        '''Here the arguments are checked to be within the given bounds'''
        # check len_scale, var and optional-arguments
        for arg in self.arg_bounds:
            bnd = list(self.arg_bounds[arg])
            val = getattr(self, arg)
            if len(bnd) == 2:
                bnd.append("oo")
            if bnd[2][0] == "c":
                if val < bnd[0]:
                    raise ValueError(str(arg)+" needs to be >= "+str(bnd[0]) +
                                     ", got: "+str(val))
            else:
                if val <= bnd[0]:
                    raise ValueError(str(arg)+" needs to be > "+str(bnd[0]) +
                                     ", got: "+str(val))
            if bnd[2][1] == "c":
                if val > bnd[1]:
                    raise ValueError(str(arg)+" needs to be <= "+str(bnd[1]) +
                                     ", got: "+str(val))
            else:
                if val >= bnd[1]:
                    raise ValueError(str(arg)+" needs to be < "+str(bnd[1]) +
                                     ", got: "+str(val))

    ###########################################################################
    # spectrum methods (can be overwritten for speedup) #######################
    ###########################################################################

    def spectrum(self, k):
        k = np.abs(np.array(k, dtype=float))
        f_t = SFT(ndim=self.dim, **self.hankel_kw)
        return f_t.transform(self.covariance, k, ret_err=False)

    def spectral_density(self, k):
        return self.spectrum(k)/self.var

    def spectral_rad_pdf(self, r):
        r = np.abs(np.array(r, dtype=float))
        if self.dim > 1:
            r_gz = r[r > 0.]
            # to prevent numerical errors, we just calculate where r>0
            res = np.zeros_like(r, dtype=float)
            res[r > 0.] = rad_fac(self.dim, r_gz)*self.spectral_density(r_gz)
            # prevent numerical errors in hankel for small r values (set 0)
            res[np.logical_not(np.isfinite(res))] = 0.
            # prevent numerical errors in hankel for big r (set non-negative)
            res = np.maximum(res, 0.)
            return res
        # TODO: this is totally hacky
        # prevent num error in hankel at r=0 in 1D
        r[r == 0.] = 0.03/self.len_scale
        res = rad_fac(self.dim, r)*self.spectral_density(r)
        # prevent numerical errors in hankel for big r (set positiv)
        res = np.maximum(res, 0.)
        return res

    def ln_spectral_rad_pdf(self, r):
        spec = np.array(self.spectral_rad_pdf(r))
        res = np.ones_like(spec, dtype=float)
        # inplace multiplication retains array-type for 0-dim arrays
        res *= -np.inf
        res[spec > 0.] = np.log(spec[spec > 0.])
        return res

    def _has_cdf(self):
        """State if a cdf is defined by the user (can be overwritten)"""
        return hasattr(self, "spectral_rad_cdf")

    def _has_ppf(self):
        """State if a ppf is defined by the user (can be overwritten)"""
        return hasattr(self, "spectral_rad_ppf")

    # fitting routine #########################################################

    def fit_variogram(self, x_data, y_data):
        '''fit the variogram-model to given data'''

        def curve(x, len_scale, var, *args):
            '''dummy function for the variogram'''
            self._len_scale = len_scale
            self._var = var
            for opt_i, opt_e in enumerate(self.opt_arg):
                setattr(self, opt_e, args[opt_i])
            return self.variogram(x)

        # set the lower boundaries for the variogram-parameters
        low_bounds = []
        low_bounds.append(self.len_scale_bounds[0])
        low_bounds.append(self.var_bounds[0])
        for opt in self.opt_arg:
            low_bounds.append(self.opt_arg_bounds[opt][0])
        # set the upper boundaries for the variogram-parameters
        top_bounds = []
        top_bounds.append(self.len_scale_bounds[1])
        top_bounds.append(self.var_bounds[1])
        for opt in self.opt_arg:
            top_bounds.append(self.opt_arg_bounds[opt][1])
        # fit the variogram
        popt, pcov = curve_fit(curve, x_data, y_data,
                               bounds=(low_bounds, top_bounds))
        out = {}
        self._len_scale = popt[0]
        out["len_scale"] = popt[0]
        self._var = popt[1]
        out["var"] = popt[1]
        for opt_i, opt_e in enumerate(self.opt_arg):
            setattr(self, opt_e, popt[opt_i+2])
            out[opt_e] = popt[opt_i+2]
        # recalculate the integral scale
        self._integral_scale = self.calc_integral_scale()
        out["integral_scale"] = self._integral_scale
        return out, pcov

    # bounds ##################################################################

    @property
    def len_scale_bounds(self):
        return self._len_scale_bounds

    @len_scale_bounds.setter
    def len_scale_bounds(self, bounds):
        if not check_bounds(bounds):
            raise ValueError("Given bounds for 'len_scale' are not " +
                             "valid, got: "+str(bounds))
        self._len_scale_bounds = bounds

    @property
    def var_bounds(self):
        return self._var_bounds

    @var_bounds.setter
    def var_bounds(self, bounds):
        if not check_bounds(bounds):
            raise ValueError("Given bounds for 'var' are not " +
                             "valid, got: "+str(bounds))
        self._var_bounds = bounds

    @property
    def opt_arg_bounds(self):
        return self._opt_arg_bounds

    @property
    def arg_bounds(self):
        res = {"len_scale": self.len_scale_bounds, "var": self.var_bounds}
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
    def len_scale_vec(self):
        '''The length scales in each direction of the spatial random field.

        Notes
        -----
        * len_scale_x = len_scale
        * len_scale_y = len_scale*anis_y
        * len_scale_z = len_scale*anis_z
        '''
        res = np.zeros(self.dim, dtype=float)
        res[0] = self.len_scale
        for i in range(1, self._dim):
            res[i] = self.len_scale*self.anis[i-1]
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
    def opt_arg(self):
        '''Names of the optional arguments'''
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
            res[i] = self.integral_scale*self.anis[i-1]
        return res

    # plotting routines #######################################################

    def plot_variogram(self, x_min=0.0, x_max=None):
        if x_max is None:
            x_max = 3*self.integral_scale
        x_s = np.linspace(x_min, x_max)
        plt.plot(x_s, self.variogram(x_s),
                 label=self.__class__.__name__+" vario")
        plt.legend()
        plt.show()

    def plot_covariance(self, x_min=0.0, x_max=None):
        if x_max is None:
            x_max = 3*self.integral_scale
        x_s = np.linspace(x_min, x_max)
        plt.plot(x_s, self.covariance(x_s),
                 label=self.__class__.__name__+" cov")
        plt.legend()
        plt.show()

    def plot_covariance_normed(self, x_min=0.0, x_max=None):
        if x_max is None:
            x_max = 3*self.integral_scale
        x_s = np.linspace(x_min, x_max)
        plt.plot(x_s, self.covariance_normed(x_s),
                 label=self.__class__.__name__+" cov normed")
        plt.legend()
        plt.show()

    def plot_variogram_normed(self, x_min=0.0, x_max=None):
        if x_max is None:
            x_max = 3*self.integral_scale
        x_s = np.linspace(x_min, x_max)
        plt.plot(x_s, self.variogram_normed(x_s),
                 label=self.__class__.__name__+" vario normed")
        plt.legend()
        plt.show()

    def plot_spectrum(self, x_min=0.0, x_max=None):
        if x_max is None:
            x_max = 3/self.integral_scale
        x_s = np.linspace(x_min, x_max)
        plt.plot(x_s, self.spectrum(x_s),
                 label=self.__class__.__name__+" " +
                 str(self.dim)+"D spec")
        plt.legend()
        plt.show()

    def plot_spectral_density(self, x_min=0.0, x_max=None):
        if x_max is None:
            x_max = 3/self.integral_scale
        x_s = np.linspace(x_min, x_max)
        plt.plot(x_s, self.spectral_density(x_s),
                 label=self.__class__.__name__+" " +
                 str(self.dim)+"D spec-dens")
        plt.legend()
        plt.show()

    def plot_spectral_rad_pdf(self, x_min=0.0, x_max=None):
        if x_max is None:
            x_max = 3/self.integral_scale
        x_s = np.linspace(x_min, x_max)
        plt.plot(x_s, self.spectral_rad_pdf(x_s),
                 label=self.__class__.__name__+" " +
                 str(self.dim)+"D spec-rad-pdf")
        plt.legend()
        plt.show()


###############################################################################
# Derived Models ##############################################################
###############################################################################

# Gaussian Model ##############################################################

class Gau(CovModel):

    def covariance_normed(self, r):
        r = np.abs(np.array(r, dtype=float))
        return np.exp(-np.pi/4*(r/self.len_scale)**2)

    def spectrum(self, k):
        return (self.var*(self.len_scale/np.pi)**self.dim *
                np.exp(-(k*self.len_scale)**2/np.pi))

    def spectral_rad_cdf(self, r):
        if self.dim == 1:
            return sps.erf(self.len_scale*r/np.sqrt(np.pi))
        elif self.dim == 2:
            return 1. - np.exp(-(r*self.len_scale)**2/np.pi)
        elif self.dim == 3:
            return (sps.erf(self.len_scale*r/np.sqrt(np.pi)) -
                    2*r*self.len_scale/np.pi *
                    np.exp(-(r*self.len_scale)**2/np.pi))
        return None

    def spectral_rad_ppf(self, u):
        if self.dim == 1:
            return sps.erfinv(u)*np.sqrt(np.pi)/self.len_scale
        elif self.dim == 2:
            return (np.sqrt(np.pi)/self.len_scale *
                    np.sqrt(-np.log(1.-u)))
        return None

    def _has_ppf(self):
        """ppf for 3 dimensions is not analytical"""
        # since the ppf is not analytical for dim=3, we have to state that
        if self.dim == 3:
            return False
        return True

    def calc_integral_scale(self):
        '''The integral scale of the gaussian model is the length scale'''
        return self.len_scale


# Exponential Model ###########################################################

class Exp(CovModel):

    def covariance_normed(self, r):
        r = np.abs(np.array(r, dtype=float))
        return np.exp(-r/self.len_scale)

    def spectrum(self, k):
        return (
            self.var*self.len_scale**self.dim*sps.gamma((self.dim+1)/2) /
            (np.pi*(1. + (k*self.len_scale)**2))**((self.dim+1)/2))

    def spectral_rad_cdf(self, r):
        if self.dim == 1:
            return np.arctan(r*self.len_scale)*2/np.pi
        elif self.dim == 2:
            return 1. - 1/np.sqrt(1 + (r*self.len_scale)**2)
        elif self.dim == 3:
            return (np.arctan(r*self.len_scale) -
                    r*self.len_scale/(1 + (r*self.len_scale)**2))*2/np.pi
        return None

    def spectral_rad_ppf(self, u):
        if self.dim == 1:
            return np.tan(np.pi/2*u)/self.len_scale
        elif self.dim == 2:
            return np.sqrt(1/u**2 - 1.)/self.len_scale
        return None

    def _has_ppf(self):
        """ppf for 3 dimensions is not analytical"""
        # since the ppf is not analytical for dim=3, we have to state that
        if self.dim == 3:
            return False
        return True

    def calc_integral_scale(self):
        '''The integral scale of the exponential model is the length scale'''
        return self.len_scale


# Spherical Model #############################################################

class Sph(CovModel):

    def covariance_normed(self, r):
        r = np.atleast_1d(np.abs(np.array(r, dtype=float)))
        res = 1. - 9./16.*r/self.len_scale + 27./1024.*(r/self.len_scale)**3
        res[r > 8./3.*self.len_scale] = 0.
        return res

    def calc_integral_scale(self):
        '''The integral scale of the spherical model is the length scale'''
        return self.len_scale


# Rational Model ##############################################################

class Rat(CovModel):

    def default_opt_arg(self):
        return {"alpha": 1.}

    def default_opt_arg_bounds(self):
        return {"alpha": [0.5, np.inf]}

    def covariance_normed(self, r):
        r = np.abs(np.array(r, dtype=float))
        return np.power(1 + 0.5/self.alpha*(r/self.len_scale)**2, -self.alpha)


# Matérn Model ################################################################

class Mat(CovModel):

    def default_opt_arg(self):
        return {"nu": 1.}

    def default_opt_arg_bounds(self):
        return {"nu": [0.5, 60., "cc"]}

    def check_opt_arg(self):
        if self.nu > 50.:
            warnings.warn("Mat: parameter 'nu' is > 50, " +
                          "calculations most likely get unstable here")

    def covariance_normed(self, r):
        r = np.abs(np.array(r, dtype=float))
        r_gz = r[r > 0.]
        res = np.ones_like(r)
        with np.errstate(over='ignore', invalid='ignore'):
            res[r > 0.] = (
                np.power(2, 1.-self.nu) / sps.gamma(self.nu) *
                np.power(np.pi/sps.beta(self.nu, .5) * r_gz/self.len_scale,
                         self.nu) *
                sps.kv(self.nu,
                       np.pi/sps.beta(self.nu, .5) * r_gz/self.len_scale))
        # if nu >> 1 we get errors for the farfield, there 0 is approached
        res[np.logical_not(np.isfinite(res))] = 0.0
        # covariance is positiv
        res = np.maximum(res, 0.)
        return res
