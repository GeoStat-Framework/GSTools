# -*- coding: utf-8 -*-
"""
GStools subpackage providing tools for the covariance-model.

.. currentmodule:: gstools.covmodel.tools

The following classes and functions are provided

.. autosummary::
   InitSubclassMeta
   rad_fac
   set_len_anis
   check_bounds
   inc_gamma
   exp_int
   inc_beta
"""
from __future__ import print_function, division, absolute_import

try:
    # only added to Python in version 3.5 and numpy version is buggy
    from math import isclose
except ImportError:
    import cmath

    def isclose(a, b, rel_tol=1e-9, abs_tol=0.0):
        if rel_tol < 0.0 or abs_tol < 0.0:
            raise ValueError("error tolerances must be non-negative")

        if a == b:
            return True
        if cmath.isinf(a) or cmath.isinf(b):
            return False
        diff = abs(b - a)
        return ((diff <= abs(rel_tol * b)) or (diff <= abs(rel_tol * a))) or (
            diff <= abs_tol
        )


import numpy as np
from scipy import special as sps

__all__ = [
    "InitSubclassMeta",
    "rad_fac",
    "set_len_anis",
    "check_bounds",
    "inc_gamma",
    "exp_int",
    "inc_beta",
]


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
            return super(InitSubclassMeta, cls).__new__(
                cls, name, bases, ns, **kwargs
            )

        def __init__(cls, name, bases, ns, **kwargs):
            super(InitSubclassMeta, cls).__init__(name, bases, ns)
            super_class = super(cls, cls)
            if hasattr(super_class, "__init_subclass__"):
                super_class.__init_subclass__.__func__(cls, **kwargs)


# Helping functions ###########################################################


def rad_fac(dim, r):
    """The volume element of the n-dimensional spherical coordinates.

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
        fac = 4 * np.pi * r ** 2
    else:  # general solution ( for the record :D )
        fac = (
            dim
            * r ** (dim - 1)
            * np.sqrt(np.pi) ** dim
            / sps.gamma(dim / 2.0 + 1.0)
        )
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
        out_anis = np.atleast_1d(anis)[: dim - 1]
        if len(out_anis) < dim - 1:
            # fill up the anisotropies with ones, such that len()==dim-1
            out_anis = np.pad(
                out_anis,
                (0, dim - len(out_anis) - 1),
                "constant",
                constant_values=1.0,
            )
    elif dim == 1:
        # there is no anisotropy in 1 dimension
        out_anis = np.empty(0)
    else:
        # fill up length-scales with main len_scale, such that len()==dim
        if len(ls_tmp) < dim:
            ls_tmp = np.pad(
                ls_tmp,
                (0, dim - len(ls_tmp)),
                "constant",
                constant_values=out_len_scale,
            )
        # if multiple length-scales are given, calculate the anisotropies
        out_anis = np.zeros(dim - 1, dtype=float)
        for i in range(1, dim):
            out_anis[i - 1] = ls_tmp[i] / ls_tmp[0]

    for ani in out_anis:
        if not ani > 0.0:
            raise ValueError(
                "anisotropy-ratios needs to be > 0, " + "got: " + str(out_anis)
            )
    return out_len_scale, out_anis


def set_angles(dim, angles):
    """Setting the angles for the given dimension.

    Parameters
    ----------
    dim : :class:`int`
        spatial dimension (anything different from 1 and 2 is interpreted as 3)
    angles : :class:`float`/list
        the angles of the SRF

    Returns
    -------
    angles : :class:`float`
        the angles fitting to the dimension
    """
    if dim == 1:
        # no rotation in 1D
        out_angles = np.empty(0)
    elif dim == 2:
        # one rotation axis in 2D
        out_angles = np.atleast_1d(angles)[:1]
    else:
        # three rotation axis in 3D
        out_angles = np.atleast_1d(angles)[:3]
        # fill up the rotation angle array with zeros
        out_angles = np.pad(
            out_angles,
            (0, 3 - len(out_angles)),
            "constant",
            constant_values=0.0,
        )
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
                * "cc" : close - close
    """
    if len(bounds) not in (2, 3):
        return False
    if bounds[1] <= bounds[0]:
        return False
    if len(bounds) == 3 and bounds[2] not in ("oo", "oc", "co", "cc"):
        return False
    return True


# special functions ###########################################################


def inc_gamma(s, x):
    r"""The (upper) incomplete gamma function

    Given by: :math:`\Gamma(s,x) = \int_x^{\infty} t^{s-1}\,e^{-t}\,{\rm d}t`

    Parameters
    ----------
    s : :class:`float`
        exponent in the integral
    x : :class:`numpy.ndarray`
        input values
    """
    if isclose(s, 0):
        return sps.exp1(x)
    if isclose(s, np.around(s)) and s < 0:
        return x ** (s - 1) * sps.expn(int(1 - np.around(s)), x)
    if s < 0:
        return (inc_gamma(s + 1, x) - x ** s * np.exp(-x)) / s
    return sps.gamma(s) * sps.gammaincc(s, x)


def exp_int(s, x):
    r"""The exponential integral :math:`E_s(x)`

    Given by: :math:`E_s(x) = \int_1^\infty \frac{e^{-xt}}{t^s}\,\mathrm dt`

    Parameters
    ----------
    s : :class:`float`
        exponent in the integral
    x : :class:`numpy.ndarray`
        input values
    """
    #    print("s, x", s, x)
    if isclose(s, 1):
        return sps.exp1(x)
    if isclose(s, np.around(s)) and s > -1:
        return sps.expn(int(np.around(s)), x)
    return inc_gamma(1 - s, x) * x ** (s - 1)


def inc_beta(a, b, x):
    r"""The incomplete Beta function

    Given by: :math:`B(a,b;\,x) = \int_0^x t^{a-1}\,(1-t)^{b-1}\,dt`

    Parameters
    ----------
    a : :class:`float`
        first exponent in the integral
    b : :class:`float`
        second exponent in the integral
    x : :class:`numpy.ndarray`
        input values
    """
    return sps.betainc(a, b, x) * sps.beta(a, b)
