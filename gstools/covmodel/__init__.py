# -*- coding: utf-8 -*-
"""
GStools subpackage providing a set of handy covariance models.

.. currentmodule:: gstools.covmodel

Subpackages
^^^^^^^^^^^

.. autosummary::
    plot

Covariance Base-Class
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   CovModel

Predefined Covariance Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   Gaussian
   Exponential
   Matern
   Rational
   Stable
   Spherical
   Linear
   MaternRescal
   SphericalRescal
   TPLGaussian
   TPLExponential
   TPLStable
"""
from __future__ import absolute_import

from gstools.covmodel.base import CovModel
from gstools.covmodel.models import (
    Gaussian,
    Exponential,
    Matern,
    Rational,
    Stable,
    Spherical,
    Linear,
    MaternRescal,
    SphericalRescal,
    TPLGaussian,
    TPLExponential,
    TPLStable,
)

__all__ = [
    "CovModel",
    "Gaussian",
    "Exponential",
    "Matern",
    "Rational",
    "Stable",
    "Spherical",
    "Linear",
    "MaternRescal",
    "SphericalRescal",
    "TPLGaussian",
    "TPLExponential",
    "TPLStable",
]
