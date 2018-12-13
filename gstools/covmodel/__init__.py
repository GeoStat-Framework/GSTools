# -*- coding: utf-8 -*-
"""
GStools subpackage providing a set of handy covariance models.

Subpackages
^^^^^^^^^^^

.. currentmodule:: gstools.covmodel

.. autosummary::
    base
    models
    tpl_models
    plot

Covariance Base-Class
^^^^^^^^^^^^^^^^^^^^^
Class to construct user defined covariance models

.. currentmodule:: gstools.covmodel.base

.. autosummary::
   CovModel

Covariance Models
^^^^^^^^^^^^^^^^^
Standard Covariance Models

.. currentmodule:: gstools.covmodel.models

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

Truncated Power Law Covariance Models

.. currentmodule:: gstools.covmodel.tpl_models

.. autosummary::
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
)
from gstools.covmodel.tpl_models import TPLGaussian, TPLExponential, TPLStable

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
