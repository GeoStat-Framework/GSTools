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
Class to construct user defined covariance models

.. autosummary::
   :toctree: generated

   CovModel

Covariance Models
^^^^^^^^^^^^^^^^^
Standard Covariance Models

.. autosummary::
   :toctree: generated

   Gaussian
   Exponential
   Matern
   Rational
   Stable
   Linear
   Circular
   Spherical
   HyperSpherical
   SuperSpherical

Truncated Power Law Covariance Models

.. autosummary::
   :toctree: generated

   TPLGaussian
   TPLExponential
   TPLStable
"""

from gstools.covmodel.base import CovModel
from gstools.covmodel.models import (
    Gaussian,
    Exponential,
    Matern,
    Rational,
    Stable,
    Linear,
    Circular,
    Spherical,
    HyperSpherical,
    SuperSpherical,
)
from gstools.covmodel.tpl_models import TPLGaussian, TPLExponential, TPLStable

__all__ = [
    "CovModel",
    "Gaussian",
    "Exponential",
    "Matern",
    "Rational",
    "Stable",
    "Linear",
    "Circular",
    "Spherical",
    "HyperSpherical",
    "SuperSpherical",
    "TPLGaussian",
    "TPLExponential",
    "TPLStable",
]
