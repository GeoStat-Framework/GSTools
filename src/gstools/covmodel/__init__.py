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
   ExpInt
   Mueller
   Stable
   Rational
   Cubic
   Linear
   Circular
   Spherical
   HyperSpherical
   SuperSpherical
   JBessel

Truncated Power Law Covariance Models

.. autosummary::
   :toctree: generated

   TPLGaussian
   TPLExponential
   TPLStable
   TPLSimple
"""

from gstools.covmodel.base import CovModel
from gstools.covmodel.models import (
    Circular,
    Cubic,
    ExpInt,
    Exponential,
    Gaussian,
    HyperSpherical,
    JBessel,
    Linear,
    Matern,
    Mueller,
    Rational,
    Spherical,
    Stable,
    SuperSpherical,
)
from gstools.covmodel.tpl_models import (
    TPLExponential,
    TPLGaussian,
    TPLSimple,
    TPLStable,
)

__all__ = [
    "CovModel",
    "Gaussian",
    "Exponential",
    "Matern",
    "ExpInt",
    "Mueller",
    "Stable",
    "Rational",
    "Cubic",
    "Linear",
    "Circular",
    "Spherical",
    "HyperSpherical",
    "SuperSpherical",
    "JBessel",
    "TPLGaussian",
    "TPLExponential",
    "TPLStable",
    "TPLSimple",
]
