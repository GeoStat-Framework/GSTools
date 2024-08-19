"""
GStools subpackage providing a set of handy covariance models.

.. currentmodule:: gstools.covmodel

Subpackages
^^^^^^^^^^^

.. autosummary::
   :toctree:

   plot

Covariance Base-Class
^^^^^^^^^^^^^^^^^^^^^
Class to construct user defined covariance models

.. autosummary::
   :toctree:

   CovModel
   SumModel

Covariance Models
^^^^^^^^^^^^^^^^^
Standard Covariance Models

.. autosummary::
   :toctree:

   Nugget
   Gaussian
   Exponential
   Matern
   Integral
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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree:

   TPLGaussian
   TPLExponential
   TPLStable
   TPLSimple
"""

from gstools.covmodel.base import CovModel, SumModel
from gstools.covmodel.models import (
    Circular,
    Cubic,
    Exponential,
    Gaussian,
    HyperSpherical,
    Integral,
    JBessel,
    Linear,
    Matern,
    Nugget,
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
    "SumModel",
    "Nugget",
    "Gaussian",
    "Exponential",
    "Matern",
    "Integral",
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
