# -*- coding: utf-8 -*-
"""
Package Content
---------------
GSTools is a library providing geostatistical tools.
The following functionalities are directly provided on module-level.

Subpackages
^^^^^^^^^^^
The following subpackages are provided

.. autosummary::
    covmodel
    field
    variogram
    random
    tools

Classes
^^^^^^^
The following classes are provided directly

.. autosummary::
   SRF
   CovModel
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

Functions
^^^^^^^^^
The following functions are provided directly

.. autosummary::
   vtk_export
   vtk_export_structured
   vtk_export_unstructured
   estimate_structured
   estimate_unstructured
"""
from __future__ import absolute_import

from gstools import field, variogram, random, covmodel, tools
from gstools.field import SRF
from gstools.tools.export import (
    vtk_export_structured,
    vtk_export_unstructured,
    vtk_export,
)
from gstools.variogram import estimate_structured, estimate_unstructured
from gstools.covmodel import (
    CovModel,
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

__all__ = ["covmodel", "field", "variogram", "random", "tools"]

__all__ += [
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

__all__ += ["estimate_structured", "estimate_unstructured"]

__all__ += [
    "SRF",
    "vtk_export_structured",
    "vtk_export_unstructured",
    "vtk_export",
]

__version__ = "1.0rc6"
