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

Spatial Random Field
~~~~~~~~~~~~~~~~~~~~
Class for random field generation

.. currentmodule:: gstools.field

.. autosummary::
   SRF

Covariance Base-Class
~~~~~~~~~~~~~~~~~~~~~
Class to construct user defined covariance models

.. currentmodule:: gstools.covmodel

.. autosummary::
   CovModel

Covariance Models
~~~~~~~~~~~~~~~~~
Predefined covariance models

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

Functions
^^^^^^^^^
The following functions are provided directly

VTK-Export
~~~~~~~~~~
Routines to export fields to the vtk format

.. currentmodule:: gstools.tools

.. autosummary::
   vtk_export
   vtk_export_structured
   vtk_export_unstructured

variogram estimation
~~~~~~~~~~~~~~~~~~~~
Estimate the variogram of a given field

.. currentmodule:: gstools.variogram

.. autosummary::
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
