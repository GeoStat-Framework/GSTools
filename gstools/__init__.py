# -*- coding: utf-8 -*-
"""
Purpose
=======

GeoStatTools is a library providing geostatistical tools for random field generation and
variogram estimation based on a list of provided or even user-defined covariance models.

The following functionalities are directly provided on module-level.

Subpackages
===========

.. autosummary::
    covmodel
    field
    variogram
    random
    tools

Classes
=======

Spatial Random Field
^^^^^^^^^^^^^^^^^^^^
Class for random field generation

.. currentmodule:: gstools.field

.. autosummary::
   SRF

Covariance Base-Class
^^^^^^^^^^^^^^^^^^^^^
Class to construct user defined covariance models

.. currentmodule:: gstools.covmodel.base

.. autosummary::
   CovModel

Covariance Models
^^^^^^^^^^^^^^^^^

Standard Covariance Models
~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: gstools.covmodel.tpl_models
.. autosummary::
   TPLGaussian
   TPLExponential
   TPLStable

Functions
=========

VTK-Export
^^^^^^^^^^
Routines to export fields to the vtk format

.. currentmodule:: gstools.tools

.. autosummary::
   vtk_export
   vtk_export_structured
   vtk_export_unstructured

variogram estimation
^^^^^^^^^^^^^^^^^^^^
Estimate the variogram of a given field

.. currentmodule:: gstools.variogram

.. autosummary::
   vario_estimate_structured
   vario_estimate_unstructured
"""
from __future__ import absolute_import

from gstools._version import __version__
from gstools import field, variogram, random, covmodel, tools
from gstools.field import SRF
from gstools.tools.export import (
    vtk_export_structured,
    vtk_export_unstructured,
    vtk_export,
)
from gstools.variogram import vario_estimate_structured, vario_estimate_unstructured
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
__all__ = ["__version__"]
__all__ += ["covmodel", "field", "variogram", "random", "tools"]

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

__all__ += ["vario_estimate_structured", "vario_estimate_unstructured"]

__all__ += [
    "SRF",
    "vtk_export_structured",
    "vtk_export_unstructured",
    "vtk_export",
]
