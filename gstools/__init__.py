# -*- coding: utf-8 -*-
"""
Purpose
=======

GeoStatTools is a library providing geostatistical tools
for random field generation, conditioned field generation,
kriging and variogram estimation
based on a list of provided or even user-defined covariance models.

The following functionalities are directly provided on module-level.

Subpackages
===========

.. autosummary::
    covmodel
    field
    variogram
    krige
    random
    tools
    transform

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
   Linear
   Circular
   Spherical
   Intersection

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
   to_vtk
   to_vtk_structured
   to_vtk_unstructured

variogram estimation
^^^^^^^^^^^^^^^^^^^^
Estimate the variogram of a given field

.. currentmodule:: gstools.variogram

.. autosummary::
   vario_estimate_structured
   vario_estimate_unstructured
"""

from gstools import field, variogram, random, covmodel, tools, krige, transform
from gstools.field import SRF
from gstools.tools import (
    vtk_export,
    vtk_export_structured,
    vtk_export_unstructured,
    to_vtk,
    to_vtk_structured,
    to_vtk_unstructured,
)
from gstools.variogram import (
    vario_estimate_structured,
    vario_estimate_unstructured,
)
from gstools.covmodel import (
    CovModel,
    Gaussian,
    Exponential,
    Matern,
    Rational,
    Stable,
    Linear,
    Circular,
    Spherical,
    Intersection,
    TPLGaussian,
    TPLExponential,
    TPLStable,
)

try:
    from gstools._version import __version__
except ModuleNotFoundError:  # pragma: nocover
    # package is not installed
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
__all__ += ["covmodel", "field", "variogram", "krige", "random", "tools"]
__all__ += ["transform"]
__all__ += [
    "CovModel",
    "Gaussian",
    "Exponential",
    "Matern",
    "Rational",
    "Stable",
    "Linear",
    "Circular",
    "Spherical",
    "Intersection",
    "TPLGaussian",
    "TPLExponential",
    "TPLStable",
]

__all__ += ["vario_estimate_structured", "vario_estimate_unstructured"]

__all__ += [
    "SRF",
    "vtk_export",
    "vtk_export_structured",
    "vtk_export_unstructured",
    "to_vtk",
    "to_vtk_structured",
    "to_vtk_unstructured",
]
