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
    normalizer

Classes
=======

Kriging
^^^^^^^
Swiss-Army-Knife for Kriging. For short cut classes see: :any:`gstools.krige`

.. currentmodule:: gstools.krige

.. autosummary::
   Krige

Spatial Random Field
^^^^^^^^^^^^^^^^^^^^
Classes for (conditioned) random field generation

.. currentmodule:: gstools.field

.. autosummary::
   SRF
   CondSRF

Covariance Base-Class
^^^^^^^^^^^^^^^^^^^^^
Class to construct user defined covariance models

.. currentmodule:: gstools.covmodel

.. autosummary::
   CovModel

Covariance Models
^^^^^^^^^^^^^^^^^

Standard Covariance Models
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   Gaussian
   Exponential
   Matern
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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   TPLGaussian
   TPLExponential
   TPLStable
   TPLSimple

Functions
=========

VTK-Export
^^^^^^^^^^
Routines to export fields to the vtk format

.. currentmodule:: gstools.tools

.. autosummary::
   vtk_export
   to_vtk

Geometric
^^^^^^^^^
Some convenient functions for geometric operations

.. autosummary::
   rotated_main_axes
   generate_grid
   generate_st_grid

Variogram Estimation
^^^^^^^^^^^^^^^^^^^^
Estimate the variogram of a given field with these routines

.. currentmodule:: gstools.variogram

.. autosummary::
   vario_estimate
   vario_estimate_axis
   standard_bins

Misc
====

.. currentmodule:: gstools.tools

.. autosummary::
   EARTH_RADIUS

"""
# Hooray!
from gstools import (
    field,
    variogram,
    random,
    covmodel,
    tools,
    krige,
    transform,
    normalizer,
)
from gstools.krige import Krige
from gstools.field import SRF, CondSRF
from gstools.tools import (
    rotated_main_axes,
    generate_grid,
    generate_st_grid,
    EARTH_RADIUS,
    vtk_export,
    vtk_export_structured,
    vtk_export_unstructured,
    to_vtk,
    to_vtk_structured,
    to_vtk_unstructured,
)
from gstools.variogram import (
    vario_estimate,
    vario_estimate_axis,
    vario_estimate_structured,
    vario_estimate_unstructured,
    standard_bins,
)
from gstools.covmodel import (
    CovModel,
    Gaussian,
    Exponential,
    Matern,
    Stable,
    Rational,
    Cubic,
    Linear,
    Circular,
    Spherical,
    HyperSpherical,
    SuperSpherical,
    JBessel,
    TPLGaussian,
    TPLExponential,
    TPLStable,
    TPLSimple,
)

try:
    from gstools._version import __version__
except ModuleNotFoundError:  # pragma: nocover
    # package is not installed
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
__all__ += ["covmodel", "field", "variogram", "krige", "random", "tools"]
__all__ += ["transform", "normalizer"]
__all__ += [
    "CovModel",
    "Gaussian",
    "Exponential",
    "Matern",
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

__all__ += [
    "vario_estimate",
    "vario_estimate_axis",
    "vario_estimate_structured",
    "vario_estimate_unstructured",
    "standard_bins",
]

__all__ += [
    "Krige",
    "SRF",
    "CondSRF",
    "rotated_main_axes",
    "generate_grid",
    "generate_st_grid",
    "EARTH_RADIUS",
    "vtk_export",
    "vtk_export_structured",
    "vtk_export_unstructured",
    "to_vtk",
    "to_vtk_structured",
    "to_vtk_unstructured",
]
