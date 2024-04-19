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
   :toctree: api

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
   KM_SCALE
   DEGREE_SCALE
   RADIAN_SCALE
"""

# Hooray!
from gstools import (
    config,
    covmodel,
    field,
    krige,
    normalizer,
    random,
    tools,
    transform,
    variogram,
)
from gstools.covmodel import (
    Circular,
    CovModel,
    Cubic,
    Exponential,
    Gaussian,
    HyperSpherical,
    Integral,
    JBessel,
    Linear,
    Matern,
    Rational,
    Spherical,
    Stable,
    SuperSpherical,
    TPLExponential,
    TPLGaussian,
    TPLSimple,
    TPLStable,
)
from gstools.field import SRF, CondSRF
from gstools.krige import Krige
from gstools.tools import (
    DEGREE_SCALE,
    EARTH_RADIUS,
    KM_SCALE,
    RADIAN_SCALE,
    generate_grid,
    generate_st_grid,
    rotated_main_axes,
    to_vtk,
    to_vtk_structured,
    to_vtk_unstructured,
    vtk_export,
    vtk_export_structured,
    vtk_export_unstructured,
)
from gstools.variogram import (
    standard_bins,
    vario_estimate,
    vario_estimate_axis,
    vario_estimate_structured,
    vario_estimate_unstructured,
)

try:
    from gstools._version import __version__
except ModuleNotFoundError:  # pragma: no cover
    # package is not installed
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
__all__ += ["covmodel", "field", "variogram", "krige", "random", "tools"]
__all__ += ["transform", "normalizer", "config"]
__all__ += [
    "CovModel",
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
    "KM_SCALE",
    "DEGREE_SCALE",
    "RADIAN_SCALE",
    "vtk_export",
    "vtk_export_structured",
    "vtk_export_unstructured",
    "to_vtk",
    "to_vtk_structured",
    "to_vtk_unstructured",
]
