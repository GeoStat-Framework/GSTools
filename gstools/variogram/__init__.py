# -*- coding: utf-8 -*-
"""
GStools subpackage providing tools for estimating and fitting variograms.

.. currentmodule:: gstools.variogram

Variogram estimation
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated

   vario_estimate
   vario_estimate_axis

Binning
^^^^^^^

.. autosummary::
   :toctree: generated

   standard_bins

----
"""

from gstools.variogram.variogram import (
    vario_estimate,
    vario_estimate_axis,
    vario_estimate_structured,
    vario_estimate_unstructured,
)
from gstools.variogram.binning import standard_bins

__all__ = [
    "vario_estimate",
    "vario_estimate_axis",
    "vario_estimate_unstructured",
    "vario_estimate_structured",
    "standard_bins",
]
