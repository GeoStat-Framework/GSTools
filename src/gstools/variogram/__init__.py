"""
GStools subpackage providing tools for estimating and fitting variograms.

.. currentmodule:: gstools.variogram

Variogram estimation
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree:

   vario_estimate
   vario_estimate_axis

Binning
^^^^^^^

.. autosummary::
   :toctree:

   standard_bins

----
"""

from gstools.variogram.binning import standard_bins
from gstools.variogram.variogram import (
    vario_estimate,
    vario_estimate_axis,
    vario_estimate_structured,
    vario_estimate_unstructured,
)

__all__ = [
    "vario_estimate",
    "vario_estimate_axis",
    "vario_estimate_unstructured",
    "vario_estimate_structured",
    "standard_bins",
]
