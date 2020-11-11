# -*- coding: utf-8 -*-
"""
GStools subpackage providing tools for estimating and fitting variograms.

.. currentmodule:: gstools.variogram

Variogram estimation
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   vario_estimate
   vario_estimate_axis

----
"""

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
]
