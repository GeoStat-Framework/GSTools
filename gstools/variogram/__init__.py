# -*- coding: utf-8 -*-
"""
GStools subpackage providing tools for estimating and fitting variograms.

.. currentmodule:: gstools.variogram

Variogram estimation
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   estimate_unstructured
   estimate_structured
"""
from __future__ import absolute_import

from gstools.variogram.variogram import (
    estimate_structured,
    estimate_unstructured,
)

__all__ = ["estimate_unstructured", "estimate_structured"]
