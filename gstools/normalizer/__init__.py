# -*- coding: utf-8 -*-
"""
GStools subpackage providing normalization routines.

.. currentmodule:: gstools.normalizer

Base-Normalizer
^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated

   Normalizer

Field-Normalizer
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated

   LogNormal
   BoxCox
   BoxCoxShift
   YeoJohnson
   Modulus
   Manly

Convenience Routines
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated

   apply_mean_norm_trend
   remove_trend_norm_mean
"""

from gstools.normalizer.base import Normalizer
from gstools.normalizer.methods import (
    LogNormal,
    BoxCox,
    BoxCoxShift,
    YeoJohnson,
    Modulus,
    Manly,
)
from gstools.normalizer.tools import (
    apply_mean_norm_trend,
    remove_trend_norm_mean,
)

__all__ = [
    "Normalizer",
    "LogNormal",
    "BoxCox",
    "BoxCoxShift",
    "YeoJohnson",
    "Modulus",
    "Manly",
    "apply_mean_norm_trend",
    "remove_trend_norm_mean",
]
