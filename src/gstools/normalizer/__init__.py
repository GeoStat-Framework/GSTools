"""
GStools subpackage providing normalization routines.

.. currentmodule:: gstools.normalizer

Base-Normalizer
^^^^^^^^^^^^^^^

.. autosummary::
   :toctree:

   Normalizer

Field-Normalizer
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree:

   LogNormal
   BoxCox
   BoxCoxShift
   YeoJohnson
   Modulus
   Manly

Convenience Routines
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree:

   apply_mean_norm_trend
   remove_trend_norm_mean
"""

from gstools.normalizer.base import Normalizer
from gstools.normalizer.methods import (
    BoxCox,
    BoxCoxShift,
    LogNormal,
    Manly,
    Modulus,
    YeoJohnson,
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
