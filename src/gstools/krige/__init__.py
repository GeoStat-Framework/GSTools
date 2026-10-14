"""
GStools subpackage providing kriging.

.. currentmodule:: gstools.krige

Kriging Classes
^^^^^^^^^^^^^^^

.. autosummary::
   :toctree:

   Krige
   Simple
   Ordinary
   Universal
   ExtDrift
   Detrended
"""

from gstools.krige.base import Krige
from gstools.krige.local import (
    LocalDetrended,
    LocalExtDrift,
    LocalKrige,
    LocalOrdinary,
    LocalSimple,
    LocalUniversal,
)
from gstools.krige.methods import (
    Detrended,
    ExtDrift,
    Ordinary,
    Simple,
    Universal,
)

__all__ = ["Krige", "LocalKrige", "Simple","LocalSimple", "Ordinary", "LocalOrdinary", "Universal", "LocalUniversal", "ExtDrift", "LocalExtDrift", "Detrended", "LocalDetrended"]
