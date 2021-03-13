# -*- coding: utf-8 -*-
"""
GStools subpackage providing kriging.

.. currentmodule:: gstools.krige

Kriging Classes
^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated

   Krige
   Simple
   Ordinary
   Universal
   ExtDrift
   Detrended
"""
from gstools.krige.base import Krige
from gstools.krige.methods import (
    Simple,
    Ordinary,
    Universal,
    ExtDrift,
    Detrended,
)

__all__ = ["Krige", "Simple", "Ordinary", "Universal", "ExtDrift", "Detrended"]
