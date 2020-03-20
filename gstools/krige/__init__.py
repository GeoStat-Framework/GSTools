# -*- coding: utf-8 -*-
"""
GStools subpackage providing kriging.

.. currentmodule:: gstools.krige

Kriging Classes
^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated

   Simple
   Ordinary
   Universal
   ExtDrift
   Detrended

----
"""
from gstools.krige.methods import (
    Simple,
    Ordinary,
    Universal,
    ExtDrift,
    Detrended,
)

__all__ = ["Simple", "Ordinary", "Universal", "ExtDrift", "Detrended"]
