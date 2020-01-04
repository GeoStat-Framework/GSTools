# -*- coding: utf-8 -*-
"""
GStools subpackage providing kriging.

.. currentmodule:: gstools.krige

Kriging Classes
^^^^^^^^^^^^^^^

.. autosummary::
   Simple
   Ordinary
   Universal
   ExtDrift
   Detrended
   DetrendedOrdinary

----
"""
from gstools.krige.methods import (
    Simple,
    Ordinary,
    Universal,
    ExtDrift,
    Detrended,
    DetrendedOrdinary,
)

__all__ = [
    "Simple",
    "Ordinary",
    "Universal",
    "ExtDrift",
    "Detrended",
    "DetrendedOrdinary",
]
