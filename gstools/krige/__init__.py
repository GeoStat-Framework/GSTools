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

----
"""
from gstools.krige.methods import Simple, Ordinary, Universal, ExtDrift

__all__ = ["Simple", "Ordinary", "Universal", "ExtDrift"]
