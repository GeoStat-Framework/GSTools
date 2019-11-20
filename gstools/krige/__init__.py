# -*- coding: utf-8 -*-
"""
GStools subpackage providing kriging.

.. currentmodule:: gstools.krige

Kriging Classes
^^^^^^^^^^^^^^^

.. autosummary::
   Simple
   Ordinary

----
"""
from gstools.krige.simple import Simple
from gstools.krige.ordinary import Ordinary

__all__ = ["Simple", "Ordinary"]
