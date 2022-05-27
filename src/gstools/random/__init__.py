# -*- coding: utf-8 -*-
"""
GStools subpackage for random number generation.

.. currentmodule:: gstools.random

Random Number Generator
^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated

   RNG

Seed Generator
^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated

    MasterRNG

Distribution factory
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated

   dist_gen

----
"""

from gstools.random.rng import RNG
from gstools.random.tools import MasterRNG, dist_gen

__all__ = ["RNG", "MasterRNG", "dist_gen"]
