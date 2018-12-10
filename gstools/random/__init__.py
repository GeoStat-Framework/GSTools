# -*- coding: utf-8 -*-
"""
GStools subpackage for random number generation.

.. currentmodule:: gstools.random

Random Number Generator
^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   RNG

Distribution factory
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   dist_gen

"""
from __future__ import absolute_import

from gstools.random.rng import RNG
from gstools.random.tools import dist_gen

__all__ = ["RNG", "dist_gen"]
