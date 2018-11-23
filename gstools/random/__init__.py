# -*- coding: utf-8 -*-
"""
GStools subpackage providing a random number generator class.

.. currentmodule:: gstools.random

Included functions
------------------
The following functions are provided

.. autosummary::
   RNG
   dist_gen
"""
from __future__ import absolute_import

from gstools.random.rng import RNG
from gstools.random.tools import dist_gen

__all__ = ["RNG", "dist_gen"]
