# -*- coding: utf-8 -*-
"""
GStools subpackage for random number generation.

.. currentmodule:: gstools.random

Random Number Generator
^^^^^^^^^^^^^^^^^^^^^^^
Class for random number generation controlled by a seed

.. autosummary::
   RNG

Distribution factory
^^^^^^^^^^^^^^^^^^^^
Routine to generate a :any:`scipy.stats.rv_continuous` distribution given
by pdf, cdf, and/or ppf.

.. autosummary::
   dist_gen
"""
from __future__ import absolute_import

from gstools.random.rng import RNG
from gstools.random.tools import dist_gen

__all__ = ["RNG", "dist_gen"]
