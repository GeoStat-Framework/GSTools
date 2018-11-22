# -*- coding: utf-8 -*-
"""
=======
GSTools
=======

Contents
--------
GeoStatTools is a library providing geostatistical tools.

Subpackages
-----------
The following subpackages are provided

.. autosummary::
    field
    variogram
"""
from __future__ import absolute_import

from gstools import field, variogram, random, covmodel

__all__ = ["field", "variogram", "random", "covmodel"]

__version__ = "0.4.0"
