# -*- coding: utf-8 -*-
"""
GStools subpackage providing tools for spatial random fields.

.. currentmodule:: gstools.field

Classes
-------
The following classes are provided

.. autosummary::
   SRF
   RandMeth
"""
from __future__ import absolute_import

from gstools.field.srf import SRF
from gstools.field.generator import RandMeth

__all__ = ["SRF", "RandMeth"]
