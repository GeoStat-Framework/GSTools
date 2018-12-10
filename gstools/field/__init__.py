# -*- coding: utf-8 -*-
"""
GStools subpackage providing tools for spatial random fields.

.. currentmodule:: gstools.field

Subpackages
^^^^^^^^^^^
The following subpackages are provided

.. autosummary::
    generator
    upscaling

Classes
^^^^^^^
The following classes are provided

Spatial Random Field
~~~~~~~~~~~~~~~~~~~~
Class for random field generation

.. autosummary::
   SRF
"""
from __future__ import absolute_import

from gstools.field.srf import SRF

__all__ = ["SRF"]
