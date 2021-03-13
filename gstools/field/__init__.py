# -*- coding: utf-8 -*-
"""
GStools subpackage providing tools for spatial random fields.

.. currentmodule:: gstools.field

Subpackages
^^^^^^^^^^^

.. autosummary::
    generator
    upscaling

Spatial Random Field
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated

   SRF
   CondSRF

Field Base Class
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated

   Field
"""

from gstools.field.base import Field
from gstools.field.srf import SRF
from gstools.field.cond_srf import CondSRF

__all__ = ["SRF", "CondSRF", "Field"]
