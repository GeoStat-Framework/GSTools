# -*- coding: utf-8 -*-
"""
GStools subpackage providing tools for spatial random fields.

.. currentmodule:: gstools.field

Subpackages
^^^^^^^^^^^

.. autosummary::
    generator
    upscaling
    base

Spatial Random Field
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   SRF
   Field
   Mesh

----
"""

from gstools.field.mesh import Mesh
from gstools.field.base import Field
from gstools.field.srf import SRF

__all__ = ["SRF", "Field", "Mesh"]
