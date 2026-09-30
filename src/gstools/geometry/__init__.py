"""
GStools subpackage providing surface-based stochastic geometry.

.. currentmodule:: gstools.geometry

Regions
^^^^^^^

.. autosummary::
   :toctree:

   Region
   Composite
   HalfSpace
   StratColumn

Layered Geometry
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree:

   Layer
   LayerStack
   Surface
   SurfaceStack

Thickness Links
^^^^^^^^^^^^^^^

.. autosummary::
   :toctree:

   ThicknessLink
   LogLink
   set_link

Facies Field
^^^^^^^^^^^^

.. autosummary::
   :toctree:

   FaciesField

Borehole Data
^^^^^^^^^^^^^

.. autosummary::
   :toctree:

   BoreholeData
   from_well_logs
   from_surface_points
   gibbs_sample

Tools
^^^^^

.. autosummary::
   :toctree:

   interfaces_to_pyvista
   topology
"""

from gstools.geometry.base import (
    NO_LABEL,
    RELATION,
    Composite,
    HalfSpace,
    Region,
    StratColumn,
)
from gstools.geometry.field import FaciesField
from gstools.geometry.gibbs import gibbs_sample
from gstools.geometry.layers import Layer, LayerStack
from gstools.geometry.observations import (
    KIND,
    BoreholeData,
    from_surface_points,
    from_well_logs,
)
from gstools.geometry.surfaces import Surface, SurfaceStack
from gstools.geometry.thickness import LINK, LogLink, ThicknessLink, set_link
from gstools.geometry.tools import interfaces_to_pyvista, topology

__all__ = [
    "NO_LABEL",
    "RELATION",
    "Region",
    "Composite",
    "HalfSpace",
    "StratColumn",
    "ThicknessLink",
    "LogLink",
    "LINK",
    "set_link",
    "Layer",
    "LayerStack",
    "Surface",
    "SurfaceStack",
    "FaciesField",
    "KIND",
    "BoreholeData",
    "from_well_logs",
    "from_surface_points",
    "gibbs_sample",
    "interfaces_to_pyvista",
    "topology",
]
