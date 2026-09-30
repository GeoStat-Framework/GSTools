Stochastic Geometry
===================

Many subsurface models are built from stratigraphic units, whose
boundaries are the main source of uncertainty.
The :any:`gstools.geometry` subpackage models these boundaries as
*height functions*: every interface is a spatial random field on the
:math:`(d-1)`-dimensional lateral domain, e.g. :math:`z = f_i(x, y)`.
A point is then labelled by comparing its elevation with the interfaces
above and below it.

Two parameterisations are provided:

- :any:`LayerStack` stacks positive layer thicknesses
  :math:`t_i = \exp(g_i)` on top of a base, with :math:`g_i` being
  Gaussian log-thickness fields. Thus, interfaces can never cross
  and borehole thicknesses condition the fields exactly by kriging on the
  lateral slice.
- :any:`SurfaceStack` models the surface elevations themselves and orders
  them by a cumulative maximum (onlap) or minimum (erode). Contact
  elevations are then exact equality data on a single surface.

Data which are only known to lie within bounds, e.g. a borehole ending
inside a unit or a unit that is absent, are imputed by a Gibbs sampler
of the truncated Gaussian distribution before the exact conditioning.
Finally, several groups of units can be combined by erosive or onlapping
relations in a :any:`StratColumn` and cut by a topography.

.. currentmodule:: gstools.geometry

The following classes and functions are provided:

.. autosummary::
   LayerStack
   Layer
   LogLink
   SurfaceStack
   Surface
   StratColumn
   HalfSpace
   Composite
   FaciesField
   BoreholeData
   from_well_logs
   from_surface_points

Examples
--------
