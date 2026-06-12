"""
GStools subpackage providing transformations to post-process normal fields.

.. currentmodule:: gstools.transform

Wrapper
^^^^^^^

.. autosummary::
   :toctree:

   apply

Field Transformations
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree:

   binary
   discrete
   boxcox
   zinnharvey
   normal_force_moments
   normal_to_lognormal
   normal_to_uniform
   normal_to_arcsin
   normal_to_uquad
   apply_function

Array Transformations
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree:

   array_discrete
   array_boxcox
   array_zinnharvey
   array_force_moments
   array_to_lognormal
   array_to_uniform
   array_to_arcsin
   array_to_uquad

----
"""

from gstools.transform.array import (
    array_boxcox,
    array_discrete,
    array_force_moments,
    array_to_arcsin,
    array_to_lognormal,
    array_to_uniform,
    array_to_uquad,
    array_zinnharvey,
)
from gstools.transform.field import (
    apply,
    apply_function,
    binary,
    boxcox,
    discrete,
    normal_force_moments,
    normal_to_arcsin,
    normal_to_lognormal,
    normal_to_uniform,
    normal_to_uquad,
    zinnharvey,
)

__all__ = [
    "apply",
    "apply_function",
    "binary",
    "discrete",
    "boxcox",
    "zinnharvey",
    "normal_force_moments",
    "normal_to_lognormal",
    "normal_to_uniform",
    "normal_to_arcsin",
    "normal_to_uquad",
    "array_discrete",
    "array_boxcox",
    "array_zinnharvey",
    "array_force_moments",
    "array_to_lognormal",
    "array_to_uniform",
    "array_to_arcsin",
    "array_to_uquad",
]
