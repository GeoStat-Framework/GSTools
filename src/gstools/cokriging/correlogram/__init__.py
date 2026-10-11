"""
GStools subpackage providing correlogram models for collocated cokriging.

.. currentmodule:: gstools.cokriging.correlogram

Correlogram models define the cross-covariance structure between primary
and secondary variables in collocated cokriging.

Base Class
^^^^^^^^^^

.. autosummary::
   :toctree:

   Correlogram

Markov Models
^^^^^^^^^^^^^

.. autosummary::
   :toctree:

   MarkovModel1

Future Models
^^^^^^^^^^^^^

Planned implementations:
    - MarkovModel2: Uses secondary variable's spatial structure
    - LinearModelCoregionalization: Full multivariate model
"""

from gstools.cokriging.correlogram.base import Correlogram
from gstools.cokriging.correlogram.markov import MarkovModel1

__all__ = ["Correlogram", "MarkovModel1"]
