"""
GStools subpackage for Multiple Point Statistics (MPS).

.. currentmodule:: gstools.mps

Multiple Point Statistics
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree:

   DirectSampling
   MPSModel
   TrainingImage
"""

from gstools.mps.direct_sampling import DirectSampling
from gstools.mps.model import MPSModel
from gstools.mps.training_image import TrainingImage

__all__ = ["DirectSampling", "MPSModel", "TrainingImage"]
