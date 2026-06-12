"""
GStools subpackage for Multiple Point Statistics (MPS).

.. currentmodule:: gstools.mps

Multiple Point Statistics
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree:

   DirectSampling
   TrainingImage
"""

from gstools.mps.direct_sampling import DirectSampling
from gstools.mps.training_image import TrainingImage

__all__ = ["DirectSampling", "TrainingImage"]
