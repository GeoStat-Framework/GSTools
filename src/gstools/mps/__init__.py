"""
GStools subpackage for Multiple Point Statistics (MPS).

.. currentmodule:: gstools.mps

Training Image
^^^^^^^^^^^^^^
Training image wrapper holding spatial pattern data and distance metrics.

.. autosummary::
   :toctree:

   TrainingImage
   Variable

MPS Model
^^^^^^^^^
Configuration class bundling a training image with Direct Sampling search parameters.

.. autosummary::
   :toctree:

   MPSModel
   PostProcessing

Simulation
^^^^^^^^^^
Direct Sampling simulation following the gstools field interface.

.. autosummary::
   :toctree:

   DirectSampling

Zonation
^^^^^^^^
Region-bound training images for zonated simulation.

.. autosummary::
   :toctree:

   Zone
"""

from gstools.mps.direct_sampling import DirectSampling
from gstools.mps.model import MPSModel, PostProcessing
from gstools.mps.training_image import TrainingImage, Variable
from gstools.mps.zone import Zone

__all__ = [
    "DirectSampling",
    "MPSModel",
    "PostProcessing",
    "TrainingImage",
    "Variable",
    "Zone",
]
