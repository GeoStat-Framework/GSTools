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

Zonation
^^^^^^^^
Bind a training image to a region of the simulation grid for zonated Direct Sampling.

.. autosummary::
   :toctree:

   Zone

Simulation
^^^^^^^^^^
Direct Sampling simulation following the gstools field interface.

.. autosummary::
   :toctree:

   DirectSampling
"""

from gstools.mps.direct_sampling import DirectSampling
from gstools.mps.model import MPSModel
from gstools.mps.post_processing import PostProcessing
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
