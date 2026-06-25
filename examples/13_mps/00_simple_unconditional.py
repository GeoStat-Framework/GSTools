r"""
A first Direct Sampling simulation
----------------------------------

This is the minimal Multiple Point Statistics example: build a training image,
wrap it in a :any:`TrainingImage`, and generate one unconditional realization
with :any:`DirectSampling`.

We use a small, synthetic *channelized* training image generated with NumPy, so
the example is fast and needs no downloads.
"""

import matplotlib.pyplot as plt
import numpy as np

import gstools as gs

###############################################################################
# Create a synthetic binary training image with curvilinear "channels".
# The two facies (0 and 1) form connected, meandering bands — exactly the kind
# of structure two-point statistics struggles to reproduce.

gx, gy = np.meshgrid(np.arange(60), np.arange(60), indexing="ij")
ti_data = ((np.sin(gx / 5.0) + np.sin((gx + gy) / 8.0)) > 0).astype(float)

###############################################################################
# Wrap the array in a :any:`TrainingImage`. For a categorical variable (facies
# codes) the distance is the fraction of mismatching neighbours, so the
# ``distance`` argument is ignored here.

ti = gs.TrainingImage(ti_data, categorical=True)
print(ti)

###############################################################################
# Create the :any:`DirectSampling` generator and simulate on a 40x40 grid.
#
# * ``n_neighbors`` — how many already-known cells define each data event.
# * ``scan_fraction`` — fraction of the training image scanned per cell
#   (smaller is faster, slightly noisier).
# * ``threshold=0.0`` — the recommended DSBC mode: always take the best match.

ds = gs.DirectSampling(
    gs.MPSModel(ti, n_neighbors=12, scan_fraction=0.3, threshold=0.0)
)
field = ds([np.arange(40, dtype=float)] * 2, seed=20250616)

###############################################################################
# Plot the training image next to the realization. The realization is not a
# copy of the TI, but it reproduces the same channel patterns.

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5))
ax0.imshow(ti_data, cmap="cividis", origin="lower")
ax0.set_title("Training image")
ax1.imshow(field, cmap="cividis", origin="lower")
ax1.set_title("DS realization")
fig.tight_layout()
