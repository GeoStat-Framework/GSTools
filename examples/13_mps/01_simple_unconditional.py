r"""
A first Direct Sampling (DS) simulation
---------------------------------------

This is the minimal Multiple Point Statistics example: build a training image,
wrap it in a :any:`TrainingImage`, and generate one unconditional realization
with :any:`DirectSampling`.

This first step deliberately uses **no conditioning data**. The goal is to see
what DS does with only a training image: it should create a new realization
that is not a copy of the TI, but still has similar spatial patterns.
At the beginning there are no known neighbors, so DS starts from TI values.
As the random simulation path advances, previously simulated cells become the
known neighbors used as the next data event.

We use a small, synthetic training image generated with NumPy, so the example
is fast and needs no downloads. Its alternating, block-like bands make it easy
to see how patterns in the TI are carried into a new realization.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import gstools as gs

###############################################################################
# Create a synthetic binary TI with alternating, block-like bands. These are
# deliberately simple patterns for this first example, not a realistic channel
# network.

gx, gy = np.meshgrid(np.arange(60), np.arange(60), indexing="ij")
ti_data = ((np.sin(gx / 5.0) + np.sin((gx + gy) / 8.0)) > 0).astype(int)

###############################################################################
# Wrap the array in a :any:`TrainingImage`. ``categorical=True`` tells GSTools
# that the values are facies labels, not continuous numbers. When DS compares
# two neighborhoods, it counts how many facies labels match or mismatch.

ti = gs.TrainingImage(ti_data, categorical=True, n_neighbors=12)
print(ti)

###############################################################################
# Create the :any:`DirectSampling` generator and simulate on a 40x40 grid.
# No measured values are supplied here; this is an unconditional simulation.
# The data events are built from values simulated earlier in the random path.
#
# * ``n_neighbors`` — how many already-known cells define each data event
#   (set on the :any:`TrainingImage`).
# * ``scan_fraction`` — fraction of the TI search window scanned per cell
#   (smaller is faster, slightly noisier).
# * ``threshold=0.0`` — the recommended DSBC mode: always take the best match.

ds = gs.DirectSampling(gs.MPSModel(ti, scan_fraction=0.3, threshold=0.0))
grid = [np.arange(40, dtype=float), np.arange(40, dtype=float)]
field = ds(grid, seed=20250616)

###############################################################################
# Plot the training image next to the realization. The result is not a copy of
# the TI, but it reproduces some of its spatial patterns. Increasing
# ``scan_fraction`` can improve the matches by searching more of the TI; this
# first example keeps it low to stay quick and focus on the workflow.

cmap = ListedColormap(["#d8c690", "#2f6f9f"])
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5))
ax0.imshow(ti_data, cmap=cmap, origin="lower", vmin=0, vmax=1)
ax0.set_title("Training image")
ax1.imshow(field, cmap=cmap, origin="lower", vmin=0, vmax=1)
ax1.set_title("DS realization")
fig.tight_layout()
