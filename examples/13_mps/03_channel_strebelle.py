r"""
A real training image: the Strebelle channels
----------------------------------------------

The previous examples used tiny synthetic training images. Here we use the
classic **Strebelle (2002) channelized fluvial training image**, the de-facto
benchmark for MPS, and condition the simulation on random hard data.

.. note::

    **Data source / license.** The training image is downloaded from the
    `GeoDataSets <https://github.com/GeostatsGuy/GeoDataSets>`_ repository by
    Michael Pyrcz (GeostatsGuy), which is distributed under the **MIT license**
    (redistribution permitted with attribution). The underlying channel TI is
    due to Strebelle, S. (2002), *Conditional simulation of complex geological
    structures using multiple-point statistics*, Mathematical Geology, 34(1),
    1-21. If the download is unavailable, the example falls back to a synthetic
    training image so it still runs offline.
"""

import os
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import gstools as gs

###############################################################################
# Load the Strebelle training image, with a synthetic fallback for offline use.

TI_URL = (
    "https://raw.githubusercontent.com/GeostatsGuy/"
    "GeoDataSets/master/MPS_Training_image_and_Realizations_500.npz"
)
CACHE = "mps_strebelle.npz"
try:
    if not os.path.exists(CACHE):
        urllib.request.urlretrieve(TI_URL, CACHE)
    ti_arr = np.load(CACHE)["array1"].astype(float)
    source = "Strebelle (2002) via GeoDataSets (MIT)"
except Exception as err:  # pragma: no cover - network fallback
    print(f"download failed ({err}); using a synthetic channel TI instead")
    gx, gy = np.meshgrid(np.arange(150), np.arange(150), indexing="ij")
    ti_arr = ((np.sin(gx / 6.0) + np.sin((gx + gy) / 10.0)) > 0).astype(float)
    source = "synthetic fallback"

ti = gs.TrainingImage(ti_arr, categorical=True)
print(f"TI {ti.shape} ({source}), sand fraction = {ti_arr.mean():.3f}")

###############################################################################
# Take 80 random conditioning points from the training image patterns.

sg_size = 80
rng = np.random.default_rng(0)
cond_x = rng.integers(0, sg_size, 80).astype(float)
cond_y = rng.integers(0, sg_size, 80).astype(float)
cond_val = ti_arr[cond_x.astype(int), cond_y.astype(int)]

###############################################################################
# Simulate with DSBC-style parameters (best-candidate + partial scan).

ds = gs.DirectSampling(
    gs.MPSModel(ti, n_neighbors=30, scan_fraction=0.2, threshold=0.0)
)
ds.set_condition([cond_x, cond_y], cond_val)
field = ds([np.arange(sg_size, dtype=float)] * 2, seed=42)

honored = int(
    (field[cond_x.astype(int), cond_y.astype(int)] == cond_val).sum()
)
print(f"conditioning honoured: {honored}/{cond_val.size}")

###############################################################################
# Plot the training image crop next to the conditional realization.

cmap = ListedColormap(["#c9a96e", "#2b6cb0"])  # shale / sand
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 5.5))
ax0.imshow(ti_arr[:sg_size, :sg_size], cmap=cmap, origin="lower")
ax0.set_title("Training image (crop)")
ax1.imshow(field, cmap=cmap, origin="lower")
ax1.scatter(cond_y, cond_x, c=cond_val, cmap=cmap, edgecolors="k", s=18)
ax1.set_title("Conditional DS realization")
fig.tight_layout()
