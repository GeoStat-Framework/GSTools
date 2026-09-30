r"""
A non-stationary real training image
------------------------------------

Here we take the Strebelle channelized fluvial training image and apply a
spatially varying rotation and scaling factor to show how Direct Sampling
can generate non-stationary features from a stationary training image.
"""

import os
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import gstools as gs

# Load the Strebelle training image
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
except Exception as err:
    print(f"download failed ({err}); using a synthetic channel TI instead")
    gx, gy = np.meshgrid(np.arange(150), np.arange(150), indexing="ij")
    ti_arr = ((np.sin(gx / 6.0) + np.sin((gx + gy) / 10.0)) > 0).astype(float)
    source = "synthetic fallback"

ti = gs.TrainingImage(ti_arr, categorical=True, n_neighbors=30)
print(f"TI {ti.shape} ({source}), sand fraction = {ti_arr.mean():.3f}")

# Grid size for simulation
sg_size = 80
xs = np.arange(sg_size, dtype=float)
ys = np.arange(sg_size, dtype=float)
gx, gy = np.meshgrid(xs, ys, indexing="ij")

# Create non-stationary rotation and anisotropy (scaling) maps. These are
# passed straight into MPSModel as model configuration (rotation=, scale=)
# rather than set on the engine after construction.
# Rotation varies smoothly along the Y-axis from 0 to 45 degrees
rotation = (gy / sg_size) * (np.pi / 4.0)

# Anisotropy (channel width) varies along the X-axis
# > 1 means wider channels, < 1 means narrower
anis = 0.8 + (gx / sg_size) * 0.4

# scale is a full per-axis vector (no implicit axis-0 pin): channels keep
# their along-x size (1.0) while the transversal width follows the map.
scale = np.stack([np.ones_like(anis), anis], axis=-1)
ds = gs.DirectSampling(
    gs.MPSModel(
        ti, scan_fraction=0.1, threshold=None, rotation=rotation, scale=scale
    )
)

print("Simulating non-stationary field...")
# Generate the field
field = ds([xs, ys], seed=42)

# Plot the result
cmap = ListedColormap(["#c9a96e", "#2b6cb0"])  # shale / sand
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 5.5))
ax0.imshow(ti_arr[:sg_size, :sg_size], cmap=cmap, origin="lower")
ax0.set_title("Training image (crop)")
ax1.imshow(field, cmap=cmap, origin="lower")
ax1.set_title("Non-stationary DS realization\n(varying rotation & scale)")
fig.tight_layout()
plt.savefig("nonstat_strebelle.png")
print("Saved nonstat_strebelle.png")
