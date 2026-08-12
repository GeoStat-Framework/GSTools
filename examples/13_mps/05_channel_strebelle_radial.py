r"""
Reproducing Mariethoz et al. (2010) Figure 7: radial non-stationarity
-----------------------------------------------------------------------

This example reproduces Figure 7 from Mariethoz et al. (2010), demonstrating
geometric non-stationarity via Direct Sampling. We use the Strebelle channel
training image and apply a radial rotation map and a distance-based affinity
(scaling) map to create channels radiating outward and becoming thinner towards
the edges.
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

# GSTools' rotation/anisotropy convention always aligns the *primary*
# (axis-0, unscaled by `anis`) direction to the `rotation` angle and scales
# axis 1 by `anis`. The Strebelle TI's channels run along its axis 1, not
# axis 0 (verified via structure-tensor analysis of the raw TI) -- left
# as-is, this makes `rotation` point the channels' *short* axis radially
# (channels end up tangential, forming concentric circles) and makes `anis`
# compress the channels' *length* instead of their width (no visible
# thinning). Transposing the TI swaps which array axis carries the channel
# direction, so both the orientation and the thinning effect come out
# right with a plain radial `rotation` and `anis` map.
ti_arr = ti_arr.T

ti = gs.TrainingImage(ti_arr, categorical=True, n_neighbors=32)
print(f"TI {ti.shape} ({source}), sand fraction = {ti_arr.mean():.3f}")

# The paper used a 1000x1000 grid. To avoid grid crowding where the thick TI
# channels collide near the center, we increase the size to 500x500.
sg_size = 1000
xs = np.arange(sg_size, dtype=float)
ys = np.arange(sg_size, dtype=float)
gx, gy = np.meshgrid(xs, ys, indexing="ij")

center_x = sg_size / 2.0
center_y = sg_size / 2.0

# 1. Rotation Map: Angle from the center, creating a radial pattern.
# Note: GSTools rotation is in radians.
# We measure the angle from index 0 (y) towards index 1 (x) to match GSTools' assumption.
rotation = np.arctan2(gy - center_y, gx - center_x)

# 2. Affinity Map: Scales the channels based on distance from center.
# The paper goes from 1.0 at the center down to ~0.4 at the corners.
# In GSTools, anis < 1 correctly makes the feature thinner.
radius = np.sqrt((gx - center_x) ** 2 + (gy - center_y) ** 2)
max_radius = np.sqrt(center_x**2 + center_y**2)
anis = 1.0 - 0.65 * (radius / max_radius)

# 3. Configure the MPS model. rotation/scale are model configuration passed
# straight into MPSModel; scale is a full per-axis vector, so axis 0 stays
# pinned at 1.0 while the affinity map scales axis 1.
scale = np.stack([np.ones_like(anis), anis], axis=-1)
ds = gs.DirectSampling(
    gs.MPSModel(
        ti, scan_fraction=0.35, threshold=None, rotation=rotation, scale=scale
    )
)

print(f"Simulating non-stationary field ({sg_size}x{sg_size})...")
field = ds([xs, ys], seed=1, num_threads=8)

# Plotting the reproduction of Mariethoz Figure 7
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# All panels are transposed so x runs horizontally and y vertically: the
# arrays are indexed [x, y], which imshow would otherwise render sideways.

# a) Rotation map
im_rot = axes[0, 0].imshow(np.rad2deg(rotation).T, cmap="gray", origin="lower")
axes[0, 0].set_title("a) Rotation (degrees)")
plt.colorbar(im_rot, ax=axes[0, 0], fraction=0.046, pad=0.04)

# b) Affinity map
im_aff = axes[0, 1].imshow(
    anis.T, cmap="gray", origin="lower", vmin=0.1, vmax=1.0
)
axes[0, 1].set_title("b) Affinity ratio")
plt.colorbar(im_aff, ax=axes[0, 1], fraction=0.046, pad=0.04)

# c) TI
cmap = ListedColormap(
    ["#000000", "#FFFFFF"]
)  # black background, white channels
axes[1, 0].imshow(ti_arr.T[:250, :250], cmap=cmap, origin="lower")
axes[1, 0].set_title("c) TI")

# d) Simulation
axes[1, 1].imshow(field.T, cmap=cmap, origin="lower")
axes[1, 1].set_title("d) One simulation")

fig.tight_layout()
plt.savefig("mariethoz_fig7_reproduction.png")
print("Saved mariethoz_fig7_reproduction.png")
