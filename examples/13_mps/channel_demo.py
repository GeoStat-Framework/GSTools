"""
Direct Sampling demo — Strebelle (2002) channelized fluvial TI.
Adapted for GSTools.
"""

import os
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from gstools import mps

# 1. Load TI
TI_URL = (
    "https://raw.githubusercontent.com/GeostatsGuy/"
    "GeoDataSets/master/MPS_Training_image_and_Realizations_500.npz"
)
CACHE = "mps_strebelle.npz"
if not os.path.exists(CACHE):
    print("Downloading Strebelle TI ...")
    urllib.request.urlretrieve(TI_URL, CACHE)

ti_arr = np.load(CACHE)["array1"].astype(int)  # (256, 256)
ti_model = mps.TrainingImage(ti_arr, categorical=True)
print(f"TI shape: {ti_model.shape}  sand={ti_arr.mean():.3f}")

# 2. Conditioning: 100 random hard-data points from the TI
SG_SIZE = 100
N_COND = 100
rng = np.random.default_rng(0)
cond_row = rng.integers(0, SG_SIZE, N_COND)
cond_col = rng.integers(0, SG_SIZE, N_COND)
cond_pos = [cond_row.astype(float), cond_col.astype(float)]
cond_val = ti_arr[cond_row, cond_col].astype(float)
print(f"Conditioning: {N_COND} pts  sand={cond_val.mean():.3f}")

# 3. Simulate (n=30, f=1.0, t=0.01)
N_NEIGH = 30
SCAN_F = 0.1
THRESH = 0.01

ds = mps.DirectSampling(
    ti_model, n_neighbors=N_NEIGH, scan_fraction=SCAN_F, threshold=THRESH
)
ds.set_condition(cond_pos, cond_val)

print("Starting simulation (this may take a moment)...")
x = np.arange(SG_SIZE, dtype=float)
y = np.arange(SG_SIZE, dtype=float)
sg = ds([x, y], seed=42).astype(int)

honored = (sg[cond_row, cond_col] == cond_val.astype(int)).sum()
print(f"Simulation complete. Conditioning: {honored}/{N_COND} honored")

# 4. Plot
cmap = ListedColormap(["#c9a96e", "#2b6cb0"])
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(ti_arr[:SG_SIZE, :SG_SIZE], cmap=cmap, origin="upper")
ax[0].set_title("Training Image (Crop)")

im = ax[1].imshow(sg, cmap=cmap, origin="upper")
ax[1].scatter(
    cond_col,
    cond_row,
    c=cond_val,
    cmap=cmap,
    edgecolors="k",
    s=20,
    label="Cond. Points",
)
ax[1].set_title(f"DS Realization (n={N_NEIGH}, f={SCAN_F}, t={THRESH})")
ax[1].legend()

plt.colorbar(im, ax=ax.ravel().tolist(), ticks=[0.25, 0.75]).set_ticklabels(
    ["shale", "sand"]
)
plt.savefig("channel_demo_mps.png", dpi=150, bbox_inches="tight")
print("Saved plot to channel_demo_mps.png")
