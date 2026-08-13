r"""
Reproducing Mariethoz et al. (2010) Figure 3: a continuous training image
--------------------------------------------------------------------------

Figure 3 of Mariethoz et al. (2010) simulates the continuous "stone" training
image conditioned on values drawn from the TI's own marginal distribution.
The paper's caption gives ``n = 80`` and ``t = 0.01``; the pattern distance is
their Distance 4 (Eq. 4, a normalized RMSE), which is ``distance="l2"`` in
GSTools' naming.

Because the conditioning values are resampled from the TI, a correct
simulation must reproduce the TI histogram closely — panel c) is the
verification, not just decoration.
"""

import os
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import gstools as gs

# 1. Training image
TI_URL = (
    "https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/"
    "master/stone.tiff"
)
CACHE = "stone.tiff"
if not os.path.exists(CACHE):
    urllib.request.urlretrieve(TI_URL, CACHE)
# The paper figure uses a 200x200 grid.
ti_data = np.array(Image.open(CACHE)).astype(float)[:200, :200]
grid_size = 200

ti = gs.TrainingImage(
    ti_data, categorical=False, distance="l2", n_neighbors=80, distance_power=0.5
)

# 2. Conditioning data: paper takes 100 values from the TI's marginal
# distribution and places them at random positions in the simulation.
rng = np.random.default_rng(1)
n_cond = 200
cond_x = rng.uniform(0, grid_size, n_cond)
cond_y = rng.uniform(0, grid_size, n_cond)
cond_val = ti_data[
    rng.integers(0, ti_data.shape[0], n_cond),
    rng.integers(0, ti_data.shape[1], n_cond),
]

# 3. Simulation
ds = gs.DirectSampling(
    gs.MPSModel(ti, scan_fraction=0.5, threshold=0.01, post_processing=1)
)
ds.set_condition([cond_x, cond_y], cond_val)

x = y = np.arange(grid_size, dtype=float)
print(f"Simulating continuous field ({grid_size}x{grid_size})...")
field = ds([x, y], seed=121, num_threads=8, progress=True, path="sequential")

# 4. Verification: DS pastes TI cells, so every simulated value must be a TI
# value (CLAUDE.md MPS validity criterion).
assert set(np.unique(field).tolist()) <= set(np.unique(ti_data).tolist())
print(f"mean -- TI {ti_data.mean():.4f} | simulation {field.mean():.4f}")

# 5. Plotting: Figure 3's layout (two images on top, histogram below)
fig = plt.figure(figsize=(12, 10))
ax1, ax2, ax3 = plt.subplot(221), plt.subplot(222), plt.subplot(212)

ax1.imshow(ti_data.T, cmap="gray", origin="lower", vmin=0, vmax=1.0)
ax1.set_title("a) Training image")

im_b = ax2.imshow(field.T, cmap="gray", origin="lower", vmin=0, vmax=1.0)
ax2.scatter(
    cond_x,
    cond_y,
    c=cond_val,
    cmap="gray",
    vmin=0,
    vmax=1.0,
    edgecolors="k",
    s=40,
    linewidths=0.5,
)
ax2.set_title("b) Simulation (conditioning points overlaid)")
plt.colorbar(im_b, ax=ax2, fraction=0.046, pad=0.04)

hist_ti, bins = np.histogram(ti_data.ravel(), bins=50, density=True)
hist_sim, _ = np.histogram(field.ravel(), bins=bins, density=True)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
ax3.plot(bin_centers, hist_sim, "k-", label="Simulation")
ax3.plot(bin_centers, hist_ti, "k--", label="Training image")
ax3.set_title("c) Comparison of histograms")
ax3.legend()

fig.tight_layout()
plt.savefig("mariethoz_fig3_reproduction.png")
print("Saved mariethoz_fig3_reproduction.png")
