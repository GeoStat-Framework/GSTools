r"""
Reproducing Mariethoz et al. (2010) Figure 6: the variation distance
-----------------------------------------------------------------------

Figure 6 contrasts the plain :math:`L_2` pattern distance with Mariethoz's
"variation" distance (Eq. 9) on a multi-Gaussian training image. The variation
distance compares the *shape* of a pattern rather than its level, and the
matched value is mean-shifted on assignment, which lets a stationary TI
reproduce a field whose local mean drifts.
"""

import matplotlib.pyplot as plt
import numpy as np

import gstools as gs

# 1. Create the Training Image (Multi-Gaussian)
# The paper uses a 250x250 multi-Gaussian field with zero mean and unit variance.
# Exponential variogram, range x = 35, range y = 25.
grid_size = 250
x = y = np.arange(grid_size, dtype=float)

# Note: GSTools len_scale is roughly range / 3 for exponential models,
# but we'll just set len_scale directly to match the spatial correlation visually.
cov_model = gs.Exponential(dim=2, var=1.0, len_scale=[35, 25], angles=0)
srf = gs.SRF(cov_model, mean=0, seed=123)
ti_data = srf((x, y), mesh_type="structured")

# We create a continuous Training Image and specify distance="variation"
# This tells GSTools to use the variation-based distance (Mariethoz Eq 9)
ti = gs.TrainingImage(
    ti_data, categorical=False, distance="variation", n_neighbors=15
)

# 2. Create the Non-Stationary Conditioning Data
# 100 points, values ranging from ~99 to ~111 with a spatial trend
np.random.seed(42)
n_cond = 100
cond_x = np.random.uniform(0, grid_size, n_cond)
cond_y = np.random.uniform(0, grid_size, n_cond)

# Create a trend that increases along the X axis (left to right)
# plus some random noise to simulate real data
cond_val = 100 + 10 * (cond_x / grid_size) + np.random.normal(0, 1.5, n_cond)

# 3. Setup the MPS Model
# From the caption: n=15, t=0.01. We use scan_fraction=0.5
model = gs.MPSModel(ti, scan_fraction=0.5, threshold=0.01)

ds = gs.DirectSampling(model)
ds.set_condition([cond_x, cond_y], cond_val)

# 4. Run the simulation
print(
    f"Simulating variation-based field ({grid_size}x{grid_size}) with 100 cond points..."
)
field = ds([x, y], seed=42, num_threads=1)

# 5. Plotting
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# a) TI
im_a = axes[0].imshow(
    ti_data.T,
    cmap="gray",
    origin="lower",
    extent=[0, 250, 0, 250],
    vmin=-3,
    vmax=3,
)
axes[0].set_title("a) Training image")
plt.colorbar(im_a, ax=axes[0], fraction=0.046, pad=0.04)

# b) Conditioning data
im_b = axes[1].scatter(
    cond_x, cond_y, c=cond_val, cmap="gray", vmin=98, vmax=112, s=20
)
axes[1].set_xlim(0, 250)
axes[1].set_ylim(0, 250)
axes[1].set_aspect("equal")
axes[1].set_title("b) Conditioning data")
plt.colorbar(im_b, ax=axes[1], fraction=0.046, pad=0.04)

# c) Simulation
im_c = axes[2].imshow(
    field.T,
    cmap="gray",
    origin="lower",
    extent=[0, 250, 0, 250],
    vmin=98,
    vmax=112,
)
# Overlay conditioning points as open circles
axes[2].scatter(
    cond_x, cond_y, facecolors="none", edgecolors="k", s=40, linewidths=1
)
axes[2].set_title("c) One simulation")
plt.colorbar(im_c, ax=axes[2], fraction=0.046, pad=0.04)

fig.tight_layout()
plt.savefig("mariethoz_fig6_reproduction.png")
print("Saved mariethoz_fig6_reproduction.png")
