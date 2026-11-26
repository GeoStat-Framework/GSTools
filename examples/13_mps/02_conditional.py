r"""
Conditioning to hard data
-------------------------

Real simulations must honour measured data (boreholes, samples). Direct
Sampling supports this through :meth:`~gstools.DirectSampling.set_condition`:
conditioning values are pinned into the grid before simulation and preserved
exactly, while the rest of the field is filled in around them.

This page adds one idea to the first Direct Sampling simulation: keep the same
TI-based pattern reproduction, but force selected grid cells to match hard data
exactly.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import gstools as gs

###############################################################################
# Recreate the same synthetic block-pattern training image used in the first
# page.

gx, gy = np.meshgrid(np.arange(60), np.arange(60), indexing="ij")
ti_data = ((np.sin(gx / 5.0) + np.sin((gx + gy) / 8.0)) > 0).astype(int)
ti = gs.TrainingImage(ti_data, categorical=True, n_neighbors=12)

###############################################################################
# Place 40 hard data points in a square pattern and set each to facies 1: the
# 28-point outline of an 8x8 square plus 12 points in its center.
patch_x, patch_y = np.meshgrid(
    np.arange(12, 20), np.arange(12, 20), indexing="ij"
)
outline = (patch_x == 12) | (patch_x == 19) | (patch_y == 12) | (patch_y == 19)
cond_x_idx = np.concatenate([patch_x[outline], np.tile(np.arange(14, 18), 3)])
cond_y_idx = np.concatenate(
    [patch_y[outline], np.repeat(np.arange(14, 17), 4)]
)
cond_x = cond_x_idx.astype(float)
cond_y = cond_y_idx.astype(float)
cond_val = np.ones(cond_x_idx.size, dtype=int)

###############################################################################
# Set the conditioning data and simulate. ``set_condition`` snaps each point to
# the nearest grid node; because our points already lie on grid nodes, the
# values are honoured exactly at those cells.

ds = gs.DirectSampling(gs.MPSModel(ti, scan_fraction=0.3, threshold=0.0))
ds.set_condition([cond_x, cond_y], cond_val)
grid = [np.arange(40, dtype=float), np.arange(40, dtype=float)]
field = ds(grid, seed=7)

assert np.all(field[cond_x_idx, cond_y_idx] == cond_val)

###############################################################################
# Plot the training image next to the conditional realization. Black-edged
# markers show the square hard-data pattern, and every marker matches facies 1.

cmap = ListedColormap(["#d8c690", "#2f6f9f"])
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5))
ax0.imshow(ti_data, cmap=cmap, origin="lower", vmin=0, vmax=1)
ax0.set_title("Training image")
ax1.imshow(field, cmap=cmap, origin="lower", vmin=0, vmax=1)
ax1.scatter(
    cond_y,
    cond_x,
    c=cond_val,
    cmap=cmap,
    edgecolors="black",
    linewidths=0.8,
    s=34,
    vmin=0,
    vmax=1,
)
ax1.set_title("Conditional DS realization (markers = hard data)")
fig.tight_layout()
