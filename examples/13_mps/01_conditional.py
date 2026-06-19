r"""
Conditioning to hard data
-------------------------

Real simulations must honour measured data (boreholes, samples). Direct
Sampling supports this through :meth:`~gstools.DirectSampling.set_condition`:
conditioning values are pinned into the grid before simulation and preserved
exactly, while the rest of the field is filled in around them.

We reuse the synthetic channel training image from the first example.
"""

import matplotlib.pyplot as plt
import numpy as np

import gstools as gs

###############################################################################
# Same synthetic channelized training image as before.

gx, gy = np.meshgrid(np.arange(60), np.arange(60), indexing="ij")
ti_data = ((np.sin(gx / 5.0) + np.sin((gx + gy) / 8.0)) > 0).astype(float)
ti = gs.TrainingImage(ti_data, categorical=True)

###############################################################################
# Draw 40 random "hard data" points and read their facies from the TI. In a
# real study these would be field measurements; here we sample the TI so the
# conditioning data are consistent with the patterns.

rng = np.random.default_rng(0)
cond_x = rng.integers(0, 40, 40).astype(float)
cond_y = rng.integers(0, 40, 40).astype(float)
cond_val = ti_data[cond_x.astype(int), cond_y.astype(int)]

###############################################################################
# Set the conditioning data and simulate. ``set_condition`` snaps each point to
# its nearest grid node, so the values are honoured exactly at those cells.

ds = gs.DirectSampling(ti, n_neighbors=12, scan_fraction=0.3, threshold=0.0)
ds.set_condition([cond_x, cond_y], cond_val)
field = ds([np.arange(40, dtype=float)] * 2, seed=7)

honored = int(
    (field[cond_x.astype(int), cond_y.astype(int)] == cond_val).sum()
)
print(f"conditioning honoured: {honored}/{cond_val.size}")

###############################################################################
# Plot the realization with the conditioning points overlaid. Every marker sits
# on a cell whose simulated facies matches the datum.

fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(field, cmap="cividis", origin="lower")
ax.scatter(cond_y, cond_x, c=cond_val, cmap="cividis", edgecolors="red", s=30)
ax.set_title("Conditional DS realization (red-edged = hard data)")
fig.tight_layout()
