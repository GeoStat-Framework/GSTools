r"""
A real training image: the Strebelle channels
----------------------------------------------

The first pages used tiny synthetic training images. Here we use the classic
**Strebelle (2002) channelized fluvial training image**, a widely used
reference example for MPS, and condition the simulation on random hard data.

.. note::

    **Data source / license.** The bundled training image is stored in
    ``input/strebelle_channel_ti.npz`` and comes from
    `GeoDataSets <https://github.com/GeostatsGuy/GeoDataSets>`_ by Michael
    Pyrcz (GeostatsGuy), distributed under the **MIT license**. The underlying
    channel TI is due to Strebelle, S. (2002), *Conditional simulation of
    complex geological structures using multiple-point statistics*,
    Mathematical Geology, 34(1), 1-21. Full redistribution notices are in
    ``input/LICENSE.txt``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import gstools as gs

###############################################################################
# Load the Strebelle TI bundled next to this example.

example_dir = (
    Path(__file__).resolve().parent if "__file__" in globals() else Path(".")
)
data_path = example_dir / "input" / "strebelle_channel_ti.npz"

if not data_path.exists():
    raise FileNotFoundError(
        f"Missing bundled training image: {data_path}. "
        "Run this example from examples/13_mps or restore the input directory."
    )

with np.load(data_path) as data:
    ti_arr = data["array1"].astype(int)

ti = gs.TrainingImage(ti_arr, categorical=True, n_neighbors=20)

###############################################################################
# Take unique conditioning points from the training image patterns. The grid is
# intentionally modest so the example remains suitable for the documentation
# gallery while still using the real 256x256 TI.

sg_size = 56
n_cond = 60
rng = np.random.default_rng(0)
cond_idx = rng.choice(sg_size * sg_size, size=n_cond, replace=False)
cond_x_idx, cond_y_idx = np.unravel_index(cond_idx, (sg_size, sg_size))
cond_x = cond_x_idx.astype(float)
cond_y = cond_y_idx.astype(float)
cond_val = ti_arr[cond_x_idx, cond_y_idx]

###############################################################################
# Simulate with DSBC-style parameters (best-candidate + partial scan).

ds = gs.DirectSampling(gs.MPSModel(ti, scan_fraction=0.08, threshold=0.0))
ds.set_condition([cond_x, cond_y], cond_val)
grid = [np.arange(sg_size, dtype=float), np.arange(sg_size, dtype=float)]
field = ds(grid, seed=42)

assert np.all(field[cond_x_idx, cond_y_idx] == cond_val)

###############################################################################
# Plot the training image crop next to the conditional realization.

cmap = ListedColormap(["#c9a96e", "#2b6cb0"])  # shale / sand
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 5.5))
ax0.imshow(
    ti_arr[:sg_size, :sg_size],
    cmap=cmap,
    origin="lower",
    vmin=0,
    vmax=1,
)
ax0.set_title("Training image (crop)")
ax1.imshow(field, cmap=cmap, origin="lower", vmin=0, vmax=1)
ax1.scatter(
    cond_y,
    cond_x,
    c=cond_val,
    cmap=cmap,
    edgecolors="k",
    linewidths=0.5,
    s=18,
    vmin=0,
    vmax=1,
)
ax1.set_title("Conditional DS realization")
fig.tight_layout()

###############################################################################
# The realization reproduces connected channel-like patterns while honoring
# every conditioning value. Increasing ``scan_fraction`` or the neighborhood
# size can make the match to the TI more detailed, at additional runtime cost.
