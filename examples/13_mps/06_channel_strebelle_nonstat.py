r"""
Nonstationary geometry with the Strebelle channels
--------------------------------------------------

The Strebelle channel page used stationary search geometry. Here the same TI is
simulated with spatially varying rotation and anisotropy maps, so the local
data-event geometry changes across the simulation grid while the training image
itself stays fixed.

.. note::

    This example reuses the bundled Strebelle TI stored in
    ``input/strebelle_channel_ti.npz``. The data source and license are documented in
    :ref:`sphx_glr_examples_13_mps_04_channel_strebelle.py` and
    ``input/LICENSE.txt``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import gstools as gs

###############################################################################
# Load the bundled Strebelle training image.
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
source = "Strebelle (2002) via bundled GeoDataSets asset"

ti = gs.mps.TrainingImage(ti_arr, categorical=True, n_neighbors=30)
print(f"TI {ti.shape} ({source}), sand fraction = {ti_arr.mean():.3f}")

###############################################################################
# Define a modest simulation grid and two smooth nonstationary maps.

sg_size = 80
xs = np.arange(sg_size, dtype=float)
ys = np.arange(sg_size, dtype=float)
gx, gy = np.meshgrid(xs, ys, indexing="ij")

# Rotation inputs are radians: here they vary along y from 0 toward pi/4
# (45 degrees). Neighborhood geometry and radii use grid-index units.
rotation = (gy / sg_size) * (np.pi / 4.0)

# Anisotropy varies along the x-axis. Values above one stretch the local search
# geometry; values below one compress it.
anis = 0.8 + (gx / sg_size) * 0.4

###############################################################################
# Simulate with the nonstationary maps attached to the model. The maps
# reorient and rescale local data-event matching; they do not change the TI.
# ``scale`` is the absolute per-axis dilation: the main axis stays 1.0 and the
# transversal axis carries the local ``anis`` ratio. The positive factors are
# dimensionless. Physical axes still control coordinates and callable inputs;
# nonuniform spacing does not change this into a physical-distance search.

scale = np.stack([np.ones_like(anis), anis], axis=-1)
ds = gs.DirectSampling(
    gs.MPSModel(
        ti,
        scan_fraction=0.1,
        threshold=0.0,
        rotation=rotation,
        scale=scale,
    )
)

print("Simulating nonstationary field...")
field = ds([xs, ys], seed=42)

###############################################################################
# Plot the original TI crop next to the nonstationary realization.

cmap = ListedColormap(["#c9a96e", "#2b6cb0"])  # shale / sand
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 5.5))
ax0.imshow(ti_arr[:sg_size, :sg_size], cmap=cmap, origin="lower")
ax0.set_title("Training image (crop)")
ax1.imshow(field, cmap=cmap, origin="lower")
ax1.set_title(
    "Nonstationary DS realization\n(varying rotation and anisotropy)"
)
fig.tight_layout()

###############################################################################
# The channel geometry remains recognizable, while the changing rotation and
# anisotropy maps steer local patterns differently across the grid. This shows
# how spatially varying search geometry can impose a broad directional trend.
