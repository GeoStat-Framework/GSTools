r"""
Zoned simulation: a different TI per region
-------------------------------------------

Real domains are often not stationary: one part of the grid follows one pattern,
another part follows a different one. **Zonation** (Mariethoz et al., 2010,
para [40]) lets a region scan its own training image instead of the primary one.

A :class:`~gstools.mps.Zone` pairs a training image with a region -- a boolean
mask or a coordinate callable. Pass a list of zones to :class:`MPSModel`; a node
inside a zone's region scans that zone's TI, every other node scans the primary.
The selected domain supplies its TI data, shape, value range, and missingness.
``n_neighbors``, ``max_radius``, distance settings, and weights come from the
effective primary or cleanup configuration; a zone TI's unused search settings
are not activated. Neighbors can cross zone boundaries, allowing interaction
without guaranteeing seamless connectivity.

Here the primary TI has horizontal-ish channels (the Strebelle image) and the
top half of the grid uses the transpose, giving vertical channels there.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import gstools as gs

###############################################################################
# Primary TI (horizontal channels) and the zone TI (its transpose -> vertical
# channels). Both are binary {0, 1} facies, so they share the same categories.

example_dir = (
    Path(__file__).resolve().parent if "__file__" in globals() else Path(".")
)
with np.load(example_dir / "input" / "strebelle_channel_ti.npz") as data:
    ti_arr = data["array1"].astype(int)

primary = gs.mps.TrainingImage(ti_arr, categorical=True, n_neighbors=16)
zone_ti = gs.mps.TrainingImage(
    ti_arr.T.copy(), categorical=True, n_neighbors=16
)

###############################################################################
# The region: the top half of the simulation grid. ``Zone`` accepts a boolean
# mask shaped like the grid (``True`` where the zone applies).

sg_size = 64
grid = [np.arange(sg_size, dtype=float), np.arange(sg_size, dtype=float)]
region = np.zeros((sg_size, sg_size), dtype=bool)
region[sg_size // 2 :, :] = True
zone = gs.mps.Zone(zone_ti, region)

field = gs.DirectSampling(
    gs.MPSModel(primary, scan_fraction=0.08, threshold=0.0, zones=[zone])
)(grid, seed=7)

###############################################################################
# Plot the two training-image crops and the zoned realization. The channel
# orientation flips at the zone boundary (dashed line).

cmap = ListedColormap(["#c9a96e", "#2b6cb0"])
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(14, 4.8))
ax0.imshow(ti_arr[:sg_size, :sg_size], cmap=cmap, origin="lower")
ax0.set_title("Primary TI (horizontal)")
ax1.imshow(ti_arr.T[:sg_size, :sg_size], cmap=cmap, origin="lower")
ax1.set_title("Zone TI (vertical)")
ax2.imshow(field, cmap=cmap, origin="lower", vmin=0, vmax=1)
ax2.axhline(sg_size // 2 - 0.5, color="k", ls="--", lw=1)
ax2.set_title("Zoned realization")
fig.tight_layout()

###############################################################################
# The lower half shows horizontal channels (primary TI) and the upper half
# vertical channels (zone TI). Events can use neighbors across the boundary.
# Assess zone interiors and boundary bands separately over an ensemble rather
# than inferring connectivity from one image. Fixed controls remain repeatable
# across thread counts within a fixed build and dependency environment.
