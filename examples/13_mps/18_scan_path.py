r"""
TI scan order: sequential vs random
------------------------------------

At each node Direct Sampling scans a window of the training image looking for a
matching pattern and stops at the first candidate under the acceptance
threshold (DSBC: it keeps the best within a budget). The order in which that
window is visited is the **scan path**.

``scan_path="sequential"`` (the default, Mariethoz et al., 2010, para [19])
walks the window in raster order from a per-node random start. ``"random"`` (a
GSTools extension) visits distinct window cells in a node-local random order, so
the visited candidate prefix avoids the raster ordering. Threshold acceptance
and ties still select according to that visited order. Set it on
:class:`MPSModel`.

Both modes are fully deterministic for a fixed seed and give bit-identical
results across thread counts within a fixed build and dependency environment.
Different visited sets can change the best distance as well as tie selection.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import gstools as gs

###############################################################################
# Strebelle channel TI, one modest grid, the same seed for both runs.

example_dir = (
    Path(__file__).resolve().parent if "__file__" in globals() else Path(".")
)
with np.load(example_dir / "input" / "strebelle_channel_ti.npz") as data:
    ti_arr = data["array1"].astype(int)

ti = gs.mps.TrainingImage(ti_arr, categorical=True, n_neighbors=16)
grid = [np.arange(64, dtype=float), np.arange(64, dtype=float)]

field_seq = gs.DirectSampling(
    gs.MPSModel(ti, scan_fraction=0.1, threshold=0.0, scan_path="sequential")
)(grid, seed=99)
field_rnd = gs.DirectSampling(
    gs.MPSModel(ti, scan_fraction=0.1, threshold=0.0, scan_path="random")
)(grid, seed=99)

# Different candidate orders can select different anchors, but different fields
# are not a required invariant (for example, a constant TI always agrees).
print("scan modes differ:", not np.array_equal(field_seq, field_rnd))

# Each mode is reproducible on its own.
field_rnd_again = gs.DirectSampling(
    gs.MPSModel(ti, scan_fraction=0.1, threshold=0.0, scan_path="random")
)(grid, seed=99)
assert np.array_equal(field_rnd, field_rnd_again)

###############################################################################
# Plot the two realizations side by side.

cmap = ListedColormap(["#c9a96e", "#2b6cb0"])
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 5.5))
ax0.imshow(field_seq, cmap=cmap, origin="lower", vmin=0, vmax=1)
ax0.set_title("scan_path='sequential'")
ax1.imshow(field_rnd, cmap=cmap, origin="lower", vmin=0, vmax=1)
ax1.set_title("scan_path='random'")
fig.tight_layout()

###############################################################################
# These realizations illustrate candidate-order differences. Statistical
# fidelity needs ensemble evidence; this comparison does not establish it.
# The sequential default remains the Mariethoz reference scan order.
