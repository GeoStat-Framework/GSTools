r"""
Continuous variables and distance metrics
-----------------------------------------

Direct Sampling is not limited to categorical facies. With
``categorical=False`` it simulates continuous variables such as permeability,
porosity, or elevation.
The distance metric then selects how two continuous neighborhoods are compared:

* ``"l1"``: Manhattan distance on the raw values.
* ``"l2"``: Euclidean distance on the raw values.
* ``"variation"``: compares relative variations, so similar local shapes can
  match even when the local mean shifts.

We keep ``threshold=0.0`` for all three runs. This is the DSBC setting: DS
takes the best selectable candidate within the configured scan budget,
with an early exit for an exact zero-distance match. ``None`` is equivalent.
That keeps this first continuous example focused on the metric choice instead
of threshold tuning.

No conditioning data are used here. The goal is to isolate one new concept:
how continuous data events are compared. The Strebelle channel page combines
conditioning again with a real training image.
"""

import matplotlib.pyplot as plt
import numpy as np

import gstools as gs

###############################################################################
# A smooth, continuous synthetic training image with connected high-value bands.

gx, gy = np.meshgrid(np.arange(60), np.arange(60), indexing="ij")
ti_data = np.sin(gx / 6.0) * np.cos(gy / 8.0)

###############################################################################
# Simulate the same grid three times, changing only the distance metric.

grid = [np.arange(32, dtype=float), np.arange(32, dtype=float)]
metrics = [
    ("l1", "Manhattan (L1)"),
    ("l2", "Euclidean (L2)"),
    ("variation", "Variation"),
]
fields = {}
for distance, label in metrics:
    ti = gs.mps.TrainingImage(
        ti_data,
        categorical=False,
        distance=distance,
        n_neighbors=12,
    )
    ds = gs.DirectSampling(gs.MPSModel(ti, scan_fraction=0.3, threshold=0.0))
    fields[label] = ds(grid, seed=3)

###############################################################################
# Plot. A shared colour scale keeps the comparison visually fair.

arrays = [ti_data, *fields.values()]
vmin = min(arr.min() for arr in arrays)
vmax = max(arr.max() for arr in arrays)

fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
axes = axes.ravel()
im = axes[0].imshow(
    ti_data,
    cmap="RdBu_r",
    origin="lower",
    vmin=vmin,
    vmax=vmax,
)
axes[0].set_title("Training image")
for ax, (label, field) in zip(axes[1:], fields.items()):
    ax.imshow(field, cmap="RdBu_r", origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(label)
fig.colorbar(im, ax=axes, shrink=0.8)

###############################################################################
# The L1 and L2 realizations are similar because both favor nearby continuous
# values. Variation distance compares local changes around means on common
# finite support and shifts pasted values by the same mean difference; outputs
# can exceed the TI range. These plots illustrate individual realizations, not
# an ensemble claim about smoothness or marginal distributions.
