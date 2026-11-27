r"""
Variation distance for continuous conditioning
----------------------------------------------

The :ref:`Continuous variables and distance metrics example
<sphx_glr_examples_13_mps_03_continuous.py>` compared continuous distances
without conditioning. Here we return to conditioning and focus on the
``"variation"`` distance: the TI has local fluctuations around zero, while the
conditioning data include a large-scale trend. Comparing local variation
rather than absolute level makes that mismatch meaningful for matching. Both
event means use common finite support. The weighted Lp distance divides
centered residuals by ``2*d_max`` and clips at one; a positive threshold is
easier to satisfy with this denominator than with ``d_max``. Thresholds are
not directly transferable between differently normalized metrics. Pasted
values shift by the same common-support mean difference and can exceed the
TI range.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

import gstools as gs

###############################################################################
# Load the generated continuous TI. It was generated once as a multi-Gaussian
# field with anisotropic spatial correlation, using an exponential covariance
# model with ``len_scale=[35, 25]`` and ``seed=123``.

example_dir = (
    Path(__file__).resolve().parent if "__file__" in globals() else Path(".")
)
data_path = example_dir / "input" / "synthetic_variation_ti.npz"

if not data_path.exists():
    raise FileNotFoundError(
        f"Missing bundled training image: {data_path}. "
        "Run this example from examples/13_mps or restore the input assets."
    )

with np.load(data_path) as data:
    ti_data = data["ti"]

grid_size = ti_data.shape[0]
x = y = np.arange(grid_size, dtype=float)

###############################################################################
# Use the variation distance for continuous data-event comparisons.

ti = gs.mps.TrainingImage(
    ti_data,
    categorical=False,
    distance="variation",
    n_neighbors=20,
)

###############################################################################
# Create conditioning data with a spatial trend and a different absolute level
# than the TI.

rng = np.random.RandomState(42)
n_cond = 100
cond_x = rng.uniform(0, grid_size, n_cond)
cond_y = rng.uniform(0, grid_size, n_cond)
cond_val = 100 + 10 * (cond_x / grid_size) + rng.normal(0, 1.5, n_cond)

###############################################################################
# Simulate with the continuous variation distance. A moderate neighborhood and
# a budget of half the TI size (capped by the valid window) keep the example
# runnable. Their effect on texture needs ensemble assessment.
model = gs.MPSModel(ti, scan_fraction=0.5, threshold=0.01)

ds = gs.DirectSampling(model)
ds.set_condition([cond_x, cond_y], cond_val)

print(
    f"Simulating variation-based field ({grid_size}x{grid_size}) "
    f"with {n_cond} conditioning points..."
)
field = ds([x, y], seed=42, num_threads=4)

###############################################################################
# Plot the TI, the conditioning data, and one conditional realization. The
# training image uses the same blue-yellow palette as the conditioned values,
# while the conditioning and simulation panels share the lower-row colorbar.

fig = plt.figure(figsize=(11, 9), constrained_layout=True)
grid = fig.add_gridspec(
    2,
    3,
    width_ratios=[1, 1, 0.045],
    height_ratios=[1, 1],
)
ax_ti = fig.add_subplot(grid[0, 0:2])
cax_ti = fig.add_subplot(grid[0, 2])
ax_cond = fig.add_subplot(grid[1, 0])
ax_sim = fig.add_subplot(grid[1, 1])
cax_val = fig.add_subplot(grid[1, 2])

ti_norm = Normalize(vmin=-3, vmax=3)
value_norm = Normalize(vmin=98, vmax=112)

# a) TI
im_a = ax_ti.imshow(
    ti_data.T,
    cmap="cividis",
    origin="lower",
    extent=[0, grid_size, 0, grid_size],
    norm=ti_norm,
)
ax_ti.set_title("a) Training image")
fig.colorbar(im_a, cax=cax_ti, label="Training value")

# b) Conditioning data
im_b = ax_cond.scatter(
    cond_x,
    cond_y,
    c=cond_val,
    cmap="cividis",
    norm=value_norm,
    s=22,
    edgecolors="white",
    linewidths=0.35,
)
ax_cond.set_xlim(0, grid_size)
ax_cond.set_ylim(0, grid_size)
ax_cond.set_aspect("equal")
ax_cond.set_title("b) Conditioning data")

# c) Simulation
im_c = ax_sim.imshow(
    field.T,
    cmap="cividis",
    origin="lower",
    extent=[0, grid_size, 0, grid_size],
    norm=value_norm,
)
# Overlay conditioning points as open circles
ax_sim.scatter(
    cond_x,
    cond_y,
    facecolors="none",
    edgecolors="white",
    s=26,
    linewidths=0.8,
)
ax_sim.set_title("c) One conditional simulation")
fig.colorbar(im_c, cax=cax_val, label="Conditioned value")

###############################################################################
# The hard data follow a smooth increase from left to right, and the simulation
# can follow that trend while drawing local fluctuations from the TI. Assess
# centered/detrended statistics over realizations rather than requiring raw TI
# marginals for this shifted case. Variation
# distance helps because the conditioning values have a very different
# absolute level from the training image.
