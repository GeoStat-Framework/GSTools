r"""
Direct Sampling parameters: ``n_neighbors``, ``scan_fraction``, ``threshold``
----------------------------------------------------------------------------------

The previous pages introduced these settings; see the gallery introduction for
their definitions. This page varies each one on the Strebelle channel TI so you
can see how it changes the result and runtime.

* ``n_neighbors`` (``n``) — how many already-known cells form each data event.
  More neighbors can capture richer patterns, but cost more per node; better
  connectivity is not guaranteed.
* ``scan_fraction`` (``f``) — the budget relative to the selected TI size,
  capped by the valid search window and floored at one candidate. Larger
  budgets can examine more candidates and cost more.
* ``threshold`` (``t``) — the acceptance mode:

  - ``t = 0.0`` is **DSBC** (Direct Sampling Best Candidate, recommended): scan
    the configured budget and keep the **best** selectable match visited
    (an exact zero-distance match is accepted immediately). ``None`` is
    equivalent; this is neither an exact-only nor necessarily a full-TI scan.
  - ``t > 0`` is **DS greedy**: accept the **first** candidate whose distance is
    strictly below ``t``, or keep the best visited if none qualifies. It can
    stop earlier per node, but the result depends on a
    well-chosen threshold; DSBC usually needs no such tuning.

A fourth knob, ``max_radius``, caps how far neighbors may sit from the
simulated node in grid-index units; it is left at its default here.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import gstools as gs

###############################################################################
# Load the Strebelle TI and set a modest grid so the parameter sweep stays
# gallery-friendly.

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

SEED = 2024
sg_size = 56
grid = [np.arange(sg_size, dtype=float), np.arange(sg_size, dtype=float)]


def run(n_neighbors, scan_fraction, threshold):
    """Simulate one field and return ``(field, runtime_seconds)``."""
    ti = gs.mps.TrainingImage(
        ti_arr, categorical=True, n_neighbors=n_neighbors
    )
    ds = gs.DirectSampling(
        gs.MPSModel(ti, scan_fraction=scan_fraction, threshold=threshold)
    )
    start = time.perf_counter()
    field = ds(grid, seed=SEED, num_threads=4)
    return field, time.perf_counter() - start


###############################################################################
# Three one-at-a-time comparisons. Each row varies a single parameter around a
# shared reference (``n=20``, ``f=0.2``, ``t=0.0`` / DSBC).

comparisons = [
    (
        "n_neighbors",
        [("n = 6", run(6, 0.2, 0.0)), ("n = 30", run(30, 0.2, 0.0))],
    ),
    (
        "scan_fraction",
        [("f = 0.02", run(20, 0.02, 0.0)), ("f = 0.4", run(20, 0.4, 0.0))],
    ),
    (
        "threshold",
        [
            ("t = 0.0 (DSBC)", run(20, 0.2, 0.0)),
            ("t = 0.25 (DS greedy)", run(20, 0.2, 0.25)),
        ],
    ),
]

for name, pair in comparisons:
    for label, (_, elapsed) in pair:
        print(f"{name:14s} {label:24s} {elapsed:5.2f} s")

###############################################################################
# Plot the pairs. Within a row the two panels share the same seed, so any
# difference is the parameter, not the randomness.

cmap = ListedColormap(["#c9a96e", "#2b6cb0"])  # shale / sand
fig, axes = plt.subplots(3, 2, figsize=(8.5, 12.0))
for row, (name, pair) in enumerate(comparisons):
    for col, (label, (field, elapsed)) in enumerate(pair):
        ax = axes[row, col]
        ax.imshow(field, cmap=cmap, origin="lower", vmin=0, vmax=1)
        ax.set_title(f"{name}: {label}\n({elapsed:.2f} s)")
        ax.set_xticks([])
        ax.set_yticks([])
fig.tight_layout()

###############################################################################
# The comparisons show the expected trade-offs: larger neighborhoods retain
# more local structure, and a larger scan fraction usually finds better
# matches at a higher cost. A positive threshold can stop sooner when it finds
# an acceptable candidate, while DSBC checks the requested fraction for the
# best candidate. The timings are specific to this machine and these settings.
