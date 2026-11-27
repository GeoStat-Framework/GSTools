r"""
Parallel execution and reproducibility
---------------------------------------

Direct Sampling is stochastic, but for a fixed seed it is **deterministic** —
and it is deterministic *independently of the thread count*. The engine builds
a dependency directed acyclic graph (DAG) over the simulation path
and parallelizes the work across threads without changing the visit order or
the candidate choices, so the same seed produces **bit-identical** output
regardless of the thread count within a fixed build and dependency
environment. This does not promise bitwise identity across operating systems,
dependency versions, or algorithm releases.

This matters in practice: you can develop and debug a simulation serially, then
run it with more threads and still get exactly the field you validated. This
page demonstrates this on the bundled Strebelle channel TI with a bit-for-bit
equality check across 1, 2, and 4 threads.

Thread resolution uses the call-time ``num_threads`` (used here), then
``gs.config.NUM_THREADS``. The current unset/falsy fallback is one thread.

.. note::

    **Data source / license.** The bundled training image is stored in
    ``input/strebelle_channel_ti.npz`` and comes from
    `GeoDataSets <https://github.com/GeostatsGuy/GeoDataSets>`_ by Michael
    Pyrcz (GeostatsGuy), distributed under the **MIT license**. See
    ``input/LICENSE.txt`` for the full redistribution notices.
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

ti = gs.mps.TrainingImage(ti_arr, categorical=True, n_neighbors=20)

###############################################################################
# Simulate the **same seed** at several thread counts. The grid is kept modest
# so the whole sweep stays gallery-friendly.

SEED = 20240517
sg_size = 64
grid = [np.arange(sg_size, dtype=float), np.arange(sg_size, dtype=float)]
thread_counts = (1, 2, 4)

fields = {}
for n_threads in thread_counts:
    ds = gs.mps.DirectSampling(gs.mps.MPSModel(ti, scan_fraction=0.1, threshold=None))
    fields[n_threads] = ds(grid, seed=SEED, num_threads=n_threads)
    print(f"num_threads={n_threads}: done")

###############################################################################
# **Reproducibility check.** Every realization must be exactly equal to the
# serial one — not merely close. A single ``array_equal`` over all thread counts
# verifies equality for this seed, workload, build, and dependency environment.

reference = fields[1]
all_identical = all(
    np.array_equal(reference, fields[n]) for n in thread_counts
)
print(f"All thread counts produced identical output: {all_identical}")
assert all_identical, (
    "Parallel output differs from serial — this must not happen."
)

###############################################################################
# Plot the (identical) realization that every thread count produced.

cmap = ListedColormap(["#c9a96e", "#2b6cb0"])  # shale / sand
fig, ax_field = plt.subplots(figsize=(5.5, 5.0))
ax_field.imshow(reference, cmap=cmap, origin="lower", vmin=0, vmax=1)
ax_field.set_title("Realization (identical for all thread counts)")
ax_field.set_xticks([])
ax_field.set_yticks([])
fig.tight_layout()

###############################################################################
# **Seed control.** DS uses the opt-in SeedSequence/MT19937 stream contract.
# ``seed`` is a master seed that supplies two separately controlled streams:
#
# * ``path_seed`` — the random order in which nodes are visited, and
# * ``node_seed`` — the TI scan start positions and fallback draws.
#
# Passing the same seeds reproduces a run exactly (just like across thread
# counts above); changing only ``path_seed`` gives a *different but still
# reproducible* realization. Splitting the seeds is handy for ablation tests —
# hold the path controls fixed and vary node randomness, or vice versa. Equal
# explicit role seeds may initialize equal streams; final fields also depend
# on path order and previously simulated values.
#
# An omitted master seed or NaN continues the sequence; an integer resets it,
# and call-time ``seed=None`` resets from fresh entropy. For path/node controls,
# omitted, NaN, or None derive a stream from the master.

ds = gs.mps.DirectSampling(gs.mps.MPSModel(ti, scan_fraction=0.1, threshold=None))
field_a = ds(grid, path_seed=1, node_seed=2, num_threads=4)
field_b = ds(grid, path_seed=1, node_seed=2, num_threads=4)
field_c = ds(grid, path_seed=9, node_seed=2, num_threads=4)
print(
    f"same path_seed/node_seed reproduce: {np.array_equal(field_a, field_b)}"
)
print(
    f"different path_seed changes result: {not np.array_equal(field_a, field_c)}"
)
assert np.array_equal(field_a, field_b)
assert np.isin(field_a, np.unique(ti_arr)).all()
