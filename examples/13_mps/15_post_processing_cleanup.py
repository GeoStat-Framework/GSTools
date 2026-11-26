r"""
Post-processing: cleanup passes
-------------------------------

A fast Direct Sampling run (small ``scan_fraction``, random path) can leave
local artefacts: early nodes use a less-informed event than later nodes.

**Post-processing** (Meerschman et al., 2013, Sect. 4) addresses this. After the
main path completes, each cleanup pass re-opens every simulated node against a
*fully informed* neighbourhood -- it now sees the values on *all* sides, not
only the ones drawn before it -- and re-draws it from the training image. Hard
conditioning components are never changed in the raw DS field. Cleanup can
change distributions and connectivity; it does not guarantee improvement of
every statistic.

Pass :any:`PostProcessing` to the model to configure the number of passes
and optionally divide the per-pass neighbourhood and scan effort.
The passes run in the Rust engine over a dependency DAG, so the result is
bit-identical across thread counts within a fixed supported build and
dependency environment. The cleanup mask restricts eligible nodes;
partially conditioned components remain fixed. Effective cleanup variables
inherit the primary settings unless explicitly overridden.

``post_processing`` is separate from call-time ``post_process=False``: the
latter disables the Field mean/normalizer/trend pipeline, not DS cleanup. A
downstream Field transformation can change displayed conditioning values.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import gstools as gs

###############################################################################
# Load the Strebelle channel training image bundled next to this example
# (see :ref:`the conditioning example <sphx_glr_examples_13_mps>` for the data
# source and license).

example_dir = (
    Path(__file__).resolve().parent if "__file__" in globals() else Path(".")
)
with np.load(example_dir / "input" / "strebelle_channel_ti.npz") as data:
    ti_arr = data["array1"].astype(int)

ti = gs.mps.TrainingImage(ti_arr, categorical=True, n_neighbors=16)
grid = [np.arange(64, dtype=float), np.arange(64, dtype=float)]

###############################################################################
# Run twice with the *same* seed: once without post-processing, once with three
# cleanup passes. Only ``post_processing`` differs, so any change in the field
# is the cleanup's doing.

common = dict(scan_fraction=0.05, threshold=0.0)
field_raw = gs.DirectSampling(gs.MPSModel(ti, **common))(grid, seed=20240520)
field_clean = gs.DirectSampling(
    gs.MPSModel(ti, post_processing=gs.mps.PostProcessing(3), **common)
)(grid, seed=20240520)


###############################################################################
# Count isolated pixels (a cell whose four neighbours all disagree with it) as a
# descriptive measure using four face-neighbors and excluding boundary cells.
# Its before/after change is an observation, not an algorithmic guarantee.


def isolated_pixels(field):
    count = 0
    for shift in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        count = count + (np.roll(field, shift, axis=(0, 1)) == field)
    # interior cells with zero agreeing neighbours
    inner = count[1:-1, 1:-1]
    return int(np.sum(inner == 0))


raw_isolated = isolated_pixels(field_raw)
clean_isolated = isolated_pixels(field_clean)
print(f"isolated pixels: raw={raw_isolated}, cleaned={clean_isolated}")

###############################################################################
# Plot the raw realization next to the cleaned one.

cmap = ListedColormap(["#c9a96e", "#2b6cb0"])  # shale / sand
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 5.5))
ax0.imshow(field_raw, cmap=cmap, origin="lower", vmin=0, vmax=1)
ax0.set_title(f"No post-processing ({raw_isolated} isolated px)")
ax1.imshow(field_clean, cmap=cmap, origin="lower", vmin=0, vmax=1)
ax1.set_title(f"3 cleanup passes ({clean_isolated} isolated px)")
fig.tight_layout()

###############################################################################
# Inspect both the visual change and the printed counts for these realizations.
# A larger ``PostProcessing.factor`` divides the requested neighbor count
# (rounded, minimum one; requested variation count at least two) and scan
# fraction; the candidate budget remains capped by the valid window. Additional
# passes can alter scientific statistics as well as isolated pixels.
# ``post_processing=0`` (the default) is a true no-op, bit-for-bit identical to
# omitting the parameter.
