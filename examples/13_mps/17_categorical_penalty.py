r"""
Categorical penalty matrix: soft facies distances
--------------------------------------------------

Plain categorical Direct Sampling treats every facies mismatch the same: a node
that "wants" facies 1 pays the identical penalty whether the candidate offers
facies 0 or facies 2. For *ordered* facies (e.g. a transition sequence
mud -> sandy-mud -> sand) that is too blunt -- a near-miss should cost less than
a jump across the whole sequence.

A per-:class:`~gstools.mps.Variable` **penalty matrix** ``T`` supplies those
soft costs. During the scan, a candidate whose facies is ``ti`` where the data
event wants ``sim`` contributes ``T[sim, ti]`` instead of a hard 0/1 mismatch.
The diagonal is zero (an exact match is free) and off-diagonal entries lie in
``(0, 1]``. The matrix may be asymmetric; the orientation is ``T[sim, ti]``.

Here we build an ordinal 3-facies TI and compare a plain run with one that makes
mismatches between nearby ordinal labels cheap and 0<->2 mismatches expensive.
These compare corresponding data-event values; they do not prohibit adjacent
category pairs in the simulated grid.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import gstools as gs

###############################################################################
# A smooth low-frequency field quantized into three ordinal facies (0, 1, 2),
# so facies transition gradually and a direct 0<->2 contact is rare.

rng = np.random.default_rng(3)
n = 80
yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
smooth = np.zeros((n, n))
for _ in range(6):  # a few random low-frequency Fourier modes
    kx, ky = rng.integers(1, 4, size=2)
    phase = rng.uniform(0, 2 * np.pi)
    smooth += np.sin(2 * np.pi * (kx * xx + ky * yy) / n + phase)
edges = np.quantile(smooth, [1 / 3, 2 / 3])
ti_arr = np.digitize(smooth, edges).astype(int)  # values in {0, 1, 2}

###############################################################################
# Plain categorical variable vs. one carrying an ordinal penalty matrix. Only
# ``penalty_matrix`` differs between the two training images.

plain_var = gs.mps.Variable(None, ti_arr, categorical=True, n_neighbors=16)
# Mismatching labels one level apart is cheap; a 0<->2 event mismatch has cost 1.
penalty = np.array(
    [
        [0.0, 0.3, 1.0],
        [0.3, 0.0, 0.3],
        [1.0, 0.3, 0.0],
    ]
)
penalty_var = gs.mps.Variable(
    None, ti_arr, categorical=True, n_neighbors=16, penalty_matrix=penalty
)

grid = [np.arange(60, dtype=float), np.arange(60, dtype=float)]
kw = dict(scan_fraction=0.2, threshold=0.0)
field_plain = gs.DirectSampling(
    gs.MPSModel(gs.mps.TrainingImage(plain_var), **kw)
)(grid, seed=11)
field_penalty = gs.DirectSampling(
    gs.MPSModel(gs.mps.TrainingImage(penalty_var), **kw)
)(grid, seed=11)

# Compare the realizations; distinct settings need not yield distinct fields.
print("fields differ:", not np.array_equal(field_plain, field_penalty))


###############################################################################
# Count face-adjacent contacts two facies apart (a 0 touching a 2). This matrix
# changes event mismatch costs, not adjacency energies. The counts describe
# these particular realizations; assess direction and magnitude over an ensemble.


def ordinal_contacts(field):
    v = 0
    for shift in ((1, 0), (0, 1)):
        v += int(np.sum(np.abs(np.diff(field, axis=shift[0])) == 2))
    return v


print(
    "0<->2 contacts:",
    f"plain={ordinal_contacts(field_plain)}",
    f"penalty={ordinal_contacts(field_penalty)}",
)

###############################################################################
# Plot the TI and both realizations.

cmap = ListedColormap(["#edf8b1", "#7fcdbb", "#2c7fb8"])
fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
for ax, img, title in zip(
    axes,
    (ti_arr[:60, :60], field_plain, field_penalty),
    ("Ordinal TI (crop)", "Plain categorical", "With penalty matrix"),
):
    ax.imshow(img, cmap=cmap, origin="lower", vmin=0, vmax=2)
    ax.set_title(title)
fig.tight_layout()

###############################################################################
# The cost matrix makes some category mismatches less important than others.
# Spatial transition statistics can change indirectly; no adjacent pair is
# forbidden and fewer 0<->2 contacts are not guaranteed. An all-ones off-diagonal
# matrix reproduces plain 0/1 mismatch costs exactly.
