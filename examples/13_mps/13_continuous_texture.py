r"""
Continuous conditioning with a bundled texture
----------------------------------------------

This compact workflow combines hard conditioning with a continuous texture. It
uses the ``stone`` continuous training image from GAIA-UNIL, samples random
hard data from that texture, and compares the conditional simulation with the
TI through both maps and histograms. The example follows the continuous
conditioning workflow in `Mariethoz et al. (2010)
<https://doi.org/10.1029/2008WR007621>`_ using a bundled input image.

The bundled figure uses a 200x200 grid, ``distance="l4"``,
``n_neighbors=75``, ``scan_fraction=0.4``, ``threshold=0.01``,
``cond_weight=2``, seed 1, four threads, and a sequential path. For a more
expensive DSBC-style experiment, try ``threshold=None`` and a larger
neighborhood in the regeneration settings.

.. note::

    This example loads the bundled derived texture
    ``input/gaia_unil_stone_texture.npz``. It was prepared from
    ``stone.tiff`` in the GAIA-UNIL training-image collection. **This data
    file is licensed under GPL-3.0**; the licence text is in
    ``input/GAIA-UNIL-GPL-3.0.txt`` and the full source and redistribution
    notices are in ``input/LICENSE.txt``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import gstools as gs

example_dir = (
    Path(__file__).resolve().parent if "__file__" in globals() else Path(".")
)
data_path = example_dir / "input" / "gaia_unil_stone_texture.npz"
output_dir = example_dir / "output"
output_path = output_dir / "continuous_stone_conditioning.png"

###############################################################################
# **Small display and plotting helpers.**
#
# Functions to display the saved figure and plot a regenerated result.
def show_saved_figure(path):
    """Display a precomputed PNG in the gallery."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing precomputed figure: {path}. "
            "Set generate_output = True to create it."
        )
    image = plt.imread(path)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(image)
    ax.axis("off")
    fig.tight_layout(pad=0)


def plot_continuous_result(ti_data, field, cond_x, cond_y, cond_val):
    """Plot the TI, conditional simulation, and marginal histograms."""
    fig = plt.figure(figsize=(12, 10))
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(212)

    ax1.imshow(ti_data.T, cmap="gray", origin="lower", vmin=0, vmax=1.0)
    ax1.set_title("a) Training image")

    im_b = ax2.imshow(field.T, cmap="gray", origin="lower", vmin=0, vmax=1.0)
    ax2.scatter(
        cond_x,
        cond_y,
        c=cond_val,
        cmap="gray",
        vmin=0,
        vmax=1.0,
        edgecolors="k",
        s=40,
        linewidths=0.5,
    )
    ax2.set_title("b) Simulation")
    plt.colorbar(im_b, ax=ax2, fraction=0.046, pad=0.04)

    hist_ti, bins = np.histogram(ti_data.flatten(), bins=50, density=True)
    hist_sim, _ = np.histogram(field.flatten(), bins=bins, density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    ax3.plot(bin_centers, hist_sim, "k-", label="Simulation")
    ax3.plot(bin_centers, hist_ti, "k--", label="Training image")
    ax3.set_title("c) Comparison of histograms")
    ax3.legend()

    fig.tight_layout()
    return fig


###############################################################################
# **Load the bundled continuous training image.**
#
# The texture is stored locally so the example does not download data during
# normal execution or regeneration.
if not data_path.exists():
    raise FileNotFoundError(
        f"Missing bundled training image: {data_path}. "
        "Run this example from examples/13_mps or restore the input assets."
    )

with np.load(data_path) as data:
    ti_data = data["texture"].astype(float)

grid_size = ti_data.shape[0]

###############################################################################
# **Build the continuous training image.**
#
# ``distance="l4"`` compares continuous data events using the fourth-power
# distance used for this texture example.
ti = gs.mps.TrainingImage(
    ti_data,
    categorical=False,
    distance="l4",
    n_neighbors=75,
    cond_weight=2,
)

###############################################################################
# **Sample hard conditioning values.**
#
# Conditioning locations are random in the simulation grid, while values are
# sampled from the training image marginal distribution.
rng = np.random.RandomState(3)
n_cond = 10
cond_x = rng.uniform(0, grid_size, n_cond)
cond_y = rng.uniform(0, grid_size, n_cond)
rand_ti_x = rng.randint(0, grid_size, n_cond)
rand_ti_y = rng.randint(0, grid_size, n_cond)
cond_val = ti_data[rand_ti_x, rand_ti_y]

model = gs.mps.MPSModel(ti, scan_fraction=0.4, threshold=0.01)

ds = gs.mps.DirectSampling(model)
ds.set_condition([cond_x, cond_y], cond_val)

x = y = np.arange(grid_size, dtype=float)

###############################################################################
# **Run Direct Sampling, or reuse the saved result.**
#
# The training-image/model setup above runs, but the simulation stays opt-in.
# Set this to True to regenerate the result (several minutes); normal gallery
# execution displays the saved figure.
generate_output = False
if generate_output:
    output_dir.mkdir(exist_ok=True)
    print(
        f"Simulating continuous field ({grid_size}x{grid_size}) "
        f"with {n_cond} conditioning points..."
    )
    field = ds([x, y], seed=1, num_threads=4, path="sequential")
    assert np.isfinite(field).all()
    # Hard data are honoured at their nearest grid node.
    ix = np.clip(np.rint(cond_x).astype(int), 0, grid_size - 1)
    iy = np.clip(np.rint(cond_y).astype(int), 0, grid_size - 1)
    _, first = np.unique(ix * grid_size + iy, return_index=True)
    assert np.allclose(field[ix[first], iy[first]], cond_val[first])

    fig = plot_continuous_result(ti_data, field, cond_x, cond_y, cond_val)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved {output_path}.")
    plt.close(fig)

###############################################################################
# **Display the saved continuous result.**
#
# In normal use this loads the bundled PNG. When regeneration is enabled, it
# displays the PNG that was just refreshed.
show_saved_figure(output_path)

###############################################################################
# The conditional realization follows the broad texture of the TI and honors
# all sampled values. The histogram comparison checks whether the simulation
# retains a similar marginal distribution.
