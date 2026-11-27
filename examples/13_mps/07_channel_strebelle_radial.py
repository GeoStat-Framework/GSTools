r"""
Radial nonstationarity
----------------------

This example follows the geometric nonstationarity idea from Mariethoz et al.
(2010), Figure 7. The Strebelle channel TI is simulated with a radial rotation
map and a distance-based anisotropy map. These maps reorient and rescale local
data-event matching so the channel patterns tend to follow the radial geometry.
This paper-based setup is also a useful check that the implementation
reproduces the expected geometry. The 1,000x1,000 simulation is a substantial
run, so it remains opt-in and can take much longer than a few seconds.

.. note::

    This example reuses the bundled Strebelle TI stored in
    ``input/strebelle_channel_ti.npz``. The data source and license are
    documented in :ref:`sphx_glr_examples_13_mps_04_channel_strebelle.py`
    and ``input/LICENSE.txt``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import gstools as gs

example_dir = (
    Path(__file__).resolve().parent if "__file__" in globals() else Path(".")
)
data_path = example_dir / "input" / "strebelle_channel_ti.npz"
output_dir = example_dir / "output"
output_path = output_dir / "radial_nonstationarity.png"


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
    fig, ax = plt.subplots(figsize=(9.5, 8))
    ax.imshow(image)
    ax.axis("off")
    fig.tight_layout(pad=0)


def plot_radial_result(ti_plot, rotation, anis, field):
    """Plot the nonstationary maps, TI crop, and regenerated realization."""
    fig = plt.figure(figsize=(9.5, 8), constrained_layout=True)
    grid_spec = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.0, 0.04, 1.0, 0.04],
        hspace=0.22,
        wspace=0.08,
    )
    ax_rot = fig.add_subplot(grid_spec[0, 0])
    cax_rot = fig.add_subplot(grid_spec[0, 1])
    ax_aff = fig.add_subplot(grid_spec[0, 2])
    cax_aff = fig.add_subplot(grid_spec[0, 3])
    ax_ti = fig.add_subplot(grid_spec[1, 0])
    ax_sim = fig.add_subplot(grid_spec[1, 2])

    im_rot = ax_rot.imshow(np.rad2deg(rotation), cmap="gray", origin="lower")
    ax_rot.set_title("a) Rotation (degrees)")
    fig.colorbar(im_rot, cax=cax_rot)

    im_aff = ax_aff.imshow(
        anis, cmap="gray", origin="lower", vmin=0.1, vmax=1.0
    )
    ax_aff.set_title("b) Affinity ratio")
    fig.colorbar(im_aff, cax=cax_aff)

    cmap = ListedColormap(["#000000", "#FFFFFF"])
    ax_ti.imshow(
        ti_plot[:250, :250],
        cmap=cmap,
        origin="lower",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    ax_ti.set_title("c) Channel facies TI")

    ax_sim.imshow(
        field,
        cmap=cmap,
        origin="lower",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    ax_sim.set_title("d) Radial DS realization")
    return fig


###############################################################################
# **Load the training image.**
#
# Load the Strebelle channel training image from the bundled NPZ file and wrap
# it in a categorical GSTools TrainingImage object.
if not data_path.exists():
    raise FileNotFoundError(
        f"Missing bundled training image: {data_path}. "
        "Run this example from examples/13_mps or restore the input directory."
    )

with np.load(data_path) as data:
    ti_arr = data["array1"].astype(float)

ti_plot = ti_arr.copy()

# GSTools' rotation/anisotropy convention aligns the primary, unscaled array
# axis with the local rotation angle. The Strebelle channels run mainly along
# the other array axis, so transposing the TI makes the radial rotation and
# anisotropy maps act on the channel direction intended for this example.
ti_arr = ti_arr.T

n_neighbors = 32
sg_size = 1000
scan_fraction = 0.35
simulation_seed = 1
ti = gs.TrainingImage(ti_arr, categorical=True, n_neighbors=n_neighbors)

###############################################################################
# **Build the nonstationary geometry.**
#
# Rotation controls the local direction of data-event matching. The affinity
# ratio controls the local anisotropy strength.
xs = np.arange(sg_size, dtype=float)
ys = np.arange(sg_size, dtype=float)
gx, gy = np.meshgrid(xs, ys, indexing="ij")

center_x = sg_size / 2.0
center_y = sg_size / 2.0

# Rotation is the angle from the center, measured in GSTools' array-axis
# convention after transposing the TI above.
rotation = np.arctan2(gy - center_y, gx - center_x)

# The anisotropy map scales the search geometry based on distance from the
# center. The affinity ratio is 1.0 in the center and lower near the corners.
radius = np.sqrt((gx - center_x) ** 2 + (gy - center_y) ** 2)
max_radius = np.sqrt(center_x**2 + center_y**2)
anis = 1.0 - 0.65 * (radius / max_radius)

###############################################################################
# **Run Direct Sampling, or reuse the saved result.**
#
# The setup above always runs so the example remains readable. This million-node
# simulation is substantial, so the default is ``False`` and normal execution
# displays the saved figure.
generate_output = False
if generate_output:
    output_dir.mkdir(exist_ok=True)
    # Absolute per-axis scale: main axis 1.0, transversal axis the anis ratio.
    scale = np.stack([np.ones_like(anis), anis], axis=-1)
    ds = gs.DirectSampling(
        gs.MPSModel(
            ti,
            scan_fraction=scan_fraction,
            threshold=0.0,
            rotation=rotation,
            scale=scale,
        )
    )

    print(f"Simulating radial nonstationary field ({sg_size}x{sg_size})...")
    field = ds([xs, ys], seed=simulation_seed, num_threads=8, path="random")

    fig = plot_radial_result(ti_plot, rotation, anis, field)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved {output_path}.")
    plt.close(fig)

###############################################################################
# **Display the saved radial result.**
#
# In normal use this loads the bundled PNG. When ``generate_output=True``, it
# displays the PNG that was just refreshed.
show_saved_figure(output_path)
