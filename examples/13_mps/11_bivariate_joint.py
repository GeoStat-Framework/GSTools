r"""
Bivariate joint simulation from a channel TI
--------------------------------------------

The context-variable page used a secondary variable as exhaustive conditioning
data. Here both variables are simulated jointly: a categorical facies variable
from the Strebelle image and a continuous secondary variable derived from a
smoothed version of the same image. This page follows the bivariate experiment
in `Mariethoz et al. (2010) <https://doi.org/10.1029/2008WR007621>`_ and
provides another check that the implementation reproduces the expected joint
patterns. See the
:ref:`context-variable example <sphx_glr_examples_13_mps_08_multivariate_context.py>`
for the case where the secondary field is known everywhere and only facies are
simulated.

The bundled figure uses a 250x250 grid, 30 neighbors for each variable,
``scan_fraction=0.5``, ``threshold=0.01``, seed 123, and four threads
(bit-identical to the original serial run). For a more expensive DSBC-style
experiment, try increasing both neighborhoods to 60 and
setting ``threshold=0.0`` in the regeneration settings.

.. note::

    This example uses the bundled derived input
    ``input/strebelle_facies_resistivity_ti.npz``. It contains the 250x250 Strebelle
    subset used in this example, plus a smoothed/noisy secondary
    variable. The source data and license are documented in
    :ref:`sphx_glr_examples_13_mps_04_channel_strebelle.py` and
    ``input/LICENSE.txt``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import gstools as gs

example_dir = (
    Path(__file__).resolve().parent if "__file__" in globals() else Path(".")
)
data_path = example_dir / "input" / "strebelle_facies_resistivity_ti.npz"
output_dir = example_dir / "output"
output_path = output_dir / "bivariate_joint_channels.png"


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
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    ax.axis("off")
    fig.tight_layout(pad=0)


def plot_bivariate_result(ti_var1, ti_var2, sim_var1, sim_var2):
    """Plot each TI variable next to its simulated counterpart."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].imshow(ti_var1, cmap="gray", origin="lower")
    axes[0, 0].set_title("a) Channel facies TI")

    im_b = axes[0, 1].imshow(
        ti_var2, cmap="gray", origin="lower", vmin=0, vmax=1.5
    )
    axes[0, 1].set_title("b) Resistivity TI")
    plt.colorbar(im_b, ax=axes[0, 1], fraction=0.046, pad=0.04)

    axes[1, 0].imshow(sim_var1, cmap="gray", origin="lower")
    axes[1, 0].set_title("c) Simulated channel facies")

    im_d = axes[1, 1].imshow(
        sim_var2, cmap="gray", origin="lower", vmin=0, vmax=1.5
    )
    axes[1, 1].set_title("d) Simulated resistivity")
    plt.colorbar(im_d, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    return fig


###############################################################################
# **Load the bundled bivariate training variables.**
#
# The input contains the Strebelle facies subset plus the derived continuous
# resistivity-like variable.
if not data_path.exists():
    raise FileNotFoundError(
        f"Missing bundled training image: {data_path}. "
        "Run this example from examples/13_mps or restore the input assets."
    )

with np.load(data_path) as data:
    ti_var1_full = data["facies"].astype(int)
    ti_var2_full = data["resistivity"].astype(float)

grid_size = 250
ti_var1 = ti_var1_full[:grid_size, :grid_size]
ti_var2 = ti_var2_full[:grid_size, :grid_size]

###############################################################################
# **Assemble the joint training image.**
#
# Both variables are simulated together, so each data event can compare
# categorical channel facies and the continuous resistivity variable.
ti = gs.TrainingImage(
    [
        gs.mps.Variable(
            "facies",
            ti_var1,
            categorical=True,
            weight=0.5,
            distance="l1",
            n_neighbors=30,
        ),
        gs.mps.Variable(
            "resistivity",
            ti_var2,
            categorical=False,
            weight=0.5,
            distance="l2",
            n_neighbors=30,
        ),
    ]
)

model = gs.MPSModel(ti, scan_fraction=0.5, threshold=0.01)
ds = gs.DirectSampling(model)

x = y = np.arange(grid_size, dtype=float)

###############################################################################
# **Run Direct Sampling, or reuse the saved result.**
#
# The setup above always runs so the example remains readable. The expensive
# simulation runs only when ``generate_output`` is set to ``True``. The default
# is ``False`` so normal execution displays the saved figure.
generate_output = False
if generate_output:
    output_dir.mkdir(exist_ok=True)
    print(
        f"Simulating unconditional bivariate field ({grid_size}x{grid_size})..."
    )
    fields = ds([x, y], seed=123, num_threads=4)

    sim_var1 = fields["facies"]
    sim_var2 = fields["resistivity"]

    fig = plot_bivariate_result(ti_var1, ti_var2, sim_var1, sim_var2)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved {output_path}.")
    plt.close(fig)

###############################################################################
# **Display the saved bivariate result.**
#
# In normal use this loads the bundled PNG. When regeneration is enabled, it
# displays the PNG that was just refreshed.
show_saved_figure(output_path)

###############################################################################
# The two simulated fields share a TI anchor at each grid node, preserving
# their spatial relationship. Both variables retain visible structure from
# their corresponding training variables.
