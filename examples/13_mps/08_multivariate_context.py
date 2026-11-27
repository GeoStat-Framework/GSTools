r"""
Multivariate channels with a context variable
---------------------------------------------

This page introduces multivariate MPS with a categorical channel variable and a
continuous context variable. The context field is provided everywhere on the
simulation grid, so Direct Sampling can use it as exhaustive secondary data
while simulating the channels.

This setup follows the multivariate example in
`Mariethoz et al. (2010) <https://doi.org/10.1029/2008WR007621>`_ and also
serves as a reproducibility example for the implementation.

**Only the channel variable is simulated here** — the context variable is
*known everywhere* and merely guides the match. Later in this gallery, the
:ref:`bivariate joint simulation example <sphx_glr_examples_13_mps_11_bivariate_joint.py>`
has no exhaustive secondary field and simulates **both** variables together.

The bundled figure uses a 250x250 grid, 30 channel neighbors, one context
neighbor, ``scan_fraction=0.5``, ``threshold=0.01``, seed 42, and four threads
(bit-identical to the original serial run).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import gstools as gs

example_dir = (
    Path(__file__).resolve().parent if "__file__" in globals() else Path(".")
)
data_path = example_dir / "input" / "synthetic_channel_context_ti.npz"
output_dir = example_dir / "output"
output_path = output_dir / "multivariate_context_channels.png"


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


def plot_multivariate_result(
    channels_ti, context_ti, channels_sim, context_sim
):
    """Plot the TI variables beside the simulated channels and context data."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].imshow(channels_ti, cmap="gray", origin="lower")
    axes[0, 0].set_title("a) Channel facies TI")

    axes[0, 1].imshow(context_ti, cmap="gray", origin="lower", vmin=0, vmax=1)
    axes[0, 1].set_title("b) Context field TI")

    axes[1, 0].imshow(channels_sim, cmap="gray", origin="lower")
    axes[1, 0].set_title("c) Simulated channel facies")

    axes[1, 1].imshow(context_sim, cmap="gray", origin="lower", vmin=0, vmax=1)
    axes[1, 1].set_title("d) Exhaustive context data")

    fig.tight_layout()
    return fig


###############################################################################
# **Load the prepared multivariate training image.**
#
# The input file contains the categorical channel TI and a continuous context
# field with matching dimensions.
if not data_path.exists():
    raise FileNotFoundError(
        f"Missing bundled training image: {data_path}. "
        "Run this example from examples/13_mps or restore the input assets."
    )

with np.load(data_path) as data:
    channels_ti = data["channels"]
    context_ti = data["context"]

###############################################################################
# **Assemble the multivariate training image.**
#
# ``channels`` is the simulated categorical variable. ``context`` is a
# continuous secondary variable that guides channel placement.
ti = gs.TrainingImage(
    [
        gs.mps.Variable(
            "channels",
            channels_ti,
            categorical=True,
            weight=0.5,
            n_neighbors=30,
        ),
        gs.mps.Variable(
            "context",
            context_ti,
            categorical=False,
            weight=0.5,
            distance="l1",
            n_neighbors=1,
        ),
    ]
)

model = gs.MPSModel(ti, scan_fraction=0.5, threshold=0.01)

###############################################################################
# **Provide exhaustive context data.**
#
# The channel variable is unknown everywhere, while the context variable is
# known at every simulation node.
sg_size = 250
xs_sg = np.arange(sg_size, dtype=float)
ys_sg = np.arange(sg_size, dtype=float)
gx_sg, gy_sg = np.meshgrid(xs_sg, ys_sg, indexing="ij")
# Use a vertical context gradient on the simulation grid, so the x-dependent
# TI orientation is replayed from bottom to top.
context_sg = gy_sg / float(sg_size - 1)

cond_pos = [gx_sg.flatten(), gy_sg.flatten()]
cond_val = {
    "channels": np.full(sg_size * sg_size, np.nan),
    "context": context_sg.flatten(),
}

ds = gs.DirectSampling(model)
ds.set_condition(cond_pos, cond_val)

###############################################################################
# **Run Direct Sampling, or reuse the saved result.**
#
# The setup above always runs so the example remains readable. The simulation
# runs only when ``generate_output`` is set to ``True``. The default
# is ``False`` so normal execution displays the saved figure.
generate_output = False
if generate_output:
    output_dir.mkdir(exist_ok=True)
    print(f"Simulating multivariate field ({sg_size}x{sg_size})...")
    fields = ds([xs_sg, ys_sg], seed=42, num_threads=4)
    channels_sim = fields["channels"]
    context_sim = fields["context"]

    fig = plot_multivariate_result(
        channels_ti, context_ti, channels_sim, context_sim
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved {output_path}.")
    plt.close(fig)

###############################################################################
# **Display the saved multivariate result.**
#
# In normal use this loads the bundled PNG. When regeneration is enabled, it
# displays the PNG that was just refreshed.
show_saved_figure(output_path)

###############################################################################
# The simulated channels follow the spatial context while retaining patterns
# from the TI. The context field itself is copied from the exhaustive input,
# which distinguishes this workflow from joint simulation.
