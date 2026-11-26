r"""
Large 3D multi-facies channel simulation
----------------------------------------

This example uses a five-facies 3D channel training image prepared from the
GAIA-UNIL ``ti_fluvsim_big_channels3D`` volume. The full source volume is much
larger than the simulation grid, so the bundled input is a channel-rich crop
that keeps all five facies while remaining compact enough for the documentation
repository.

Rare facies can disappear in a purely unconditional simulation when one
background facies dominates the TI. To keep the multi-facies structure visible,
this example samples sparse hard data from a nearest-neighbor rescaling of the
TI crop and conditions the Direct Sampling realization on those values.

The bundled output was regenerated on a 64x64x48 grid with 1020 conditioning
points, ``n_neighbors=100``, ``max_radius=20``, ``scan_fraction=0.5``,
``threshold=0.0``, ``cond_weight=3``, seed 11, and four threads. The simulation
took approximately **41 minutes on our machine with four threads** in the
previous full regeneration. Runtime depends on the hardware and installed
backend. Normal gallery execution displays the bundled precomputed figures.
You can reproduce the simulation and regenerate all four figures by running
this command from the repository root in an environment with GSTools and its
Rust backend installed:

.. code-block:: bash

    GSTOOLS_GENERATE_EXAMPLE_OUTPUT=1 PYTHONPATH=src \
      python examples/13_mps/13_3d_fluvsim_multifacies.py

Regeneration writes the figures into ``examples/13_mps/output/``. The full
settings are retained to preserve the five-facies structure; the faster tuning
candidates distorted facies proportions and secondary-facies geometry.

.. note::

    **Data source / license.** The bundled training image is stored in
    ``input/gaia_unil_fluvsim_5facies_crop.npz`` and was prepared from
    ``MPS_book_data/Part2/ti_fluvsim_big_channels3D.zip`` in the
    `GAIA-UNIL trainingimages repository
    <https://github.com/GAIA-UNIL/trainingimages>`_, distributed under
    **GPL-3.0**. Full redistribution notices are in ``input/LICENSE.txt``.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

import gstools as gs

example_dir = (
    Path(__file__).resolve().parent if "__file__" in globals() else Path(".")
)
data_path = example_dir / "input" / "gaia_unil_fluvsim_5facies_crop.npz"
output_dir = example_dir / "output"
output_paths = {
    "3d": output_dir / "three_dimensional_fluvsim_multifacies_3d.png",
    "slices": output_dir / "three_dimensional_fluvsim_multifacies_slices.png",
    "proportions": output_dir
    / "three_dimensional_fluvsim_multifacies_proportions.png",
    "correlation": output_dir
    / "three_dimensional_fluvsim_multifacies_two_point_correlation.png",
}

FACIES_COLORS = ["#d8c69f", "#1f78b4", "#74a661", "#b85c38", "#7b61a8"]
FACIES_LABELS = [
    "0 background",
    "1 channel facies",
    "2 margin facies",
    "3 secondary facies",
    "4 rare facies",
]
FACIES_CMAP = ListedColormap(FACIES_COLORS)
SIMULATION_SHAPE = (64, 64, 48)
COND_COUNTS = {0: 200, 1: 320, 2: 200, 3: 200, 4: 100}
N_NEIGHBORS = 100
MAX_RADIUS = 20
SCAN_FRACTION = 0.5
THRESHOLD = 0.0
COND_WEIGHT = 3
SEED = 11
NUM_THREADS = 4
VALIDATION_MAX_LAG = 24


###############################################################################
# **Small display and plotting helpers.**
#
# These helpers keep the gallery body compact: each section below either saves
# one figure when regeneration is requested, or displays the corresponding
# bundled PNG.
def show_saved_figure(path, width=10.5):
    """Display a precomputed PNG in the gallery."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing precomputed figure: {path}. "
            "Set GSTOOLS_GENERATE_EXAMPLE_OUTPUT=1 to create it."
        )
    image = plt.imread(path)
    height = width * image.shape[0] / image.shape[1]
    fig, ax = plt.subplots(figsize=(width, height))
    ax.imshow(image)
    ax.axis("off")
    fig.tight_layout(pad=0)


def save_figure(fig, path):
    """Save one gallery figure and close the Matplotlib object."""
    output_dir.mkdir(exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    print(f"Saved {path}.")
    plt.close(fig)


def facies_handles():
    """Legend handles shared by the slice and bar-plot figures."""
    return [
        Patch(facecolor=color, edgecolor="0.35", label=label)
        for color, label in zip(FACIES_COLORS, FACIES_LABELS)
    ]


def decimate_volume(volume, step):
    """Return a smaller volume for responsive Matplotlib voxel plotting."""
    return volume[::step, ::step, ::step]


def cutaway_mask(shape):
    """Hide one corner so internal facies geometry stays visible."""
    ix, iy, iz = np.indices(shape)
    return (
        (ix > 0.52 * shape[0])
        & (iy < 0.48 * shape[1])
        & (iz > 0.42 * shape[2])
    )


def plot_multifacies_voxels(ax, volume, title):
    """Plot all non-background facies in a cutaway 3D voxel view."""
    active = (volume > 0) & ~cutaway_mask(volume.shape)
    colors = np.empty(volume.shape, dtype=object)
    colors[:] = "#00000000"
    for facies, color in enumerate(FACIES_COLORS):
        if facies == 0:
            continue
        colors[(volume == facies) & active] = color + "d0"
    ax.voxels(
        active,
        facecolors=colors,
        edgecolor="#24313a44",
        linewidth=0.015,
    )
    ax.set_title(title, pad=8)
    ax.set_box_aspect(volume.shape)
    ax.set_axis_off()
    ax.set_proj_type("ortho")
    ax.view_init(elev=22, azim=-48)
    ax.set_xlim(0, volume.shape[0])
    ax.set_ylim(0, volume.shape[1])
    ax.set_zlim(0, volume.shape[2])


def center_slices(volume):
    """Return orthogonal centre slices from a 3D array."""
    cx, cy, cz = (size // 2 for size in volume.shape)
    return [
        (volume[cx, :, :].T, f"x={cx}", "x", cx),
        (volume[:, cy, :].T, f"y={cy}", "y", cy),
        (volume[:, :, cz].T, f"z={cz}", "z", cz),
    ]


def overlay_conditioning(
    ax, cond_idx, cond_val, axis_name, slice_coord, window=1
):
    """Overlay conditioning points that lie near an orthogonal slice."""
    if cond_idx is None or cond_val is None:
        return
    axis = {"x": 0, "y": 1, "z": 2}[axis_name]
    near = np.abs(cond_idx[:, axis] - slice_coord) <= window
    if not np.any(near):
        return
    coords = cond_idx[near]
    values = cond_val[near].astype(int)
    if axis_name == "x":
        xs, ys = coords[:, 1], coords[:, 2]
    elif axis_name == "y":
        xs, ys = coords[:, 0], coords[:, 2]
    else:
        xs, ys = coords[:, 0], coords[:, 1]
    for facies in np.unique(values):
        mask = values == facies
        ax.scatter(
            xs[mask],
            ys[mask],
            s=18,
            marker="o",
            facecolors="none",
            edgecolors=FACIES_COLORS[int(facies)],
            linewidths=1.1,
        )


def plot_slices(row_axes, volume, prefix, cond_idx=None, cond_val=None):
    """Plot the three centre slices of a categorical volume."""
    for ax, (slc, title, axis_name, slice_coord) in zip(
        row_axes, center_slices(volume)
    ):
        ax.imshow(
            slc,
            cmap=FACIES_CMAP,
            origin="lower",
            interpolation="nearest",
            vmin=0,
            vmax=4,
        )
        if cond_idx is not None:
            overlay_conditioning(
                ax, cond_idx, cond_val, axis_name, slice_coord
            )
            ax.set_title(f"{prefix} {title}; hard data +/-1 cell")
        else:
            ax.set_title(f"{prefix} {title}")
        ax.set_xticks([])
        ax.set_yticks([])


def facies_proportions(volume):
    """Return facies proportions for the configured facies codes."""
    values = np.asarray(volume, dtype=int).ravel()
    counts = np.bincount(values, minlength=len(FACIES_LABELS)).astype(float)
    return counts[: len(FACIES_LABELS)] / counts[: len(FACIES_LABELS)].sum()


def two_point_correlation_x(volume, facies, max_lag):
    """Indicator two-point correlation along x, normalized by lag zero."""
    indicator = volume == facies
    values = []
    for lag in range(max_lag + 1):
        if lag == 0:
            corr = indicator.mean()
        else:
            corr = (indicator[:-lag, :, :] & indicator[lag:, :, :]).mean()
        values.append(corr)
    values = np.asarray(values, dtype=float)
    if values[0] > 0:
        values /= values[0]
    return values


def rescale_ti_nearest(ti_data, shape):
    """Nearest-neighbor resampling used only to choose conditioning data."""
    idx = [
        np.linspace(0, ti_data.shape[axis] - 1, shape[axis])
        .round()
        .astype(int)
        for axis in range(3)
    ]
    return ti_data[np.ix_(*idx)]


def sample_conditioning_from_ti(ti_data, shape, seed=11):
    """Sample deterministic hard data from all facies in a resized TI crop."""
    target = rescale_ti_nearest(ti_data, shape)
    rng = np.random.default_rng(seed)
    cond_idx = []
    cond_val = []
    for facies, count in COND_COUNTS.items():
        coords = np.argwhere(target == facies)
        selected = coords[rng.choice(len(coords), size=count, replace=False)]
        cond_idx.append(selected)
        cond_val.append(np.full(count, facies, dtype=float))
    cond_idx = np.vstack(cond_idx)
    cond_val = np.concatenate(cond_val)
    return cond_idx, cond_val


def build_pyvista_facies_mesh(volume):
    """Build a PyVista volume and non-background mesh for local exploration."""
    import pyvista as pv

    grid = pv.ImageData(dimensions=np.array(volume.shape) + 1)
    grid.cell_data["facies"] = volume.ravel(order="F")
    channel_mesh = grid.threshold([0.5, 4.5], scalars="facies")
    return grid, channel_mesh


###############################################################################
# **Load the five-facies 3D channel training image.**
#
# The bundled array is a crop from the larger GAIA-UNIL Fluvsim SGEMS volume.
# The crop origin and source shape are stored inside the NPZ for provenance.
if not data_path.exists():
    raise FileNotFoundError(
        f"Missing bundled training image: {data_path}. "
        "Run this example from examples/13_mps or restore the input assets."
    )

with np.load(data_path) as data:
    ti_data = data["facies"].astype(int)

assert ti_data.shape == (120, 120, 60)
assert set(np.unique(ti_data)) == {0, 1, 2, 3, 4}

###############################################################################
# **Configure the conditioned 3D data event.**
#
# The five facies are unbalanced in the TI. Sparse hard data sampled from all
# facies keeps the rare categories present in the realization while the MPS
# data event still controls the connected 3D channel patterns.
ti = gs.TrainingImage(
    ti_data,
    categorical=True,
    n_neighbors=N_NEIGHBORS,
    max_radius=MAX_RADIUS,
    cond_weight=COND_WEIGHT,
)
model = gs.MPSModel(
    ti,
    scan_fraction=SCAN_FRACTION,
    threshold=THRESHOLD,
    boundary="partial",
)
ds = gs.DirectSampling(model)

###############################################################################
# **Prepare sparse hard data.**
#
# The hard data are deterministic samples from all facies in a resized copy of
# the TI crop. They keep rare facies visible and give the lower simulation
# slices meaningful markers, similar to the validation-oriented figures used in
# applied MPS papers.
cond_idx, cond_val = sample_conditioning_from_ti(
    ti_data, SIMULATION_SHAPE, seed=SEED
)
ds.set_condition(
    [
        cond_idx[:, 0].astype(float),
        cond_idx[:, 1].astype(float),
        cond_idx[:, 2].astype(float),
    ],
    cond_val,
)

grid = [np.arange(size, dtype=float) for size in SIMULATION_SHAPE]

###############################################################################
# **Optional PyVista mesh construction.**
#
# PyVista is convenient for interactive local exploration of the five-facies
# volume. The helper below mirrors PyVista's ``ImageData`` workflow but is
# disabled by default because off-screen VTK rendering is not reliable in all
# documentation environments.
use_pyvista = False
if use_pyvista:
    pv_grid, channel_mesh = build_pyvista_facies_mesh(ti_data)
    # channel_mesh.plot(show_edges=False)

###############################################################################
# **Run Direct Sampling, or reuse the saved figures.**
#
# The setup above always runs so the example remains readable. The simulation
# runs only when the environment variable below is set.
generate_output = os.environ.get("GSTOOLS_GENERATE_EXAMPLE_OUTPUT") == "1"
if generate_output:
    print(
        "Simulating five-facies Fluvsim field "
        f"{SIMULATION_SHAPE} with {cond_val.size} conditioning points..."
    )
    field = np.rint(ds(grid, seed=SEED, num_threads=NUM_THREADS)).astype(int)
    assert field.shape == SIMULATION_SHAPE
    assert set(np.unique(field)) == {0, 1, 2, 3, 4}
    assert np.all(
        field[cond_idx[:, 0], cond_idx[:, 1], cond_idx[:, 2]] == cond_val
    )


###############################################################################
# **3D comparison.**
#
# The 3D view shows the connected channel bodies directly. Background cells are
# hidden so the multi-facies structures are easier to see.
def plot_3d_views(ti_data, field):
    """Compare cutaway voxel renderings of the TI and simulation."""
    fig = plt.figure(figsize=(11.2, 5.5), constrained_layout=True)
    ax_ti = fig.add_subplot(1, 2, 1, projection="3d")
    ax_sim = fig.add_subplot(1, 2, 2, projection="3d")
    plot_multifacies_voxels(
        ax_ti,
        decimate_volume(ti_data, step=3),
        "Training image",
    )
    plot_multifacies_voxels(
        ax_sim,
        decimate_volume(field, step=2),
        "Conditional simulation",
    )
    fig.suptitle("Non-background facies in 3D", fontsize=14)
    return fig


if generate_output:
    save_figure(plot_3d_views(ti_data, field), output_paths["3d"])
show_saved_figure(output_paths["3d"], width=10.5)


###############################################################################
# **Orthogonal slices.**
#
# Matching slices make it easier to inspect local facies contacts. Conditioning
# data near the simulated slices are drawn as open circles.
def plot_slice_comparison(ti_data, field, cond_idx, cond_val):
    """Compare centre slices through the TI and conditional simulation."""
    fig = plt.figure(figsize=(11.5, 6.8), constrained_layout=True)
    layout = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.18])
    ti_axes = [fig.add_subplot(layout[0, i]) for i in range(3)]
    sim_axes = [fig.add_subplot(layout[1, i]) for i in range(3)]
    legend_ax = fig.add_subplot(layout[2, :])
    plot_slices(ti_axes, ti_data, "TI")
    plot_slices(sim_axes, field, "Simulation", cond_idx, cond_val)
    legend_ax.axis("off")
    legend_ax.legend(
        handles=facies_handles(),
        loc="center",
        ncol=len(FACIES_LABELS),
        frameon=False,
        fontsize=10,
    )
    return fig


if generate_output:
    save_figure(
        plot_slice_comparison(ti_data, field, cond_idx, cond_val),
        output_paths["slices"],
    )
show_saved_figure(output_paths["slices"], width=10.5)


###############################################################################
# **Facies proportions.**
#
# Each stacked bar is one full volume. The plot checks whether the conditional
# simulation keeps approximately the same facies mix as the training image.
def plot_proportion_comparison(ti_data, field):
    """Compare facies proportions as stacked TI/simulation bars."""
    proportions = np.vstack(
        [facies_proportions(ti_data), facies_proportions(field)]
    )
    x = np.arange(2)
    bottoms = np.zeros(2)
    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    for facies, color in enumerate(FACIES_COLORS):
        values = proportions[:, facies]
        ax.bar(
            x,
            values,
            bottom=bottoms,
            width=0.58,
            color=color,
            edgecolor="0.25",
            linewidth=0.35,
        )
        for xpos, bottom, value in zip(x, bottoms, values):
            if value >= 0.035:
                ax.text(
                    xpos,
                    bottom + value / 2,
                    f"{100 * value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="0.08",
                )
        bottoms += values
    ax.set_title("Facies proportions", fontsize=13)
    ax.set_ylabel("fraction of cells")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(["Training image", "Simulation"])
    ax.grid(axis="y", color="0.88", linewidth=0.8)
    ax.legend(
        handles=facies_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    return fig


if generate_output:
    save_figure(
        plot_proportion_comparison(ti_data, field), output_paths["proportions"]
    )
show_saved_figure(output_paths["proportions"], width=8.8)


###############################################################################
# **Two-point correlation.**
#
# The two-point correlation along x is a compact continuity check for the
# non-background facies. Solid lines are the simulation; dashed lines are the
# training image.
def plot_two_point_comparison(ti_data, field):
    """Compare TI and simulation continuity with x-direction S2 curves."""
    max_lag = min(VALIDATION_MAX_LAG, ti_data.shape[0] - 1, field.shape[0] - 1)
    lags = np.arange(max_lag + 1)
    fig, ax = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    for facies in range(1, len(FACIES_LABELS)):
        color = FACIES_COLORS[facies]
        ax.plot(
            lags,
            two_point_correlation_x(ti_data, facies, max_lag),
            linestyle="--",
            color=color,
            linewidth=1.4,
            alpha=0.85,
        )
        ax.plot(
            lags,
            two_point_correlation_x(field, facies, max_lag),
            linestyle="-",
            color=color,
            linewidth=1.8,
            label=FACIES_LABELS[facies],
        )
    ax.set_title("Two-point correlation along x", fontsize=13)
    ax.set_xlabel("lag [cells]")
    ax.set_ylabel("S2(h) / S2(0)")
    ax.set_ylim(0, 1.05)
    ax.grid(color="0.88", linewidth=0.8)
    ax.legend(
        title="solid=simulation, dashed=TI", frameon=False, fontsize=10, ncol=2
    )
    ax.tick_params(labelsize=9)
    return fig


if generate_output:
    save_figure(
        plot_two_point_comparison(ti_data, field), output_paths["correlation"]
    )
show_saved_figure(output_paths["correlation"], width=9.5)

###############################################################################
# **About these figures.**
#
# These four figures show the results of one Direct Sampling realization.
# Normal gallery execution displays the saved PNGs; it does not run the 3D
# simulation. Set ``GSTOOLS_GENERATE_EXAMPLE_OUTPUT=1`` to generate the field
# and refresh all four figures. The previous run took about 41 minutes on our
# machine with four threads.
