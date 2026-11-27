r"""
Herten outcrop: multi-facies simulation with a rotation gradient
------------------------------------------------------------------------

This example uses the supplied Herten section S1 facies map as a training image.
The left-hand crop contains eight of the ten facies in the full section. We
compare two unconditional simulations with identical settings and seed: one with
stationary search geometry and one with an x-dependent rotation map. Rotation
changes the local simulation-to-training-image lag mapping; it does not rotate
the input raster or guarantee a particular dip for every simulated boundary.
This is one categorical variable, ``facies``, with several labels, rather than
multivariate co-simulation. This example changes rotation only. Separate examples
demonstrate scale.

The angle ranges from 0 to -12 degrees across the x samples of the 8 m-wide crop.
The positive threshold 0.05 accepts approximate matches; the library default
remains zero (best candidate within budget). See :ref:`mps-contracts`. One cleanup
pass revisits simulated nodes using their current neighborhoods; see
:ref:`sphx_glr_examples_13_mps_15_post_processing_cleanup.py`.
Continue with :ref:`sphx_glr_examples_13_mps_07_channel_strebelle_radial.py`
for radial geometry, then :ref:`sphx_glr_examples_13_mps_08_multivariate_context.py`
for multiple variables.

**The default preview displays a curated figure from a prior simulation.** It
uses no network and performs no new numerical validation. To regenerate both
22,400-cell simulations explicitly, run from any working directory::

    MPLBACKEND=Agg GSTOOLS_MPS_HERTEN_REGENERATE=1 \
      GSTOOLS_MPS_HERTEN_OUTPUT_DIR=/tmp/new-herten-result \
      python examples/13_mps/19_herten_nonstationary.py

The output directory must be new. ``GSTOOLS_MPS_HERTEN_THREADS`` optionally changes
the requested two threads and is recorded in provenance. Regeneration may take
several minutes. It writes validated raw fields, a figure, and a manifest with
parameters, source and binary identity. Fixed-seed repetition is supported within
a fixed build/dependency environment, without claiming portable byte identity or
that the two fields are geometric warps of each other.

The data are a supplied compressed NumPy representation of a digitized outcrop,
not a reproduction of the publication's full 3D study. Bayer, Comunian, Höyng,
and Mariethoz (2015), *Physicochemical properties and 3D geostatistical simulations
of the Herten and the Descalvado aquifer analogs*,
`PANGAEA <https://doi.pangaea.de/10.1594/PANGAEA.844167>`_, CC BY 3.0.
Related article: *High resolution multi-facies realizations of sedimentary
reservoir and aquifer analogs*, Scientific Data 2, 150033,
`doi:10.1038/sdata.2015.33 <https://doi.org/10.1038/sdata.2015.33>`_.
The section-to-NPZ conversion was supplied, not independently reconstructed.

For a downloaded script, retain the ``input/`` and ``output/`` directories beside
it. The gallery ZIP includes these assets. Individual inputs are available as
:download:`Herten NPZ <input/herten_s1_section.npz>` and
:download:`data attribution <input/HERTEN-CC-BY-3.0.txt>`.
Local cell extents are plotted in metres, without georeferenced coordinates or
an inferred geological name-to-code legend.
"""

import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

import gstools as gs

# Sphinx-Gallery executes blocks without __file__ from the example directory.
EXAMPLE_DIR = (
    Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
)
INPUT_HASH = "c88baaeff13f58539889477180fcf685107dcf78d9e6561308d04f196d1fcc74"
SEED = 13
CONTRACT = "ds-seedsequence-mt19937-v1"


def file_hash(path):
    """SHA-256 identity of a local artifact."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_input():
    """Read the unchanged supplied [x, z] section and select its left crop."""
    path = EXAMPLE_DIR / "input/herten_s1_section.npz"
    if file_hash(path) != INPUT_HASH:
        raise ValueError("Herten input checksum mismatch")
    with np.load(path, allow_pickle=False) as data:
        section = data["section"]
        if section.shape != (320, 140) or section.dtype != np.int8:
            raise ValueError("Unexpected Herten section layout")
        spacing = float(data["spacing"])
        crop = section[:160, :].copy()
    return crop, spacing


###############################################################################
# The rotation map is in radians, with one angle per grid node in 2D.
# It varies along x and is constant along z. DS uses grid-index geometry;
# physical units enter only through the plotted cell-edge extent.
def rotation_map(shape):
    """Prescribe endpoint angles across the x samples."""
    nx, nz = shape
    rotation_degrees = np.broadcast_to(
        np.linspace(0.0, -12.0, nx)[:, None], (nx, nz)
    )
    return np.deg2rad(rotation_degrees)[..., None]


def validate_field(field, shape, categories):
    """Validate raw categorical output before any display/storage cast."""
    field = np.asarray(field)
    if (
        field.shape != tuple(shape)
        or not np.isfinite(field).all()
        or not np.equal(field, np.rint(field)).all()
        or not np.isin(field, categories).all()
    ):
        raise ValueError("Invalid Herten categorical field")
    return field


###############################################################################
# Each case creates a fresh generator and explicitly resets the same master
# seed. The smoke profile in tests overrides controls; the full comparison uses
# these reference defaults, including the inherited cleanup defaults.
def simulate_case(
    ti_data, rotated, shape=None, threads=2, n_neighbors=50, scan_fraction=0.3
):
    """Generate one raw univariate realization, including a cleanup sweep."""
    shape = ti_data.shape if shape is None else shape
    variable = gs.mps.Variable(
        "facies", ti_data, categorical=True, n_neighbors=n_neighbors
    )
    geometry = {"rotation": rotation_map(shape)} if rotated else {}
    model = gs.MPSModel(
        gs.mps.TrainingImage(variable),
        scan_fraction=scan_fraction,
        threshold=0.05,
        scan_path="random",
        post_processing=1,
        boundary="partial",
        **geometry,
    )
    ds = gs.DirectSampling(model)
    raw = ds(
        [np.arange(n, dtype=float) for n in shape],
        seed=SEED,
        path="random",
        num_threads=threads,
        post_process=False,
        store=False,
    )
    return validate_field(raw, shape, np.unique(ti_data))


def plot_result(ti_data, stationary, rotated, spacing):
    """Shared category colors, orientation, and physical cell extents."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    cmap = ListedColormap(plt.get_cmap("tab10").colors)
    norm = BoundaryNorm(np.arange(-0.5, 10.5, 1.0), cmap.N)
    nx, nz = ti_data.shape
    extent = (0.0, nx * spacing, 0.0, nz * spacing)
    fig, axes = plt.subplots(3, 1, figsize=(7, 10), layout="constrained")
    panels = [
        (ti_data, "Training image: Herten S1, left crop"),
        (stationary, f"Stationary search, seed {SEED}"),
        (rotated, f"Rotation gradient 0 to -12 degrees, seed {SEED}"),
    ]
    for ax, (field, title) in zip(axes, panels):
        image = ax.imshow(
            field.T,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
        ax.set(title=title, xlabel="x (m)", ylabel="z (m)")
    codes = np.arange(10)
    colorbar = fig.colorbar(image, ax=list(axes), ticks=codes, shrink=0.75)
    colorbar.set_label("Facies code (categorical)")
    colorbar.ax.set_yticklabels(
        [f"{code} (absent)" if code in {0, 6} else str(code) for code in codes]
    )
    return fig


def source_identity(module):
    """Record the imported file and checkout state when available."""
    path = Path(module.__file__).resolve()
    result = {"origin": str(path), "version": module.__version__}
    for parent in path.parents:
        if (parent / ".git").exists():
            for key, args in (
                ("head", ["rev-parse", "HEAD"]),
                ("status", ["status", "--porcelain"]),
            ):
                result[key] = subprocess.check_output(
                    ["git", "-C", str(parent), *args], text=True, timeout=30
                ).strip()
            break
    return result


def regenerate(output, threads=2):
    """Run the full reference comparison into a new, explicit output directory."""
    import importlib

    import gstools_core

    if threads < 1:
        raise ValueError("Requested threads must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    ti_data, spacing = load_input()
    manifest = {
        "status": "running",
        "input_sha256": INPUT_HASH,
        "crop": "section[:160, :]",
        "shape": list(ti_data.shape),
        "spacing_m": spacing,
        "categories": np.unique(ti_data).tolist(),
        "coordinate_convention": "index axes; physical cell-edge plot extent",
        "seed_contract": CONTRACT,
        "seed": SEED,
        "requested_threads": threads,
        "controls": {
            "n_neighbors": 50,
            "scan_fraction": 0.3,
            "threshold": 0.05,
            "scan_path": "random",
            "path": "random",
            "post_processing": 1,
            "cleanup_controls": "inherited defaults",
            "boundary": "partial",
            "scale": None,
            "rotation_degrees": [0.0, -12.0],
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "frontend": source_identity(gs),
        "source_checkouts": {},
        "core": source_identity(gstools_core),
        "elapsed_seconds": {},
    }
    source_paths = {"frontend": EXAMPLE_DIR.parents[1]}
    core_source = os.environ.get("GSTOOLS_MPS_HERTEN_CORE_SOURCE")
    if core_source:
        source_paths["core"] = Path(core_source).resolve()
    for name, repo in source_paths.items():
        if (repo / ".git").exists():
            manifest["source_checkouts"][name] = {
                key: subprocess.check_output(
                    ["git", "-C", str(repo), *args], text=True, timeout=30
                ).strip()
                for key, args in (
                    ("head", ["rev-parse", "HEAD"]),
                    ("status", ["status", "--porcelain"]),
                )
            }
    native = importlib.import_module(gstools_core.mps_simulate.__module__)
    manifest["native"] = {
        "origin": native.__file__,
        "sha256": file_hash(native.__file__),
        "interface_revision": gstools_core.mps_api_version(),
    }
    manifest_path = output / "herten_nonstationary.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        fields = {}
        for name, rotated in (("stationary", False), ("rotated", True)):
            start = time.perf_counter()
            fields[name] = simulate_case(ti_data, rotated, threads=threads)
            manifest["elapsed_seconds"][name] = time.perf_counter() - start
        # Validation has already happened on the raw floating output.
        cache = output / "herten_nonstationary.npz"
        np.savez_compressed(
            cache, **{k: v.astype(np.int8) for k, v in fields.items()}
        )
        fig = plot_result(
            ti_data, fields["stationary"], fields["rotated"], spacing
        )
        image = output / "herten_nonstationary.png"
        fig.savefig(image, dpi=150)
        manifest["outputs"] = {p.name: file_hash(p) for p in (cache, image)}
        manifest["status"] = "passed"
    except BaseException as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return fig


def preview():
    """Display the curated figure shipped with the gallery example."""
    import matplotlib.pyplot as plt

    image = plt.imread(EXAMPLE_DIR / "output" / "herten_nonstationary.png")
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.imshow(image)
    ax.set_axis_off()
    fig.tight_layout()
    return fig


###############################################################################
# Preview execution does not overwrite the curated assets. Regeneration writes
# to a new directory selected by the caller; review results before curating them.
def main():
    """Select the documented offline preview or full regeneration mode."""
    import matplotlib.pyplot as plt

    mode = os.environ.get("GSTOOLS_MPS_HERTEN_REGENERATE", "0")
    if mode not in {"0", "1"}:
        raise ValueError("GSTOOLS_MPS_HERTEN_REGENERATE must be 0 or 1")
    if mode == "1":
        output = os.environ.get("GSTOOLS_MPS_HERTEN_OUTPUT_DIR")
        if not output:
            raise ValueError(
                "Set GSTOOLS_MPS_HERTEN_OUTPUT_DIR to a new directory"
            )
        threads = int(os.environ.get("GSTOOLS_MPS_HERTEN_THREADS", "2"))
        regenerate(output, threads)
    else:
        print("Precomputed Herten preview; no new numerical simulation.")
        preview()
    plt.show()


if __name__ == "__main__":
    main()
