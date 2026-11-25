r"""
HESS Roussillon aquifer reproduction
------------------------------------

This example shows how GSTools can reproduce the main ingredients of the
Roussillon Continental Pliocene aquifer workflow from
`Dall'Alba et al. (2020) <https://hess.copernicus.org/articles/24/4997/2020/>`_.
The original open dataset is available from
`Zenodo <https://doi.org/10.5281/zenodo.4110278>`_.

The paper used DeeSse Direct Sampling with a 2D nonstationary training image,
a 3D trend map, continuous rotation maps, hard conditioning data, and a
layer-by-layer vertical sampling strategy. The reproduction command below uses
the same public TI, transformed simulation grid, trend map, rotation map, and
hard data, then writes generated TI, input-map, 2D-result, facies/vertical,
3D-volume, and cross-section figures to ``output/``.

For normal documentation/gallery execution this file only displays the saved
figures. Regenerate them explicitly with::

    python examples/13_mps/15_roussillon_hess_reproduction.py small \
        --archive <download dir>/roussillon_mps_2020_dallalba.zip \
        --realizations 2 --num-threads 4

After a run exists, redraw only the figures with::

    python examples/13_mps/15_roussillon_hess_reproduction.py figures \
        --archive <download dir>/roussillon_mps_2020_dallalba.zip

If the archive is missing, the command downloads it automatically. For
byte-identical reproducibility checks, keep ``--num-threads 1`` and add
``--repeat-check``. The heavy implementation lives in
``input/roussillon_hess_tools.inc`` so the example stays readable.

.. note::

    Current GSTools differs from the paper's DeeSse setup in two relevant
    places: GSTools currently has one global threshold instead of separate
    facies/trend thresholds, and the Direct Sampling engine has no true
    inactive-cell mask. The small mode therefore selects a fully active real
    subdomain from the Roussillon transformed grid.
"""

from pathlib import Path
import sys


example_dir = (
    Path(__file__).resolve().parent if "__file__" in globals() else Path(".")
)
output_dir = example_dir / "output"
ti_path = output_dir / "roussillon_hess_small_ti.png"
input_path = output_dir / "roussillon_hess_small_inputs.png"
input_3d_path = output_dir / "roussillon_hess_small_input_3d.png"
output_path = output_dir / "roussillon_hess_small_reproduction.png"
facies_vertical_path = output_dir / "roussillon_hess_small_facies_vertical.png"
volume_path = output_dir / "roussillon_hess_small_3d.png"
cross_section_path = output_dir / "roussillon_hess_small_cross_sections.png"


def _cli_requested():
    """Return True when the file is being used as the reproduction CLI."""
    return len(sys.argv) > 1 and sys.argv[1] in {
        "audit",
        "small",
        "full",
        "figures",
        "-h",
        "--help",
    }


if __name__ == "__main__" and _cli_requested():
    import importlib.machinery
    import importlib.util

    helper_path = example_dir / "input" / "roussillon_hess_tools.inc"
    loader = importlib.machinery.SourceFileLoader(
        "roussillon_hess_tools", str(helper_path)
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)

    raise SystemExit(module.main())

import matplotlib.pyplot as plt


###############################################################################
# **Display the generated inputs and the GSTools reproduction.**
#
# The first saved figure shows the selected TI facies. The second and third
# saved figures show the 2D and 3D input maps: transformed-grid trend and
# gridded rotation. The fourth saved figure shows the GSTools small-domain 2D
# diagnostics. The fifth saved figure follows paper Fig. 10 with facies
# percentages and vertical-run dissimilarity. The sixth saved figure follows
# the paper-style 3D volume presentation. The seventh saved figure mirrors the
# paper's cross-section comparison for the vertical-sampling strategy.
def show_saved_figure(path, figsize):
    """Display a precomputed HESS reproduction figure."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing precomputed figure: {path}. Run:\n"
            "python examples/13_mps/15_roussillon_hess_reproduction.py small "
            "--archive <download dir>/roussillon_mps_2020_dallalba.zip "
            "--realizations 2 --num-threads 4"
        )
    image = plt.imread(path)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)
    ax.axis("off")
    fig.tight_layout(pad=0)


show_saved_figure(ti_path, figsize=(7.2, 7.6))
show_saved_figure(input_path, figsize=(12.8, 5.6))
show_saved_figure(input_3d_path, figsize=(13.5, 5.8))
show_saved_figure(output_path, figsize=(16, 8.8))
show_saved_figure(facies_vertical_path, figsize=(16.2, 6.1))
show_saved_figure(volume_path, figsize=(15, 10))
show_saved_figure(cross_section_path, figsize=(15, 8.5))
