r"""
HESS Roussillon aquifer reproduction helper
-------------------------------------------

This script prepares and runs GSTools Direct Sampling experiments based on the
open HESS Roussillon case study by Dall'Alba et al. (2020):

* paper: https://hess.copernicus.org/articles/24/4997/2020/
* data: https://doi.org/10.5281/zenodo.4110278

It is intentionally a command line reproduction helper rather than an example
that executes automatically during documentation builds.

Typical usage from the repository root::

    python examples/13_mps/15_roussillon_hess_reproduction.py audit \
        --archive <download dir>/roussillon_mps_2020_dallalba.zip

    python examples/13_mps/15_roussillon_hess_reproduction.py small \
        --archive <download dir>/roussillon_mps_2020_dallalba.zip \
        --realizations 2 --num-threads 4

    python examples/13_mps/15_roussillon_hess_reproduction.py figures \
        --archive <download dir>/roussillon_mps_2020_dallalba.zip

    python examples/13_mps/15_roussillon_hess_reproduction.py full \
        --archive <download dir>/roussillon_mps_2020_dallalba.zip \
        --realizations 50 --num-threads 8

Important fidelity notes
~~~~~~~~~~~~~~~~~~~~~~~~

The workflow uses the paper TI, transformed simulation grid, trend map,
rotation map, and hard data. It reproduces the paper's layer-by-layer 2D
stacking strategy, including optional vertical sampling between layers.

Current GSTools differs from DeeSse in a few important ways:

* GSTools currently has one global Direct Sampling threshold, while the paper
  used facies threshold 0.05 and trend threshold 0.25.
* The paper used two rotation maps as angular bounds. GSTools accepts one
  rotation map, so this script uses their cellwise mean by default.
* GSTools Direct Sampling currently simulates rectangular structured grids and
  has no inactive-cell mask. The ``small`` mode therefore selects fully active
  rectangular subdomains. The ``full`` mode preserves and reports the paper
  mask in outputs, but true active-mask handling would need engine support for
  exact full-domain fidelity.

Reproducibility notes
~~~~~~~~~~~~~~~~~~~~~

The archive MD5, data alignment, domain selection, seed schedule, and output
hashes are recorded in JSON summaries. For byte-identical reruns, keep
``--num-threads 1`` (the default) and use ``--repeat-check`` on a small run.
Timing values are printed to the terminal but are not saved unless
``--record-timing`` is passed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import pickle
import sys
import tempfile
import textwrap
import time
import urllib.request
import warnings
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


ARCHIVE_URL = (
    "https://zenodo.org/api/records/4110278/files/"
    "roussillon_mps_2020_dallalba.zip/content"
)
ARCHIVE_MD5 = "4d598a59c64bf5eb28362987dceba432"
ARCHIVE_NAME = "roussillon_mps_2020_dallalba.zip"

ORIGIN_X = 664328.1865
ORIGIN_Y = 6153000.2413
DX = 100.0
DY = 100.0
DZ = 2.0
DEFAULT_Z_ORIGIN = -201.0

AUX_Y_OFFSET = 6
AUX_X_OFFSET = 1
FACIES = tuple(range(6))
INACTIVE = np.uint8(255)

REL_TI = "data/tis/ti_roussillon.gslib"
REL_SG = "data/simulation_grid/pickle/simulation_grid_transfo.pickle"
REL_TREND = "data/auxiliary_variables/pickle/trend_map.pickle"
REL_ROTATION = "data/auxiliary_variables/pickle/rotation_map.pickle"
REL_HD = "data/hard_data/hd_merge.csv"
REL_PAPER_PROB = "data/simulation_outputs/pickle/output_simulation_probability.pickle"
REL_PAPER_ENTROPY = "data/simulation_outputs/pickle/output_simulation_entropy.pickle"
REL_PAPER_WITH_VS = "data/simulation_outputs/pickle/output_simulation_with_vs.pickle"
REL_PAPER_WITHOUT_VS = (
    "data/simulation_outputs/pickle/output_simulation_without_vs.pickle"
)
# Facies palette matching the HESS Roussillon facies style.
# The order matches the facies codes in ti_roussillon.gslib.
FACIES_COLORS = (
    "#6f8891",
    "#d3bc7c",
    "#d38255",
    "#1e847e",
    "#28587e",
    "#85bdb4",
)
INACTIVE_COLOR = "#ffffff"
FACIES_LABELS = (
    "0 floodplain",
    "1 braided river",
    "2 meandering river",
    "3 crevasse splay",
    "4 alluvial fan",
    "5 levee",
)


@dataclass(frozen=True)
class Domain:
    """3D subdomain in archive array order: z, y, x."""

    z0: int
    y0: int
    x0: int
    nz: int
    ny: int
    nx: int

    @property
    def zslice(self):
        return slice(self.z0, self.z0 + self.nz)

    @property
    def yslice(self):
        return slice(self.y0, self.y0 + self.ny)

    @property
    def xslice(self):
        return slice(self.x0, self.x0 + self.nx)

    @property
    def shape(self):
        return (self.nz, self.ny, self.nx)


@dataclass
class HESSData:
    """Loaded and aligned HESS archive inputs."""

    ti_facies: np.ndarray
    ti_trend: np.ndarray
    sim_grid: np.ndarray
    trend: np.ndarray
    rotation_maps: np.ndarray
    hard_data: dict[str, np.ndarray]
    z_origin: float
    archive: Path


def example_dir() -> Path:
    path = Path(__file__).resolve()
    if path.parent.name == "input":
        return path.parents[1]
    return path.parent


def default_output_dir() -> Path:
    return example_dir() / "output" / "roussillon_hess_reproduction"


def default_archive_path() -> Path:
    input_archive = example_dir() / "input" / ARCHIVE_NAME
    if input_archive.exists():
        return input_archive
    tmp_archive = Path(tempfile.gettempdir()) / ARCHIVE_NAME
    if tmp_archive.exists():
        return tmp_archive
    return Path.home() / ".cache" / "gstools" / ARCHIVE_NAME


def md5sum(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(block_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {ARCHIVE_URL}")
    print(f"       to {path}")
    with urllib.request.urlopen(ARCHIVE_URL) as response, path.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def archive_candidates(requested: Path) -> list[Path]:
    """Return archive locations in the order they should be tried."""
    candidates = [
        requested,
        example_dir() / "input" / ARCHIVE_NAME,
        Path(tempfile.gettempdir()) / ARCHIVE_NAME,
        Path.home() / ".cache" / "gstools" / ARCHIVE_NAME,
    ]
    unique = []
    seen = set()
    for candidate in candidates:
        key = str(candidate.expanduser())
        if key not in seen:
            unique.append(candidate.expanduser())
            seen.add(key)
    return unique


def validate_archive(path: Path) -> Path:
    got = md5sum(path)
    if got != ARCHIVE_MD5:
        raise ValueError(
            f"Archive MD5 mismatch for {path}: got {got}, expected {ARCHIVE_MD5}"
        )
    return path


def ensure_archive(path: Path, download: bool) -> Path:
    for candidate in archive_candidates(path):
        if candidate.exists():
            print(f"Using HESS archive: {candidate}")
            return validate_archive(candidate)

    searched = "\n  - ".join(str(candidate) for candidate in archive_candidates(path))
    if not download:
        raise FileNotFoundError(
            "HESS archive not found. Searched:\n  - "
            f"{searched}\nPass --download or place the zip in examples/13_mps/input."
        )

    print(
        "HESS archive not found locally; downloading it now. "
        "Pass --no-download to require a local file."
    )
    download_archive(path)
    return validate_archive(path)


def archive_member(zf: zipfile.ZipFile, rel_path: str) -> str:
    matches = [name for name in zf.namelist() if name.endswith(rel_path)]
    if len(matches) != 1:
        raise KeyError(
            f"Expected exactly one archive member ending with {rel_path!r}, "
            f"found {len(matches)}"
        )
    return matches[0]


def read_text(zf: zipfile.ZipFile, rel_path: str) -> str:
    return zf.read(archive_member(zf, rel_path)).decode("utf-8")


def read_pickle(zf: zipfile.ZipFile, rel_path: str):
    return pickle.load(io.BytesIO(zf.read(archive_member(zf, rel_path))))



def read_gslib_ti(zf: zipfile.ZipFile, rel_path: str) -> dict[str, np.ndarray]:
    lines = read_text(zf, rel_path).splitlines()
    header = lines[0].split()
    nx, ny, nz = (int(header[i]) for i in range(3))
    n_var = int(lines[1].strip())
    names = [lines[2 + i].strip() for i in range(n_var)]
    values = np.loadtxt(io.StringIO("\n".join(lines[2 + n_var :])))
    if values.ndim == 1:
        values = values[:, np.newaxis]
    expected = nx * ny * nz
    if values.shape != (expected, n_var):
        raise ValueError(
            f"Unexpected GSLIB value shape {values.shape}; expected "
            f"{(expected, n_var)}"
        )
    out = {}
    for i, name in enumerate(names):
        # GSLIB rows are x-fastest. Keep arrays in row-major layer order:
        # (z, y, x), squeezed to (y, x) for this 2D TI.
        arr = values[:, i].reshape(nz, ny, nx)
        out[name] = arr[0] if nz == 1 else arr
    return out


def read_hard_data(zf: zipfile.ZipFile) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(io.StringIO(read_text(zf, REL_HD))))
    return {
        "x": np.array([float(row["X"]) for row in rows], dtype=float),
        "y": np.array([float(row["Y"]) for row in rows], dtype=float),
        "z": np.array([float(row["Z"]) for row in rows], dtype=float),
        "facies": np.array([int(float(row["facies"])) for row in rows], dtype=int),
    }


def map_hard_data(
    hard_data: dict[str, np.ndarray],
    z_origin: float,
    shape: tuple[int, int, int],
) -> dict[str, np.ndarray]:
    ix_aux = np.rint((hard_data["x"] - ORIGIN_X) / DX).astype(int)
    iy_aux = np.rint((hard_data["y"] - ORIGIN_Y) / DY).astype(int)
    mapped = {
        "ix": ix_aux - AUX_X_OFFSET,
        "iy": iy_aux - AUX_Y_OFFSET,
        "iz": np.rint((hard_data["z"] - z_origin) / DZ).astype(int),
        "facies": hard_data["facies"].astype(int),
    }
    nz, ny, nx = shape
    mapped["inside"] = (
        (0 <= mapped["iz"])
        & (mapped["iz"] < nz)
        & (0 <= mapped["iy"])
        & (mapped["iy"] < ny)
        & (0 <= mapped["ix"])
        & (mapped["ix"] < nx)
    )
    return mapped


def calibrate_z_origin(
    zf: zipfile.ZipFile,
    hard_data: dict[str, np.ndarray],
    shape: tuple[int, int, int],
    search_min: float,
    search_max: float,
    search_step: float,
) -> tuple[float, dict[str, float]]:
    reference = read_pickle(zf, REL_PAPER_WITH_VS)
    if reference.ndim == 4:
        reference = reference[0]
    best = {
        "z_origin": DEFAULT_Z_ORIGIN,
        "matches": -1,
        "total": 0,
        "ratio": 0.0,
    }
    candidates = np.arange(search_min, search_max + 0.5 * search_step, search_step)
    for z_origin in candidates:
        mapped = map_hard_data(hard_data, z_origin, shape)
        inside = mapped["inside"]
        if not np.any(inside):
            continue
        vals = reference[
            mapped["iz"][inside],
            mapped["iy"][inside],
            mapped["ix"][inside],
        ]
        finite = np.isfinite(vals)
        if not np.any(finite):
            continue
        facies = mapped["facies"][inside][finite]
        matches = int(np.count_nonzero(vals[finite] == facies))
        total = int(finite.sum())
        ratio = matches / total
        if matches > best["matches"]:
            best = {
                "z_origin": float(z_origin),
                "matches": matches,
                "total": total,
                "ratio": float(ratio),
            }
    return float(best["z_origin"]), best


def aligned_rotation_mean(rotation_maps: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(rotation_maps, axis=0)


def load_hess_data(args) -> HESSData:
    archive = ensure_archive(Path(args.archive).expanduser(), args.download)
    with zipfile.ZipFile(archive) as zf:
        ti = read_gslib_ti(zf, REL_TI)
        sim_grid = read_pickle(zf, REL_SG).astype(bool)
        trend_raw = read_pickle(zf, REL_TREND)
        rotation_raw = read_pickle(zf, REL_ROTATION)
        hard_data = read_hard_data(zf)

        trend = trend_raw[
            :,
            AUX_Y_OFFSET : AUX_Y_OFFSET + sim_grid.shape[1],
            AUX_X_OFFSET : AUX_X_OFFSET + sim_grid.shape[2],
        ].astype(float)
        rotation_maps = rotation_raw[
            :,
            :,
            AUX_Y_OFFSET : AUX_Y_OFFSET + sim_grid.shape[1],
            AUX_X_OFFSET : AUX_X_OFFSET + sim_grid.shape[2],
        ].astype(float)
        if trend.shape != sim_grid.shape:
            raise ValueError(
                f"Aligned trend shape {trend.shape} does not match SG {sim_grid.shape}"
            )
        if rotation_maps.shape[1:] != sim_grid.shape:
            raise ValueError(
                "Aligned rotation maps shape "
                f"{rotation_maps.shape[1:]} does not match SG {sim_grid.shape}"
            )

        if args.z_origin == "auto":
            try:
                z_origin, calibration = calibrate_z_origin(
                    zf,
                    hard_data,
                    sim_grid.shape,
                    args.z_search_min,
                    args.z_search_max,
                    args.z_search_step,
                )
                print(
                    "Calibrated hard-data z origin from archived with-VS "
                    f"output: {z_origin:g} m "
                    f"({calibration['matches']}/{calibration['total']} matches, "
                    f"ratio={calibration['ratio']:.3f})"
                )
            except Exception as exc:
                z_origin = DEFAULT_Z_ORIGIN
                print(
                    "Could not auto-calibrate hard-data z origin; using "
                    f"{DEFAULT_Z_ORIGIN:g} m. Reason: {exc}"
                )
        else:
            z_origin = float(args.z_origin)

    return HESSData(
        ti_facies=ti["facies"].astype(int),
        ti_trend=ti["trend"].astype(float),
        sim_grid=sim_grid,
        trend=trend,
        rotation_maps=rotation_maps,
        hard_data=hard_data,
        z_origin=z_origin,
        archive=archive,
    )


def finite_stats(arr: np.ndarray) -> dict[str, float | int]:
    finite = np.isfinite(arr)
    out: dict[str, float | int] = {
        "shape": tuple(int(s) for s in arr.shape),
        "finite": int(finite.sum()),
        "nan": int((~finite).sum()),
    }
    if np.any(finite):
        out.update(
            {
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
                "mean": float(np.nanmean(arr)),
            }
        )
    return out


def facies_counts(arr: np.ndarray, mask: np.ndarray | None = None) -> dict[str, int]:
    if mask is not None:
        arr = arr[mask]
    arr = arr[np.isfinite(arr)]
    values, counts = np.unique(arr.astype(int), return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, counts)}


def block_sum3d(mask: np.ndarray, window: tuple[int, int, int]) -> np.ndarray:
    nz, ny, nx = window
    csum = mask.astype(np.int32).cumsum(0).cumsum(1).cumsum(2)
    csum = np.pad(csum, ((1, 0), (1, 0), (1, 0)), constant_values=0)
    return (
        csum[nz:, ny:, nx:]
        - csum[:-nz, ny:, nx:]
        - csum[nz:, :-ny, nx:]
        - csum[nz:, ny:, :-nx]
        + csum[:-nz, :-ny, nx:]
        + csum[:-nz, ny:, :-nx]
        + csum[nz:, :-ny, :-nx]
        - csum[:-nz, :-ny, :-nx]
    )


def select_domain(data: HESSData, args) -> Domain:
    if args.mode == "full":
        nz, ny, nx = data.sim_grid.shape
        return Domain(0, 0, 0, nz, ny, nx)

    nz, ny, nx = args.small_shape
    if nz <= 0 or ny <= 0 or nx <= 0:
        raise ValueError("--small-shape values must be positive")

    rotation_mean = aligned_rotation_mean(data.rotation_maps)
    valid = data.sim_grid & np.isfinite(data.trend) & np.isfinite(rotation_mean)
    valid_sums = block_sum3d(valid, (nz, ny, nx))

    mapped = map_hard_data(data.hard_data, data.z_origin, data.sim_grid.shape)
    hard_counts = np.zeros(data.sim_grid.shape, dtype=np.int16)
    inside = mapped["inside"]
    np.add.at(
        hard_counts,
        (mapped["iz"][inside], mapped["iy"][inside], mapped["ix"][inside]),
        1,
    )
    hard_sums = block_sum3d(hard_counts, (nz, ny, nx))

    target = nz * ny * nx
    full_blocks = valid_sums == target
    if np.any(full_blocks):
        scores = np.where(full_blocks, hard_sums, -1)
        z0, y0, x0 = np.unravel_index(int(scores.argmax()), scores.shape)
        hard_count = int(scores[z0, y0, x0])
        print(
            "Selected fully active small domain "
            f"z={z0}:{z0+nz}, y={y0}:{y0+ny}, x={x0}:{x0+nx} "
            f"with {hard_count} mapped hard-data rows."
        )
        return Domain(int(z0), int(y0), int(x0), nz, ny, nx)

    z0, y0, x0 = np.unravel_index(int(valid_sums.argmax()), valid_sums.shape)
    fraction = int(valid_sums[z0, y0, x0]) / target
    print(
        "WARNING: no fully active small domain found for requested shape; "
        f"using best available block with valid fraction {fraction:.3f}."
    )
    return Domain(int(z0), int(y0), int(x0), nz, ny, nx)


def fill_nan_2d(arr: np.ndarray, fallback: float) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if np.all(np.isfinite(arr)):
        return arr
    out = arr.copy()
    missing = ~np.isfinite(out)
    if np.all(missing):
        out[:] = fallback
        return out
    try:
        from scipy import ndimage

        nearest = ndimage.distance_transform_edt(
            missing, return_distances=False, return_indices=True
        )
        out[missing] = out[tuple(nearest[:, missing])]
    except Exception:
        out[missing] = fallback
    return out


def build_training_image(data: HESSData, args):
    import gstools as gs

    ti_facies = data.ti_facies.astype(float)
    ti_trend = data.ti_trend.astype(float)
    if args.clip_trend:
        ti_trend = np.clip(ti_trend, args.clip_trend[0], args.clip_trend[1])
    return gs.mps.TrainingImage(
        [
            gs.mps.Variable(
                "facies",
                ti_facies,
                categorical=True,
                n_neighbors=args.facies_neighbors,
                max_radius=args.max_radius,
                cond_weight=args.cond_weight,
            ),
            gs.mps.Variable(
                "trend_map",
                ti_trend,
                categorical=False,
                distance="l1",
                n_neighbors=args.trend_neighbors,
                max_radius=args.max_radius,
                cond_weight=args.cond_weight,
            ),
        ]
    )


def hard_data_layer_conditions(
    data: HESSData,
    domain: Domain,
    local_z: int,
) -> tuple[np.ndarray, int]:
    mapped = map_hard_data(data.hard_data, data.z_origin, data.sim_grid.shape)
    global_z = domain.z0 + local_z
    in_layer = (
        mapped["inside"]
        & (mapped["iz"] == global_z)
        & (domain.y0 <= mapped["iy"])
        & (mapped["iy"] < domain.y0 + domain.ny)
        & (domain.x0 <= mapped["ix"])
        & (mapped["ix"] < domain.x0 + domain.nx)
    )
    facies = np.full((domain.ny, domain.nx), np.nan, dtype=float)
    count = 0
    for iy, ix, val in zip(
        mapped["iy"][in_layer] - domain.y0,
        mapped["ix"][in_layer] - domain.x0,
        mapped["facies"][in_layer],
    ):
        facies[int(iy), int(ix)] = float(val)
        count += 1
    return facies, count


def transition_matrix_from_hard_data(
    data: HESSData,
) -> np.ndarray:
    mapped = map_hard_data(data.hard_data, data.z_origin, data.sim_grid.shape)
    inside = mapped["inside"]
    cols: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for iz, iy, ix, facies in zip(
        mapped["iz"][inside],
        mapped["iy"][inside],
        mapped["ix"][inside],
        mapped["facies"][inside],
    ):
        cols.setdefault((int(iy), int(ix)), []).append((int(iz), int(facies)))

    counts = np.zeros((len(FACIES), len(FACIES)), dtype=float)
    for vals in cols.values():
        by_layer: dict[int, list[int]] = {}
        for iz, facies in vals:
            by_layer.setdefault(iz, []).append(facies)
        series = []
        for iz in sorted(by_layer):
            values, counts_i = np.unique(by_layer[iz], return_counts=True)
            series.append((iz, int(values[counts_i.argmax()])))
        for (iz0, fac0), (iz1, fac1) in zip(series[:-1], series[1:]):
            if iz1 - iz0 == 1 and fac0 in FACIES and fac1 in FACIES:
                counts[fac0, fac1] += 1.0

    probs = np.zeros_like(counts)
    for facies in FACIES:
        row_sum = counts[facies].sum()
        if row_sum > 0:
            probs[facies] = counts[facies] / row_sum
        else:
            probs[facies, facies] = 1.0
    return probs


def sample_vertical_conditions(
    layer: np.ndarray,
    active: np.ndarray,
    rng: np.random.Generator,
    transition_probs: np.ndarray,
    facies_codes: tuple[int, ...],
    rate: float,
    min_per_facies: int,
) -> tuple[np.ndarray, int]:
    facies = np.full(layer.shape, np.nan, dtype=float)
    total = 0
    for code in facies_codes:
        coords = np.argwhere((layer == code) & active)
        if coords.size == 0:
            continue
        n_sample = int(np.ceil(rate * len(coords)))
        if min_per_facies and len(coords):
            n_sample = max(min_per_facies, n_sample)
        n_sample = min(len(coords), n_sample)
        if n_sample <= 0:
            continue
        chosen = coords[rng.choice(len(coords), size=n_sample, replace=False)]
        for iy, ix in chosen:
            probs = transition_probs[int(code)]
            next_code = int(rng.choice(FACIES, p=probs))
            facies[int(iy), int(ix)] = float(next_code)
        total += n_sample
    return facies, total


def choose_rotation_layer(
    data: HESSData,
    global_z: int,
    domain: Domain,
    mode: str,
) -> np.ndarray:
    raw = data.rotation_maps[:, global_z, domain.yslice, domain.xslice]
    if mode == "lower":
        angle = raw[0]
    elif mode == "upper":
        angle = raw[1]
    elif mode == "mean":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            angle = np.nanmean(raw, axis=0)
    else:
        raise ValueError(f"Unknown rotation mode: {mode}")
    angle = fill_nan_2d(angle, fallback=0.0)
    return np.deg2rad(angle)


def simulate_layer(
    ti,
    args,
    trend: np.ndarray,
    rotation: np.ndarray,
    facies_condition: np.ndarray,
    seed: int,
) -> np.ndarray:
    import gstools as gs

    model = gs.mps.MPSModel(
        ti,
        scan_fraction=args.scan_fraction,
        threshold=args.threshold if args.threshold > 0 else None,
        rotation=rotation[..., None],
    )
    ds = gs.mps.DirectSampling(model)
    iy, ix = np.indices(trend.shape)
    # The trend is known at every node; facies only at the sparse hard data.
    ds.set_condition(
        [iy.ravel().astype(float), ix.ravel().astype(float)],
        {"trend_map": trend.ravel(), "facies": facies_condition.ravel()},
    )
    grid = [np.arange(n, dtype=float) for n in trend.shape]
    result = ds(
        grid, seed=seed, path=args.path, num_threads=args.num_threads
    )
    names = [v.name for v in ti.variables]
    return np.rint(result[names.index("facies")]).astype(np.uint8)


def simulate_volume(
    data: HESSData,
    ti,
    domain: Domain,
    args,
    seed: int,
    vertical_sampling: bool,
    transition_probs: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(seed)
    volume = np.full(domain.shape, INACTIVE, dtype=np.uint8)
    vertical_condition = np.full((domain.ny, domain.nx), np.nan, dtype=float)
    vertical_condition_count = 0
    vertical_streak = 0
    hard_counts = []
    vertical_counts = []
    layer_times = []

    for local_z in range(domain.nz):
        global_z = domain.z0 + local_z
        t0 = time.time()
        active = data.sim_grid[global_z, domain.yslice, domain.xslice]
        trend = data.trend[global_z, domain.yslice, domain.xslice]
        trend = fill_nan_2d(trend, fallback=float(np.nanmean(data.ti_trend)))
        if args.clip_trend:
            trend = np.clip(trend, args.clip_trend[0], args.clip_trend[1])
        rotation = choose_rotation_layer(data, global_z, domain, args.rotation_map)

        hard_condition, hard_count = hard_data_layer_conditions(
            data, domain, local_z
        )
        facies_condition = np.full((domain.ny, domain.nx), np.nan, dtype=float)

        use_vertical = vertical_sampling and vertical_condition_count > 0
        if use_vertical and vertical_streak >= args.max_successive_layers:
            use_vertical = False
            vertical_streak = 0

        if use_vertical:
            facies_condition[np.isfinite(vertical_condition)] = vertical_condition[
                np.isfinite(vertical_condition)
            ]
            vertical_streak += 1

        facies_condition[np.isfinite(hard_condition)] = hard_condition[
            np.isfinite(hard_condition)
        ]

        layer = simulate_layer(
            ti,
            args,
            trend=trend,
            rotation=rotation,
            facies_condition=facies_condition,
            seed=seed + local_z,
        )
        layer[~active] = INACTIVE
        volume[local_z] = layer

        if vertical_sampling:
            vertical_condition, vertical_condition_count = sample_vertical_conditions(
                layer=layer,
                active=active,
                rng=rng,
                transition_probs=transition_probs,
                facies_codes=args.vertical_facies,
                rate=args.vertical_sample_rate,
                min_per_facies=args.min_vertical_samples_per_facies,
            )
        else:
            vertical_condition[:] = np.nan
            vertical_condition_count = 0

        hard_counts.append(int(hard_count))
        vertical_counts.append(int(vertical_condition_count))
        layer_times.append(time.time() - t0)
        print(
            f"  layer {local_z + 1:03d}/{domain.nz:03d} "
            f"(global z={global_z:03d}): hard={hard_count}, "
            f"next_vertical={vertical_condition_count}, "
            f"active={int(active.sum())}, time={layer_times[-1]:.1f}s"
        )

    diagnostics = {
        "seed": int(seed),
        "vertical_sampling": bool(vertical_sampling),
        "hard_conditions_per_layer": hard_counts,
        "next_vertical_conditions_per_layer": vertical_counts,
    }
    if args.record_timing:
        diagnostics["layer_seconds"] = [float(value) for value in layer_times]
    return volume, diagnostics


def update_counts(counts: np.ndarray, volume: np.ndarray) -> None:
    for facies in FACIES:
        counts[facies] += volume == facies


def probability_from_counts(counts: np.ndarray, n_realizations: int) -> np.ndarray:
    return counts.astype(np.float32) / float(n_realizations)


def entropy_from_probability(prob: np.ndarray, active: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(prob > 0.0, np.log(prob), 0.0)
        entropy = -np.sum(np.where(prob > 0.0, prob * logp, 0.0), axis=0)
    entropy = entropy / np.log(len(FACIES))
    entropy = entropy.astype(np.float32)
    entropy[~active] = np.nan
    return entropy


def proportions_from_counts(
    counts: np.ndarray,
    n_realizations: int,
    active: np.ndarray,
) -> dict[str, float]:
    denom = float(n_realizations * int(active.sum()))
    if denom == 0.0:
        return {str(facies): 0.0 for facies in FACIES}
    return {str(facies): float(counts[facies, active].sum() / denom) for facies in FACIES}


def vertical_run_histogram(
    volume: np.ndarray,
    active: np.ndarray,
) -> dict[str, list[int]]:
    hist = {str(facies): np.zeros(volume.shape[0] + 1, dtype=np.int64) for facies in FACIES}
    nz, ny, nx = volume.shape
    for iy in range(ny):
        for ix in range(nx):
            vals = volume[:, iy, ix]
            mask = active[:, iy, ix] & (vals != INACTIVE)
            if not np.any(mask):
                continue
            start = None
            current = None
            run_len = 0
            for iz in range(nz):
                if not mask[iz]:
                    if current is not None:
                        hist[str(int(current))][run_len] += 1
                    current = None
                    run_len = 0
                    start = None
                    continue
                val = int(vals[iz])
                if start is None or val != current:
                    if current is not None:
                        hist[str(int(current))][run_len] += 1
                    start = iz
                    current = val
                    run_len = 1
                else:
                    run_len += 1
            if current is not None:
                hist[str(int(current))][run_len] += 1
    return {key: val.astype(int).tolist() for key, val in hist.items()}


def hard_data_facies_counts(
    data: HESSData,
    domain: Domain,
) -> dict[str, int]:
    mapped = map_hard_data(data.hard_data, data.z_origin, data.sim_grid.shape)
    in_domain = (
        mapped["inside"]
        & (domain.z0 <= mapped["iz"])
        & (mapped["iz"] < domain.z0 + domain.nz)
        & (domain.y0 <= mapped["iy"])
        & (mapped["iy"] < domain.y0 + domain.ny)
        & (domain.x0 <= mapped["ix"])
        & (mapped["ix"] < domain.x0 + domain.nx)
    )
    facies = mapped["facies"][in_domain]
    return {str(code): int(np.count_nonzero(facies == code)) for code in FACIES}


def proportions_from_plain_counts(counts: dict[str, int]) -> dict[str, float]:
    denom = float(sum(int(counts.get(str(code), 0)) for code in FACIES))
    if denom == 0.0:
        return {str(code): 0.0 for code in FACIES}
    return {str(code): float(counts.get(str(code), 0) / denom) for code in FACIES}


def vertical_run_histogram_from_hard_data(
    data: HESSData,
    domain: Domain,
) -> dict[str, list[int]]:
    mapped = map_hard_data(data.hard_data, data.z_origin, data.sim_grid.shape)
    in_domain = (
        mapped["inside"]
        & (domain.z0 <= mapped["iz"])
        & (mapped["iz"] < domain.z0 + domain.nz)
        & (domain.y0 <= mapped["iy"])
        & (mapped["iy"] < domain.y0 + domain.ny)
        & (domain.x0 <= mapped["ix"])
        & (mapped["ix"] < domain.x0 + domain.nx)
    )
    columns: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for iz, iy, ix, facies in zip(
        mapped["iz"][in_domain],
        mapped["iy"][in_domain],
        mapped["ix"][in_domain],
        mapped["facies"][in_domain],
    ):
        columns.setdefault((int(iy), int(ix)), []).append((int(iz), int(facies)))

    hist = {str(facies): np.zeros(domain.nz + 1, dtype=np.int64) for facies in FACIES}
    for values in columns.values():
        values.sort()
        current = None
        run_len = 0
        for _, facies in values:
            if facies not in FACIES:
                if current is not None:
                    hist[str(int(current))][min(run_len, domain.nz)] += 1
                current = None
                run_len = 0
                continue
            if current is None or facies != current:
                if current is not None:
                    hist[str(int(current))][min(run_len, domain.nz)] += 1
                current = int(facies)
                run_len = 1
            else:
                run_len += 1
        if current is not None:
            hist[str(int(current))][min(run_len, domain.nz)] += 1
    return {key: val.astype(int).tolist() for key, val in hist.items()}


def normalized_run_distribution(hist: list[int], n_bins: int) -> np.ndarray:
    arr = np.asarray(hist, dtype=float)
    if arr.size < n_bins:
        arr = np.pad(arr, (0, n_bins - arr.size))
    arr = arr[:n_bins].copy()
    arr[0] = 0.0
    total = arr.sum()
    if total <= 0.0:
        return arr
    return arr / total


def run_histogram_dissimilarity(
    candidate: list[int],
    reference: list[int],
    n_bins: int,
) -> float:
    cand = normalized_run_distribution(candidate, n_bins)
    ref = normalized_run_distribution(reference, n_bins)
    if ref.sum() <= 0.0 or cand.sum() <= 0.0:
        return np.nan
    return float(np.sqrt(np.mean((cand - ref) ** 2)))


def compare_to_paper_outputs(
    data: HESSData,
    domain: Domain,
    probability: np.ndarray,
    entropy: np.ndarray,
    active: np.ndarray,
) -> dict[str, object]:
    comparisons: dict[str, object] = {}
    with zipfile.ZipFile(data.archive) as zf:
        try:
            paper_prob = read_pickle(zf, REL_PAPER_PROB)[
                :, domain.zslice, domain.yslice, domain.xslice
            ]
            diff = probability - paper_prob
            valid = np.isfinite(paper_prob) & active[np.newaxis, ...]
            comparisons["probability_rmse_by_facies"] = {
                str(facies): float(
                    np.sqrt(np.nanmean(diff[facies][valid[facies]] ** 2))
                )
                if np.any(valid[facies])
                else np.nan
                for facies in FACIES
            }
            comparisons["probability_mean_abs_diff_by_facies"] = {
                str(facies): float(np.nanmean(np.abs(diff[facies][valid[facies]])))
                if np.any(valid[facies])
                else np.nan
                for facies in FACIES
            }
        except Exception as exc:
            comparisons["probability_error"] = str(exc)

        try:
            paper_entropy = read_pickle(zf, REL_PAPER_ENTROPY)
            if paper_entropy.ndim == 4:
                paper_entropy = paper_entropy[0]
            paper_entropy = paper_entropy[
                domain.zslice, domain.yslice, domain.xslice
            ]
            valid = np.isfinite(paper_entropy) & np.isfinite(entropy) & active
            comparisons["entropy_rmse"] = (
                float(np.sqrt(np.nanmean((entropy[valid] - paper_entropy[valid]) ** 2)))
                if np.any(valid)
                else np.nan
            )
            comparisons["entropy_mean_abs_diff"] = (
                float(np.nanmean(np.abs(entropy[valid] - paper_entropy[valid])))
                if np.any(valid)
                else np.nan
            )
        except Exception as exc:
            comparisons["entropy_error"] = str(exc)
    return comparisons


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def array_sha256(arr: np.ndarray) -> str:
    """Stable content hash for reproducibility checks."""
    contiguous = np.ascontiguousarray(arr)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def save_quicklook(
    path: Path,
    data: HESSData,
    domain: Domain,
    first_volume: np.ndarray,
    probability: np.ndarray,
    entropy: np.ndarray,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except Exception as exc:
        print(f"Skipping quicklook figure; matplotlib unavailable: {exc}")
        return

    colors = list(FACIES_COLORS) + [INACTIVE_COLOR]
    cmap = ListedColormap(colors)
    mid_z = domain.nz // 2
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    axes[0, 0].imshow(data.ti_facies, origin="lower", cmap=cmap, vmin=0, vmax=6)
    axes[0, 0].set_title("TI facies")
    active = data.sim_grid[domain.z0 + mid_z, domain.yslice, domain.xslice]
    axes[0, 1].imshow(active, origin="lower", cmap="gray_r", vmin=0, vmax=1)
    axes[0, 1].set_title(f"Active SG cells, local z={mid_z}")
    trend = data.trend[domain.z0 + mid_z, domain.yslice, domain.xslice]
    axes[0, 2].imshow(trend, origin="lower", cmap="viridis")
    axes[0, 2].set_title(f"SG trend, local z={mid_z}")
    display = first_volume[mid_z].astype(float)
    display[display == INACTIVE] = 6
    axes[1, 0].imshow(display, origin="lower", cmap=cmap, vmin=0, vmax=6)
    axes[1, 0].set_title("First realization")
    axes[1, 1].imshow(probability[1, mid_z], origin="lower", cmap="magma", vmin=0, vmax=1)
    axes[1, 1].set_title("P(facies 1)")
    axes[1, 2].imshow(entropy[mid_z], origin="lower", cmap="magma", vmin=0, vmax=1)
    axes[1, 2].set_title("Shannon entropy")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def gallery_training_image_path(mode: str) -> Path:
    """Top-level generated training-image figure."""
    return example_dir() / "output" / f"roussillon_hess_{mode}_ti.png"


def gallery_input_path(mode: str) -> Path:
    """Top-level generated input-map figure."""
    return example_dir() / "output" / f"roussillon_hess_{mode}_inputs.png"


def gallery_input_3d_path(mode: str) -> Path:
    """Top-level generated 3D input-map figure."""
    return example_dir() / "output" / f"roussillon_hess_{mode}_input_3d.png"


def gallery_output_path(mode: str) -> Path:
    """Top-level generated GSTools result figure."""
    return example_dir() / "output" / f"roussillon_hess_{mode}_reproduction.png"


def gallery_facies_vertical_path(mode: str) -> Path:
    """Top-level generated Fig. 10-style facies and vertical-run figure."""
    return example_dir() / "output" / f"roussillon_hess_{mode}_facies_vertical.png"


def gallery_volume_path(mode: str) -> Path:
    """Top-level generated 3D GSTools volume figure."""
    return example_dir() / "output" / f"roussillon_hess_{mode}_3d.png"


def gallery_cross_section_path(mode: str) -> Path:
    """Top-level generated vertical-sampling cross-section figure."""
    return example_dir() / "output" / f"roussillon_hess_{mode}_cross_sections.png"


def save_training_image_figure(path: Path, data: HESSData) -> None:
    """Create a standalone selected-TI facies figure."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
    except Exception as exc:
        print(f"Skipping training-image figure; matplotlib unavailable: {exc}")
        return

    facies_colors = list(FACIES_COLORS)
    facies_cmap = ListedColormap(facies_colors)
    fig, ax = plt.subplots(figsize=(7.2, 7.6), constrained_layout=True)
    ax.imshow(
        data.ti_facies,
        origin="lower",
        cmap=facies_cmap,
        vmin=0,
        vmax=len(FACIES) - 1,
        interpolation="nearest",
    )
    ax.set_title("a) Selected TI facies")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.legend(
        handles=[
            Patch(facecolor=c, edgecolor="0.4", label=l)
            for c, l in zip(facies_colors, FACIES_LABELS)
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
        fontsize=8,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training-image figure to {path}")


def input_map_layers(data: HESSData, args, global_z: int):
    """Return masked full-domain trend and rotation maps for one layer."""
    active_layer = data.sim_grid[global_z]
    trend_layer = np.ma.masked_where(
        ~active_layer | ~np.isfinite(data.trend[global_z]), data.trend[global_z]
    )
    raw_rotation = data.rotation_maps[:, global_z]
    if args.rotation_map == "lower":
        rotation_deg_raw = raw_rotation[0]
    elif args.rotation_map == "upper":
        rotation_deg_raw = raw_rotation[1]
    elif args.rotation_map == "mean":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rotation_deg_raw = np.nanmean(raw_rotation, axis=0)
    else:
        raise ValueError(f"Unknown rotation mode: {args.rotation_map}")
    rotation_layer = np.ma.masked_where(
        ~active_layer, fill_nan_2d(rotation_deg_raw, fallback=0.0)
    )
    return active_layer, trend_layer, rotation_layer


def save_input_overview_figure(
    path: Path,
    data: HESSData,
    args,
    summaries: dict[str, dict[str, object]],
) -> None:
    """Create paper-equivalent 2D trend and rotation input maps."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping input overview figure; matplotlib unavailable: {exc}")
        return

    if "with_vertical_sampling" not in summaries:
        return

    domain = Domain(**summaries["with_vertical_sampling"]["domain"])
    global_z = domain.z0
    _, trend_layer, rotation_deg = input_map_layers(data, args, global_z)

    fig, (ax_trend, ax_rotation) = plt.subplots(
        1, 2, figsize=(12.8, 5.6), constrained_layout=True
    )
    trend_cmap = plt.get_cmap("viridis").copy()
    trend_cmap.set_bad("white")
    rotation_cmap = plt.get_cmap("twilight").copy()
    rotation_cmap.set_bad("white")

    im_trend = ax_trend.imshow(trend_layer, origin="lower", cmap=trend_cmap)
    ax_trend.set_title(f"b) Simulation-grid trend map, layer z={global_z}")
    ax_trend.set_xticks([])
    ax_trend.set_yticks([])
    fig.colorbar(im_trend, ax=ax_trend, fraction=0.046, pad=0.02)

    im_rot = ax_rotation.imshow(
        rotation_deg, origin="lower", cmap=rotation_cmap, vmin=-110, vmax=75
    )
    ax_rotation.set_title(f"c) Gridded rotation map, layer z={global_z} (degrees)")
    ax_rotation.set_xticks([])
    ax_rotation.set_yticks([])
    fig.colorbar(im_rot, ax=ax_rotation, fraction=0.046, pad=0.02)

    fig.suptitle("Generated input maps for the HESS Roussillon GSTools run", fontsize=13)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(
        f"Saved input overview figure to {path} "
        f"(layer z={global_z}; subdomain z/y/x={domain.z0}:{domain.z0 + domain.nz}, "
        f"{domain.y0}:{domain.y0 + domain.ny}, "
        f"{domain.x0}:{domain.x0 + domain.nx}; full SG shape={data.sim_grid.shape})"
    )


def save_input_3d_figure(
    path: Path,
    data: HESSData,
    args,
    summaries: dict[str, dict[str, object]],
) -> None:
    """Create clean 3D masked views of the trend and rotation input maps."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception as exc:
        print(f"Skipping 3D input-map figure; matplotlib unavailable: {exc}")
        return

    if "with_vertical_sampling" not in summaries:
        return

    domain = Domain(**summaries["with_vertical_sampling"]["domain"])
    global_z = domain.z0
    active_layer, trend_layer, rotation_layer = input_map_layers(data, args, global_z)
    stride = 3
    yy, xx = np.mgrid[0 : active_layer.shape[0] : stride, 0 : active_layer.shape[1] : stride]
    active_ds = active_layer[::stride, ::stride]
    trend_ds = np.asarray(trend_layer.filled(np.nan))[::stride, ::stride]
    rotation_ds = np.asarray(rotation_layer.filled(np.nan))[::stride, ::stride]

    fig = plt.figure(figsize=(14.8, 6.2), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, wspace=0.02)

    def show_map_surface(ax, values, title, cmap_name, vmin=None, vmax=None, label=""):
        values = values.astype(float)
        valid = active_ds & np.isfinite(values)
        if not np.any(valid):
            ax.set_axis_off()
            ax.set_title(title)
            return
        if vmin is None:
            vmin = float(np.nanmin(values[valid]))
        if vmax is None:
            vmax = float(np.nanmax(values[valid]))
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap(cmap_name)
        colors = cmap(norm(np.where(valid, values, vmin)))
        colors[..., 3] = np.where(valid, 0.96, 0.0)
        z = np.where(valid, 0.0, np.nan)
        ax.plot_surface(
            xx,
            yy,
            z,
            facecolors=colors,
            rstride=1,
            cstride=1,
            linewidth=0.0,
            edgecolor="none",
            antialiased=False,
            shade=False,
        )
        ax.set_title(title, pad=8)
        ax.set_axis_off()
        ax.view_init(elev=38, azim=-58)
        try:
            ax.set_proj_type("ortho")
            ax.set_box_aspect((active_layer.shape[1], active_layer.shape[0], 28))
        except Exception:
            pass
        ax.set_xlim(0, active_layer.shape[1])
        ax.set_ylim(0, active_layer.shape[0])
        ax.set_zlim(-1, 1)
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.72, fraction=0.035, pad=0.01)
        if label:
            cbar.set_label(label, fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    ax = fig.add_subplot(grid[0, 0], projection="3d")
    show_map_surface(
        ax,
        trend_ds,
        f"a) 3D view of trend map, layer z={global_z}",
        "viridis",
        label="trend",
    )
    ax = fig.add_subplot(grid[0, 1], projection="3d")
    show_map_surface(
        ax,
        rotation_ds,
        f"b) 3D view of rotation map, layer z={global_z}",
        "twilight",
        vmin=-110,
        vmax=75,
        label="degrees",
    )

    fig.suptitle("Masked 3D views of the HESS Roussillon input maps", fontsize=13)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 3D input-map figure to {path}")

def save_result_figure(
    path: Path,
    data: HESSData,
    args,
    summaries: dict[str, dict[str, object]],
) -> None:
    """Create generated plots of the GSTools realization diagnostics."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
    except Exception as exc:
        print(f"Skipping result figure; matplotlib unavailable: {exc}")
        return

    if "with_vertical_sampling" not in summaries:
        return

    domain = Domain(**summaries["with_vertical_sampling"]["domain"])
    run_root = Path(args.output_dir).expanduser() / args.mode
    with_dir = run_root / "with_vertical_sampling"
    with np.load(with_dir / "summary.npz") as with_npz:
        probability = with_npz["probability"]
        entropy = with_npz["entropy"]
        active = with_npz["active"].astype(bool)

    volume_path = with_dir / "realization_000.npy"
    if not volume_path.exists():
        print("Skipping result figure; realization_000.npy is missing.")
        return
    with_volume = np.load(volume_path)

    # Use the paper layers in full mode when they exist. For small mode, which
    # only has a compact 8-layer validation domain, use the same number of
    # representative relative layers.
    paper_global_layers = [0, 20, 40, 60]
    if all(domain.z0 <= z < domain.z0 + domain.nz for z in paper_global_layers):
        slice_ids = [z - domain.z0 for z in paper_global_layers]
    else:
        slice_ids = np.linspace(0, domain.nz - 1, 4).round().astype(int).tolist()
    analysis_z = 0

    facies_colors = list(FACIES_COLORS) + [INACTIVE_COLOR]
    facies_labels = list(FACIES_LABELS) + ["inactive"]
    facies_cmap = ListedColormap(facies_colors)
    prob_cmap = plt.get_cmap("magma").copy()
    prob_cmap.set_bad("white")

    def show_facies(ax, arr, active_layer, title):
        display = np.asarray(arr, dtype=float).copy()
        display[(display == INACTIVE) | ~active_layer] = 6
        ax.imshow(
            display,
            origin="lower",
            cmap=facies_cmap,
            vmin=0,
            vmax=6,
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    fig = plt.figure(figsize=(17.2, 10.0), constrained_layout=True)
    grid = fig.add_gridspec(
        3, 12, height_ratios=(1.0, 0.22, 1.0), hspace=0.07, wspace=0.08
    )
    top_axes = [fig.add_subplot(grid[0, 3 * j : 3 * (j + 1)]) for j in range(4)]
    legend_ax = fig.add_subplot(grid[1, :])
    bottom_axes = [fig.add_subplot(grid[2, 4 * j : 4 * (j + 1)]) for j in range(3)]

    for j, local_z in enumerate(slice_ids):
        show_facies(
            top_axes[j],
            with_volume[local_z],
            active[local_z],
            f"{chr(ord('a') + j)}) Realization layer z={domain.z0 + local_z}",
        )

    legend_ax.axis("off")
    legend_ax.legend(
        handles=[
            Patch(facecolor=c, edgecolor="0.4", label=l)
            for c, l in zip(facies_colors, facies_labels)
        ],
        loc="center",
        ncol=len(facies_labels),
        frameon=False,
        fontsize=11,
        columnspacing=1.2,
        handlelength=1.8,
        handleheight=1.1,
    )

    prob_braided = np.ma.masked_where(~active[analysis_z], probability[1, analysis_z])
    im_prob = bottom_axes[0].imshow(
        prob_braided, origin="lower", cmap=prob_cmap, vmin=0, vmax=1
    )
    bottom_axes[0].set_title(f"e) P(braided river), layer z={domain.z0 + analysis_z}")
    bottom_axes[0].set_xticks([])
    bottom_axes[0].set_yticks([])
    fig.colorbar(im_prob, ax=bottom_axes[0], fraction=0.046, pad=0.02)

    entropy_layer = np.ma.masked_where(~active[analysis_z], entropy[analysis_z])
    im_entropy = bottom_axes[1].imshow(
        entropy_layer, origin="lower", cmap=prob_cmap, vmin=0, vmax=1
    )
    bottom_axes[1].set_title(f"f) Shannon entropy, layer z={domain.z0 + analysis_z}")
    bottom_axes[1].set_xticks([])
    bottom_axes[1].set_yticks([])
    fig.colorbar(im_entropy, ax=bottom_axes[1], fraction=0.046, pad=0.02)

    prob_meander = np.ma.masked_where(~active[analysis_z], probability[2, analysis_z])
    im_meander = bottom_axes[2].imshow(
        prob_meander, origin="lower", cmap=prob_cmap, vmin=0, vmax=1
    )
    bottom_axes[2].set_title(f"g) P(meandering river), layer z={domain.z0 + analysis_z}")
    bottom_axes[2].set_xticks([])
    bottom_axes[2].set_yticks([])
    fig.colorbar(im_meander, ax=bottom_axes[2], fraction=0.046, pad=0.02)

    fig.suptitle(
        "GSTools small-domain 2D diagnostics: realization layers, probabilities, and entropy",
        fontsize=13,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved result figure to {path}")

def save_facies_vertical_figure(
    path: Path,
    data: HESSData,
    args,
    summaries: dict[str, dict[str, object]],
) -> None:
    """Create a Fig. 10-style facies proportion and vertical-run comparison."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping facies/vertical figure; matplotlib unavailable: {exc}")
        return

    if "with_vertical_sampling" not in summaries:
        return

    domain = Domain(**summaries["with_vertical_sampling"]["domain"])
    ti_props = facies_counts(data.ti_facies.astype(float))
    hard_props = proportions_from_plain_counts(hard_data_facies_counts(data, domain))
    with_props = summaries["with_vertical_sampling"].get("facies_proportions", {})
    without_props = summaries.get("without_vertical_sampling", {}).get(
        "facies_proportions", {}
    )

    hard_hist = vertical_run_histogram_from_hard_data(data, domain)
    with_hists = summaries["with_vertical_sampling"].get("vertical_run_histograms", [])
    without_hists = summaries.get("without_vertical_sampling", {}).get(
        "vertical_run_histograms", []
    )
    n_bins = domain.nz + 1

    def mean_dissimilarity(run_hists, facies):
        values = [
            run_histogram_dissimilarity(hist[str(facies)], hard_hist[str(facies)], n_bins)
            for hist in run_hists
        ]
        values = [value for value in values if np.isfinite(value)]
        return float(np.mean(values)) if values else np.nan

    dissim_with = [mean_dissimilarity(with_hists, code) for code in FACIES]
    dissim_without = [mean_dissimilarity(without_hists, code) for code in FACIES]
    facies_names = (
        "Flood\nplain",
        "Braided\nriver",
        "Meandering\nriver",
        "Crevasse\nsplay",
        "Alluvial\nfan",
        "Levee",
    )

    fig, (ax_prop, ax_run) = plt.subplots(
        1, 2, figsize=(16.2, 6.1), constrained_layout=True
    )
    x = np.arange(len(FACIES))
    width = 0.19
    ax_prop.bar(
        x - 1.5 * width,
        [100.0 * ti_props[str(code)] for code in FACIES],
        width,
        label="training image",
        color="0.72",
        edgecolor="0.35",
        linewidth=0.5,
    )
    ax_prop.bar(
        x - 0.5 * width,
        [100.0 * hard_props[str(code)] for code in FACIES],
        width,
        label="hard data",
        color="0.35",
        edgecolor="0.35",
        linewidth=0.5,
    )
    ax_prop.bar(
        x + 0.5 * width,
        [100.0 * with_props.get(str(code), 0.0) for code in FACIES],
        width,
        label="with vertical sampling",
        color="#4c78a8",
        edgecolor="0.35",
        linewidth=0.5,
    )
    if without_props:
        ax_prop.bar(
            x + 1.5 * width,
            [100.0 * without_props.get(str(code), 0.0) for code in FACIES],
            width,
            label="without vertical sampling",
            color="#f58518",
            edgecolor="0.35",
            linewidth=0.5,
        )
    ax_prop.set_title("a) Facies percentages")
    ax_prop.set_xticks(x)
    ax_prop.set_xticklabels(facies_names)
    ax_prop.set_ylabel("percentage")
    ax_prop.grid(axis="y", alpha=0.25)
    ax_prop.legend(frameon=False, fontsize=8, ncol=2)

    run_codes = [
        code
        for code in FACIES
        if np.asarray(hard_hist[str(code)], dtype=float)[1:].sum() > 0.0
    ]
    if run_codes:
        xr = np.arange(len(run_codes))
        ax_run.bar(
            xr - width / 2,
            [dissim_with[code] for code in run_codes],
            width,
            label="with vertical sampling",
            color="#4c78a8",
            edgecolor="0.35",
            linewidth=0.5,
        )
        ax_run.bar(
            xr + width / 2,
            [dissim_without[code] for code in run_codes],
            width,
            label="without vertical sampling",
            color="#f58518",
            edgecolor="0.35",
            linewidth=0.5,
        )
        ax_run.set_xticks(xr)
        ax_run.set_xticklabels([facies_names[code] for code in run_codes])
    else:
        ax_run.text(
            0.5,
            0.5,
            "no vertical hard-data runs in selected domain",
            transform=ax_run.transAxes,
            ha="center",
            va="center",
        )
    ax_run.set_title("b) Vertical run-length dissimilarity")
    ax_run.set_ylabel("distance to hard-data run lengths")
    ax_run.grid(axis="y", alpha=0.25)
    ax_run.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Generated facies proportion and vertical-transition comparison",
        fontsize=13,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved facies/vertical comparison figure to {path}")

def save_cross_section_figure(
    path: Path,
    data: HESSData,
    args,
    summaries: dict[str, dict[str, object]],
) -> None:
    """Create a Fig. 11-style vertical-sampling cross-section comparison."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
    except Exception as exc:
        print(f"Skipping cross-section figure; matplotlib unavailable: {exc}")
        return

    if "with_vertical_sampling" not in summaries:
        return

    domain = Domain(**summaries["with_vertical_sampling"]["domain"])
    run_root = Path(args.output_dir).expanduser() / args.mode
    with_dir = run_root / "with_vertical_sampling"
    without_dir = run_root / "without_vertical_sampling"
    with_path = with_dir / "realization_000.npy"
    without_path = without_dir / "realization_000.npy"
    if not with_path.exists() or not without_path.exists():
        print("Skipping cross-section figure; both with/without VS realizations are required.")
        return

    with_volume = np.load(with_path)
    without_volume = np.load(without_path)
    mid_x = domain.nx // 2
    mid_y = domain.ny // 2

    facies_colors = list(FACIES_COLORS) + [INACTIVE_COLOR]
    facies_labels = list(FACIES_LABELS) + ["inactive"]
    facies_cmap = ListedColormap(facies_colors)

    def show_facies(ax, arr, title, xlabel=None, ylabel=None):
        display = np.asarray(arr, dtype=float)
        display[display == INACTIVE] = 6
        ax.imshow(
            display,
            origin="lower",
            cmap=facies_cmap,
            vmin=0,
            vmax=6,
            interpolation="nearest",
            aspect="auto",
        )
        ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xticks([])
        if ylabel:
            ax.set_ylabel(ylabel)
        else:
            ax.set_yticks([])

    fig = plt.figure(figsize=(15, 8.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=(1, 1, 0.9), hspace=0.14, wspace=0.08)
    ax_without_x = fig.add_subplot(grid[0, 0])
    ax_without_y = fig.add_subplot(grid[0, 1])
    ax_with_x = fig.add_subplot(grid[1, 0])
    ax_with_y = fig.add_subplot(grid[1, 1])
    ax_map = fig.add_subplot(grid[:, 2])

    show_facies(
        ax_without_x,
        without_volume[:, mid_y, :],
        "a) Without vertical sampling: x-z section",
        xlabel="x",
        ylabel="z",
    )
    show_facies(
        ax_without_y,
        without_volume[:, :, mid_x],
        "b) Without vertical sampling: y-z section",
        xlabel="y",
        ylabel="z",
    )
    show_facies(
        ax_with_x,
        with_volume[:, mid_y, :],
        "c) With vertical sampling: x-z section",
        xlabel="x",
        ylabel="z",
    )
    show_facies(
        ax_with_y,
        with_volume[:, :, mid_x],
        "d) With vertical sampling: y-z section",
        xlabel="y",
        ylabel="z",
    )

    show_facies(
        ax_map,
        with_volume[0],
        "e) Bottom layer and section traces",
    )
    ax_map.axhline(mid_y, color="white", linewidth=2.0)
    ax_map.axvline(mid_x, color="black", linewidth=2.0)
    ax_map.text(
        0.03,
        0.96,
        "white: x-z\nblack: y-z",
        transform=ax_map.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )

    fig.legend(
        handles=[
            Patch(facecolor=c, edgecolor="0.4", label=l)
            for c, l in zip(facies_colors, facies_labels)
        ],
        loc="lower center",
        ncol=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=8,
    )
    fig.suptitle(
        "Generated cross-section comparison for the vertical-sampling strategy",
        fontsize=13,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved cross-section figure to {path}")

def save_3d_volume_figure(
    path: Path,
    data: HESSData,
    args,
    summaries: dict[str, dict[str, object]],
) -> None:
    """Create 3D views similar in spirit to the paper uncertainty figures."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize, to_rgba
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception as exc:
        print(f"Skipping 3D volume figure; matplotlib unavailable: {exc}")
        return

    if "with_vertical_sampling" not in summaries:
        return

    domain = Domain(**summaries["with_vertical_sampling"]["domain"])
    run_root = Path(args.output_dir).expanduser() / args.mode
    with_dir = run_root / "with_vertical_sampling"

    with np.load(with_dir / "summary.npz") as with_npz:
        probability = with_npz["probability"]
        entropy = with_npz["entropy"]
        active = with_npz["active"].astype(bool)

    volume_path = with_dir / "realization_000.npy"
    if not volume_path.exists():
        print("Skipping 3D volume figure; realization_000.npy is missing.")
        return
    volume = np.load(volume_path)

    def as_xyz(arr):
        return np.transpose(arr, (2, 1, 0))

    def colors_as_xyz(arr):
        return np.transpose(arr, (2, 1, 0, 3))

    def style_3d(ax, title):
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=25, azim=-55)
        try:
            ax.set_box_aspect((domain.nx, domain.ny, max(domain.nz * 5, 1)))
        except Exception:
            pass

    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.1, 0.9), hspace=0.16, wspace=0.08)

    facies_mask = active & (volume != 0) & (volume != INACTIVE)
    facies_rgba = np.zeros((*volume.shape, 4), dtype=float)
    for code, color in zip(FACIES, FACIES_COLORS):
        facies_rgba[volume == code] = to_rgba(color, alpha=0.72)
    ax = fig.add_subplot(grid[0, 0], projection="3d")
    if np.any(facies_mask):
        ax.voxels(
            as_xyz(facies_mask),
            facecolors=colors_as_xyz(facies_rgba),
            edgecolor="none",
        )
    style_3d(ax, "a) 3D facies realization\n(non-floodplain facies shown)")

    entropy_mask = active & (entropy > 0.0)
    entropy_rgba = plt.get_cmap("magma")(Normalize(0.0, 1.0)(entropy))
    entropy_rgba[..., 3] = np.where(entropy_mask, 0.70, 0.0)
    ax = fig.add_subplot(grid[0, 1], projection="3d")
    if np.any(entropy_mask):
        ax.voxels(
            as_xyz(entropy_mask),
            facecolors=colors_as_xyz(entropy_rgba),
            edgecolor="none",
        )
    style_3d(ax, "b) 3D Shannon entropy\n(uncertain cells shown)")

    prob = probability[1]
    prob_mask = active & (prob > 0.0)
    prob_rgba = plt.get_cmap("viridis")(Normalize(0.0, 1.0)(prob))
    prob_rgba[..., 3] = np.where(prob_mask, 0.65, 0.0)
    ax = fig.add_subplot(grid[0, 2], projection="3d")
    if np.any(prob_mask):
        ax.voxels(
            as_xyz(prob_mask),
            facecolors=colors_as_xyz(prob_rgba),
            edgecolor="none",
        )
    style_3d(ax, "c) 3D P(braided river)\n(nonzero probabilities shown)")

    layer0 = 0
    cmap2d = plt.get_cmap("magma").copy()
    cmap2d.set_bad("white")
    lower_panels = [
        (np.ma.masked_where(~active[layer0], entropy[layer0]), "d) Entropy layer"),
        (
            np.ma.masked_where(~active[layer0], probability[1, layer0]),
            "e) P(braided river) layer",
        ),
        (
            np.ma.masked_where(~active[layer0], probability[2, layer0]),
            "f) P(meandering river) layer",
        ),
    ]
    for j, (arr, title) in enumerate(lower_panels):
        ax = fig.add_subplot(grid[1, j])
        im = ax.imshow(arr, origin="lower", cmap=cmap2d, vmin=0, vmax=1)
        ax.set_title(f"{title} z={domain.z0 + layer0}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(
        "Generated 3D diagnostics for the HESS Roussillon GSTools small-domain run",
        fontsize=13,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 3D volume figure to {path}")


def run_audit(args) -> None:
    data = load_hess_data(args)
    rotation_mean = aligned_rotation_mean(data.rotation_maps)
    mapped = map_hard_data(data.hard_data, data.z_origin, data.sim_grid.shape)
    active = data.sim_grid
    alignment = {
        "sim_grid_shape": tuple(int(s) for s in data.sim_grid.shape),
        "sim_grid_active": int(data.sim_grid.sum()),
        "trend": finite_stats(data.trend),
        "rotation_mean_degrees": finite_stats(rotation_mean),
        "active_cells_with_finite_trend": int((active & np.isfinite(data.trend)).sum()),
        "active_cells_with_finite_rotation": int(
            (active & np.isfinite(rotation_mean)).sum()
        ),
        "hard_data_rows": int(len(data.hard_data["facies"])),
        "hard_data_inside_mapped_grid": int(mapped["inside"].sum()),
        "hard_data_facies_counts": facies_counts(data.hard_data["facies"].astype(float)),
        "z_origin": float(data.z_origin),
        "aux_alignment_offsets": {"y": AUX_Y_OFFSET, "x": AUX_X_OFFSET},
    }
    ti_info = {
        "ti_facies_shape": tuple(int(s) for s in data.ti_facies.shape),
        "ti_trend_shape": tuple(int(s) for s in data.ti_trend.shape),
        "ti_facies_counts": facies_counts(data.ti_facies.astype(float)),
        "ti_trend": finite_stats(data.ti_trend),
    }
    domain = select_domain(data, args)
    domain_active = data.sim_grid[domain.zslice, domain.yslice, domain.xslice]
    domain_info = {
        **asdict(domain),
        "active": int(domain_active.sum()),
        "size": int(np.prod(domain.shape)),
        "active_fraction": float(domain_active.mean()),
    }
    report = {
        "source": {
            "paper": "https://hess.copernicus.org/articles/24/4997/2020/",
            "dataset": "https://doi.org/10.5281/zenodo.4110278",
            "archive": str(data.archive),
            "archive_md5": ARCHIVE_MD5,
        },
        "ti": ti_info,
        "alignment": alignment,
        "selected_small_domain": domain_info,
        "gstools_deesse_gaps": [
            "GSTools currently has one global threshold; paper used facies=0.05 and trend=0.25.",
            "GSTools uses one rotation map; paper used two maps as angular bounds.",
            "GSTools currently has no inactive-cell mask in Direct Sampling.",
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output_dir:
        out = Path(args.output_dir).expanduser()
        write_json(out / "audit.json", report)


def run_ensemble(
    data: HESSData,
    args,
    vertical_sampling: bool,
    label: str,
) -> dict[str, object]:
    ti = build_training_image(data, args)
    domain = select_domain(data, args)
    active = data.sim_grid[domain.zslice, domain.yslice, domain.xslice]
    transition_probs = transition_matrix_from_hard_data(data)
    run_dir = Path(args.output_dir).expanduser() / args.mode / label
    run_dir.mkdir(parents=True, exist_ok=True)

    counts = np.zeros((len(FACIES), *domain.shape), dtype=np.uint16)
    first_volume = None
    run_diagnostics = []
    run_histograms = []

    print(
        f"Running {args.realizations} realization(s) for {label} on domain "
        f"{asdict(domain)}; active fraction={active.mean():.3f}"
    )
    for i_real in range(args.realizations):
        seed = args.seed + i_real * 100_000
        print(f"realization {i_real + 1}/{args.realizations}, seed={seed}")
        volume, diagnostics = simulate_volume(
            data=data,
            ti=ti,
            domain=domain,
            args=args,
            seed=seed,
            vertical_sampling=vertical_sampling,
            transition_probs=transition_probs,
        )
        update_counts(counts, volume)
        diagnostics["facies_counts"] = facies_counts(volume.astype(float), volume != INACTIVE)
        run_diagnostics.append(diagnostics)
        run_histograms.append(vertical_run_histogram(volume, active))
        if first_volume is None:
            first_volume = volume.copy()
        if args.save_realizations:
            np.save(run_dir / f"realization_{i_real:03d}.npy", volume)

    probability = probability_from_counts(counts, args.realizations)
    entropy = entropy_from_probability(probability, active)
    proportions = proportions_from_counts(counts, args.realizations, active)
    comparisons = compare_to_paper_outputs(data, domain, probability, entropy, active)
    hashes = {"counts_sha256": array_sha256(counts)}
    if first_volume is not None:
        hashes["first_realization_sha256"] = array_sha256(first_volume)

    repeat_check = {"enabled": False}
    if args.repeat_check and first_volume is not None:
        print("Running repeat reproducibility check for first realization...")
        repeat_volume, _ = simulate_volume(
            data=data,
            ti=ti,
            domain=domain,
            args=args,
            seed=args.seed,
            vertical_sampling=vertical_sampling,
            transition_probs=transition_probs,
        )
        repeat_hash = array_sha256(repeat_volume)
        repeat_check = {
            "enabled": True,
            "seed": int(args.seed),
            "equal": bool(np.array_equal(first_volume, repeat_volume)),
            "first_realization_sha256": hashes["first_realization_sha256"],
            "repeat_sha256": repeat_hash,
        }
        if not repeat_check["equal"]:
            raise RuntimeError(
                "Repeat reproducibility check failed: first realization changed. "
                "Use --num-threads 1 and keep all settings fixed."
            )

    np.savez_compressed(
        run_dir / "summary.npz",
        counts=counts,
        probability=probability,
        entropy=entropy,
        active=active,
        domain=np.array(
            [domain.z0, domain.y0, domain.x0, domain.nz, domain.ny, domain.nx],
            dtype=int,
        ),
    )
    summary = {
        "source": {
            "paper": "https://hess.copernicus.org/articles/24/4997/2020/",
            "dataset": "https://doi.org/10.5281/zenodo.4110278",
            "archive": str(data.archive),
        },
        "mode": args.mode,
        "label": label,
        "domain": asdict(domain),
        "settings": {
            "realizations": args.realizations,
            "seed": args.seed,
            "num_threads": args.num_threads,
            "path": args.path,
            "scan_fraction": args.scan_fraction,
            "threshold": args.threshold,
            "cond_weight": args.cond_weight,
            "facies_neighbors": args.facies_neighbors,
            "trend_neighbors": args.trend_neighbors,
            "max_radius": args.max_radius,
            "rotation_map": args.rotation_map,
            "vertical_sampling": vertical_sampling,
            "vertical_facies": args.vertical_facies,
            "vertical_sample_rate": args.vertical_sample_rate,
            "max_successive_layers": args.max_successive_layers,
            "z_origin": data.z_origin,
        },
        "facies_proportions": proportions,
        "hashes": hashes,
        "paper_comparison": comparisons,
        "reproducibility_check": repeat_check,
        "realization_diagnostics": run_diagnostics,
        "vertical_run_histograms": run_histograms,
    }
    write_json(run_dir / "summary.json", summary)
    if first_volume is not None:
        save_quicklook(
            run_dir / "quicklook.png",
            data=data,
            domain=domain,
            first_volume=first_volume,
            probability=probability,
            entropy=entropy,
            title=f"HESS Roussillon GSTools {args.mode} ({label})",
        )
    print(f"Wrote {run_dir}")
    return summary


def load_combined_summary(args, mode: str) -> dict[str, dict[str, object]]:
    path = Path(args.output_dir).expanduser() / mode / "combined_summary.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing existing summary: {path}. Run `{mode}` before `figures`."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_figures(args) -> None:
    data = load_hess_data(args)
    summaries = load_combined_summary(args, args.figure_source_mode)
    plot_args = argparse.Namespace(**vars(args))
    plot_args.mode = args.figure_source_mode
    save_training_image_figure(gallery_training_image_path(args.figure_source_mode), data)
    save_input_overview_figure(
        gallery_input_path(args.figure_source_mode), data, plot_args, summaries
    )
    save_input_3d_figure(
        gallery_input_3d_path(args.figure_source_mode), data, plot_args, summaries
    )
    save_result_figure(
        gallery_output_path(args.figure_source_mode), data, plot_args, summaries
    )
    save_facies_vertical_figure(
        gallery_facies_vertical_path(args.figure_source_mode), data, plot_args, summaries
    )
    save_3d_volume_figure(
        gallery_volume_path(args.figure_source_mode), data, plot_args, summaries
    )
    save_cross_section_figure(
        gallery_cross_section_path(args.figure_source_mode), data, plot_args, summaries
    )


def run_mode(args) -> None:
    data = load_hess_data(args)
    labels = [("with_vertical_sampling", True)]
    if not args.skip_without_vertical:
        labels.append(("without_vertical_sampling", False))
    summaries = {}
    for label, enabled in labels:
        summaries[label] = run_ensemble(data, args, enabled, label)
    out = Path(args.output_dir).expanduser() / args.mode / "combined_summary.json"
    write_json(out, summaries)
    print(f"Wrote combined summary to {out}")
    if args.gallery_figure:
        save_training_image_figure(gallery_training_image_path(args.mode), data)
        save_input_overview_figure(gallery_input_path(args.mode), data, args, summaries)
        save_input_3d_figure(gallery_input_3d_path(args.mode), data, args, summaries)
        save_result_figure(gallery_output_path(args.mode), data, args, summaries)
        save_facies_vertical_figure(
            gallery_facies_vertical_path(args.mode), data, args, summaries
        )
        save_3d_volume_figure(gallery_volume_path(args.mode), data, args, summaries)
        save_cross_section_figure(gallery_cross_section_path(args.mode), data, args, summaries)


def parse_vertical_facies(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    out = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    bad = [facies for facies in out if facies not in FACIES]
    if bad:
        raise argparse.ArgumentTypeError(f"Invalid facies code(s): {bad}")
    return out


def parse_clip_trend(value: str | None) -> tuple[float, float] | None:
    if value is None or value.lower() in {"none", "false", "no"}:
        return None
    parts = [float(item.strip()) for item in value.split(",")]
    if len(parts) != 2 or parts[0] >= parts[1]:
        raise argparse.ArgumentTypeError("--clip-trend expects min,max")
    return (parts[0], parts[1])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(__doc__ or ""),
    )
    parser.add_argument(
        "mode",
        choices=("audit", "small", "full", "figures"),
        help=(
            "audit inputs, run a small real subdomain, run the full domain, "
            "or redraw figures from an existing run."
        ),
    )
    parser.add_argument(
        "--figure-source-mode",
        choices=("small", "full"),
        default="small",
        help="Existing run to use when mode is `figures`.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=default_archive_path(),
        help="Path to roussillon_mps_2020_dallalba.zip.",
    )
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download the Zenodo archive if it is not found locally.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Directory for diagnostics, realizations, summaries, and figures.",
    )
    parser.add_argument(
        "--z-origin",
        default="auto",
        help=(
            "Transformed hard-data z origin in metres, or 'auto' to calibrate "
            "against the archived with-vertical-sampling output."
        ),
    )
    parser.add_argument("--z-search-min", type=float, default=-240.0)
    parser.add_argument("--z-search-max", type=float, default=-160.0)
    parser.add_argument("--z-search-step", type=float, default=1.0)

    parser.add_argument(
        "--small-shape",
        type=int,
        nargs=3,
        metavar=("NZ", "NY", "NX"),
        default=(8, 64, 64),
        help="Small-mode real subdomain shape in archive axis order z y x.",
    )
    parser.add_argument("--realizations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=202010)
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Thread count for Direct Sampling; keep 1 for byte-identical reruns.",
    )
    parser.add_argument(
        "--repeat-check",
        action="store_true",
        help="Rerun the first realization and fail unless it is byte-identical.",
    )
    parser.add_argument(
        "--record-timing",
        action="store_true",
        help="Store per-layer wall-clock timings in JSON summaries.",
    )
    parser.add_argument(
        "--path",
        choices=("random", "sequential"),
        default="random",
        help="Direct Sampling simulation path.",
    )
    parser.add_argument("--scan-fraction", type=float, default=0.75)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Acceptance threshold; 0 selects the best candidate (None).",
    )
    parser.add_argument("--cond-weight", type=float, default=1.0)
    parser.add_argument("--facies-neighbors", type=int, default=24)
    parser.add_argument("--trend-neighbors", type=int, default=5)
    parser.add_argument("--max-radius", type=float, default=20.0)
    parser.add_argument(
        "--clip-trend",
        type=parse_clip_trend,
        default=None,
        metavar="MIN,MAX",
        help="Optionally clip TI and SG trend values, e.g. --clip-trend 0,1.",
    )
    parser.add_argument(
        "--rotation-map",
        choices=("mean", "lower", "upper"),
        default="mean",
        help="How to collapse the two archived rotation maps to GSTools' one map.",
    )

    parser.add_argument(
        "--vertical-facies",
        type=parse_vertical_facies,
        default=(1, 2, 4),
        help=(
            "Comma-separated facies codes sampled for vertical conditioning. "
            "Default 1,2,4 follows braided/meandering/alluvial-fan interpretation."
        ),
    )
    parser.add_argument("--vertical-sample-rate", type=float, default=0.01)
    parser.add_argument("--max-successive-layers", type=int, default=6)
    parser.add_argument(
        "--min-vertical-samples-per-facies",
        type=int,
        default=1,
        help="Keeps small mode visibly exercising vertical sampling.",
    )
    parser.add_argument(
        "--skip-without-vertical",
        action="store_true",
        help="Do not run the comparison ensemble without vertical sampling.",
    )
    parser.add_argument(
        "--save-realizations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save each realization as .npy in addition to ensemble summaries.",
    )
    parser.add_argument(
        "--gallery-figure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the generated PNG figures displayed by the gallery example.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.realizations is None:
        args.realizations = (
            1 if args.mode == "audit" else (2 if args.mode in {"small", "figures"} else 50)
        )
    try:
        if args.mode == "audit":
            run_audit(args)
        elif args.mode == "figures":
            run_figures(args)
        else:
            run_mode(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


def _called_as_cli() -> bool:
    """Avoid accidental execution when Sphinx-gallery imports/runs examples."""
    return len(sys.argv) > 1 and sys.argv[1] in {
        "audit",
        "small",
        "full",
        "-h",
        "--help",
    }


if __name__ == "__main__" and _called_as_cli():
    raise SystemExit(main())
