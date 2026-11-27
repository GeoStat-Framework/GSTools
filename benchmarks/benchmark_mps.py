"""Benchmark MPS (Multiple Point Statistics) workflows: TrainingImage and DirectSampling.

See benchmarks/README.md for ASV setup and the current MPS benchmark limits.

MPS Direct Sampling is Rust-only: it requires gstools_core>=1.4.0 and computes
its numerical distances and scans in Rust. The DirectSamplingBenchmarks
therefore run a single "core" backend (the Rust engine); there is no pure-Python
MPS backend to compare against.

Benchmarks that run against a gstools commit that does not include
gstools.mps (e.g. the main branch before MPS was merged) are automatically
skipped rather than failing, so this file can live in the repo before MPS
lands on main.

All Direct Sampling simulation work uses the complete Rust engine.
"""

from __future__ import annotations

import numpy as np

# MPS classes are imported directly from gstools.mps so this benchmark file
# works regardless of whether they are re-exported from the top-level gstools
# package. The try/except lets ASV skip when MPS is absent or its required
# GSTools-Core MPS kernels are unavailable.
try:
    from gstools.mps import DirectSampling, MPSModel, TrainingImage, Variable

    _MPS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _MPS_AVAILABLE = False
    DirectSampling = MPSModel = TrainingImage = Variable = None  # type: ignore[assignment,misc]


def _check_mps_available():
    """Skip when MPS or its required GSTools-Core kernels are unavailable."""
    if not _MPS_AVAILABLE:
        raise NotImplementedError(
            "gstools.mps is not available in this gstools version; "
            "benchmark skipped for pre-MPS commits."
        )


# ---------------------------------------------------------------------------
# Case definitions
# ---------------------------------------------------------------------------

TI_CASES = (
    "cat_60x60",  # categorical 60×60 synthetic channel TI
    "cat_150x150",  # categorical 150×150 (larger TI, measures scaling)
    "cont_60x60",  # continuous 60×60 with l1 distance
    "multivar_60x60",  # multivariate 2-variable (categorical + continuous) 60×60
)

DS_CASES = (
    "cat_dsbc_small",  # categorical TI=40×40, SG=20×20, n=8,  f=0.3, t=0.0
    "cat_dsbc_medium",  # categorical TI=60×60, SG=30×30, n=12, f=0.3, t=0.0  (baseline)
    "cat_dsbc_large",  # categorical TI=120×120, SG=40×40, n=16, f=0.3, t=0.0
    "cat_ds_medium",  # categorical TI=60×60, SG=30×30, n=12, f=0.3, t=0.1  (DS mode)
    "cont_l1_medium",  # continuous   TI=60×60, SG=30×30, n=12, f=0.3, t=0.0, l1
    "cat_dsbc_hiscan",  # categorical TI=60×60, SG=30×30, n=12, f=0.8, t=0.0  (high scan)
    "cat_dsbc_highk",  # categorical TI=60×60, SG=30×30, n=24, f=0.3, t=0.0  (more neighbors)
    "cat_dsbc_cond",  # categorical TI=60×60, SG=30×30, n=12, f=0.3, t=0.0, conditioned
    "cat_dsbc_cleanup",  # categorical TI=60×60, SG=30×30, n=12, f=0.3, t=0.0, post_processing=3
)

# Full parameter specification for each DS case.
_DS_SPECS = {
    "cat_dsbc_small": {
        "ti_shape": (40, 40),
        "sg_shape": (20, 20),
        "categorical": True,
        "distance": "l1",
        "n_neighbors": 8,
        "scan_fraction": 0.3,
        "threshold": 0.0,
        "conditioned": False,
    },
    "cat_dsbc_medium": {
        "ti_shape": (60, 60),
        "sg_shape": (30, 30),
        "categorical": True,
        "distance": "l1",
        "n_neighbors": 12,
        "scan_fraction": 0.3,
        "threshold": 0.0,
        "conditioned": False,
    },
    "cat_dsbc_large": {
        "ti_shape": (120, 120),
        "sg_shape": (40, 40),
        "categorical": True,
        "distance": "l1",
        "n_neighbors": 16,
        "scan_fraction": 0.3,
        "threshold": 0.0,
        "conditioned": False,
    },
    "cat_ds_medium": {
        "ti_shape": (60, 60),
        "sg_shape": (30, 30),
        "categorical": True,
        "distance": "l1",
        "n_neighbors": 12,
        "scan_fraction": 0.3,
        "threshold": 0.1,
        "conditioned": False,
    },
    "cont_l1_medium": {
        "ti_shape": (60, 60),
        "sg_shape": (30, 30),
        "categorical": False,
        "distance": "l1",
        "n_neighbors": 12,
        "scan_fraction": 0.3,
        "threshold": 0.0,
        "conditioned": False,
    },
    "cat_dsbc_hiscan": {
        "ti_shape": (60, 60),
        "sg_shape": (30, 30),
        "categorical": True,
        "distance": "l1",
        "n_neighbors": 12,
        "scan_fraction": 0.8,
        "threshold": 0.0,
        "conditioned": False,
    },
    "cat_dsbc_highk": {
        "ti_shape": (60, 60),
        "sg_shape": (30, 30),
        "categorical": True,
        "distance": "l1",
        "n_neighbors": 24,
        "scan_fraction": 0.3,
        "threshold": 0.0,
        "conditioned": False,
    },
    "cat_dsbc_cond": {
        "ti_shape": (60, 60),
        "sg_shape": (30, 30),
        "categorical": True,
        "distance": "l1",
        "n_neighbors": 12,
        "scan_fraction": 0.3,
        "threshold": 0.0,
        "conditioned": True,
        "n_cond": 30,
    },
    "cat_dsbc_cleanup": {
        "ti_shape": (60, 60),
        "sg_shape": (30, 30),
        "categorical": True,
        "distance": "l1",
        "n_neighbors": 12,
        "scan_fraction": 0.3,
        "threshold": 0.0,
        "conditioned": False,
        "post_processing": 3,
    },
}

# One deterministic seed per DS case; offset from a fixed base so they differ.
_DS_SEEDS = {case: 20250101 + i for i, case in enumerate(DS_CASES)}


# ---------------------------------------------------------------------------
# Data-construction helpers
# ---------------------------------------------------------------------------


def _make_categorical_ti(ti_shape):
    """Synthetic binary channel TI matching the MPS tutorial examples (example 00)."""
    gx, gy = np.meshgrid(
        np.arange(ti_shape[0]), np.arange(ti_shape[1]), indexing="ij"
    )
    return ((np.sin(gx / 5.0) + np.sin((gx + gy) / 8.0)) > 0).astype(np.int32)


def _make_continuous_ti(ti_shape):
    """Smooth continuous TI (sine-cosine product), range [-1, 1] (example 02)."""
    gx, gy = np.meshgrid(
        np.arange(ti_shape[0]), np.arange(ti_shape[1]), indexing="ij"
    )
    return np.sin(gx / 6.0) * np.cos(gy / 8.0)


def _make_conditioning_data(ti_data, sg_shape, n_cond, seed):
    """Draw n_cond conditioning points on the SG; values read from ti_data.

    Returns (cond_pos, cond_val) matching the set_condition() API.
    """
    rng = np.random.RandomState(seed)
    n_nodes = sg_shape[0] * sg_shape[1]
    flat_idx = rng.choice(n_nodes, size=min(n_cond, n_nodes), replace=False)
    cond_xi, cond_yi = np.unravel_index(flat_idx, sg_shape)
    # Read TI values at the corresponding positions (SG is always smaller than TI
    # in the current specs, so no wrapping is needed; kept for safety).
    ti_xi = cond_xi % ti_data.shape[0]
    ti_yi = cond_yi % ti_data.shape[1]
    cond_val = ti_data[ti_xi, ti_yi].astype(float)
    cond_pos = [cond_xi.astype(float), cond_yi.astype(float)]
    return cond_pos, cond_val


# ---------------------------------------------------------------------------
# TrainingImageBenchmarks
# ---------------------------------------------------------------------------


class TrainingImageBenchmarks:
    """Measure TrainingImage construction for different sizes and data types.

    Benchmarks the cost of wrapping raw numpy arrays in a TrainingImage.
    Construction includes internal bookkeeping — variable creation, shape
    validation, distance-function dispatch, and NaN-presence detection.
    The numerical distance kernels used during simulation are in Rust.
    """

    params = [TI_CASES]
    param_names = ["case"]

    def setup_cache(self):
        """Build raw TI numpy arrays once per ASV environment."""
        ti_cat_60 = _make_categorical_ti((60, 60))
        return {
            "cat_60x60": ti_cat_60,
            "cat_150x150": _make_categorical_ti((150, 150)),
            "cont_60x60": _make_continuous_ti((60, 60)),
            "multivar_60x60": (ti_cat_60, _make_continuous_ti((60, 60))),
        }

    def setup(self, data, case):
        """Skip when benchmarking a gstools commit that predates gstools.mps."""
        _check_mps_available()

    def time_training_image_construct(self, data, case):
        """Construct a TrainingImage from raw arrays."""
        self._build(data, case)

    def peakmem_training_image_construct(self, data, case):
        """Peak memory for TrainingImage construction."""
        self._build(data, case)

    def _build(self, data, case):
        raw = data[case]
        if case in ("cat_60x60", "cat_150x150"):
            TrainingImage(raw, categorical=True)
        elif case == "cont_60x60":
            TrainingImage(raw, categorical=False, distance="l1")
        elif case == "multivar_60x60":
            ti_cat, ti_cont = raw
            v0 = Variable("facies", ti_cat, categorical=True)
            v1 = Variable(
                "porosity", ti_cont, categorical=False, distance="l1"
            )
            TrainingImage([v0, v1])
        else:
            raise ValueError(f"Unknown TI case: {case!r}")


# ---------------------------------------------------------------------------
# DirectSamplingBenchmarks
# ---------------------------------------------------------------------------


class DirectSamplingBenchmarks:
    """Measure DirectSampling simulation time and peak memory.

    Each case exercises a specific combination of TI size, simulation-grid
    size, algorithm mode, variable type, search parameters, and optional
    conditioning.  The DirectSampling object is constructed in setup() so
    that time_simulate() and peakmem_simulate() measure only the simulation
    call itself, isolating the per-node scan from one-time construction cost.

    Case encoding: ``{var_type}_{mode}_{size_modifier}``
      - var_type: ``cat`` (categorical) or ``cont`` (continuous)
      - mode:     ``dsbc`` (threshold=0, DSBC best-candidate) or
                  ``ds`` (threshold>0, greedy DS)
      - size/mod: ``small``, ``medium``, ``large``, ``hiscan``, ``highk``,
                  ``cond``

    MPS is Rust-only, so the single "core" backend runs the Rust engine (there
    is no pure-Python MPS backend to compare against).
    """

    # Repeat each Rust-backed simulation three times for a stable median.
    # repeat=3 limits total measurement time while still giving a stable median.
    # Override with ASV's --repeat flag when higher resolution is needed.
    number = 1
    repeat = 3
    # Allow up to 5 minutes per case; the large categorical case can be slow.
    timeout = 300

    BACKENDS = ("core",)

    params = [DS_CASES, BACKENDS]
    param_names = ["case", "backend"]

    def setup_cache(self):
        """Build raw TI arrays and conditioning data once per ASV environment."""
        cache = {}
        for case, spec in _DS_SPECS.items():
            if spec["categorical"]:
                ti_data = _make_categorical_ti(spec["ti_shape"])
            else:
                ti_data = _make_continuous_ti(spec["ti_shape"])

            entry = {"ti_data": ti_data}
            if spec["conditioned"]:
                cond_pos, cond_val = _make_conditioning_data(
                    ti_data,
                    spec["sg_shape"],
                    spec.get("n_cond", 30),
                    seed=20250104,
                )
                entry["cond_pos"] = cond_pos
                entry["cond_val"] = cond_val

            cache[case] = entry
        return cache

    def setup(self, data, case, backend):
        """Build TrainingImage, MPSModel, and DirectSampling ready for simulation.

        Called by ASV before each timing/memory measurement.  Object construction
        is excluded from the timed region; only the simulation call is measured.
        Skips when benchmarking a gstools commit that predates gstools.mps.
        """
        _check_mps_available()
        import gstools.config as _cfg

        # MPS is Rust-only; the single "core" backend runs the Rust engine.
        try:
            import gstools_core  # noqa: F401
        except ImportError:
            raise NotImplementedError(
                "gstools_core not installed; MPS benchmark skipped."
            )
        _cfg.USE_GSTOOLS_CORE = True

        spec = _DS_SPECS[case]
        ti = TrainingImage(
            data[case]["ti_data"],
            categorical=spec["categorical"],
            distance=spec["distance"],
            n_neighbors=spec["n_neighbors"],
        )
        model = MPSModel(
            ti,
            scan_fraction=spec["scan_fraction"],
            threshold=spec["threshold"],
            post_processing=spec.get("post_processing", 0),
        )
        self.ds = DirectSampling(model)
        self.grid = [np.arange(s, dtype=float) for s in spec["sg_shape"]]
        self.seed = _DS_SEEDS[case]

        if spec["conditioned"]:
            self.ds.set_condition(
                data[case]["cond_pos"], data[case]["cond_val"]
            )

    def time_simulate(self, data, case, backend):
        """Run one Direct Sampling simulation."""
        self.ds(self.grid, seed=self.seed, store=False)

    def peakmem_simulate(self, data, case, backend):
        """Peak memory for one Direct Sampling simulation."""
        self.ds(self.grid, seed=self.seed, store=False)
