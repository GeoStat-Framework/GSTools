"""Benchmark representative two-point statistics workflows.

Usage:
    cd /path/to/MPS-Tools/GSTools
    # See benchmarks/README.md for ASV and optional cProfile setup.
    asv machine --yes
    asv run --quick --show-stderr --bench benchmark_two_point_statistics
    asv run main^! --bench benchmark_two_point_statistics
    asv run
    asv publish
    asv preview
    asv compare main~1 main

Backend speedup should be interpreted as:
    speedup = cython_fallback_time / rust_core_time

Values greater than 1.0 mean the Rust backend is faster on the same machine
for the same benchmark, case, commit, and thread count.

By default the suite uses one GSTools thread. For local OpenMP scaling
experiments, set GSTOOLS_BENCHMARK_THREADS before ASV discovers the
benchmarks, for example:
    GSTOOLS_BENCHMARK_THREADS=1,2,4,8 \
        asv --config asv.openmp.conf.json run 'main^!'
"""

from __future__ import annotations

import contextlib
import os

import gstools as gs
import numpy as np

AVAILABLE_BACKENDS = ("cython_fallback", "rust_core")


def _configured_backends():
    """Return backend labels requested through the environment."""
    raw = os.environ.get("GSTOOLS_BENCHMARK_BACKENDS", "")
    if not raw.strip():
        return AVAILABLE_BACKENDS
    backends = tuple(item.strip() for item in raw.split(",") if item.strip())
    unknown = sorted(set(backends) - set(AVAILABLE_BACKENDS))
    if unknown:
        raise ValueError(
            "GSTOOLS_BENCHMARK_BACKENDS contains unknown backends: "
            + ", ".join(unknown)
        )
    if not backends:
        raise ValueError("GSTOOLS_BENCHMARK_BACKENDS did not define backends")
    return backends


BACKENDS = _configured_backends()


def _configured_thread_counts():
    """Return thread count integers requested through the environment."""
    raw = os.environ.get("GSTOOLS_BENCHMARK_THREADS", "1")
    thread_counts = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if item.startswith("threads_"):
            value = item[len("threads_") :]
        else:
            value = item
        thread_counts.append(int(value))
    if not thread_counts:
        raise ValueError("GSTOOLS_BENCHMARK_THREADS did not define threads")
    return tuple(thread_counts)


THREAD_COUNTS = _configured_thread_counts()
VARIOGRAM_CASES = (
    "full_900",
    "sampled_5000_to_1500",
    "sampled_15000_to_4500",
)
KRIGE_CASES = ("small_30x500", "large_120x2000", "extra_large_360x6000")
FIELD_CASES = (
    "srf_unstructured_randmeth",
    "srf_structured_randmeth",
    "srf_structured_fourier",
    "condsrf_unstructured",
)


@contextlib.contextmanager
def gstools_backend(use_core, num_threads):
    """Temporarily select the GSTools backend and thread count."""
    previous = (
        gs.config._GSTOOLS_CORE_AVAIL,
        gs.config.USE_GSTOOLS_CORE,
        gs.config.NUM_THREADS,
    )
    try:
        if use_core:
            if not previous[0]:
                raise NotImplementedError("gstools_core is not available")
            gs.config._GSTOOLS_CORE_AVAIL = True
            gs.config.USE_GSTOOLS_CORE = True
        else:
            gs.config._GSTOOLS_CORE_AVAIL = False
            gs.config.USE_GSTOOLS_CORE = False
        gs.config.NUM_THREADS = num_threads
        yield
    finally:
        (
            gs.config._GSTOOLS_CORE_AVAIL,
            gs.config.USE_GSTOOLS_CORE,
            gs.config.NUM_THREADS,
        ) = previous


def _use_core(backend):
    """Map a benchmark backend label to the GSTools core flag."""
    if backend == "rust_core":
        return True
    if backend == "cython_fallback":
        return False
    raise ValueError(f"Unknown backend: {backend}")


def _random_points(seed, count, scale):
    """Create a deterministic two-dimensional point cloud."""
    rng = np.random.RandomState(seed)
    return rng.rand(count) * scale, rng.rand(count) * scale


def _smooth_field(x, y):
    """Evaluate a smooth deterministic field on matching coordinates."""
    return np.sin(x / 10.0) + np.cos(y / 15.0)


def _make_variogram_data(seed, count, scale=100.0):
    """Create positions, values, and lag bins for variogram estimation."""
    x, y = _random_points(seed, count, scale)
    field = _smooth_field(x, y)
    bins = np.linspace(0.0, scale * 0.6, 16)
    return (x, y), field, bins


def _make_krige_data(seed, cond_count, target_count, scale=50.0):
    """Create conditioning and target data for kriging workflows."""
    rng = np.random.RandomState(seed)
    cond_x = rng.rand(cond_count) * scale
    cond_y = rng.rand(cond_count) * scale
    cond_val = _smooth_field(cond_x, cond_y)
    target_pos = (
        rng.rand(target_count) * scale,
        rng.rand(target_count) * scale,
    )
    return (cond_x, cond_y), cond_val, target_pos


def _check_backend_available(backend):
    """Skip Rust benchmark rows when ``gstools_core`` is unavailable."""
    if backend == "rust_core" and not gs.config._GSTOOLS_CORE_AVAIL:
        raise NotImplementedError("gstools_core is not available")


class VariogramBenchmarks:
    """Measure variogram estimation cases for each backend, case, and thread count."""

    params = [BACKENDS, VARIOGRAM_CASES, THREAD_COUNTS]
    param_names = ["backend", "case", "threads"]

    def setup_cache(self):
        """Create deterministic variogram inputs once per ASV environment."""
        return {
            "full_900": _make_variogram_data(20220501, 900),
            "sampled_5000_to_1500": _make_variogram_data(20220502, 5000),
            "sampled_15000_to_4500": _make_variogram_data(20220503, 15000),
        }

    def setup(self, data, backend, case, threads):
        """Skip backend rows that cannot run in the current environment."""
        _check_backend_available(backend)

    def time_variogram_estimate(self, data, backend, case, threads):
        """Run one variogram estimation case."""
        pos, field, bins = data[case]
        kwargs = {}
        if case == "sampled_5000_to_1500":
            kwargs = {"sampling_size": 1500, "sampling_seed": 20220504}
        if case == "sampled_15000_to_4500":
            kwargs = {"sampling_size": 4500, "sampling_seed": 20220505}
        with gstools_backend(_use_core(backend), threads):
            gs.vario_estimate(
                pos,
                field,
                bins,
                mesh_type="unstructured",
                return_counts=True,
                **kwargs,
            )

    def peakmem_variogram_estimate(self, data, backend, case, threads):
        """Measure peak memory for variogram estimation case."""
        pos, field, bins = data[case]
        kwargs = {}
        if case == "sampled_5000_to_1500":
            kwargs = {"sampling_size": 1500, "sampling_seed": 20220504}
        if case == "sampled_15000_to_4500":
            kwargs = {"sampling_size": 4500, "sampling_seed": 20220505}
        with gstools_backend(_use_core(backend), threads):
            gs.vario_estimate(
                pos,
                field,
                bins,
                mesh_type="unstructured",
                return_counts=True,
                **kwargs,
            )


class KrigingBenchmarks:
    """Measure global kriging cases for each backend, case, and thread count."""

    params = [BACKENDS, KRIGE_CASES, THREAD_COUNTS]
    param_names = ["backend", "case", "threads"]

    def setup_cache(self):
        """Create deterministic kriging inputs once per ASV environment."""
        return {
            "small_30x500": _make_krige_data(20220506, 30, 500),
            "large_120x2000": _make_krige_data(20220507, 120, 2000),
            "extra_large_360x6000": _make_krige_data(20220508, 360, 6000),
        }

    def setup(self, data, backend, case, threads):
        """Skip backend rows that cannot run in the current environment."""
        _check_backend_available(backend)

    def time_global_krige(self, data, backend, case, threads):
        """Run one global kriging case."""
        cond_pos, cond_val, target_pos = data[case]
        model = gs.Exponential(dim=2, var=1.5, len_scale=12.0, nugget=0.05)
        krige = gs.Krige(
            model,
            cond_pos,
            cond_val,
            exact=False,
            cond_err=0.05,
        )
        with gstools_backend(_use_core(backend), threads):
            krige(
                target_pos,
                mesh_type="unstructured",
                return_var=True,
                store=False,
            )

    def peakmem_global_krige(self, data, backend, case, threads):
        """Measure peak memory for global kriging case."""
        cond_pos, cond_val, target_pos = data[case]
        model = gs.Exponential(dim=2, var=1.5, len_scale=12.0, nugget=0.05)
        krige = gs.Krige(
            model,
            cond_pos,
            cond_val,
            exact=False,
            cond_err=0.05,
        )
        with gstools_backend(_use_core(backend), threads):
            krige(
                target_pos,
                mesh_type="unstructured",
                return_var=True,
                store=False,
            )


class RandomFieldBenchmarks:
    """Measure SRF and CondSRF generation cases for each backend, case, and thread count."""

    params = [BACKENDS, FIELD_CASES, THREAD_COUNTS]
    param_names = ["backend", "case", "threads"]

    def setup_cache(self):
        """Create deterministic random-field inputs once per ASV environment."""
        return {
            "unstructured_pos": _random_points(20220509, 2000, 100.0),
            "structured_pos": (
                np.linspace(0.0, 100.0, 64),
                np.linspace(0.0, 100.0, 64),
            ),
            "condsrf": _make_krige_data(20220510, 40, 1000),
        }

    def setup(self, data, backend, case, threads):
        """Skip backend rows that cannot run in the current environment."""
        _check_backend_available(backend)

    def time_field_generation(self, data, backend, case, threads):
        """Run one SRF or CondSRF case by label."""
        with gstools_backend(_use_core(backend), threads):
            if case == "srf_unstructured_randmeth":
                self._run_srf_unstructured(data)
            elif case == "srf_structured_randmeth":
                self._run_srf_structured(data)
            elif case == "srf_structured_fourier":
                self._run_srf_fourier(data)
            elif case == "condsrf_unstructured":
                self._run_condsrf(data)
            else:
                raise ValueError(f"Unknown field benchmark case: {case}")

    def peakmem_field_generation(self, data, backend, case, threads):
        """Measure peak memory for one SRF or CondSRF case."""
        with gstools_backend(_use_core(backend), threads):
            if case == "srf_unstructured_randmeth":
                self._run_srf_unstructured(data)
            elif case == "srf_structured_randmeth":
                self._run_srf_structured(data)
            elif case == "srf_structured_fourier":
                self._run_srf_fourier(data)
            elif case == "condsrf_unstructured":
                self._run_condsrf(data)
            else:
                raise ValueError(f"Unknown field benchmark case: {case}")

    def _run_srf_unstructured(self, data):
        """Run unstructured SRF generation with the RandMeth generator."""
        model = gs.Exponential(dim=2, var=2.0, len_scale=8.0)
        srf = gs.SRF(model, mean=1.0, seed=20220508, mode_no=512)
        return srf(data["unstructured_pos"], mesh_type="unstructured")

    def _run_srf_structured(self, data):
        """Run structured SRF generation with the RandMeth generator."""
        model = gs.Exponential(dim=2, var=2.0, len_scale=8.0)
        srf = gs.SRF(model, mean=1.0, seed=20220509, mode_no=512)
        return srf(data["structured_pos"], mesh_type="structured")

    def _run_srf_fourier(self, data):
        """Run structured SRF generation with the Fourier generator."""
        model = gs.Gaussian(dim=2, var=2.0, len_scale=30.0)
        srf = gs.SRF(
            model,
            generator="Fourier",
            period=[100.0, 100.0],
            mode_no=[32, 32],
            seed=20220510,
        )
        return srf(data["structured_pos"], mesh_type="structured")

    def _run_condsrf(self, data):
        """Run conditioned random-field generation on unstructured targets."""
        cond_pos, cond_val, target_pos = data["condsrf"]
        model = gs.Exponential(dim=2, var=1.5, len_scale=12.0, nugget=0.05)
        krige = gs.Krige(
            model,
            cond_pos,
            cond_val,
            exact=False,
            cond_err=0.05,
        )
        cond_srf = gs.CondSRF(krige, seed=20220511, mode_no=512)
        return cond_srf(
            target_pos,
            mesh_type="unstructured",
            seed=20220512,
            store=False,
            krige_store=False,
        )
