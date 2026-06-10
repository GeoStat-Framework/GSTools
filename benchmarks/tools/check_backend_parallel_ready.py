#!/usr/bin/env python
"""Check whether GSTools benchmark backends are ready for parallel runs.

This is a fast CI probe. It verifies that GSTools-Cython accepts explicit
OpenMP thread counts and that the Rust backend can run a small workflow while
GSTools is configured with more than one thread.
"""

from __future__ import annotations

import argparse
import importlib
import sys

import numpy as np

MODULES = {
    "variogram": "gstools_cython.variogram",
    "field": "gstools_cython.field",
    "krige": "gstools_cython.krige",
}
EXPLICIT_THREAD_COUNTS = (2, 4, 8)


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--threads",
        default=2,
        type=int,
        help="Thread count to request for the Rust backend smoke workflow.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-module Cython OpenMP thread details.",
    )
    parser.add_argument(
        "--cython-only",
        action="store_true",
        help="Only check GSTools-Cython OpenMP readiness.",
    )
    parser.add_argument(
        "--require-default-openmp",
        action="store_true",
        help=(
            "Also fail when GSTools-Cython reports only one default thread. "
            "CI does not use this because some runners default to one thread "
            "while still accepting explicit OpenMP thread counts."
        ),
    )
    return parser.parse_args()


def package_version(package_name):
    """Return an installed package version or a clear missing marker."""
    try:
        package = importlib.import_module(package_name)
    except ModuleNotFoundError:
        return "not installed"
    return getattr(package, "__version__", "unknown")


def check_module(label, module_name):
    """Check default and explicit OpenMP thread counts for one module."""
    module = importlib.import_module(module_name)
    default_threads = module.set_num_threads(None)
    explicit = {
        count: module.set_num_threads(count)
        for count in EXPLICIT_THREAD_COUNTS
    }
    return label, default_threads, explicit


def check_cython_parallel(verbose=False, require_default_openmp=False):
    """Validate GSTools-Cython OpenMP readiness across benchmark modules."""
    default_values = []
    for label, module_name in MODULES.items():
        try:
            label, default_threads, explicit = check_module(label, module_name)
        except ModuleNotFoundError as err:
            print(f"Cython OpenMP readiness: FAIL. Missing module: {err.name}")
            return False
        default_values.append(default_threads)
        if verbose:
            explicit_text = ", ".join(
                f"{request}->{actual}" for request, actual in explicit.items()
            )
            print(f"{label} default None -> {default_threads}")
            print(f"{label} explicit -> {explicit_text}")
        mismatches = {
            request: actual
            for request, actual in explicit.items()
            if actual != request
        }
        if mismatches:
            mismatch_text = ", ".join(
                f"{request}->{actual}"
                for request, actual in mismatches.items()
            )
            print(
                "Cython OpenMP readiness: FAIL. "
                f"{label} did not accept requested thread counts: "
                f"{mismatch_text}."
            )
            return False

    if require_default_openmp and min(default_values) <= 1:
        print(
            "Cython OpenMP readiness: FAIL. GSTools-Cython reports one "
            "default thread."
        )
        return False

    print("Cython OpenMP readiness: PASS")
    return True


def check_rust_backend(threads):
    """Run a small Rust-backed GSTools workflow with ``threads`` configured."""
    try:
        import gstools as gs
    except ModuleNotFoundError as err:
        print(f"Rust backend readiness: FAIL. Missing module: {err.name}")
        return False

    try:
        importlib.import_module("gstools_core")
    except ModuleNotFoundError as err:
        print(f"Rust backend readiness: FAIL. Missing module: {err.name}")
        return False

    if not gs.config._GSTOOLS_CORE_AVAIL:
        print(
            "Rust backend readiness: FAIL. GSTools did not detect gstools_core."
        )
        return False

    previous = (
        gs.config._GSTOOLS_CORE_AVAIL,
        gs.config.USE_GSTOOLS_CORE,
        gs.config.NUM_THREADS,
    )
    try:
        gs.config._GSTOOLS_CORE_AVAIL = True
        gs.config.USE_GSTOOLS_CORE = True
        gs.config.NUM_THREADS = threads

        x = np.linspace(0.0, 10.0, 12)
        y = np.linspace(0.0, 5.0, 12)
        field = np.sin(x) + np.cos(y)
        bins = np.linspace(0.0, 8.0, 6)

        # variogram
        gs.vario_estimate(
            (x, y),
            field,
            bins,
            mesh_type="unstructured",
            return_counts=True,
        )

        # field generation
        model = gs.Gaussian(dim=2, var=1.0, len_scale=3.0)
        srf = gs.SRF(model, seed=1)
        srf((x, y))

        # kriging
        krige = gs.Krige(
            model,
            cond_pos=(x[:6], y[:6]),
            cond_val=field[:6],
        )
        krige((x[6:], y[6:]))

    finally:
        (
            gs.config._GSTOOLS_CORE_AVAIL,
            gs.config.USE_GSTOOLS_CORE,
            gs.config.NUM_THREADS,
        ) = previous

    print(f"Rust backend readiness: PASS with NUM_THREADS={threads}")
    return True


def main():
    """Run the parallel backend readiness checks."""
    args = parse_args()
    print(f"python: {sys.executable}")
    print(f"gstools: {package_version('gstools')}")
    print(f"gstools_cython: {package_version('gstools_cython')}")
    print(f"gstools_core: {package_version('gstools_core')}")

    cython_ready = check_cython_parallel(
        verbose=args.verbose,
        require_default_openmp=args.require_default_openmp,
    )
    rust_ready = True if args.cython_only else check_rust_backend(args.threads)
    return 0 if cython_ready and rust_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
