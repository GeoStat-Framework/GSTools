#!/usr/bin/env python
"""Smoke-check that the GSTools benchmark backend runs with threads set.

This is a fast CI guard, not a parallel-performance proof. It verifies that the
Rust backend (gstools_core) is installed and that a small GSTools workflow
completes with ``gs.config.NUM_THREADS`` set above one. It does NOT measure
speedup or confirm that multiple OS threads were actually used — the benchmark
thread-scaling results are the real evidence of parallel execution.
"""

from __future__ import annotations

import argparse
import importlib
import sys

import numpy as np


def positive_int(value):
    """Parse a command-line integer that must be >= 1."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("thread count must be at least 1")
    return parsed


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--threads",
        default=2,
        type=positive_int,
        help="Thread count (>=1) to request for the Rust backend smoke check.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the detected gstools_core version.",
    )
    return parser.parse_args()


def package_version(package_name):
    """Return an installed package version or a clear missing marker."""
    try:
        package = importlib.import_module(package_name)
    except ModuleNotFoundError:
        return "not installed"
    return getattr(package, "__version__", "unknown")


def check_rust_backend(threads, verbose=False):
    """Run a small Rust-backed GSTools workflow with ``threads`` configured."""
    try:
        import gstools as gs
    except ModuleNotFoundError as err:
        print(f"Rust backend readiness: FAIL. Missing module: {err.name}")
        return False

    try:
        core = importlib.import_module("gstools_core")
    except ModuleNotFoundError as err:
        print(f"Rust backend readiness: FAIL. Missing module: {err.name}")
        return False

    if verbose:
        print(f"gstools_core: {getattr(core, '__version__', 'unknown')}")

    previous = gs.config.NUM_THREADS
    try:
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
        gs.config.NUM_THREADS = previous

    print(f"Rust backend readiness: PASS with NUM_THREADS={threads}")
    return True


def main():
    """Run the parallel backend readiness check."""
    args = parse_args()
    print(f"python: {sys.executable}")
    print(f"gstools: {package_version('gstools')}")
    print(f"gstools_core: {package_version('gstools_core')}")

    rust_ready = check_rust_backend(args.threads, verbose=args.verbose)
    return 0 if rust_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
