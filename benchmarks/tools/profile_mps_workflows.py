#!/usr/bin/env python
"""Profile representative MPS benchmark workflows with cProfile.

MPS is Rust-only (requires gstools_core>=1.4.0); the DirectSamplingBenchmarks
run a single "core" backend, which this helper passes automatically. Python
profiling observes the call into Rust, not its internal node work.

Usage:
    cd /path/to/MPS-Tools/GSTools
    ASV_ENV="$(ls -td .asv/env/* | head -n 1)"
    "$ASV_ENV/bin/python" benchmarks/tools/profile_mps_workflows.py --list
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

# Map profile case names to (BenchmarkClass, method_name, ds_or_ti_case_label).
# TI cases profile construction; DS cases profile the simulation call.
CASES = {
    # TrainingImage construction cases
    "ti-cat-60x60": (
        "TrainingImageBenchmarks",
        "time_training_image_construct",
        "cat_60x60",
    ),
    "ti-cat-150x150": (
        "TrainingImageBenchmarks",
        "time_training_image_construct",
        "cat_150x150",
    ),
    "ti-cont-60x60": (
        "TrainingImageBenchmarks",
        "time_training_image_construct",
        "cont_60x60",
    ),
    "ti-multivar-60x60": (
        "TrainingImageBenchmarks",
        "time_training_image_construct",
        "multivar_60x60",
    ),
    # DirectSampling simulation cases
    "ds-cat-dsbc-small": (
        "DirectSamplingBenchmarks",
        "time_simulate",
        "cat_dsbc_small",
    ),
    "ds-cat-dsbc-medium": (
        "DirectSamplingBenchmarks",
        "time_simulate",
        "cat_dsbc_medium",
    ),
    "ds-cat-dsbc-large": (
        "DirectSamplingBenchmarks",
        "time_simulate",
        "cat_dsbc_large",
    ),
    "ds-cat-ds-medium": (
        "DirectSamplingBenchmarks",
        "time_simulate",
        "cat_ds_medium",
    ),
    "ds-cont-l1-medium": (
        "DirectSamplingBenchmarks",
        "time_simulate",
        "cont_l1_medium",
    ),
    "ds-cat-hiscan": (
        "DirectSamplingBenchmarks",
        "time_simulate",
        "cat_dsbc_hiscan",
    ),
    "ds-cat-highk": (
        "DirectSamplingBenchmarks",
        "time_simulate",
        "cat_dsbc_highk",
    ),
    "ds-cat-cond": (
        "DirectSamplingBenchmarks",
        "time_simulate",
        "cat_dsbc_cond",
    ),
}


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--case",
        default="ds-cat-dsbc-medium",
        choices=["all", *CASES],
        help="Workflow to profile. Default: ds-cat-dsbc-medium.",
    )
    parser.add_argument(
        "--repeat",
        default=1,
        type=int,
        help="Number of times to run each selected workflow inside cProfile.",
    )
    parser.add_argument(
        "--limit",
        default=25,
        type=int,
        help="Number of cProfile rows to print per workflow.",
    )
    parser.add_argument(
        "--sort",
        default="cumtime",
        choices=["cumtime", "tottime", "calls"],
        help="pstats sort key. cumtime is usually the best first view.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available workflow cases and exit.",
    )
    return parser.parse_args()


def iter_selected(case):
    """Yield the requested workflow case definitions."""
    if case == "all":
        yield from CASES.items()
        return
    yield case, CASES[case]


def load_suite_class(class_name):
    """Import a benchmark class from benchmark_mps."""
    try:
        from benchmarks import benchmark_mps
    except ModuleNotFoundError as err:
        print(
            "Could not import MPS benchmark dependencies. Activate the "
            "GSTools benchmark environment, run this script with an ASV env "
            "Python from .asv/env/<env-id>/bin/python, or install the project "
            f"dependencies first. Original error: {err}",
            file=sys.stderr,
        )
        raise SystemExit(1) from err
    return getattr(benchmark_mps, class_name)


def run_case(name, class_name, method_base_name, case, repeat, limit, sort):
    """Profile one MPS benchmark workflow case.

    setup() is called once outside the profiler so that only the measured
    operation (simulation or TI construction) appears in the cProfile output.
    This mirrors what ASV does: setup() is excluded from the timed region.
    """
    suite_cls = load_suite_class(class_name)
    suite = suite_cls()
    data = suite.setup_cache()

    # Classes parametrized by "backend" (DirectSamplingBenchmarks) take the
    # single Rust "core" backend; TI-construction benchmarks are not.
    extra = ("core",) if "backend" in getattr(suite, "param_names", []) else ()

    # Call setup() outside the profiler so construction overhead is excluded.
    if hasattr(suite, "setup"):
        suite.setup(data, case, *extra)

    method = getattr(suite, method_base_name)

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(repeat):
        method(data, case, *extra)
    profiler.disable()

    print(f"\n== {name} ==")
    stats = pstats.Stats(profiler, stream=sys.stdout)
    stats.strip_dirs().sort_stats(sort).print_stats(limit)


def main():
    """Run the MPS cProfile helper."""
    args = parse_args()

    if args.list:
        col = max(len(k) for k in CASES) + 2
        for name, (cls, _method, case_label) in CASES.items():
            tag = "TI" if cls == "TrainingImageBenchmarks" else "DS"
            print(f"{name:<{col}} [{tag}]  case={case_label}")
        return

    for name, (suite_cls, method_name, params) in iter_selected(args.case):
        run_case(
            name,
            suite_cls,
            method_name,
            params,
            args.repeat,
            args.limit,
            args.sort,
        )


if __name__ == "__main__":
    main()
