#!/usr/bin/env python
"""Build a case/backend comparison report from ASV result JSON files.

The default ASV browser remains useful for commit-history plots. This helper
creates a focused static HTML view for questions ASV does not show cleanly:
case-by-case backend comparisons and OpenMP thread comparisons.

The script has no plotting dependency. It reads ASV result JSON and writes a
self-contained HTML file with a small inline SVG renderer.

Usage:
    python benchmarks/tools/plot_case_backend_comparison.py
    python benchmarks/tools/plot_case_backend_comparison.py --results-dir .asv/results
    python benchmarks/tools/plot_case_backend_comparison.py --results-dir .asv-openmp/results
    python benchmarks/tools/plot_case_backend_comparison.py --metric time
    python benchmarks/tools/plot_case_backend_comparison.py --metric memory
    python benchmarks/tools/plot_case_backend_comparison.py --benchmark krige
    python benchmarks/tools/plot_case_backend_comparison.py --output comparison.html
"""

from __future__ import annotations

import argparse
import base64
import html
import itertools
import json
import math
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
BACKENDS = ("cython_fallback", "rust_core")
THREAD_PREFIX = "threads_"
BENCHMARK_PREFIXES = (
    "time_variogram_estimate",
    "peakmem_variogram_estimate",
    "time_global_krige",
    "peakmem_global_krige",
    "time_field_generation",
    "peakmem_field_generation",
)
BENCHMARK_ALIASES = {
    "variogram": "variogram",
    "vario": "variogram",
    "krige": "krige",
    "kriging": "krige",
    "field": "field",
    "random_field": "field",
    "random-field": "field",
    "srf": "field",
    "condsrf": "field",
}


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        action="append",
        type=Path,
        default=None,
        help=(
            "Path to an ASV results directory. Can be supplied more than "
            "once. Default: .asv/results if present, otherwise "
            ".asv-openmp/results."
        ),
    )
    parser.add_argument(
        "--benchmark",
        choices=sorted(BENCHMARK_ALIASES),
        default=None,
        help="Filter to one benchmark family. Default: include all families.",
    )
    parser.add_argument(
        "--metric",
        choices=("all", "time", "memory"),
        default="all",
        help="Metric to include. Default: all, which includes time and memory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "HTML output path. Default: case-backend-comparison.html next to "
            "the first selected results directory."
        ),
    )
    parser.add_argument(
        "--max-commits",
        type=positive_int,
        default=None,
        help=(
            "Include only the newest N commits in the report. Raw ASV result "
            "files are not changed. Default: include all commits."
        ),
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="Print a compact terminal table instead of writing HTML.",
    )
    return parser.parse_args()


def positive_int(value):
    """Parse a positive integer command-line value."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def default_results_dirs():
    """Return the default ASV results directory list."""
    for candidate in (Path(".asv/results"), Path(".asv-openmp/results")):
        if candidate.exists():
            return [candidate]
    raise FileNotFoundError(
        "Could not find .asv/results or .asv-openmp/results"
    )


def existing_results_dirs(paths):
    """Validate and return result directories."""
    if paths is None:
        return default_results_dirs()
    missing = [path for path in paths if not path.exists()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"ASV results directory not found: {names}")
    return paths


def iter_result_files(results_dir):
    """Yield ASV result JSON files below ``results_dir``."""
    for path in sorted(results_dir.glob("**/*.json")):
        if path.name in {"benchmarks.json", "machine.json"}:
            continue
        yield path


def get_git_tags():
    """Return a mapping from full commit hash to tag name, version-sorted."""
    try:
        tag_out = subprocess.run(
            ["git", "tag", "-l", "--sort=version:refname"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if not tag_out:
            return {}
        result = {}
        for tag in tag_out.splitlines():
            tag = tag.strip()
            if not tag:
                continue
            rev = subprocess.run(
                ["git", "rev-parse", f"{tag}^{{}}"],
                capture_output=True,
                text=True,
            )
            if rev.returncode == 0:
                result[rev.stdout.strip()] = tag
        return result
    except (subprocess.SubprocessError, FileNotFoundError):
        return {}


def load_json(path):
    """Load one ASV JSON file, ignoring invalid files."""
    try:
        with path.open(encoding="utf8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def result_entry(raw_result, result_columns):
    """Normalize ASV result payloads across schema versions."""
    if isinstance(raw_result, dict):
        return raw_result
    if isinstance(raw_result, list) and result_columns:
        return dict(zip(result_columns, raw_result))
    return {"result": raw_result, "params": []}


def flatten_values(values):
    """Yield scalar values from nested ASV result lists."""
    if isinstance(values, list):
        for value in values:
            yield from flatten_values(value)
        return
    yield values


def is_number(value):
    """Return whether ``value`` is a numeric, non-NaN result."""
    return isinstance(value, (int, float)) and not math.isnan(value)


def short_benchmark_name(name):
    """Return the method name from a fully qualified ASV benchmark name."""
    return name.rsplit(".", maxsplit=1)[-1]


def metric_from_name(name):
    """Return the metric represented by one benchmark name."""
    short_name = short_benchmark_name(name)
    if short_name.startswith("peakmem_"):
        return "memory"
    if short_name.startswith("time_"):
        return "time"
    return None


def family_from_name(name):
    """Return the benchmark family label represented by one benchmark name."""
    short_name = short_benchmark_name(name)
    if "variogram" in short_name:
        return "variogram"
    if "krig" in short_name:
        return "krige"
    if "field" in short_name:
        return "field"
    return None


def parse_benchmark_name(name):
    """Split a benchmark name into base benchmark, case, and thread labels.

    Older results may encode case and thread in generated method names, while
    current results keep them as ASV parameters. This parser extracts encoded
    values when present and leaves parameterized results for ``result_rows``
    to fill in.
    """
    short_name = short_benchmark_name(name)
    threads = None
    match = re.search(r"_(threads_\d+)$", short_name)
    if match:
        threads = match.group(1)
        short_name = short_name[: match.start()]

    for prefix in BENCHMARK_PREFIXES:
        if short_name == prefix:
            return prefix, None, threads
        marker = f"{prefix}_"
        if short_name.startswith(marker):
            return prefix, short_name[len(marker) :], threads

    return short_name, None, threads


def normalize_thread(value):
    """Normalize ASV thread values to a ``threads_N`` label."""
    if value is None:
        return "threads_1"
    text = str(value).strip("'\"")
    if text.startswith(THREAD_PREFIX):
        return text
    if text.isdigit():
        return f"{THREAD_PREFIX}{text}"
    return text


def is_thread_value(value):
    """Return whether a parameter value looks like a thread count."""
    text = str(value).strip("'\"")
    return text.startswith(THREAD_PREFIX) or text.isdigit()


def clean_param_value(value):
    """Convert an ASV repr-like parameter value into display text."""
    return str(value).strip("'\"")


def result_rows(benchmark, entry):
    """Return normalized rows for one ASV benchmark result entry."""
    result = entry.get("result")
    params = entry.get("params") or []
    if not isinstance(result, list):
        return []

    base_name, encoded_case, encoded_threads = parse_benchmark_name(benchmark)
    rows = []
    values = list(flatten_values(result))
    combinations = itertools.product(*params) if params else [()]

    for combo, value in zip(combinations, values):
        if not is_number(value):
            continue
        combo_values = [clean_param_value(item) for item in combo]
        backend = next(
            (candidate for candidate in BACKENDS if candidate in combo_values),
            None,
        )
        if backend is None:
            continue

        case_values = [
            item
            for item in combo_values
            if item not in BACKENDS and not is_thread_value(item)
        ]
        thread = next(
            (
                normalize_thread(item)
                for item in combo_values
                if is_thread_value(item)
            ),
            normalize_thread(encoded_threads),
        )
        rows.append(
            {
                "benchmark": base_name,
                "family": family_from_name(base_name),
                "metric": metric_from_name(base_name),
                "case": "/".join(case_values) if case_values else encoded_case,
                "threads": thread,
                "backend": backend,
                "value": float(value),
            }
        )
    return rows


def collect_rows(
    results_dirs, benchmark_filter=None, metric_filter="all", tag_map=None
):
    """Collect normalized benchmark rows from ASV result folders."""
    rows = []
    benchmark_filter = (
        BENCHMARK_ALIASES[benchmark_filter] if benchmark_filter else None
    )
    for results_dir in results_dirs:
        for path in iter_result_files(results_dir):
            data = load_json(path)
            if data is None:
                continue
            result_columns = data.get("result_columns", [])
            commit_hash = data.get("commit_hash", "unknown")
            commit = commit_hash[:8]
            date = data.get("date", 0)
            env_name = data.get("env_name", path.stem)
            params = data.get("params", {})
            python = data.get("python") or params.get("python", "-")
            machine_data = load_json(path.parent / "machine.json") or {}
            machine = {
                key: machine_data.get(key, params.get(key, "-"))
                for key in (
                    "machine",
                    "os",
                    "arch",
                    "cpu",
                    "num_cpu",
                    "ram",
                )
            }
            for benchmark, raw_result in data.get("results", {}).items():
                entry = result_entry(raw_result, result_columns)
                for row in result_rows(benchmark, entry):
                    if (
                        not row["metric"]
                        or not row["family"]
                        or not row["case"]
                    ):
                        continue
                    if benchmark_filter and row["family"] != benchmark_filter:
                        continue
                    if (
                        metric_filter != "all"
                        and row["metric"] != metric_filter
                    ):
                        continue
                    row.update(
                        {
                            "source": str(results_dir),
                            "commit": commit,
                            "commit_hash": commit_hash,
                            "tag": (tag_map or {}).get(commit_hash, ""),
                            "date": date,
                            "date_label": format_date(date),
                            "env": env_name,
                            "python": python,
                            **machine,
                        }
                    )
                    rows.append(row)
    return sort_rows(rows)


def limit_recent_commits(rows, max_commits=None):
    """Return rows for only the newest requested commits."""
    if max_commits is None:
        return rows
    ordered_commits = sorted(
        {(row["date"], row["commit_hash"]) for row in rows},
        reverse=True,
    )
    included = {
        commit_hash for _, commit_hash in ordered_commits[:max_commits]
    }
    return [row for row in rows if row["commit_hash"] in included]


def sort_rows(rows):
    """Sort rows for stable reports."""
    return sorted(
        rows,
        key=lambda row: (
            row["source"],
            row["metric"],
            row["family"],
            row["benchmark"],
            row["case"],
            thread_number(row["threads"]),
            row["date"],
            row["commit"],
            row["backend"],
        ),
    )


def thread_number(label):
    """Return the numeric part of a ``threads_N`` label."""
    if isinstance(label, str) and label.startswith(THREAD_PREFIX):
        try:
            return int(label[len(THREAD_PREFIX) :])
        except ValueError:
            return -1
    return -1


def format_date(timestamp):
    """Format an ASV millisecond timestamp as a UTC date."""
    if not timestamp:
        return "-"
    return datetime.fromtimestamp(
        timestamp / 1000.0,
        tz=timezone.utc,
    ).strftime("%Y-%m-%d")


def default_output_path(results_dirs):
    """Return the default HTML report path."""
    return results_dirs[0].parent / "case-backend-comparison.html"


def print_table(rows):
    """Print normalized rows as a compact terminal table."""
    if not rows:
        print("No matching ASV backend comparison rows found.")
        return
    headers = [
        "source",
        "commit",
        "date",
        "metric",
        "family",
        "benchmark",
        "case",
        "threads",
        "backend",
        "value",
    ]
    table = [
        [
            row["source"],
            row["commit"],
            row["date_label"],
            row["metric"],
            row["family"],
            row["benchmark"],
            row["case"],
            row["threads"],
            row["backend"],
            f"{row['value']:.6g}",
        ]
        for row in rows
    ]
    widths = [
        max(len(str(item)) for item in column)
        for column in zip(headers, *table)
    ]

    def fmt(row):
        return "  ".join(
            str(item).ljust(width) for item, width in zip(row, widths)
        )

    print(fmt(headers))
    print(fmt(["-" * width for width in widths]))
    for row in table:
        print(fmt(row))


def gstools_logo_markup():
    """Return a compact embedded GSTools logo for the HTML report."""
    logo_path = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "source"
        / "pics"
        / "gstools_150.png"
    )
    try:
        encoded = base64.b64encode(logo_path.read_bytes()).decode("ascii")
    except OSError:
        return '<div class="brand-fallback" aria-hidden="true">GS</div>'
    return (
        '<img class="brand-logo" '
        f'src="data:image/png;base64,{encoded}" alt="GSTools logo">'
    )


def format_ram(value):
    """Format ASV RAM for the machine summary.

    ASV 0.6+ stores RAM in bytes; older versions stored the raw /proc/meminfo
    kB value.  Any amount below 1 GiB expressed as bytes is implausible for a
    benchmark machine, so treat it as kB and convert.
    """
    try:
        amount = int(value)
    except (TypeError, ValueError):
        return str(value)
    if amount < 1024**3:
        amount *= 1024
    return f"{amount / (1024**3):.1f} GiB"


def machine_summary_markup(rows):
    """Return report cards for the distinct benchmark machines."""
    machines = {}
    for row in rows:
        key = tuple(
            str(row.get(field, "-"))
            for field in (
                "machine",
                "os",
                "arch",
                "cpu",
                "num_cpu",
                "ram",
                "python",
            )
        )
        machines[key] = {
            "machine": key[0],
            "os": key[1],
            "arch": key[2],
            "cpu": key[3],
            "num_cpu": key[4],
            "ram": format_ram(key[5]),
            "python": key[6],
        }

    cards = []
    for machine in machines.values():
        title = html.escape(machine["machine"])
        details = html.escape(
            f"{machine['os']} · {machine['arch']} · {machine['cpu']} · "
            f"{machine['num_cpu']} CPUs · {machine['ram']} · "
            f"Python {machine['python']}"
        )
        cards.append(
            '<div class="machine-card">'
            f"<strong>{title}</strong><span>{details}</span>"
            "</div>"
        )
    return "".join(cards)


def read_template(name):
    """Read one file from the report template directory."""
    return (TEMPLATE_DIR / name).read_text(encoding="utf8")


def render_html(rows):
    """Render a self-contained interactive HTML report.

    The page skeleton, stylesheet and script live in ``templates/`` and are
    inlined here so the output stays a single file.
    """
    payload = json.dumps(rows, separators=(",", ":")).replace("</", "<\\/")
    template = read_template("backend_comparison.html")
    return (
        template.replace(
            "__STYLE__\n", read_template("backend_comparison.css")
        )
        .replace("__SCRIPT__\n", read_template("backend_comparison.js"))
        .replace("__PAYLOAD__", payload)
        .replace("__LOGO_MARKUP__", gstools_logo_markup())
        .replace("__MACHINE_MARKUP__", machine_summary_markup(rows))
    )


def main():
    """Run the report generator."""
    args = parse_args()
    results_dirs = existing_results_dirs(args.results_dir)
    tag_map = get_git_tags()
    rows = collect_rows(
        results_dirs,
        benchmark_filter=args.benchmark,
        metric_filter=args.metric,
        tag_map=tag_map,
    )
    rows = limit_recent_commits(rows, args.max_commits)
    if args.table:
        try:
            print_table(rows)
        except BrokenPipeError:
            with open(os.devnull, "w", encoding="utf8") as devnull:
                os.dup2(devnull.fileno(), 1)
            return
        return
    if not rows:
        print("No matching ASV backend comparison rows found.")
        return

    output = args.output or default_output_path(results_dirs)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(rows), encoding="utf8")
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
