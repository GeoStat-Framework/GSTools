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
            capture_output=True, text=True, check=True,
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
                capture_output=True, text=True,
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

    Current OpenMP results may encode case and thread into generated method
    names, while baseline results may keep them as ASV parameters. This parser
    extracts the encoded part when present and leaves parameterized results for
    ``result_rows`` to fill in.
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


def collect_rows(results_dirs, benchmark_filter=None, metric_filter="all", tag_map=None):
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
    if amount < 1024 ** 3:
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
            f'{machine["os"]} · {machine["arch"]} · {machine["cpu"]} · '
            f'{machine["num_cpu"]} CPUs · {machine["ram"]} · '
            f'Python {machine["python"]}'
        )
        cards.append(
            '<div class="machine-card">'
            f"<strong>{title}</strong><span>{details}</span>"
            "</div>"
        )
    return "".join(cards)


def render_html(rows):
    """Render a self-contained interactive HTML report."""
    payload = json.dumps(rows, separators=(",", ":")).replace("</", "<\\/")
    template = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GSTools Two-Point Statistics Backend Comparison</title>
<style>
:root {
  color-scheme: light dark;
  --bg: #eef4f7;
  --bg-gradient: radial-gradient(circle at top left, rgba(15, 118, 110, 0.16), transparent 34rem), linear-gradient(180deg, #f7fbfc 0%, #eef4f7 100%);
  --panel: #ffffff;
  --panel-muted: #f7fafb;
  --line: #d8e3e9;
  --line-strong: #aebfca;
  --text: #14212b;
  --muted: #607384;
  --accent: #0f766e;
  --accent-soft: #e2f3f0;
  --accent-strong: #0b5f59;
  --cython: #2f72b7;
  --rust: #c75a3c;
  --success: #14804a;
  --danger: #b42318;
  --shadow: 0 18px 48px rgba(26, 44, 56, 0.10);
  --radius: 14px;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #101820;
    --bg-gradient: radial-gradient(circle at top left, rgba(20, 184, 166, 0.14), transparent 34rem), linear-gradient(180deg, #111c24 0%, #0d141b 100%);
    --panel: #17222b;
    --panel-muted: #111b23;
    --line: #263744;
    --line-strong: #405767;
    --text: #e8f0f4;
    --muted: #a4b5c1;
    --accent: #2dd4bf;
    --accent-soft: rgba(45, 212, 191, 0.12);
    --accent-strong: #7dded2;
    --shadow: 0 18px 48px rgba(0, 0, 0, 0.28);
  }
}
* { box-sizing: border-box; }
body {
  background: var(--bg);
  background-image: var(--bg-gradient);
  color: var(--text);
  font: 14px/1.5 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  margin: 0;
}
a { color: var(--accent-strong); }
a:hover { text-decoration-thickness: 2px; }
button, input, select { font: inherit; }
button { touch-action: manipulation; }
:focus-visible {
  outline: 3px solid color-mix(in srgb, var(--accent) 42%, transparent);
  outline-offset: 2px;
}
.skip-link {
  background: var(--accent-strong);
  border-radius: 0 0 8px 0;
  color: white;
  left: 0;
  padding: 8px 12px;
  position: fixed;
  top: 0;
  transform: translateY(-120%);
  z-index: 10;
}
.skip-link:focus { transform: translateY(0); }
header, main {
  margin: 0 auto;
  max-width: none;
  padding: 26px clamp(16px, 2.2vw, 34px);
  width: 100%;
}
header { padding-bottom: 12px; }
.brand-row {
  align-items: center;
  background: color-mix(in srgb, var(--panel) 96%, transparent);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  display: flex;
  gap: 28px;
  justify-content: space-between;
  padding: clamp(18px, 2.4vw, 28px);
}
.brand-copy { min-width: 0; }
.brand-logo,
.brand-fallback {
  flex: 0 0 auto;
  height: clamp(64px, 9vw, 82px);
  width: clamp(64px, 9vw, 82px);
}
.brand-logo {
  filter: drop-shadow(0 10px 18px rgba(26, 44, 56, 0.16));
}
.brand-fallback {
  align-items: center;
  background: var(--accent-soft);
  border: 1px solid var(--line);
  border-radius: 50%;
  color: var(--accent-strong);
  display: flex;
  font-weight: 800;
  justify-content: center;
}
h1 {
  font-size: clamp(24px, 3.6vw, 40px);
  letter-spacing: -0.03em;
  line-height: 1.08;
  margin: 8px 0 12px;
  max-width: 900px;
}
p {
  color: var(--muted);
  margin: 0;
  max-width: 780px;
}
.eyebrow {
  color: var(--accent);
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.10em;
  text-transform: uppercase;
}
.lede { font-size: 15px; }
.report-links {
  display: flex;
  flex-wrap: wrap;
  gap: 10px 18px;
  margin-top: 14px;
}
.report-links a {
  align-items: center;
  color: var(--accent-strong);
  display: inline-flex;
  font-weight: 700;
  text-decoration: none;
}
.report-links a::after { content: "↗"; font-size: 12px; margin-left: 5px; opacity: 0.7; }
.report-links a:hover { text-decoration: underline; }
.machine-summary {
  display: grid;
  gap: 8px;
  margin: 14px 0 0;
}
.machine-card {
  background: color-mix(in srgb, var(--panel) 94%, transparent);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  box-shadow: 0 10px 24px rgba(26, 44, 56, 0.05);
  display: grid;
  gap: 3px;
  padding: 12px 14px;
}
.machine-card span {
  color: var(--muted);
  font-size: 12px;
}
.controls, .panel {
  background: color-mix(in srgb, var(--panel) 96%, transparent);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
}
.controls {
  display: grid;
  gap: 18px;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  padding: 20px;
}
.wide-control { grid-column: span 2; }
.filter-control {
  border: 0;
  margin: 0;
  min-width: 0;
  padding: 0;
}
.filter-control legend {
  color: var(--muted);
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.08em;
  margin-bottom: 9px;
  padding: 0;
  text-transform: uppercase;
}
.checkbox-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.check-label {
  align-items: center;
  background: var(--panel-muted);
  border: 1px solid var(--line);
  border-radius: 999px;
  color: var(--text);
  cursor: pointer;
  display: inline-flex;
  font-size: 13px;
  font-weight: 650;
  gap: 7px;
  line-height: 1.2;
  min-height: 30px;
  padding: 5px 10px 5px 8px;
  transition: background 150ms ease, border-color 150ms ease, transform 150ms ease;
}
.check-label:hover { border-color: var(--line-strong); transform: translateY(-1px); }
.check-label:has(input:checked) {
  background: var(--accent-soft);
  border-color: color-mix(in srgb, var(--accent) 48%, var(--line));
  color: var(--accent-strong);
}
.check-label input {
  accent-color: var(--accent);
  margin: 0;
  width: 13px;
}
.panel {
  margin-top: 20px;
  overflow: hidden;
  padding: 0;
}
.panel-header {
  align-items: center;
  border-bottom: 1px solid var(--line);
  display: flex;
  gap: 12px;
  justify-content: space-between;
  padding: 18px 20px;
}
.panel-title {
  font-size: 17px;
  font-weight: 800;
  letter-spacing: -0.01em;
}
.panel-note {
  background: var(--accent-soft);
  border: 1px solid color-mix(in srgb, var(--accent) 32%, var(--line));
  border-radius: 999px;
  color: var(--accent-strong);
  font-size: 12px;
  font-weight: 800;
  padding: 4px 10px;
}
#chart {
  background: var(--panel-muted);
  border-bottom: 1px solid var(--line);
  min-height: 280px;
  overflow-x: auto;
  padding: 16px;
}
svg {
  display: block;
  height: auto;
  max-width: none;
  min-width: min(100%, 1120px);
  width: auto;
}
.axis, .grid {
  stroke: var(--line);
  stroke-width: 1;
}
.grid { stroke-dasharray: 2 4; }
.tick, .bar-label, .axis-label { fill: var(--muted); font-size: 12px; }
.title { fill: var(--text); font-size: 16px; font-weight: 800; }
.line-path { fill: none; stroke-width: 2.4; }
.point { fill: var(--panel); stroke-width: 2; }
.chart-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 9px 14px;
  padding: 14px 20px 18px;
}
.legend-chip {
  align-items: center;
  color: var(--muted);
  display: inline-flex;
  font-size: 12px;
  font-weight: 650;
  gap: 7px;
}
.legend-swatch {
  border-radius: 999px;
  display: inline-block;
  height: 4px;
  width: 24px;
}
.multi-select {
  background: var(--panel-muted);
  border: 1px solid var(--line);
  border-radius: 10px;
  box-sizing: border-box;
  color: var(--text);
  min-height: 96px;
  padding: 5px;
  width: 100%;
}
.multi-select option {
  border-radius: 6px;
  padding: 5px 8px;
}
.select-actions {
  display: flex;
  gap: 8px;
  margin-top: 7px;
}
.select-action-btn, .toggle-btn {
  background: var(--panel-muted);
  border: 1px solid var(--line);
  border-radius: 8px;
  color: var(--muted);
  cursor: pointer;
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.04em;
  padding: 4px 10px;
  text-transform: uppercase;
}
.select-action-btn:hover, .toggle-btn:hover { border-color: var(--accent); color: var(--accent-strong); }
.toggle-btn.active {
  background: var(--accent-soft);
  border-color: var(--accent);
  color: var(--accent-strong);
}
.toggle-btn + .toggle-btn { margin-left: 4px; }
@media (max-width: 900px) {
  .wide-control { grid-column: auto; }
}
@media (max-width: 700px) {
  header, main { padding: 18px; }
  .brand-row {
    align-items: flex-start;
    gap: 16px;
    padding: 16px;
  }
  .brand-logo, .brand-fallback { height: 60px; width: 60px; }
  .controls { grid-template-columns: 1fr; padding: 16px; }
  .panel-header { align-items: flex-start; flex-direction: column; }
}
</style>
</head>
<body>
<a class="skip-link" href="#main">Skip to benchmark controls</a>
<header>
  <div class="brand-row">
    <div class="brand-copy">
      <div class="eyebrow">ASV benchmark report</div>
      <h1>GSTools Two-Point Statistics Backend Comparison</h1>
      <p class="lede">
        Cython and Rust backend results grouped side by side for selected
        two-point statistics cases, thread counts, commits, and metrics.
      </p>
      <div class="report-links">
        <a href="asv/">Complete native ASV history</a>
        <a href="https://asv.readthedocs.io/" rel="noreferrer">
          Powered by Airspeed Velocity (ASV)
        </a>
      </div>
    </div>
    __LOGO_MARKUP__
  </div>
  <div class="machine-summary">__MACHINE_MARKUP__</div>
</header>
<main id="main">
  <section class="controls">
    <fieldset class="filter-control">
      <legend>Metric</legend>
      <div id="metric" class="checkbox-list"></div>
    </fieldset>
	    <fieldset class="filter-control">
	      <legend>Family</legend>
	      <div id="family" class="checkbox-list"></div>
	    </fieldset>
	    <fieldset class="filter-control">
	      <legend>View</legend>
	      <div id="viewMode" class="checkbox-list"></div>
	    </fieldset>
	    <fieldset class="filter-control wide-control">
	      <legend>Case</legend>
	      <div id="case" class="checkbox-list"></div>
	    </fieldset>
	    <fieldset class="filter-control">
	      <legend>Threads</legend>
	      <div id="threads" class="checkbox-list"></div>
	    </fieldset>
	    <fieldset class="filter-control wide-control">
	      <legend>
	        <button id="modeCommit" class="toggle-btn active" onclick="setReferenceMode('commit')">Commits</button>
	        <button id="modeTag" class="toggle-btn" onclick="setReferenceMode('tag')">Tags</button>
	      </legend>
	      <select id="commit" class="multi-select" multiple size="5"></select>
	      <select id="tag" class="multi-select" multiple size="5" style="display:none"></select>
	      <div class="select-actions">
	        <button class="select-action-btn" onclick="selectAllOptions(referenceModeValue())">All</button>
	        <button class="select-action-btn" onclick="clearAllOptions(referenceModeValue())">None</button>
	      </div>
	    </fieldset>
	    <fieldset class="filter-control">
	      <legend>Backend</legend>
	      <div id="backend" class="checkbox-list"></div>
	    </fieldset>
	  </section>
	  <section class="panel">
	    <div class="panel-header">
	      <div class="panel-title" id="chart-title">Benchmark visualization</div>
	      <div class="panel-note">Lower values are better</div>
	    </div>
	    <div id="chart"></div>
	    <div id="legend" class="chart-legend"></div>
	  </section>
</main>
<script>
const rows = __PAYLOAD__;
const backends = ["cython_fallback", "rust_core"];
const colors = {cython_fallback: "#2f75b5", rust_core: "#d05a45"};
const linePalette = [
  "#2f75b5", "#d05a45", "#0f766e", "#9467bd", "#8c6d31", "#6b7280",
  "#1f9fb2", "#b44e9d", "#6f9e3f", "#d38b2f", "#4b5cc4", "#a14f3f"
];
const labels = {
  cython_fallback: "Cython fallback",
  rust_core: "Rust core"
};
const modeLabels = {
  bar: "Bar plot",
  line: "Line plot"
};
let xMode = "commit";
const metricUnits = {
  time: {label: "time", unit: "ms", factor: 1000},
  memory: {label: "peak memory", unit: "MiB", factor: 1 / (1024 * 1024)}
};

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function unique(values) {
  return Array.from(new Set(values)).sort((a, b) => {
    const ta = Number(String(a).replace("threads_", ""));
    const tb = Number(String(b).replace("threads_", ""));
    if (!Number.isNaN(ta) && !Number.isNaN(tb)) return ta - tb;
    return String(a).localeCompare(String(b));
  });
}

function optionText(kind, value) {
  if (kind === "commit") {
    const row = rows.find(item => item.commit === value);
    if (!row) return value;
    const tagPart = row.tag ? ` · ${row.tag}` : "";
    return `${value}${tagPart} (${row.date_label})`;
  }
  if (kind === "tag") {
    const row = rows.find(item => item.tag === value);
    return row ? `${value} — ${row.date_label}` : value;
  }
  return value;
}

function commitOrder(values) {
  return unique(values).sort((a, b) => {
    const ra = rows.find(item => item.commit === a);
    const rb = rows.find(item => item.commit === b);
    const da = ra ? ra.date : 0;
    const db = rb ? rb.date : 0;
    if (da !== db) return da - db;
    return String(a).localeCompare(String(b));
  });
}

function tagOrder(values) {
  return Array.from(new Set(values)).sort((a, b) => {
    const ra = rows.find(r => r.tag === a);
    const rb = rows.find(r => r.tag === b);
    const da = ra ? ra.date : 0;
    const db = rb ? rb.date : 0;
    if (da !== db) return da - db;
    return String(a).localeCompare(String(b));
  });
}

function checkedValues(id) {
  return Array.from(
    document.querySelectorAll(`#${id} input[type="checkbox"]:checked`)
  ).map(input => input.value);
}

function selectedOptions(id) {
  const el = document.getElementById(id);
  if (!el) return [];
  if (el.tagName === "SELECT") {
    return Array.from(el.selectedOptions).map(opt => opt.value);
  }
  return checkedValues(id);
}

function setSelectOptions(id, values, previousValues = [], textByValue = {}, defaultValues = null) {
  const el = document.getElementById(id);
  if (!el) return;
  const retained = values.filter(v => previousValues.includes(v));
  const defaults = defaultValues === null ? values : defaultValues;
  const selected = new Set(retained.length ? retained : defaults.filter(v => values.includes(v)));
  if (!selected.size && values.length) selected.add(values[0]);
  el.innerHTML = values.map(value => {
    const sel = selected.has(value) ? " selected" : "";
    const text = textByValue[value] || optionText(id, value);
    return `<option value="${escapeHtml(value)}"${sel}>${escapeHtml(text)}</option>`;
  }).join("");
}

function selectAllOptions(id) {
  const el = document.getElementById(id);
  if (!el) return;
  for (const opt of el.options) opt.selected = true;
  refreshOptions();
  renderAll();
}

function clearAllOptions(id) {
  const el = document.getElementById(id);
  if (!el) return;
  for (const opt of el.options) opt.selected = false;
  refreshOptions();
  renderAll();
}

function referenceModeValue() {
  return xMode;
}

function setReferenceMode(mode) {
  xMode = mode;
  document.getElementById("modeCommit").classList.toggle("active", mode === "commit");
  document.getElementById("modeTag").classList.toggle("active", mode === "tag");
  document.getElementById("commit").style.display = mode === "tag" ? "none" : "";
  document.getElementById("tag").style.display = mode === "tag" ? "" : "none";
  refreshOptions();
  renderAll();
}

function setCheckboxes(
  id,
  values,
  previousValues = [],
  textByValue = {},
  defaultValues = null
) {
  const container = document.getElementById(id);
  const retained = values.filter(value => previousValues.includes(value));
  const defaults = defaultValues === null ? values : defaultValues;
  const selected = new Set(retained.length ? retained : defaults.filter(value => values.includes(value)));
  if (!selected.size && values.length) selected.add(values[0]);
  container.innerHTML = values.map(value => {
    const checked = selected.has(value) ? " checked" : "";
    const text = textByValue[value] || optionText(id, value);
    return (
      `<label class="check-label">` +
      `<input type="checkbox" value="${escapeHtml(value)}"${checked}>` +
      `<span>${escapeHtml(text)}</span>` +
      `</label>`
    );
  }).join("");
}

function enforceSingleChoice(id, event) {
  if (!event.target.matches("input[type='checkbox']")) return;
  const inputs = Array.from(document.querySelectorAll(`#${id} input[type="checkbox"]`));
  if (event.target.checked) {
    for (const input of inputs) {
      if (input !== event.target) input.checked = false;
    }
    return;
  }
  if (!inputs.some(input => input.checked)) {
    event.target.checked = true;
  }
}

function enforceAtLeastOne(id, event) {
  if (!event.target.matches("input[type='checkbox']")) return;
  const inputs = Array.from(document.querySelectorAll(`#${id} input[type="checkbox"]`));
  if (!inputs.some(input => input.checked)) {
    event.target.checked = true;
  }
}

function metricValue() {
  return checkedValues("metric")[0] || "";
}

function familyValue() {
  return checkedValues("family")[0] || "";
}

function viewModeValue() {
  return checkedValues("viewMode")[0] || "line";
}

function selectedBenchmark() {
  const candidates = unique(rows
    .filter(row => row.metric === metricValue() && row.family === familyValue())
    .map(row => row.benchmark));
  return candidates[0] || "";
}

function benchmarkRows() {
  const benchmark = selectedBenchmark();
  return rows.filter(row =>
    row.metric === metricValue() &&
    row.family === familyValue() &&
    row.benchmark === benchmark
  );
}

function filteredRows() {
  const metric = metricValue();
  const family = familyValue();
  const benchmark = selectedBenchmark();
  const cases = checkedValues("case");
  const threads = checkedValues("threads");
  const selectedBackends = checkedValues("backend");
  const isTagMode = referenceModeValue() === "tag";
  const xFilter = isTagMode
    ? (row => row.tag && selectedOptions("tag").includes(row.tag))
    : (row => selectedOptions("commit").includes(row.commit));
  return rows.filter(row =>
    (!metric || row.metric === metric) &&
    (!family || row.family === family) &&
    (!benchmark || row.benchmark === benchmark) &&
    cases.includes(row.case) &&
    threads.includes(row.threads) &&
    xFilter(row) &&
    selectedBackends.includes(row.backend)
  );
}

function lastValue(values) {
  return values.length ? values[values.length - 1] : "";
}

function refreshOptions() {
  const previousMain = {
    metrics: checkedValues("metric"),
    families: checkedValues("family"),
    viewModes: checkedValues("viewMode"),
    cases: checkedValues("case"),
    threads: checkedValues("threads"),
    commits: selectedOptions("commit"),
    tags: selectedOptions("tag"),
    backends: checkedValues("backend")
  };
  const metricOptions = unique(rows.map(row => row.metric));
  const defaultMetric = metricOptions.includes("time") ? "time" : metricOptions[0];
  setCheckboxes("metric", metricOptions, previousMain.metrics, {}, [defaultMetric]);
  const familyOptions = unique(rows
    .filter(row => row.metric === metricValue())
    .map(row => row.family));
  const defaultFamily = familyOptions.includes("krige") ? "krige" : familyOptions[0];
  setCheckboxes("family", familyOptions, previousMain.families, {}, [defaultFamily]);
  setCheckboxes("viewMode", ["line", "bar"], previousMain.viewModes, modeLabels, ["line"]);
  const isTagMode = referenceModeValue() === "tag";
  const benchRows = benchmarkRows();
  const caseOptions = unique(benchRows.map(row => row.case));
  const defaultCase =
    caseOptions.find(v => v.includes("extra_large")) ||
    caseOptions.find(v => v.includes("sampled_15000")) ||
    caseOptions.find(v => v.includes("srf_unstructured")) ||
    caseOptions[0];
  setCheckboxes("case", caseOptions, previousMain.cases, {}, [defaultCase]);
  const selectedCases = checkedValues("case");
  const threadOptions = unique(benchRows
    .filter(row => selectedCases.includes(row.case))
    .map(row => row.threads));
  const defaultThread = threadOptions.includes("threads_1") ? "threads_1" : threadOptions[0];
  setCheckboxes("threads", threadOptions, previousMain.threads, {}, [defaultThread]);
  const selectedThreads = checkedValues("threads");
  const baseRows = benchRows.filter(row =>
    selectedCases.includes(row.case) && selectedThreads.includes(row.threads)
  );
  if (isTagMode) {
    const tagOptions = tagOrder(baseRows.filter(row => row.tag).map(row => row.tag));
    setSelectOptions("tag", tagOptions, previousMain.tags, {}, tagOptions);
    const selectedTags = selectedOptions("tag");
    const backendOptions = backends.filter(backend => baseRows.some(row =>
      row.backend === backend && row.tag && selectedTags.includes(row.tag)
    ));
    setCheckboxes("backend", backendOptions, previousMain.backends, labels, backendOptions);
  } else {
    const commitOptions = commitOrder(baseRows.map(row => row.commit));
    setSelectOptions("commit", commitOptions, previousMain.commits, {}, [lastValue(commitOptions)]);
    const selectedCommits = selectedOptions("commit");
    const backendOptions = backends.filter(backend => baseRows.some(row =>
      row.backend === backend && selectedCommits.includes(row.commit)
    ));
    setCheckboxes("backend", backendOptions, previousMain.backends, labels, backendOptions);
  }
  const benchmark = selectedBenchmark();
  const mode = viewModeValue() === "line" ? "Line plot" : "Bar plot";
  document.getElementById("chart-title").textContent =
    benchmark ? `${mode} - ${benchmark}` : mode;
}

function formatValue(row) {
  const unit = metricUnits[row.metric];
  return row.value * unit.factor;
}

function formatNumber(value) {
  if (value >= 100) return value.toFixed(0);
  if (value >= 10) return value.toFixed(1);
  return value.toFixed(2);
}

function axisMax(values) {
  const maxValue = Math.max(...values, 0);
  return maxValue > 0 ? maxValue * 1.16 : 1;
}

function orderedKeys(values) {
  const seen = new Set();
  const keys = [];
  for (const value of values) {
    if (!seen.has(value)) {
      seen.add(value);
      keys.push(value);
    }
  }
  return keys;
}

function barGroupKey(row) {
  const xKey = referenceModeValue() === "tag" ? row.tag : row.commit;
  return `${row.case}|${row.threads}|${xKey}`;
}

function barGroupLabel(key, showThread, showCommit) {
  const [caseName, threads, commit] = key.split("|");
  const parts = [caseName];
  if (showThread) parts.push(threads);
  if (showCommit) parts.push(commit);
  return parts.join(" · ");
}

function selectionSummary(values, noun) {
  return values.length === 1 ? values[0] : `${values.length} ${noun}`;
}

function seriesLabel(key) {
  const [backend, caseName, threads] = key.split("|");
  return `${labels[backend] || backend} · ${caseName} · ${threads}`;
}

function chartSubtitle(data) {
  const cases = checkedValues("case");
  const threads = checkedValues("threads");
  const isTagMode = referenceModeValue() === "tag";
  const xValues = isTagMode ? selectedOptions("tag") : selectedOptions("commit");
  const noun = isTagMode ? "tags" : "commits";
  const selectedBackends = checkedValues("backend").map(value => labels[value] || value);
  return [
    data[0].benchmark,
    selectionSummary(cases, "cases"),
    selectionSummary(threads, "thread groups"),
    selectionSummary(xValues, noun),
    selectionSummary(selectedBackends, "backends")
  ].join(" · ");
}

function plotWidthFor(container, minimum, perItemWidth, itemCount) {
  const available = Math.max(0, Math.floor(container.clientWidth - 18));
  return Math.max(minimum, available, perItemWidth * itemCount + 180);
}

function barChartHeight(width) {
  return Math.max(580, Math.min(720, Math.round(width * 0.34)));
}

function lineChartHeight(width) {
  return Math.max(540, Math.min(680, Math.round(width * 0.30)));
}

function renderBarChart(data) {
  const chart = document.getElementById("chart");
  const legend = document.getElementById("legend");
  const metric = data[0].metric;
  const unit = metricUnits[metric];
  const selectedThreads = checkedValues("threads");
  const selectedCommits = selectedOptions("commit");
  const selectedBackends = checkedValues("backend");
  const showThread = selectedThreads.length > 1;
  const showCommit = selectedCommits.length > 1;
  const groups = orderedKeys(data.map(barGroupKey));
  const values = data.map(formatValue);
  const maxValue = axisMax(values);
  const margin = {top: 60, right: 34, bottom: 124, left: 88};
  const backendCount = Math.max(1, selectedBackends.length);
  const barGap = 0;
  const barWidth = backendCount === 1 ? 48 : 40;
  const clusterWidth = backendCount * barWidth + (backendCount - 1) * barGap;
  const groupStep = clusterWidth + 34;
  const availableWidth = Math.max(960, Math.floor(chart.clientWidth - 18));
  const requiredWidth = margin.left + groups.length * groupStep + margin.right;
  const width = Math.max(availableWidth, requiredWidth);
  const height = barChartHeight(width);
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const usedPlotWidth = groups.length * groupStep;
  const plotLeft = margin.left + Math.max(0, (plotWidth - usedPlotWidth) / 2);
  const y = value => margin.top + plotHeight - (value / maxValue) * plotHeight;

  let svg = `<svg viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" role="img">`;
  svg += `<text x="${margin.left}" y="26" class="title">${escapeHtml(chartSubtitle(data))}</text>`;
  svg += `<text x="${margin.left}" y="45" class="axis-label">${unit.label} (${unit.unit}), lower is better</text>`;

  for (let i = 0; i <= 5; i++) {
    const value = (maxValue / 5) * i;
    const yy = y(value);
    svg += `<line x1="${margin.left}" x2="${width - margin.right}" y1="${yy}" y2="${yy}" class="grid" />`;
    svg += `<text x="${margin.left - 10}" y="${yy + 4}" text-anchor="end" class="tick">${formatNumber(value)}</text>`;
  }

  groups.forEach((groupKey, groupIndex) => {
    const groupX = plotLeft + groupIndex * groupStep + groupStep / 2;
    const label = barGroupLabel(groupKey, showThread, showCommit);
    svg += `<text x="${groupX}" y="${height - 46}" text-anchor="middle" class="tick" transform="rotate(-25 ${groupX} ${height - 46})">${escapeHtml(label)}</text>`;
    selectedBackends.forEach((backend, backendIndex) => {
      const row = data.find(item => barGroupKey(item) === groupKey && item.backend === backend);
      if (!row) return;
      const value = formatValue(row);
      const x = groupX - clusterWidth / 2 + backendIndex * (barWidth + barGap) + barWidth / 2;
      const yy = y(value);
      const barHeight = margin.top + plotHeight - yy;
      svg += `<rect x="${x - barWidth / 2}" y="${yy}" width="${barWidth}" height="${barHeight}" rx="3" fill="${colors[backend]}" opacity="0.86"><title>${escapeHtml(label)} · ${labels[backend]}: ${formatNumber(value)} ${unit.unit}</title></rect>`;
      svg += `<text x="${x}" y="${yy - 8}" text-anchor="middle" class="bar-label">${formatNumber(value)}</text>`;
    });
  });

  svg += `<line x1="${margin.left}" x2="${width - margin.right}" y1="${margin.top + plotHeight}" y2="${margin.top + plotHeight}" class="axis" />`;
  svg += `<line x1="${margin.left}" x2="${margin.left}" y1="${margin.top}" y2="${margin.top + plotHeight}" class="axis" />`;

  // % change annotations when exactly 2 commits/tags are compared
  const isTagMode = referenceModeValue() === "tag";
  const xVals = isTagMode
    ? tagOrder(selectedOptions("tag"))
    : commitOrder(selectedOptions("commit"));
  if (xVals.length === 2) {
    const baseVals = new Map();
    groups.forEach((groupKey, groupIndex) => {
      const parts = groupKey.split("|");
      const xKey = parts[2];
      const groupX = plotLeft + groupIndex * groupStep + groupStep / 2;
      selectedBackends.forEach((backend, backendIndex) => {
        const row = data.find(item => barGroupKey(item) === groupKey && item.backend === backend);
        if (!row) return;
        const value = formatValue(row);
        const barX = groupX - clusterWidth / 2 + backendIndex * (barWidth + barGap) + barWidth / 2;
        const key = `${parts[0]}|${parts[1]}|${backend}`;
        if (xKey === xVals[0]) {
          baseVals.set(key, value);
        } else {
          const base = baseVals.get(key);
          if (!base) return;
          const pct = ((value - base) / base) * 100;
          const sign = pct >= 0 ? "+" : "";
          const fill = Math.abs(pct) <= 5 ? "var(--muted)" : pct > 0 ? "var(--danger)" : "var(--success)";
          const annotY = Math.max(margin.top + 4, y(value) - 20);
          svg += `<text x="${barX}" y="${annotY}" text-anchor="middle" font-size="10" fill="${fill}" font-weight="700">${sign}${pct.toFixed(1)}%</text>`;
        }
      });
    });
  }

  svg += `</svg>`;
  chart.innerHTML = svg;
  legend.innerHTML = selectedBackends.map(backend =>
    `<span class="legend-chip"><span class="legend-swatch" style="background:${colors[backend]}"></span>${escapeHtml(labels[backend] || backend)}</span>`
  ).join("");
}

function renderLineChart(data) {
  const chart = document.getElementById("chart");
  const legend = document.getElementById("legend");
  const metric = data[0].metric;
  const unit = metricUnits[metric];
  const isTagMode = referenceModeValue() === "tag";
  const rowXKey = row => isTagMode ? row.tag : row.commit;
  const xKeys = isTagMode
    ? tagOrder(data.map(rowXKey))
    : commitOrder(data.map(row => row.commit));
  const values = data.map(formatValue);
  const maxValue = axisMax(values);
  const width = plotWidthFor(chart, 960, 150, xKeys.length);
  const height = lineChartHeight(width);
  const margin = {top: 60, right: 42, bottom: 108, left: 88};
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const x = key => {
    if (xKeys.length === 1) return margin.left + plotWidth / 2;
    return margin.left + xKeys.indexOf(key) * (plotWidth / (xKeys.length - 1));
  };
  const y = value => margin.top + plotHeight - (value / maxValue) * plotHeight;
  const groups = new Map();
  for (const row of data) {
    const key = `${row.backend}|${row.case}|${row.threads}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }

  let svg = `<svg viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" role="img">`;
  svg += `<text x="${margin.left}" y="26" class="title">${escapeHtml(chartSubtitle(data))}</text>`;
  svg += `<text x="${margin.left}" y="45" class="axis-label">${unit.label} (${unit.unit}), lower is better</text>`;
  for (let i = 0; i <= 5; i++) {
    const value = (maxValue / 5) * i;
    const yy = y(value);
    svg += `<line x1="${margin.left}" x2="${width - margin.right}" y1="${yy}" y2="${yy}" class="grid" />`;
    svg += `<text x="${margin.left - 10}" y="${yy + 4}" text-anchor="end" class="tick">${formatNumber(value)}</text>`;
  }
  xKeys.forEach(xKey => {
    const xx = x(xKey);
    const row = rows.find(item => rowXKey(item) === xKey);
    svg += `<text x="${xx}" y="${height - 56}" text-anchor="middle" class="tick" transform="rotate(-25 ${xx} ${height - 56})">${escapeHtml(xKey)}</text>`;
    if (row && row.date_label !== "-") {
      svg += `<text x="${xx}" y="${height - 30}" text-anchor="middle" class="tick">${escapeHtml(row.date_label)}</text>`;
    }
  });

  Array.from(groups.entries()).forEach(([key, group], index) => {
    const color = linePalette[index % linePalette.length];
    const byXKey = new Map(group.map(row => [rowXKey(row), row]));
    const points = xKeys
      .filter(xKey => byXKey.has(xKey))
      .map(xKey => {
        const row = byXKey.get(xKey);
        return {x: x(xKey), y: y(formatValue(row)), row, xKey};
      });
    if (!points.length) return;
    svg += `<polyline class="line-path" stroke="${color}" points="${points.map(point => `${point.x},${point.y}`).join(" ")}"><title>${escapeHtml(seriesLabel(key))}</title></polyline>`;
    for (const point of points) {
      const value = formatValue(point.row);
      svg += `<circle class="point" cx="${point.x}" cy="${point.y}" r="3.5" stroke="${color}"><title>${escapeHtml(seriesLabel(key))} · ${escapeHtml(point.xKey)}: ${formatNumber(value)} ${unit.unit}</title></circle>`;
    }
  });

  svg += `<line x1="${margin.left}" x2="${width - margin.right}" y1="${margin.top + plotHeight}" y2="${margin.top + plotHeight}" class="axis" />`;
  svg += `<line x1="${margin.left}" x2="${margin.left}" y1="${margin.top}" y2="${margin.top + plotHeight}" class="axis" />`;
  svg += `</svg>`;
  chart.innerHTML = svg;
  legend.innerHTML = Array.from(groups.keys()).map((key, index) =>
    `<span class="legend-chip"><span class="legend-swatch" style="background:${linePalette[index % linePalette.length]}"></span>${escapeHtml(seriesLabel(key))}</span>`
  ).join("");
}

function renderChart() {
  const data = filteredRows();
  const chart = document.getElementById("chart");
  const legend = document.getElementById("legend");
  if (!data.length) {
    const emptyMsg = referenceModeValue() === "tag"
      ? "No benchmarked commits carry a tag in these results. " +
        "Tags appear once a tagged commit has been through the publish workflow."
      : "No matching ASV rows for this selection.";
    chart.innerHTML = `<p style="color:var(--muted);padding:12px 4px">${emptyMsg}</p>`;
    legend.innerHTML = "";
    return;
  }
  if (viewModeValue() === "line") {
    renderLineChart(data);
  } else {
    renderBarChart(data);
  }
}

function renderAll() {
  renderChart();
}

for (const id of ["metric", "family", "viewMode"]) {
  document.getElementById(id).addEventListener("change", event => {
    enforceSingleChoice(id, event);
    refreshOptions();
    renderAll();
  });
}
for (const id of ["case", "threads", "backend"]) {
  document.getElementById(id).addEventListener("change", event => {
    enforceAtLeastOne(id, event);
    refreshOptions();
    renderAll();
  });
}
for (const id of ["commit", "tag"]) {
  const el = document.getElementById(id);
  if (el) el.addEventListener("change", () => { refreshOptions(); renderAll(); });
}
let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(renderAll, 120);
});
refreshOptions();
renderAll();
</script>
</body>
</html>
"""
    return (
        template.replace("__PAYLOAD__", payload)
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
