#!/usr/bin/env python
"""Summarize Rust-vs-Cython speedups from local ASV result files.

The summary is optional. ASV itself remains the source of truth for benchmark
storage and visualization. The helper understands both current generated
benchmark method names, which encode case and thread labels, and older
parameterized ASV result files.

Usage:
    python benchmarks/tools/asv_speedup_summary.py
    python benchmarks/tools/asv_speedup_summary.py --results-dir .asv/results
    python benchmarks/tools/asv_speedup_summary.py --format markdown
    python benchmarks/tools/asv_speedup_summary.py --format html --output report.html
    python benchmarks/tools/asv_speedup_summary.py --include-legacy

Speedup is calculated as:
    cython_fallback_time / rust_core_time

Values greater than 1.0 mean Rust was faster on the same machine, commit,
environment, benchmark, case, and thread-count combination.
"""

from __future__ import annotations

import argparse
import html
import itertools
import json
import math
import re
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
LEGACY_BENCHMARKS = {
    "time_srf",
    "peakmem_srf",
    "time_variogram",
    "peakmem_variogram",
    "time_krige",
    "peakmem_krige",
}


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        action="append",
        type=Path,
        help=(
            "Path to an ASV results directory. Can be supplied more than once."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("table", "markdown", "html"),
        default="table",
        help="Output format.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write Markdown or HTML output to this file instead of stdout.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include non-time benchmarks as ratios too.",
    )
    parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="Include removed BackendBenchmarks rows from older saved results.",
    )
    return parser.parse_args()


def iter_result_files(results_dir):
    """Yield ASV result JSON files below ``results_dir``."""
    for path in sorted(results_dir.glob("**/*.json")):
        if path.name in {"benchmarks.json", "machine.json"}:
            continue
        yield path


def load_json(path):
    """Load one ASV result file, ignoring unreadable or invalid JSON files."""
    try:
        with path.open(encoding="utf8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def result_entry(raw_result, result_columns):
    """Normalize ASV result payloads across result schema versions."""
    if isinstance(raw_result, dict):
        return raw_result
    if isinstance(raw_result, list) and result_columns:
        return dict(zip(result_columns, raw_result))
    return {"result": raw_result, "params": []}


def is_number(value):
    """Return whether ``value`` is a numeric, non-NaN result."""
    return isinstance(value, (int, float)) and not math.isnan(value)


def flatten_values(values):
    """Yield scalar values from nested ASV result lists."""
    if isinstance(values, list):
        for value in values:
            yield from flatten_values(value)
        return
    yield values


def parse_benchmark_name(name):
    """Split a benchmark name into benchmark, case, and thread labels."""
    short_name = short_benchmark_name(name)
    threads = "-"
    match = re.search(r"_(threads_\d+)$", short_name)
    if match:
        threads = match.group(1)
        short_name = short_name[: match.start()]

    for prefix in BENCHMARK_PREFIXES:
        if short_name == prefix:
            return prefix, "-", threads
        marker = f"{prefix}_"
        if short_name.startswith(marker):
            return prefix, short_name[len(marker) :], threads

    return short_name, "-", threads


def backend_rows(benchmark, entry):
    """Return backend/value rows for one ASV benchmark entry."""
    parsed_benchmark, parsed_case, parsed_threads = parse_benchmark_name(
        benchmark
    )
    result = entry.get("result")
    params = entry.get("params") or []
    if not isinstance(result, list) or not params:
        return []

    rows = []
    combinations = itertools.product(*params)
    for combo, value in zip(combinations, flatten_values(result)):
        if not is_number(value):
            continue
        combo_values = [str(item).strip("'\"") for item in combo]
        backend = next(
            (candidate for candidate in BACKENDS if candidate in combo_values),
            None,
        )
        if backend is None:
            continue
        case_values = [
            item
            for item in combo_values
            if item not in BACKENDS and not item.startswith(THREAD_PREFIX)
        ]
        threads = next(
            (item for item in combo_values if item.startswith(THREAD_PREFIX)),
            parsed_threads,
        )
        rows.append(
            {
                "backend": backend,
                "benchmark": parsed_benchmark,
                "case": "/".join(case_values) if case_values else parsed_case,
                "threads": threads,
                "value": float(value),
            }
        )
    return rows


def short_benchmark_name(name):
    """Return the method name from a fully qualified ASV benchmark name."""
    return name.rsplit(".", maxsplit=1)[-1]


def collect_speedups(results_dirs, include_all, include_legacy):
    """Collect Rust-vs-Cython ratios from one or more ASV result folders."""
    rows = []
    for results_dir in results_dirs:
        for path in iter_result_files(results_dir):
            data = load_json(path)
            if data is None:
                continue
            result_columns = data.get("result_columns", [])
            commit = data.get("commit_hash", "unknown")[:8]
            date = data.get("date", 0)
            env_name = data.get("env_name", path.stem)
            results = data.get("results", {})
            for benchmark, raw_result in results.items():
                benchmark_name, _, _ = parse_benchmark_name(benchmark)
                if not include_legacy and benchmark_name in LEGACY_BENCHMARKS:
                    continue
                if not include_all and ".time_" not in benchmark:
                    continue
                by_case = {}
                entry = result_entry(raw_result, result_columns)
                for row in backend_rows(benchmark, entry):
                    key = (row["benchmark"], row["case"], row["threads"])
                    by_case.setdefault(key, {})[row["backend"]] = row["value"]
                for (name, case, threads), values in by_case.items():
                    cython = values.get("cython_fallback")
                    rust = values.get("rust_core")
                    if (
                        not is_number(cython)
                        or not is_number(rust)
                        or rust == 0
                    ):
                        continue
                    rows.append(
                        {
                            "source": str(results_dir),
                            "commit": commit,
                            "date": date,
                            "env": env_name,
                            "benchmark": name,
                            "case": case,
                            "threads": threads,
                            "cython": cython,
                            "rust": rust,
                            "speedup": cython / rust,
                        }
                    )
    return rows


def sort_rows(rows):
    """Sort rows for stable terminal and report output."""
    return sorted(
        rows,
        key=lambda row: (
            row["source"],
            row["benchmark"],
            row["case"],
            thread_number(row["threads"]),
            row["date"],
            row["commit"],
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


def latest_rows(rows):
    """Keep the newest row for each source, env, benchmark, case, and thread."""
    latest = {}
    for row in rows:
        key = (
            row["source"],
            row["env"],
            row["benchmark"],
            row["case"],
            row["threads"],
        )
        if key not in latest or row["date"] >= latest[key]["date"]:
            latest[key] = row
    return sort_rows(latest.values())


def format_date(timestamp):
    """Format an ASV millisecond timestamp as a UTC date."""
    if not timestamp:
        return "-"
    return datetime.fromtimestamp(
        timestamp / 1000.0,
        tz=timezone.utc,
    ).strftime("%Y-%m-%d")


def print_table(rows):
    """Print speedup rows as an aligned terminal table."""
    if not rows:
        print("No matching Rust-vs-Cython ASV results found.")
        return

    headers = [
        "source",
        "commit",
        "date",
        "env",
        "benchmark",
        "case",
        "threads",
        "cython",
        "rust",
        "speedup",
    ]
    table = [
        [
            row["source"],
            row["commit"],
            format_date(row["date"]),
            row["env"],
            row["benchmark"],
            row["case"],
            row["threads"],
            f"{row['cython']:.6g}",
            f"{row['rust']:.6g}",
            f"{row['speedup']:.3f}x",
        ]
        for row in rows
    ]
    widths = [
        max(len(str(item)) for item in column)
        for column in zip(headers, *table)
    ]

    def fmt(row):
        """Format one aligned table row."""
        return "  ".join(
            str(item).ljust(width) for item, width in zip(row, widths)
        )

    print(fmt(headers))
    print(fmt(["-" * width for width in widths]))
    for row in table:
        print(fmt(row))


def markdown_table(headers, rows):
    """Render a simple Markdown table."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def render_markdown(rows):
    """Render the latest speedups and thread scaling as Markdown."""
    if not rows:
        return "No matching Rust-vs-Cython ASV results found.\n"

    rows = latest_rows(rows)
    lines = [
        "# GSTools ASV Backend Report",
        "",
        "Values greater than `1.0x` mean `rust_core` was faster than "
        "`cython_fallback` for the same benchmark, case, commit, and thread "
        "label.",
        "",
    ]

    for benchmark in sorted({row["benchmark"] for row in rows}):
        benchmark_rows = [row for row in rows if row["benchmark"] == benchmark]
        lines.extend([f"## {benchmark}", ""])
        table = [
            [
                row["source"],
                row["commit"],
                format_date(row["date"]),
                row["case"],
                row["threads"],
                f"{row['cython']:.6g}",
                f"{row['rust']:.6g}",
                f"{row['speedup']:.3f}x",
            ]
            for row in benchmark_rows
        ]
        lines.extend(
            [
                markdown_table(
                    [
                        "source",
                        "commit",
                        "date",
                        "case",
                        "threads",
                        "cython",
                        "rust",
                        "speedup",
                    ],
                    table,
                ),
                "",
            ]
        )

    scaling = thread_scaling_rows(rows)
    if scaling:
        lines.extend(["## Thread Scaling", ""])
        lines.extend(
            [
                markdown_table(
                    [
                        "source",
                        "commit",
                        "benchmark",
                        "case",
                        "backend",
                        "threads",
                        "time",
                        "vs min thread",
                    ],
                    scaling,
                ),
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def thread_scaling_rows(rows):
    """Build rows that compare each backend against its lowest thread count."""
    by_key = {}
    for row in rows:
        for backend, value in (
            ("cython_fallback", row["cython"]),
            ("rust_core", row["rust"]),
        ):
            key = (
                row["source"],
                row["commit"],
                row["benchmark"],
                row["case"],
                backend,
            )
            by_key.setdefault(key, {})[row["threads"]] = value

    table = []
    for key, values in sorted(by_key.items()):
        if len(values) < 2:
            continue
        source, commit, benchmark, case, backend = key
        base_thread = min(values, key=thread_number)
        base_time = values[base_thread]
        if not base_time:
            continue
        for threads in sorted(values, key=thread_number):
            table.append(
                [
                    source,
                    commit,
                    benchmark,
                    case,
                    backend,
                    threads,
                    f"{values[threads]:.6g}",
                    f"{base_time / values[threads]:.3f}x",
                ]
            )
    return table


def render_html(rows):
    """Render the Markdown report as a small standalone HTML page."""
    markdown = render_markdown(rows)
    body_lines = []
    in_table = False
    for line in markdown.splitlines():
        if line.startswith("# "):
            body_lines.append(f"<h1>{html.escape(line[2:])}</h1>")
            continue
        if line.startswith("## "):
            body_lines.append(f"<h2>{html.escape(line[3:])}</h2>")
            continue
        if line.startswith("| "):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if all(cell == "---" for cell in cells):
                continue
            tag = "th" if not in_table else "td"
            if tag == "th":
                body_lines.append("<table>")
                in_table = True
            body_lines.append(
                "<tr>"
                + "".join(
                    f"<{tag}>{html.escape(cell)}</{tag}>" for cell in cells
                )
                + "</tr>"
            )
            continue
        if in_table:
            body_lines.append("</table>")
            in_table = False
        if line:
            body_lines.append(f"<p>{html.escape(line)}</p>")
    if in_table:
        body_lines.append("</table>")

    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        "<title>GSTools ASV Backend Report</title>\n"
        "<style>\n"
        "body{font-family:system-ui,sans-serif;margin:2rem;line-height:1.4;}\n"
        "table{border-collapse:collapse;margin:1rem 0 2rem;width:100%;}\n"
        "th,td{border:1px solid #ddd;padding:.35rem .5rem;text-align:left;}\n"
        "th{background:#f4f4f4;}\n"
        "code{background:#f4f4f4;padding:.1rem .25rem;}\n"
        "</style>\n"
        "</head>\n"
        "<body>\n" + "\n".join(body_lines) + "\n</body>\n</html>\n"
    )


def write_or_print(text, output):
    """Write rendered output to a file or print it to stdout."""
    if output:
        output.write_text(text, encoding="utf8")
        print(f"Wrote {output}")
        return
    print(text, end="")


def main():
    """Run the ASV speedup summary command."""
    args = parse_args()
    results_dirs = args.results_dir or [Path(".asv/results")]
    rows = collect_speedups(results_dirs, args.all, args.include_legacy)
    rows = sort_rows(rows)
    if args.format == "table":
        print_table(rows)
    elif args.format == "markdown":
        write_or_print(render_markdown(rows), args.output)
    else:
        write_or_print(render_html(rows), args.output)


if __name__ == "__main__":
    main()
