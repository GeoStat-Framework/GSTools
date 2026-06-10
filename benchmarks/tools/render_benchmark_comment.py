#!/usr/bin/env python
"""Render a GitHub PR comment body from an ASV benchmark comparison.

Used by both the pull-request comparison workflow (for same-repo PRs) and the
workflow_run comment workflow (for cross-fork PRs) so the badge wording and
comment format stay in sync across both code paths.

Usage:
    python benchmarks/tools/render_benchmark_comment.py \\
        --base <sha> --head <sha> \\
        [--artifact-url <url>] \\
        [--comparison <file>] \\
        --output <file>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base", required=True, help="Base commit SHA.")
    parser.add_argument("--head", required=True, help="Head commit SHA.")
    parser.add_argument(
        "--artifact-url",
        default="",
        help="URL to the full HTML report artifact.",
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        default=None,
        help="Path to the ASV comparison text file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Write the rendered comment body to this file.",
    )
    return parser.parse_args()


def render(base, head, artifact_url, comparison):
    """Return the full PR comment body as a string."""
    base = base[:8]
    head = head[:8]
    report_link = (
        f" · [Full HTML Report ↗]({artifact_url})" if artifact_url else ""
    )

    lines = comparison.splitlines()
    # ASV marks "+" (regressed) or "-" (improved) only when BOTH the
    # ratio > 1.05 AND the Mann-Whitney U test agree. Lines with "~"
    # exceeded the ratio threshold but were not statistically significant.
    regressed = sum(1 for l in lines if l.startswith("+") and l[1:2] == " ")
    improved = sum(1 for l in lines if l.startswith("-") and l[1:2] == " ")
    note = "Mann-Whitney U · 5% threshold"
    if regressed:
        badge = f"⚠️ {regressed} benchmark(s) regressed · {note}"
    elif improved:
        badge = f"✅ {improved} benchmark(s) improved, none regressed · {note}"
    else:
        badge = f"✅ No significant changes detected · {note}"

    return (
        "<!-- gstools-openmp-benchmark -->\n"
        "## OpenMP Benchmark Results\n\n"
        f"**Base:** `{base}` → **Head:** `{head}`{report_link}\n\n"
        f"{badge}\n\n"
        "<details>\n<summary>Full ASV comparison</summary>\n\n"
        f"```\n{comparison}\n```\n\n"
        "</details>\n"
    )


def main():
    """Render and write the PR comment body."""
    args = parse_args()

    if args.comparison is not None:
        try:
            comparison = args.comparison.read_text(encoding="utf8")
        except OSError as err:
            print(f"Cannot read comparison file: {err}", file=sys.stderr)
            return 1
    else:
        comparison = "_Comparison output not available._"

    body = render(args.base, args.head, args.artifact_url, comparison)
    args.output.write_text(body, encoding="utf8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
