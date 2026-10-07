#!/usr/bin/env python
r"""Render a GitHub PR comment body from an ASV benchmark comparison.

Used by both the pull-request comparison workflow (for same-repo PRs) and the
workflow_run comment workflow (for cross-fork PRs) so the badge wording and
comment format stay in sync across both code paths.

For cross-fork pull requests the comparison text is produced by an untrusted
workflow run and later posted with a privileged token, so every input is
treated as hostile: the comparison can never escape its code block, commit
identifiers must be hexadecimal, and the report link is only kept when it
points at an artifact of the expected repository.

Usage:
    python benchmarks/tools/render_benchmark_comment.py \\
        --base <sha> --head <sha> \\
        --repository <owner/name> \\
        [--artifact-url <url>] \\
        [--comparison <file>] \\
        --output <file>
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# GitHub rejects comment bodies above 65536 characters; leave room for the
# surrounding markup.
MAX_COMPARISON_CHARS = 60_000
# ASV compare change-column markers:
#   "+" regressed, "-" improved,
#   "!" a benchmark that worked on the base now FAILS on the head,
#   "*" a benchmark that failed on the base now works,
#   "x" not comparable (the benchmark signature changed between commits).
# ("~" marks a statistically insignificant ratio in the Ratio column, not the
# change column, so it never appears here — kept for the legacy text layout.)
CHANGE_MARKERS = frozenset("+-~x!*")
SHA_PATTERN = re.compile(r"[0-9a-f]{7,40}")
ARTIFACT_URL_PATTERN = re.compile(
    r"https://github\.com/(?P<repository>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)"
    r"/actions/runs/[0-9]+/artifacts/[0-9]+"
)


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base", required=True, help="Base commit SHA.")
    parser.add_argument("--head", required=True, help="Head commit SHA.")
    parser.add_argument(
        "--repository",
        required=True,
        help="GitHub repository (owner/name) the artifact URL must belong to.",
    )
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


def validate_sha(value):
    """Return ``value`` lower-cased if it is an abbreviated or full git SHA."""
    value = value.strip().lower()
    if not SHA_PATTERN.fullmatch(value):
        raise ValueError(f"Not a git commit SHA: {value!r}")
    return value


def safe_artifact_url(url, repository):
    """Return ``url`` if it is an Actions artifact URL of ``repository``."""
    match = ARTIFACT_URL_PATTERN.fullmatch(url.strip())
    if match is None or match.group("repository") != repository:
        return ""
    return match.group(0)


def change_marker(line):
    """Return the ``asv compare`` change marker of one output line.

    ASV >= 0.6 prints a Markdown table whose first cell holds the marker
    (``| +        | ...``); older releases put the marker in the first
    column followed by whitespace (``+   1.00ms  2.00ms  ...``). Header and
    separator rows yield ``""``.
    """
    if line.startswith("|"):
        cells = line.split("|")
        marker = cells[1].strip() if len(cells) > 2 else ""
    else:
        marker = line[:1] if line[1:2] == " " else ""
    return marker if marker in CHANGE_MARKERS else ""


def code_fence(text):
    """Return a backtick fence that no line inside ``text`` can close."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def render(base, head, artifact_url, repository, comparison):
    """Return the full PR comment body as a string."""
    base = validate_sha(base)[:8]
    head = validate_sha(head)[:8]
    artifact_url = safe_artifact_url(artifact_url, repository)
    report_link = (
        f" · [Full HTML Report ↗]({artifact_url})" if artifact_url else ""
    )

    if len(comparison) > MAX_COMPARISON_CHARS:
        comparison = (
            comparison[:MAX_COMPARISON_CHARS] + "\n... (comparison truncated)"
        )

    # ASV marks "+" (regressed) or "-" (improved) only when BOTH the ratio
    # > 1.05 AND the Mann-Whitney U test agree. "!" marks a benchmark that
    # worked on the base but now fails on the head — a failure must be
    # surfaced, never hidden behind "no significant changes".
    markers = [change_marker(line) for line in comparison.splitlines()]
    regressed = markers.count("+")
    improved = markers.count("-") + markers.count("*")
    failed = markers.count("!")
    note = "Mann-Whitney U · 5% threshold"
    if failed or regressed:
        parts = []
        if failed:
            parts.append(f"{failed} benchmark(s) failed")
        if regressed:
            parts.append(f"{regressed} benchmark(s) regressed")
        badge = f"⚠️ {', '.join(parts)} · {note}"
    elif improved:
        badge = f"✅ {improved} benchmark(s) improved, none regressed · {note}"
    else:
        badge = f"✅ No significant changes detected · {note}"

    fence = code_fence(comparison)
    return (
        "<!-- gstools-openmp-benchmark -->\n"
        "## OpenMP Benchmark Results\n\n"
        f"**Base:** `{base}` → **Head:** `{head}`{report_link}\n\n"
        f"{badge}\n\n"
        "<details>\n<summary>Full ASV comparison</summary>\n\n"
        f"{fence}\n{comparison}\n{fence}\n\n"
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

    if args.artifact_url and not safe_artifact_url(
        args.artifact_url, args.repository
    ):
        print(
            f"Ignoring unexpected artifact URL: {args.artifact_url!r}",
            file=sys.stderr,
        )

    try:
        body = render(
            args.base,
            args.head,
            args.artifact_url,
            args.repository,
            comparison,
        )
    except ValueError as err:
        print(err, file=sys.stderr)
        return 1
    args.output.write_text(body, encoding="utf8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
