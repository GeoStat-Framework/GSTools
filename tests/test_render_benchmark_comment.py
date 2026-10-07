"""Tests for the benchmark PR comment renderer."""

import importlib.util
import unittest
from pathlib import Path

SCRIPT = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "tools"
    / "render_benchmark_comment.py"
)
SPEC = importlib.util.spec_from_file_location(
    "render_benchmark_comment",
    SCRIPT,
)
COMMENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMMENT)

BASE = "a" * 40
HEAD = "b" * 40
REPO = "GeoStat-Framework/GSTools"
ARTIFACT_URL = f"https://github.com/{REPO}/actions/runs/1/artifacts/2"


def fenced_block(body):
    """Return the code fence and content of the comparison block."""
    start = body.index("<summary>Full ASV comparison</summary>\n\n") + len(
        "<summary>Full ASV comparison</summary>\n\n"
    )
    fence = body[start:].split("\n", 1)[0]
    content = body[start + len(fence) + 1 :].rsplit(fence, 1)[0]
    return fence, content


class TestRenderBenchmarkComment(unittest.TestCase):
    """Untrusted comparison text must never escape its code block."""

    def test_plain_comparison_uses_three_backtick_fence(self):
        """Ordinary ASV output keeps the familiar triple-backtick fence."""
        body = COMMENT.render(BASE, HEAD, ARTIFACT_URL, REPO, "+ 1.0s 2.0s")

        fence, content = fenced_block(body)
        self.assertEqual(fence, "```")
        self.assertEqual(content, "+ 1.0s 2.0s\n")

    def test_backtick_runs_cannot_close_fence(self):
        """A fence longer than any run in the content cannot be closed."""
        payload = "x\n```\n# injected heading\n`````\n<img src=x>\n"

        body = COMMENT.render(BASE, HEAD, ARTIFACT_URL, REPO, payload)

        fence, content = fenced_block(body)
        self.assertEqual(fence, "`" * 6)
        self.assertEqual(content, payload + "\n")
        self.assertNotIn("\n# injected heading", body.split(fence)[0])

    def test_oversized_comparison_is_truncated(self):
        """The comment stays below GitHub's body size limit."""
        payload = "x" * (COMMENT.MAX_COMPARISON_CHARS + 100)

        body = COMMENT.render(BASE, HEAD, ARTIFACT_URL, REPO, payload)

        self.assertLess(len(body), 65536)
        self.assertIn("truncated", body)

    def test_sha_must_be_hexadecimal(self):
        """Commit identifiers are rendered as code and must be plain hex."""
        for bad in ("`x`", "abc", "g" * 40, ""):
            with self.assertRaises(ValueError):
                COMMENT.validate_sha(bad)
        self.assertEqual(COMMENT.validate_sha("ABCDEF0"), "abcdef0")

    def test_artifact_url_must_belong_to_repository(self):
        """Only artifact URLs of the expected repository are linked."""
        self.assertEqual(
            COMMENT.safe_artifact_url(ARTIFACT_URL, REPO), ARTIFACT_URL
        )
        for bad in (
            "https://evil.example/report",
            "https://github.com/evil/GSTools/actions/runs/1/artifacts/2",
            f"{ARTIFACT_URL})[click](https://evil.example",
            f"{ARTIFACT_URL}/../../evil",
            "",
        ):
            self.assertEqual(COMMENT.safe_artifact_url(bad, REPO), "")

    def test_change_markers_from_table_and_legacy_output(self):
        """Regressions are counted from both ASV comparison layouts."""
        table = (
            "| Change   | Before [aaaaaaaa]   | After [bbbbbbbb]   | Ratio |\n"
            "|----------|---------------------|--------------------|-------|\n"
            "| +        | 1.00±0ms            | 2.00±0ms           | 2.00  |\n"
            "| -        | 2.00±0ms            | 1.00±0ms           | 0.50  |\n"
            "| ~        | 1.00±0ms            | 1.10±0ms           | 1.10  |\n"
            "|          | 1.00±0ms            | 1.00±0ms           | 1.00  |\n"
        )
        legacy = (
            "       before           after         ratio\n"
            "+     1.00±0ms      2.00±0ms     2.00  bench_a\n"
            "+     1.00±0ms      2.00±0ms     2.00  bench_b\n"
            "-     2.00±0ms      1.00±0ms     0.50  bench_c\n"
            "- not a marker line\n"
        )

        self.assertEqual(
            [COMMENT.change_marker(l) for l in table.splitlines()],
            ["", "", "+", "-", "~", ""],
        )
        self.assertEqual(
            [COMMENT.change_marker(l) for l in legacy.splitlines()],
            ["", "+", "+", "-", "-"],
        )
        self.assertIn(
            "1 benchmark(s) regressed",
            COMMENT.render(BASE, HEAD, "", REPO, table),
        )
        self.assertIn(
            "1 benchmark(s) improved, none regressed",
            COMMENT.render(BASE, HEAD, "", REPO, table.replace("| +", "|  ")),
        )

    def test_introduced_failure_is_surfaced_not_hidden(self):
        """A benchmark that now fails ("!") must not read as "no changes"."""
        table = (
            "| Change   | Before [aaaaaaaa]   | After [bbbbbbbb]   | Ratio |\n"
            "|----------|---------------------|--------------------|-------|\n"
            "| !        | 1.00±0ms            | failed             | n/a   |\n"
            "|          | 1.00±0ms            | 1.00±0ms           | 1.00  |\n"
        )
        self.assertEqual(
            [COMMENT.change_marker(l) for l in table.splitlines()],
            ["", "", "!", ""],
        )
        body = COMMENT.render(BASE, HEAD, "", REPO, table)
        self.assertIn("1 benchmark(s) failed", body)
        self.assertNotIn("No significant changes detected", body)
        self.assertIn("⚠️", body)

    def test_rejected_artifact_url_drops_link(self):
        """An unexpected URL results in no report link at all."""
        body = COMMENT.render(
            BASE, HEAD, "https://evil.example/report", REPO, "ok"
        )

        self.assertNotIn("evil.example", body)
        self.assertNotIn("Full HTML Report", body)


if __name__ == "__main__":
    unittest.main()
