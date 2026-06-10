"""Tests for the custom ASV case/backend comparison report."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "tools"
    / "plot_case_backend_comparison.py"
)
SPEC = importlib.util.spec_from_file_location(
    "plot_case_backend_comparison",
    SCRIPT,
)
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def result_row(commit_hash, date):
    """Create one normalized row for report-only tests."""
    return {
        "source": "results",
        "commit": commit_hash[:8],
        "commit_hash": commit_hash,
        "date": date,
        "date_label": REPORT.format_date(date),
        "env": "conda-py3.14",
        "python": "3.14",
        "machine": "github-actions-ubuntu",
        "os": "Ubuntu 24.04",
        "arch": "x86_64",
        "cpu": "Test CPU",
        "num_cpu": "4",
        "ram": str(8 * 1024**3),
        "benchmark": "time_global_krige",
        "family": "krige",
        "metric": "time",
        "case": "small_30x500",
        "threads": "threads_1",
        "backend": "rust_core",
        "value": 0.1,
    }


class TestPlotCaseBackendComparison(unittest.TestCase):
    """Test ASV result normalization and report rendering."""

    def test_collect_rows_includes_machine_metadata(self):
        """Adjacent ASV machine metadata is retained for the report."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir)
            machine_dir = results_dir / "github-actions-ubuntu"
            machine_dir.mkdir()
            machine = {
                "machine": "github-actions-ubuntu",
                "os": "Ubuntu 24.04",
                "arch": "x86_64",
                "cpu": "Test CPU",
                "num_cpu": "4",
                "ram": str(8 * 1024**3),
            }
            result = {
                "commit_hash": "a" * 40,
                "date": 1_700_000_000_000,
                "env_name": "conda-py3.14",
                "python": "3.14",
                "params": {"machine": "github-actions-ubuntu"},
                "results": {
                    (
                        "benchmark_two_point_statistics.KrigingBenchmarks."
                        "time_global_krige_small_30x500_threads_1"
                    ): {
                        "result": [0.2, 0.1],
                        "params": [["cython_fallback", "rust_core"]],
                    }
                },
            }
            (machine_dir / "machine.json").write_text(
                json.dumps(machine),
                encoding="utf8",
            )
            (machine_dir / "result.json").write_text(
                json.dumps(result),
                encoding="utf8",
            )

            rows = REPORT.collect_rows([results_dir])

        self.assertEqual(
            {row["machine"] for row in rows}, {"github-actions-ubuntu"}
        )
        self.assertEqual({row["python"] for row in rows}, {"3.14"})
        self.assertEqual({row["cpu"] for row in rows}, {"Test CPU"})

    def test_limit_recent_commits_keeps_only_newest_commits(self):
        """The custom report can remain bounded without deleting results."""
        rows = [
            result_row("a" * 40, 1_700_000_000_000),
            result_row("b" * 40, 1_800_000_000_000),
            result_row("c" * 40, 1_900_000_000_000),
        ]

        limited = REPORT.limit_recent_commits(rows, 2)

        self.assertEqual(
            {row["commit_hash"] for row in limited},
            {"b" * 40, "c" * 40},
        )

    def test_render_html_shows_machine_and_asv_links(self):
        """The report identifies its machine and links to complete ASV data."""
        rendered = REPORT.render_html(
            [result_row("a" * 40, 1_700_000_000_000)]
        )

        self.assertIn("github-actions-ubuntu", rendered)
        self.assertIn("Ubuntu 24.04", rendered)
        self.assertIn("Python 3.14", rendered)
        self.assertIn('href="asv/"', rendered)
        self.assertIn("Powered by Airspeed Velocity (ASV)", rendered)


if __name__ == "__main__":
    unittest.main()
