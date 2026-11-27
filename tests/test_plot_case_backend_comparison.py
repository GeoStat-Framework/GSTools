"""Tests for the custom ASV case/backend comparison report."""

import contextlib
import importlib.util
import io
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
                        "result": [0.1],
                        "params": [["rust_core"]],
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

    def test_load_json_warns_on_broken_file(self):
        """A corrupt result file is reported, not silently ignored."""
        with tempfile.TemporaryDirectory() as temp_dir:
            bad = Path(temp_dir) / "broken.json"
            bad.write_text("{not valid json", encoding="utf8")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                result = REPORT.load_json(bad)
        self.assertIsNone(result)
        self.assertIn("could not read", stderr.getvalue())

    def test_result_rows_reports_missing_measurements(self):
        """Null/NaN measurements are skipped with an explicit warning."""
        entry = {
            "result": [0.1, None, float("nan")],
            "params": [
                ["rust_core"],
                ["small_30x500", "large_120x2000", "extra_large_360x6000"],
                ["threads_1"],
            ],
        }
        name = "benchmark_two_point_statistics.KrigingBenchmarks.time_global_krige"
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            rows = REPORT.result_rows(name, entry)
        self.assertEqual(len(rows), 1)  # only the one numeric value survives
        self.assertIn("skipped 2 missing/failed", stderr.getvalue())

    def test_result_rows_warns_on_length_mismatch(self):
        """A value/combination count mismatch is surfaced, not hidden."""
        entry = {
            "result": [0.1],  # one value
            "params": [
                ["rust_core"],
                ["small_30x500"],
                ["threads_1", "threads_2"],
            ],  # two combinations
        }
        name = "benchmark_two_point_statistics.KrigingBenchmarks.time_global_krige"
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            REPORT.result_rows(name, entry)
        self.assertIn("do not match", stderr.getvalue())

    def test_warns_when_one_machine_name_spans_hardware(self):
        """Two CPUs under the same machine name are flagged, not hidden."""
        row_a = result_row("a" * 40, 1_700_000_000_000)
        row_b = result_row("b" * 40, 1_800_000_000_000)
        row_a.update(machine="github-actions-ubuntu", cpu="Intel X", os="L1")
        row_b.update(machine="github-actions-ubuntu", cpu="AMD Y", os="L2")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            REPORT.warn_on_mixed_hardware([row_a, row_b])
        self.assertIn("spans 2 different", stderr.getvalue())

    def test_no_hardware_warning_for_single_machine(self):
        """A single consistent machine produces no warning."""
        rows = [result_row("a" * 40, 1_700_000_000_000)]
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            REPORT.warn_on_mixed_hardware(rows)
        self.assertEqual(stderr.getvalue(), "")

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
