"""Integration tests against the installed Airspeed Velocity (ASV).

The benchmark tooling in ``benchmarks/tools`` depends on ASV's result JSON
layout, its ``asv compare`` text output and a handful of command-line
options. These tests exercise the real ``asv`` executable on the real
benchmark suite so that a new ASV release which changes any of those
surfaces fails CI instead of silently producing empty or misleading reports.

The tests are skipped when ASV is not installed or the checkout has no git
history (for example when running from an sdist).
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
TOOLS = REPO_ROOT / "benchmarks" / "tools"
FAST_CASE = "time_variogram_estimate\\('cython_fallback', 'full_900'"


def load_tool(name):
    """Import one script from ``benchmarks/tools`` as a module."""
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def asv_available():
    """Return True if ASV can be imported by the test interpreter."""
    return importlib.util.find_spec("asv") is not None


def git(*args):
    """Run git in the repository root and return stripped stdout."""
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_history_available():
    """Return True if the checkout has at least two commits."""
    if shutil.which("git") is None:
        return False
    try:
        git("rev-parse", "HEAD~1")
    except subprocess.CalledProcessError:
        return False
    return True


@unittest.skipUnless(asv_available(), "asv is not installed")
@unittest.skipUnless(git_history_available(), "no git history available")
class TestAsvIntegration(unittest.TestCase):
    """Run the installed ASV once and check every surface we depend on."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.results_dir = root / "results"
        cls.machine = "asv-integration-test"
        cls.head = git("rev-parse", "HEAD")
        cls.base = git("rev-parse", "HEAD~1")

        # Derive the test configuration from the real one so that a config
        # key the installed ASV no longer accepts is caught as well.
        config = json.loads((REPO_ROOT / "asv.conf.json").read_text())
        config["repo"] = str(REPO_ROOT)
        config["benchmark_dir"] = str(REPO_ROOT / "benchmarks")
        config["branches"] = [cls.head]
        config["results_dir"] = str(cls.results_dir)
        config["html_dir"] = str(root / "html")
        config["env_dir"] = str(root / "env")
        cls.config = root / "asv.conf.json"
        cls.config.write_text(json.dumps(config), encoding="utf8")

        # Keep ASV's machine file and the benchmark matrix away from the
        # developer's home directory and from the Rust backend.
        cls.env = dict(os.environ)
        cls.env.update(
            HOME=str(root / "home"),
            USERPROFILE=str(root / "home"),
            GSTOOLS_BENCHMARK_BACKENDS="cython_fallback",
            GSTOOLS_BENCHMARK_THREADS="1",
        )
        (root / "home").mkdir()

        # Same helper as CI: stores detected hardware under a stable name
        # through ASV's Python API (``asv machine --machine`` alone would
        # record only the name).
        subprocess.run(
            [
                sys.executable,
                str(TOOLS / "configure_asv_machine.py"),
                cls.machine,
            ],
            cwd=cls.tmp.name,
            env=cls.env,
            check=True,
            capture_output=True,
        )
        cls.asv(
            "run",
            "--python=same",
            "--quick",
            "--machine",
            cls.machine,
            "--set-commit-hash",
            cls.head,
            "--bench",
            FAST_CASE,
        )
        cls.head_result = cls.result_file(cls.head)
        cls.write_slower_base_result()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def asv(cls, *args):
        """Run one ASV command against the temporary configuration."""
        completed = subprocess.run(
            [sys.executable, "-m", "asv", "--config", str(cls.config), *args],
            cwd=cls.tmp.name,
            env=cls.env,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise AssertionError(
                f"asv {' '.join(args)} failed:\n{completed.stdout}\n"
                f"{completed.stderr}"
            )
        return completed.stdout

    @classmethod
    def result_file(cls, commit_hash):
        """Return the single ASV result file written for ``commit_hash``."""
        matches = list(
            (cls.results_dir / cls.machine).glob(f"{commit_hash[:8]}-*.json")
        )
        if len(matches) != 1:
            raise AssertionError(f"Expected one result file, got {matches}")
        return matches[0]

    @classmethod
    def write_slower_base_result(cls):
        """Fabricate base-commit results at half the head timings.

        ``asv compare`` then has to flag the head commit as a regression,
        which is the signal the PR comment badge relies on.
        """
        payload = json.loads(cls.head_result.read_text(encoding="utf8"))
        payload["commit_hash"] = cls.base
        payload["date"] -= 1000
        columns = payload["result_columns"]
        for entry in payload["results"].values():
            values = entry[columns.index("result")]
            entry[columns.index("result")] = [
                None if value is None else value / 2 for value in values
            ]
        base_file = cls.head_result.with_name(
            cls.head_result.name.replace(cls.head[:8], cls.base[:8])
        )
        base_file.write_text(json.dumps(payload), encoding="utf8")

    def test_result_json_is_understood_by_report_tool(self):
        """The result layout written by ASV maps onto our normalized rows."""
        report = load_tool("plot_case_backend_comparison")

        rows = report.collect_rows([self.results_dir])
        head_rows = [row for row in rows if row["commit_hash"] == self.head]

        self.assertEqual(len(head_rows), 1, rows)
        row = head_rows[0]
        self.assertEqual(row["benchmark"], "time_variogram_estimate")
        self.assertEqual(row["family"], "variogram")
        self.assertEqual(row["metric"], "time")
        self.assertEqual(row["backend"], "cython_fallback")
        self.assertEqual(row["case"], "full_900")
        self.assertEqual(row["threads"], "threads_1")
        self.assertEqual(row["machine"], self.machine)
        self.assertGreater(row["value"], 0.0)

    def test_result_json_keeps_machine_metadata(self):
        """``machine.json`` still carries the fields shown in the report."""
        machine = json.loads(
            (self.results_dir / self.machine / "machine.json").read_text()
        )
        for key in ("machine", "os", "arch", "cpu", "num_cpu", "ram"):
            self.assertIn(key, machine)

    def test_compare_output_is_understood_by_comment_renderer(self):
        """A regression flagged by ``asv compare`` reaches the PR badge."""
        comment = load_tool("render_benchmark_comment")
        comparison = self.asv(
            "compare",
            self.base,
            self.head,
            "--machine",
            self.machine,
            "--factor",
            "1.05",
            "--split",
        )

        body = comment.render(
            self.base, self.head, "", "GeoStat-Framework/GSTools", comparison
        )

        self.assertIn("time_variogram_estimate", comparison)
        self.assertIn("1 benchmark(s) regressed", body)

    def test_publish_builds_html_site(self):
        """``asv publish`` accepts our options and writes the site."""
        html_dir = Path(self.tmp.name) / "published"

        self.asv("publish", "--no-pull", "--html-dir", str(html_dir))

        self.assertTrue((html_dir / "index.html").is_file())

    def test_repository_configs_load_with_installed_asv(self):
        """Both ASV configuration files are accepted by the installed ASV."""
        from asv.config import Config

        for name in ("asv.conf.json", "asv.openmp.conf.json"):
            config = Config.load(str(REPO_ROOT / name))
            self.assertEqual(config.benchmark_dir, "benchmarks")
            self.assertTrue(config.build_command, name)
            self.assertTrue(config.install_command, name)


if __name__ == "__main__":
    unittest.main()
