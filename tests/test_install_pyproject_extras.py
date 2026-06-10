"""Tests for the benchmark pyproject extras installer."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "tools"
    / "install_pyproject_extras.py"
)
SPEC = importlib.util.spec_from_file_location(
    "install_pyproject_extras", SCRIPT
)
INSTALLER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSTALLER)


class TestInstallPyprojectExtras(unittest.TestCase):
    """Test loading and installing pyproject optional dependencies."""

    def setUp(self):
        """Create a small pyproject fixture."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.pyproject = Path(self.temp_dir.name) / "pyproject.toml"
        self.pyproject.write_text(
            """
[project]
name = "example"

[project.optional-dependencies]
first = ["one>=1", "shared>=1"]
second = ["shared>=1", "two>=2"]
""".strip(),
            encoding="utf8",
        )

    def tearDown(self):
        """Remove the temporary pyproject fixture."""
        self.temp_dir.cleanup()

    def test_extra_requirements_preserves_order_and_deduplicates(self):
        """Requirements follow requested-extra order without duplicates."""
        requirements = INSTALLER.extra_requirements(
            self.pyproject,
            ["first", "second"],
        )

        self.assertEqual(requirements, ["one>=1", "shared>=1", "two>=2"])

    def test_extra_requirements_rejects_unknown_extra(self):
        """Unknown extras fail with a useful list of available extras."""
        with self.assertRaisesRegex(
            ValueError,
            "Available extras: first, second",
        ):
            INSTALLER.extra_requirements(self.pyproject, ["missing"])

    def test_install_requirements_uses_current_python(self):
        """Installation calls pip through the helper's Python executable."""
        with mock.patch.object(INSTALLER.subprocess, "run") as run:
            INSTALLER.install_requirements(["one>=1", "two>=2"])

        run.assert_called_once_with(
            [
                INSTALLER.sys.executable,
                "-m",
                "pip",
                "install",
                "one>=1",
                "two>=2",
            ],
            check=True,
        )
