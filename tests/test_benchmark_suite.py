"""Reproducibility tests for the benchmark input generators.

These are non-timed, infrastructure-level checks: ASV compares the *runtime* of
equivalent work across commits and thread counts, which is only meaningful if
each case feeds deterministic, finite inputs. These tests guard that property
without running the (timed) benchmark methods themselves.
"""

import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

MODULE_PATH = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "benchmark_two_point_statistics.py"
)
SPEC = importlib.util.spec_from_file_location(
    "benchmark_two_point_statistics", MODULE_PATH
)
SUITE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUITE)


def _assert_identical(first, second):
    """Assert two nested tuples of arrays are element-for-element identical."""
    for a, b in zip(first, second):
        if isinstance(a, tuple):
            _assert_identical(a, b)
        else:
            np.testing.assert_array_equal(a, b)


class TestBenchmarkInputDeterminism(unittest.TestCase):
    """Fixed seeds must yield byte-identical, finite benchmark inputs."""

    def test_variogram_inputs_are_deterministic(self):
        """The same seed reproduces identical variogram inputs."""
        first = SUITE._make_variogram_data(20220501, 900)
        second = SUITE._make_variogram_data(20220501, 900)
        _assert_identical(first, second)

    def test_krige_inputs_are_deterministic(self):
        """The same seed reproduces identical kriging inputs."""
        first = SUITE._make_krige_data(20220506, 30, 500)
        second = SUITE._make_krige_data(20220506, 30, 500)
        _assert_identical(first, second)

    def test_different_seeds_differ(self):
        """Different seeds produce different point clouds."""
        (x1, _), _, _ = SUITE._make_variogram_data(20220501, 900)
        (x2, _), _, _ = SUITE._make_variogram_data(20220502, 900)
        self.assertFalse(np.array_equal(x1, x2))

    def test_inputs_are_finite(self):
        """Generated fields and positions contain no NaN/inf."""
        (vx, vy), vfield, vbins = SUITE._make_variogram_data(20220501, 900)
        for arr in (vx, vy, vfield, vbins):
            self.assertTrue(np.isfinite(arr).all())
        (cx, cy), cval, (tx, ty) = SUITE._make_krige_data(20220506, 30, 500)
        for arr in (cx, cy, cval, tx, ty):
            self.assertTrue(np.isfinite(arr).all())


class TestBenchmarkCoverage(unittest.TestCase):
    """setup_cache must build exactly the declared cases for each family."""

    def test_variogram_cache_matches_declared_cases(self):
        """VariogramBenchmarks caches one dataset per declared case."""
        cache = SUITE.VariogramBenchmarks().setup_cache()
        self.assertEqual(set(cache), set(SUITE.VARIOGRAM_CASES))

    def test_krige_cache_matches_declared_cases(self):
        """KrigingBenchmarks caches one dataset per declared case."""
        cache = SUITE.KrigingBenchmarks().setup_cache()
        self.assertEqual(set(cache), set(SUITE.KRIGE_CASES))

    def test_single_rust_backend(self):
        """The backend axis exposes exactly the Rust core."""
        self.assertEqual(SUITE.AVAILABLE_BACKENDS, ("rust_core",))


class TestThreadCountParsing(unittest.TestCase):
    """GSTOOLS_BENCHMARK_THREADS must only accept positive thread counts."""

    def test_rejects_zero_and_negative(self):
        """Zero or negative thread counts are rejected with a clear error."""
        for bad in ("0", "1,0", "-2", "1,-1,4", "threads_0"):
            with mock.patch.dict(
                os.environ, {"GSTOOLS_BENCHMARK_THREADS": bad}
            ):
                with self.assertRaises(ValueError):
                    SUITE._configured_thread_counts()

    def test_accepts_positive_counts(self):
        """A comma-separated list of positive counts parses in order."""
        with mock.patch.dict(
            os.environ, {"GSTOOLS_BENCHMARK_THREADS": "1,2,4,8"}
        ):
            self.assertEqual(SUITE._configured_thread_counts(), (1, 2, 4, 8))


if __name__ == "__main__":
    unittest.main()
