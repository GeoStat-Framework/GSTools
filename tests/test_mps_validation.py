"""Tests for gstools.mps._validation."""

import unittest

import numpy as np

from gstools.mps._validation import check_float64_integers, exact_count


class TestExactCount(unittest.TestCase):
    def test_integral_values_accepted(self):
        for v, want in [(3, 3), (3.0, 3), (np.int64(5), 5), (True, 1)]:
            self.assertEqual(exact_count(v, minimum=0, context="n"), want)

    def test_fractional_rejected(self):
        for v in (0.9, 1.9, 2.5):
            with self.assertRaisesRegex(ValueError, "must be an integer"):
                exact_count(v, minimum=0, context="n")

    def test_nonfinite_and_text_rejected(self):
        for v in (np.nan, np.inf, "3"):
            with self.assertRaises(ValueError):
                exact_count(v, minimum=0, context="n")

    def test_minimum_enforced(self):
        with self.assertRaisesRegex(ValueError, ">= 1"):
            exact_count(0, minimum=1, context="n")


class TestFloat64Integers(unittest.TestCase):
    def test_lossy_value_rejected(self):
        with self.assertRaisesRegex(ValueError, "not exactly representable"):
            check_float64_integers(np.array([2**60 + 3]), True, "x")

    def test_exact_power_of_two_accepted(self):
        check_float64_integers(np.array([2**60, -(2**53)]), True, "x")

    def test_float_input_untouched(self):
        check_float64_integers(np.array([2.0**60 + 3]), True, "x")


if __name__ == "__main__":
    unittest.main()
