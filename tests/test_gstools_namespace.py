"""Tests for the gstools package namespace."""

import unittest

import gstools as gs


class TestNamespace(unittest.TestCase):
    """The gstools namespace exports mps without duplicates."""

    def test_all_has_no_duplicates_and_lists_mps(self):
        self.assertEqual(len(gs.__all__), len(set(gs.__all__)))
        self.assertIn("mps", gs.__all__)

    def test_mps_names_are_only_reachable_under_gs_mps(self):
        mps_names = [
            "DirectSampling",
            "MPSModel",
            "PostProcessing",
            "TrainingImage",
            "Variable",
            "Zone",
        ]
        for name in mps_names:
            self.assertNotIn(name, gs.__all__)
            self.assertFalse(hasattr(gs, name))
        self.assertCountEqual(gs.mps.__all__, mps_names)


if __name__ == "__main__":
    unittest.main()
