"""Tests for gstools.mps.data_event."""

import unittest

import numpy as np

from gstools.mps.data_event import DataEvent


class TestDataEvent(unittest.TestCase):
    def _make(self, k=3, dim=2):
        lags_sg = np.arange(k * dim, dtype=np.float64).reshape(k, dim)
        values = np.arange(k, dtype=np.float64)
        cond_mask = np.array([True, False, False][:k], dtype=bool)
        lag_norms = np.linalg.norm(lags_sg, axis=1)
        return DataEvent(lags_sg, values, cond_mask, lag_norms)

    def test_len_is_lag_count(self):
        de = self._make(k=3)
        self.assertEqual(len(de), 3)

    def test_lags_ti_defaults_none(self):
        de = self._make()
        self.assertIsNone(de.lags_ti)

    def test_with_lags_ti_sets_field_and_keeps_sg(self):
        de = self._make(k=3)
        ti = de.lags_sg * 2
        de2 = de.with_lags_ti(ti)
        np.testing.assert_array_equal(de2.lags_ti, ti)
        np.testing.assert_array_equal(de2.lags_sg, de.lags_sg)
        self.assertIsNone(de.lags_ti)  # original unchanged

    def test_truncate_slices_all_arrays(self):
        de = self._make(k=3).with_lags_ti(np.arange(6.0).reshape(3, 2))
        t = de.truncate(2)
        self.assertEqual(len(t), 2)
        np.testing.assert_array_equal(t.lags_sg, de.lags_sg[:2])
        np.testing.assert_array_equal(t.values, de.values[:2])
        np.testing.assert_array_equal(t.cond_mask, de.cond_mask[:2])
        np.testing.assert_array_equal(t.lag_norms, de.lag_norms[:2])
        np.testing.assert_array_equal(t.lags_ti, de.lags_ti[:2])

    def test_truncate_does_not_mutate_lags_sg_via_lags_ti(self):
        # Aliasing invariant: truncating must not let lags_ti writes hit lags_sg.
        de = self._make(k=3)
        de = de.with_lags_ti(
            de.lags_sg.copy()
        )  # caller must copy for independence
        t = de.truncate(2)
        t.lags_ti[:] = -999.0  # mutate the truncated TI-frame copy
        # original SG lags must be untouched
        self.assertFalse(np.any(de.lags_sg == -999.0))

    def test_truncate_preserves_none_lags_ti(self):
        de = self._make(k=3)
        self.assertIsNone(de.truncate(2).lags_ti)


if __name__ == "__main__":
    unittest.main()
