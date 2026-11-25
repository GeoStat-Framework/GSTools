"""Tests for gstools.mps.scan."""

import unittest

import numpy as np

from gstools.mps.direct_sampling import DirectSampling
from gstools.mps.model import MPSModel
from gstools.mps.scan import _scan_window
from gstools.mps.training_image import TrainingImage


class TestScanWindowThreshold(unittest.TestCase):
    def test_ds_mode_strict_threshold(self):
        # Mariethoz2010 ¶23: DS acceptance is strict d < t. A candidate at
        # exactly the threshold (0.1) must be rejected; the 0.05 candidate is
        # accepted instead. With an inclusive d <= t the at-threshold
        # candidate would win first (scan order).
        win_shape = (2,)
        lo = np.array([0])
        dists = np.array([0.1, 0.05])

        def dist_fn(y_blk):
            return dists[: len(y_blk)]

        y = _scan_window(lo, win_shape, 0, 2, 0.1, dist_fn)
        self.assertEqual(int(y[0]), 1)  # the 0.05 candidate, not the 0.1 one

    def test_scan_window_greedy_first_under_threshold(self):
        # DS mode (threshold > 0) must return the FIRST candidate in scan order
        # with d < threshold — NOT the global-argmin candidate.
        # Design: candidate 0 has d=0.15 (under threshold 0.2, not the argmin);
        # candidate 1 has d=0.05 (under threshold 0.2, IS the argmin).
        # np.argmax(under) returns the first True in [True, True] == index 0,
        # confirming greedy (first-under-threshold) semantics, not argmin semantics.
        win_shape = (3,)
        lo = np.array([0])
        # distances for positions 0, 1, 2 in scan order
        dists = np.array([0.15, 0.05, 0.30])

        def dist_fn(y_blk):
            idxs = (y_blk[:, 0] - lo[0]).astype(int)
            return dists[idxs]

        # Argmin candidate is position 1 (d=0.05); first-under-threshold is
        # position 0 (d=0.15 < 0.2). The function must return position 0.
        y = _scan_window(lo, win_shape, 0, 3, 0.2, dist_fn)
        self.assertEqual(
            int(y[0]), 0
        )  # first-under-threshold, not argmin (pos 1)

    def test_dsbc_accepts_exact_match(self):
        # DSBC (t=0): exact match d == 0 is still accepted (d <= 0).
        win_shape = (2,)
        lo = np.array([0])
        dists = np.array([0.0, 0.5])

        def dist_fn(y_blk):
            return dists[: len(y_blk)]

        y = _scan_window(lo, win_shape, 0, 2, 0.0, dist_fn)
        self.assertEqual(int(y[0]), 0)  # the exact-match candidate


class TestRandomScanPath(unittest.TestCase):
    """MPSModel(scan_path='random'): GSTools extension (M10 para [19] is sequential)."""

    def test_random_order_samples_the_whole_window_without_repeats(self):
        # Contract: same number of DISTINCT candidates as sequential mode,
        # spread over the window instead of one contiguous run, and full
        # coverage when max_scan == win_size. Grid the 40x40 window into 16
        # 10x10 blocks: 160 uniform draws leave a block empty with negligible
        # probability, while 160 contiguous cells span 4 rows -> <= 8 blocks.
        def visited(u_start_i, max_scan, scan_path, win_shape=(40, 40)):
            seen = []

            def dist_fn(y_blk):  # never accepted (DSBC needs d <= 0)
                seen.append(y_blk.copy())
                return np.ones(len(y_blk))

            _scan_window(
                np.zeros(len(win_shape), dtype=int),
                win_shape,
                u_start_i,
                max_scan,
                0.0,
                dist_fn,
                scan_path,
            )
            return np.concatenate(seen, axis=0)

        rnd = visited(0.42, 160, "random")
        self.assertEqual(len(np.unique(rnd, axis=0)), 160)
        blocks = lambda v: {(int(r) // 10, int(c) // 10) for r, c in v}
        self.assertEqual(len(blocks(rnd)), 16)
        self.assertLessEqual(len(blocks(visited(0.42, 160, "sequential"))), 8)
        full = visited(0.81, 64, "random", win_shape=(8, 8))
        np.testing.assert_array_equal(
            np.sort(np.ravel_multi_index(tuple(full.T), (8, 8))), np.arange(64)
        )

    def test_random_order_is_pinned(self):
        # The order comes from numpy's Generator.choice,
        # whose stream numpy does not freeze across versions (NEP 19).
        seen = []

        def dist_fn(y_blk):
            seen.append(y_blk.copy())
            return np.ones(len(y_blk))

        _scan_window(
            np.zeros(2, dtype=int), (40, 40), 0.42, 160, 0.0, dist_fn, "random"
        )
        flat = np.ravel_multi_index(tuple(np.concatenate(seen).T), (40, 40))
        self.assertEqual(
            flat[:8].tolist(), [44, 962, 102, 856, 1271, 923, 1327, 253]
        )
        self.assertEqual(int(flat.sum()), 128841)

    def test_option_reaches_the_engine_and_keeps_output_valid(self):
        rng = np.random.default_rng(0)
        data = rng.integers(0, 3, (20, 20)).astype(float)
        ti = TrainingImage(data, categorical=True, n_neighbors=4)
        pos = [np.arange(8, dtype=float)] * 2
        seeds = dict(path_seed=1, node_seed=10)

        self.assertEqual(MPSModel(ti).scan_path, "sequential")
        with self.assertRaises(ValueError):
            MPSModel(ti, scan_path="spiral")

        ds_seq = DirectSampling(MPSModel(ti, scan_fraction=0.3))
        ds_rnd = DirectSampling(
            MPSModel(ti, scan_fraction=0.3, scan_path="random")
        )
        f_rnd = ds_rnd(pos, num_threads=1, **seeds)
        # Plumbed through (otherwise identical), reproducible, a subset of the
        # TI values, and thread-count independent: the scan order is keyed on
        # the pre-drawn per-node u_start, never on execution order.
        self.assertFalse(np.array_equal(ds_seq(pos, **seeds), f_rnd))
        np.testing.assert_array_equal(
            f_rnd, ds_rnd(pos, num_threads=1, **seeds)
        )
        np.testing.assert_array_equal(
            f_rnd, ds_rnd(pos, num_threads=4, **seeds)
        )
        self.assertTrue(np.isin(f_rnd, np.unique(data)).all())


if __name__ == "__main__":
    unittest.main()
