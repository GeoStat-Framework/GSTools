"""Tests for gstools.mps.neighbors."""

import unittest
import warnings

import numpy as np

import gstools as gs
from gstools.mps.model import MPSModel
from gstools.mps.neighbors import (
    _lag_transform_matrix,
    _precompute_offsets,
    _reduce_to_fit,
    _transform_lags,
    _window_bounds,
)


class TestPrecomputeOffsets(unittest.TestCase):
    """Shape and limits of the precomputed neighbour offsets."""

    def test_offsets_shape(self):
        off = _precompute_offsets((5, 5))
        # shape: (N, 2) for 2-D, no zero row
        self.assertEqual(off.ndim, 2)
        self.assertEqual(off.shape[1], 2)
        self.assertFalse(np.any(np.all(off == 0, axis=1)))
        # sorted by Euclidean norm
        norms = np.linalg.norm(off, axis=1)
        self.assertTrue(np.all(norms[:-1] <= norms[1:]))

    def test_offsets_1d(self):
        off = _precompute_offsets((10,))
        self.assertEqual(off.shape[1], 1)
        self.assertFalse(np.any(off == 0))

    def test_offsets_max_offset(self):
        off = _precompute_offsets((5, 5), max_offset=1)
        self.assertLessEqual(np.abs(off).max(), 1)
        # 2-D, max_offset=1: 3^2 - 1 = 8 neighbours
        self.assertEqual(off.shape, (8, 2))


class TestTransformLagsCollapse(unittest.TestCase):
    def test_collapsed_lags_warn(self):
        # Strong anisotropy flattens the y-axis: SG lags (0,1) and (0,2) both
        # round to TI lag (0,0). The dropped neighbour must be warned about.
        M = np.array([[1.0, 0.0], [0.0, 0.1]])
        lags = np.array([[0.0, 1.0], [0.0, 2.0]])
        de = np.array([5.0, 9.0])
        with self.assertWarns(RuntimeWarning):
            lags_ti, de_out = _transform_lags(lags, M, de)
        self.assertEqual(len(lags_ti), 1)
        self.assertEqual(de_out.tolist(), [5.0])  # first occurrence kept

    def test_no_collapse_no_warning(self):
        # Identity transform keeps both lags distinct → no warning.
        M = np.eye(2)
        lags = np.array([[0.0, 1.0], [0.0, 2.0]])
        de = np.array([5.0, 9.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            lags_ti, de_out = _transform_lags(lags, M, de)
        self.assertEqual(len(lags_ti), 2)


class TestReduceToFit(unittest.TestCase):
    """Global reduce-to-fit: drop the most-outside node by rank (Mariethoz et al. 2010)."""

    def test_drops_oob_regardless_of_rank(self):
        # The near (low-rank) lag maps out of a 4x4 TI; the far (high-rank) lag
        # is feasible. para [43] keeps the feasible one and drops the OOB one —
        # the opposite of partial-mode furthest-first truncation.
        lags_ti = np.array([[0.0, 10.0], [1.0, 0.0]])
        de = np.array([5.0, 9.0])
        out_lags, out_de = _reduce_to_fit(lags_ti, (4, 4), de)
        np.testing.assert_array_equal(out_lags, [[1.0, 0.0]])
        np.testing.assert_array_equal(
            out_de, [9.0]
        )  # parallel array sliced too

    def test_both_extremes_overspan_reduced(self):
        # Two lags each fit alone but jointly over-span a 4-wide TI (ti-1==3):
        # extent 3-(-3)=6 > 3. One outermost node is dropped until it fits.
        lags_ti = np.array([[0.0, 3.0], [0.0, -3.0]])
        (out_lags,) = _reduce_to_fit(lags_ti, (4, 4))
        self.assertEqual(len(out_lags), 1)

    def test_collocated_never_dropped_and_inbounds_kept(self):
        # h=0 and an in-bounds lag survive; the over-long lag is removed.
        lags_ti = np.array([[0.0, 0.0], [3.0, 0.0], [9.0, 0.0]])
        (out_lags,) = _reduce_to_fit(lags_ti, (4, 4))  # ti-1 == 3
        self.assertIn([0.0, 0.0], out_lags.tolist())  # collocated kept
        self.assertNotIn([9.0, 0.0], out_lags.tolist())  # most-outside dropped

    def test_already_fits_is_noop(self):
        lags_ti = np.array([[1.0, 0.0], [0.0, -1.0]])
        (out_lags,) = _reduce_to_fit(lags_ti, (20, 20))
        np.testing.assert_array_equal(out_lags, lags_ti)

    def test_empty(self):
        (out,) = _reduce_to_fit(np.empty((0, 2)), (4, 4))
        self.assertEqual(out.size, 0)

    def test_strong_anis_retains_pattern_neighbour(self):
        # End to end: strong anisotropy on a small TI must not collapse the
        # window to a random draw; with para [43] the feasible neighbour is
        # retained, so output stays valid and finite (no crash, no NaN).
        rng = np.random.default_rng(0)
        ti = gs.mps.TrainingImage(rng.integers(0, 2, (6, 6)))
        ds = gs.mps.DirectSampling(
            MPSModel(ti, scan_fraction=1.0, scale=[1.0, 0.1])
        )
        field = ds([np.arange(8, dtype=float)] * 2, seed=0)
        self.assertTrue(np.all(np.isfinite(field)))
        self.assertTrue(set(np.unique(field)).issubset({0.0, 1.0}))


class TestLagTransformMatrix(unittest.TestCase):
    """Canonical-map transform semantics."""

    def test_scale_half_doubles_ti_lags(self):
        # scale=0.5 -> structures half the TI size -> TI-frame lags
        # are 2x the SG lags (lag_ti = (1/s) * R^-1 * lag_sg).
        scale_map = np.broadcast_to(np.array([0.5, 0.5]), (4, 4, 2))
        M = _lag_transform_matrix(2, None, scale_map, np.array([0, 0]))
        lags = np.array([[3.0, 0.0], [0.0, 5.0]])
        lags_ti = np.rint(lags @ M.T)
        np.testing.assert_array_equal(lags_ti, [[6.0, 0.0], [0.0, 10.0]])

    def test_uniform_dilation_isotropic(self):
        # Uniform dilation was inexpressible in the axis-0-pinned anis form.
        scale_map = np.broadcast_to(np.array([2.0, 2.0]), (3, 3, 2))
        M = _lag_transform_matrix(2, None, scale_map, np.array([1, 1]))
        np.testing.assert_allclose(M, 0.5 * np.eye(2))

    def test_rotation_only_matches_derotate(self):
        from gstools.tools.geometric import matrix_derotate

        rot_map = np.full((4, 4, 1), np.pi / 6)
        M = _lag_transform_matrix(2, rot_map, None, np.array([2, 3]))
        np.testing.assert_array_equal(M, matrix_derotate(2, [np.pi / 6]))

    def test_per_node_map_indexed_at_node(self):
        rot_map = np.zeros((4, 4, 1))
        rot_map[3, 1, 0] = np.pi / 2
        M_id = _lag_transform_matrix(2, rot_map, None, np.array([0, 0]))
        M_rot = _lag_transform_matrix(2, rot_map, None, np.array([3, 1]))
        np.testing.assert_allclose(M_id, np.eye(2), atol=1e-15)
        self.assertFalse(np.allclose(M_rot, np.eye(2)))

    def test_geometry_drop_ignores_sg_rank(self):
        # The NEAREST lag (SG rank 0) lands outside the TI -> it is dropped
        # (geometry rule), not the furthest (rank rule).
        lags_ti = np.array([[9.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        values = np.array([10.0, 20.0, 30.0])
        out_lags, out_vals = _reduce_to_fit(lags_ti, (5, 5), values)
        np.testing.assert_array_equal(out_lags, [[1.0, 1.0], [2.0, 0.0]])
        np.testing.assert_array_equal(out_vals, [20.0, 30.0])


class TestWindowFitWarning(unittest.TestCase):
    def test_infeasible_warns_and_drops_neighbours(self):
        # A lag larger than the TI cannot fit any anchor, so the function must
        # warn before dropping the over-long lag.
        ti_shape = np.array([5])
        lags_ti = np.array([[10.0]])  # |lag| 10 > TI size 5 → infeasible
        with self.assertWarns(RuntimeWarning):
            lo, hi, keep = _window_bounds(lags_ti, ti_shape)
        # the loop drops the over-long lag → keep < len(lags_ti)
        self.assertEqual(keep, -1)

    def test_feasible_no_warning(self):
        # A lag that fits → no warning.
        ti_shape = np.array([20])
        lags_ti = np.array([[1.0]])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            lo, hi, keep = _window_bounds(lags_ti, ti_shape)
        self.assertEqual(keep, 1)


class TestOffsetGridSizeLimit(unittest.TestCase):
    def test_large_3d_grid_raises_valueerror(self):
        # 200^3 with default max_offset=200 → (401)^3 > 5M → guard fires
        with self.assertRaises(ValueError) as ctx:
            _precompute_offsets((200, 200, 200))
        msg = str(ctx.exception)
        self.assertIn("max_offset", msg)
        self.assertIn("max_radius", msg)

    def test_large_3d_grid_with_max_radius_ok(self):
        # explicit small max_offset bypasses the guard
        off = _precompute_offsets((200, 200, 200), max_offset=10)
        self.assertEqual(off.shape[1], 3)

    def test_small_grid_ok(self):
        # small 2-D grid stays under threshold
        off = _precompute_offsets((20, 20))
        self.assertFalse(np.any(np.all(off == 0, axis=1)))


if __name__ == "__main__":
    unittest.main()
