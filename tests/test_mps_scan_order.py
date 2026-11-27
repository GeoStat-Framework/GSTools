"""Training-image scan order, reproducibility, and thread invariance."""

import numpy as np
import pytest

import gstools as gs


def _ti(shape=(40, 40), seed=0):
    rng = np.random.default_rng(seed)
    return gs.mps.TrainingImage(rng.integers(0, 3, shape), n_neighbors=12)


def _ds(scan_path):
    return gs.mps.DirectSampling(
        gs.mps.MPSModel(_ti(), scan_fraction=0.3, scan_path=scan_path)
    )


# --- validation ---
def test_scan_path_rejects_invalid():
    with pytest.raises(ValueError, match="scan_path"):
        gs.mps.MPSModel(_ti(), scan_path="spiral")


def test_scan_path_default_is_sequential():
    assert gs.mps.MPSModel(_ti()).scan_path == "sequential"


# --- random scan runs and is valid ---
def test_random_scan_runs():
    field = _ds("random")([np.arange(12, dtype=float)] * 2, seed=3)
    assert field.shape == (12, 12)
    assert np.all(np.isin(field, [0.0, 1.0, 2.0]))


# --- random scan is deterministic for a fixed seed ---
def test_random_scan_is_reproducible():
    pos = [np.arange(12, dtype=float)] * 2
    f1 = _ds("random")(pos, seed=7)
    f2 = _ds("random")(pos, seed=7)
    np.testing.assert_array_equal(f1, f2)


# --- random scan differs from sequential (same seed) ---
def test_random_scan_differs_from_sequential():
    pos = [np.arange(12, dtype=float)] * 2
    f_seq = _ds("sequential")(pos, seed=1)
    f_rand = _ds("random")(pos, seed=1)
    assert not np.array_equal(f_seq, f_rand)


# --- random scan: serial == parallel (determinism across thread counts) ---
def test_random_scan_serial_parallel_identical_above_cutoff():
    # > 512 path nodes to actually exercise the multi-threaded DAG schedule.
    pos = [np.arange(24, dtype=float)] * 2  # 576 nodes
    ds1 = gs.mps.DirectSampling(
        gs.mps.MPSModel(_ti(), scan_fraction=0.3, scan_path="random")
    )
    f1 = ds1(pos, seed=5, num_threads=1)
    ds4 = gs.mps.DirectSampling(
        gs.mps.MPSModel(_ti(), scan_fraction=0.3, scan_path="random")
    )
    f4 = ds4(pos, seed=5, num_threads=4)
    np.testing.assert_array_equal(f1, f4)


# --- sequential scan also stays serial/parallel identical (guard) ---
def test_sequential_scan_serial_parallel_identical_above_cutoff():
    pos = [np.arange(24, dtype=float)] * 2
    ds1 = gs.mps.DirectSampling(gs.mps.MPSModel(_ti(), scan_fraction=0.3))
    ds4 = gs.mps.DirectSampling(gs.mps.MPSModel(_ti(), scan_fraction=0.3))
    np.testing.assert_array_equal(
        ds1(pos, seed=5, num_threads=1), ds4(pos, seed=5, num_threads=4)
    )
