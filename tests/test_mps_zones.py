"""Zone construction, validation, and zoned simulation behavior."""

import numpy as np
import pytest

import gstools as gs
from gstools.mps import Zone
from gstools.mps.nonstationary import build_zone_selector, resolve_where


def _primary():
    return gs.mps.TrainingImage(np.zeros((20, 20)), n_neighbors=8)


# --- Zone construction / validation ---
def test_zone_rejects_non_boolean_where_array():
    with pytest.raises(ValueError, match="boolean"):
        Zone(_primary(), where=np.zeros((20, 20), dtype=int))


def test_zone_accepts_bool_array_and_callable():
    z1 = Zone(_primary(), where=np.ones((20, 20), dtype=bool))
    z2 = Zone(_primary(), where=lambda x, y: x >= 10)
    assert z1.ti.shape == (20, 20)
    assert callable(z2.where)


# --- resolve_where ---
def test_resolve_where_callable_and_array():
    gx, gy = np.meshgrid(np.arange(3.0), np.arange(3.0), indexing="ij")
    pos = (gx.ravel(), gy.ravel())
    mask = resolve_where(lambda x, y: x >= 1, (3, 3), pos)
    assert mask.shape == (3, 3)
    assert mask[0].sum() == 0 and mask[1].all()
    with pytest.raises(ValueError, match="boolean"):
        resolve_where(lambda x, y: x, (3, 3), pos)  # float, not bool


# --- build_zone_selector: labels + overlap rejection ---
def test_build_zone_selector_labels_and_overlap():
    gx, gy = np.meshgrid(np.arange(3.0), np.arange(3.0), indexing="ij")
    pos = (gx.ravel(), gy.ravel())
    za = Zone(gs.mps.TrainingImage(np.ones((3, 3))), where=lambda x, y: x < 1)
    zb = Zone(
        gs.mps.TrainingImage(np.ones((3, 3)) * 2), where=lambda x, y: x >= 2
    )
    sel = build_zone_selector([za, zb], (3, 3), pos)
    np.testing.assert_array_equal(
        sel, np.array([[1, 1, 1], [0, 0, 0], [2, 2, 2]])
    )
    # overlapping regions must raise
    z_all = Zone(
        gs.mps.TrainingImage(np.ones((3, 3))), where=lambda x, y: x >= 0
    )
    with pytest.raises(ValueError, match="overlap"):
        build_zone_selector([za, z_all], (3, 3), pos)


# --- MPSModel._validate_zones ---
def test_mpsmodel_rejects_zone_kind_mismatch():
    primary = gs.mps.TrainingImage(np.zeros((20, 20)), categorical=True)
    bad = gs.mps.TrainingImage(np.zeros((20, 20)), categorical=False)
    with pytest.raises(ValueError, match="categorical/continuous"):
        gs.mps.MPSModel(primary, zones=[Zone(bad, where=lambda x, y: x < 5)])


def test_mpsmodel_rejects_zone_dim_mismatch():
    primary = gs.mps.TrainingImage(np.zeros((20, 20)))
    bad = gs.mps.TrainingImage(np.zeros((20, 20, 20)))
    with pytest.raises(ValueError, match="dim"):
        gs.mps.MPSModel(
            primary, zones=[Zone(bad, where=np.ones((20, 20), bool))]
        )


def test_mpsmodel_zones_default_empty_and_roundtrip():
    primary = _primary()
    assert gs.mps.MPSModel(primary).zones == []
    z = Zone(_primary(), where=lambda x, y: x < 5)
    model = gs.mps.MPSModel(primary, zones=[z])
    assert model.zones == [z]


# --- native zoned simulation ---
def test_zoned_simulation_selects_zone_ti_per_region():
    # Primary TI all 0, zone TI all 1 over x >= 5: nodes in the zone must paste
    # 1, nodes outside must stay 0 (proves the right TI is scanned per node).
    primary = gs.mps.TrainingImage(np.zeros((20, 20)), n_neighbors=8)
    zone_ti = gs.mps.TrainingImage(np.ones((20, 20)), n_neighbors=8)
    z = Zone(zone_ti, where=lambda x, y: x >= 5)
    ds = gs.mps.DirectSampling(gs.mps.MPSModel(primary, zones=[z]))
    field = ds([np.arange(8, dtype=float)] * 2, seed=1)
    assert np.all(field[:5] == 0.0)
    assert np.all(field[5:] == 1.0)


def test_zoned_simulation_differently_shaped_zone_ti_runs():
    # Zone TI need not match the primary TI's shape.
    primary = gs.mps.TrainingImage(np.zeros((20, 20)), n_neighbors=6)
    zone_ti = gs.mps.TrainingImage(np.ones((12, 9)), n_neighbors=6)
    z = Zone(zone_ti, where=lambda x, y: y >= 4)
    ds = gs.mps.DirectSampling(gs.mps.MPSModel(primary, zones=[z]))
    field = ds([np.arange(8, dtype=float)] * 2, seed=2)
    assert np.all(field[:, :4] == 0.0)
    assert np.all(field[:, 4:] == 1.0)


def test_no_zone_machinery_is_bit_identical_to_plain():
    # Passing an empty zone list must not change anything vs no zones at all.
    rng = np.random.default_rng(0)
    data = rng.integers(0, 3, (20, 20))
    pos = [np.arange(10, dtype=float)] * 2
    f_plain = gs.mps.DirectSampling(
        gs.mps.MPSModel(
            gs.mps.TrainingImage(data, n_neighbors=8), scan_fraction=0.3
        )
    )(pos, seed=4)
    f_empty = gs.mps.DirectSampling(
        gs.mps.MPSModel(
            gs.mps.TrainingImage(data, n_neighbors=8),
            scan_fraction=0.3,
            zones=[],
        )
    )(pos, seed=4)
    np.testing.assert_array_equal(f_plain, f_empty)


def test_zoned_simulation_serial_parallel_identical_above_cutoff():
    primary = gs.mps.TrainingImage(np.zeros((30, 30)), n_neighbors=8)
    zone_ti = gs.mps.TrainingImage(np.ones((30, 30)), n_neighbors=8)
    pos = [np.arange(24, dtype=float)] * 2  # 576 nodes > 512 cutoff
    f1 = gs.mps.DirectSampling(
        gs.mps.MPSModel(
            primary, zones=[Zone(zone_ti, where=lambda x, y: x >= 12)]
        )
    )(pos, seed=5, num_threads=1)
    f4 = gs.mps.DirectSampling(
        gs.mps.MPSModel(
            primary, zones=[Zone(zone_ti, where=lambda x, y: x >= 12)]
        )
    )(pos, seed=5, num_threads=4)
    np.testing.assert_array_equal(f1, f4)


def test_zoned_simulation_reports_overlap_at_call_time():
    primary = _primary()
    z1 = Zone(_primary(), where=lambda x, y: x >= 3)
    z2 = Zone(_primary(), where=lambda x, y: x >= 5)  # overlaps z1
    ds = gs.mps.DirectSampling(gs.mps.MPSModel(primary, zones=[z1, z2]))
    with pytest.raises(ValueError, match="overlap"):
        ds([np.arange(8, dtype=float)] * 2, seed=1)
