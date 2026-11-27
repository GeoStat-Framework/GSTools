"""Cleanup configuration, numerical behavior, and parallel execution."""

import numpy as np
import pytest

import gstools as gs
from gstools.mps import Variable
from gstools.mps import simulate as _sim
from gstools.mps.simulate import ds_simulate
from gstools.mps.training_image import _post_pass_n_neighbors, _resolve_post_ti
from gstools.random.rng import RNG


def _ti(**kw):
    return gs.mps.TrainingImage(np.zeros((20, 20)), n_neighbors=16, **kw)


# --- Explicitly configured variable settings ---
def test_variable_records_explicit_fields():
    v = Variable("c", np.zeros((4, 4)), distance_power=0.5)
    assert v._explicit == frozenset({"distance_power"})
    plain = Variable("c", np.zeros((4, 4)))
    assert plain._explicit == frozenset()
    # replace() carries the full search config as explicit (it passes them all)
    repl = plain.replace(n_neighbors=5)
    assert "n_neighbors" in repl._explicit


# --- validators ---
@pytest.mark.parametrize("bad", [-1, -5])
def test_post_processing_rejects_negative(bad):
    with pytest.raises(ValueError, match="post_processing must be >= 0"):
        gs.mps.MPSModel(_ti(), post_processing=bad)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.inf, np.nan])
def test_post_processing_factor_rejects_nonfinite_or_nonpositive(bad):
    with pytest.raises(ValueError, match="post_processing_factor"):
        gs.mps.MPSModel(_ti(), post_processing_factor=bad)


def test_post_processing_path_validation():
    # valid string modes + None + explicit array accepted
    for p in (None, "same", "random", "sequential"):
        gs.mps.MPSModel(_ti(), post_processing_path=p)
    gs.mps.MPSModel(_ti(), post_processing_path=np.zeros((5, 2), dtype=int))
    with pytest.raises(ValueError, match="post_processing_path"):
        gs.mps.MPSModel(_ti(), post_processing_path="spiral")
    with pytest.raises(ValueError, match="2-D"):
        gs.mps.MPSModel(_ti(), post_processing_path=np.zeros(5, dtype=int))


def test_post_processing_mask_rejects_non_boolean():
    with pytest.raises(TypeError, match="post_processing_mask"):
        gs.mps.MPSModel(_ti(), post_processing_mask=np.zeros((20, 20), int))


# --- override validation (must keep primary data + kind) ---
def test_post_processing_variables_must_match_primary_data_and_kind():
    ti = _ti()
    bad = Variable(None, np.ones((20, 20)))  # different data
    with pytest.raises(ValueError, match="keep the primary"):
        gs.mps.MPSModel(ti, post_processing=1, post_processing_variables=[bad])


def test_post_processing_variables_override_via_replace_is_accepted():
    ti = _ti()
    ov = ti.variable().replace(n_neighbors=8)
    model = gs.mps.MPSModel(
        ti, post_processing=1, post_processing_variables=[ov]
    )
    assert len(model.post_processing_variables) == 1


# --- effort check: variation cleanup must keep >= 2 neighbours ---
def test_post_pass_effort_rejects_variation_below_two_neighbours():
    ti = gs.mps.TrainingImage(
        np.zeros((20, 20)),
        categorical=False,
        distance="variation",
        n_neighbors=2,
    )
    # factor 2 -> round(2/2) = 1 post-pass neighbour -> invalid for variation
    with pytest.raises(ValueError, match="variation"):
        gs.mps.MPSModel(ti, post_processing=1, post_processing_factor=2.0)


# --- helpers ---
def test_resolve_post_ti_applies_only_explicit_overrides():
    ti = _ti()  # n_neighbors=16, l1 by default
    ov = ti.variable().replace(n_neighbors=4)
    post = _resolve_post_ti(ti, [ov])
    assert post.variable().n_neighbors == 4


def test_post_pass_n_neighbors_rounds_and_floors():
    ti = _ti()  # n_neighbors=16
    nk = _post_pass_n_neighbors(ti, 4.0)  # 16/4 = 4
    assert nk[None] == 4
    nk1 = _post_pass_n_neighbors(ti, 100.0)  # floored at 1
    assert nk1[None] == 1


# --- defaults ---
def test_post_processing_defaults():
    model = gs.mps.MPSModel(_ti())
    assert model.post_processing == 0
    assert model.post_processing_factor == 1.0
    assert model.post_processing_variables == []


# --- Independent cleanup oracle and conditioning invariants ---------------
#
# The oracle below composes single-node resimulations through the *same* Rust
# scan engine: with every other cell conditioned, a one-node main pass gathers
# exactly the cleanup pass's fully-informed neighbourhood and scans it
# identically. In DSBC mode (threshold=0) with a full window scan the anchor is
# the global-distance minimum, so the per-node scan-start randomness drops out
# and the result is deterministic. Running those single-node draws serially in
# raster order, writing each back before the next (Gauss-Seidel), must
# reproduce the Rust cleanup pass. This checks cleanup ordering and write-back;
# the scan kernel has separate numerical reference tests.


_ORACLE_KW = dict(threshold=0.0, scan_fraction=1.0, boundary="partial")


def _cont_ti():
    rng = np.random.default_rng(12345)
    return gs.mps.TrainingImage(
        rng.random((12, 12)), categorical=False, distance="l2", n_neighbors=8
    )


def _resim_one_node(ti, sim_shape, field, node):
    """Resimulate a single node with every other cell conditioned (DSBC)."""
    conditions = {
        idx: {None: float(field[idx])}
        for idx in np.ndindex(*sim_shape)
        if idx != node
    }
    rng = RNG(0).random
    result = ds_simulate(
        ti,
        sim_shape,
        rng_path=rng,
        rng_nodes=rng,
        conditions=conditions,
        path=np.array([list(node)], dtype=int),
        scan_path="sequential",
        **_ORACLE_KW,
    )
    return result[None][node]


def _oracle_one_pass(ti, sim_shape, field0):
    field = np.array(field0, dtype=float)
    for node in np.ndindex(*sim_shape):  # raster order = sequential path
        field[node] = _resim_one_node(ti, sim_shape, field, node)
    return field


def test_post_processing_serial_matches_oracle():
    # TI smaller than the sim grid: no exact TI-sized crop exists, so the
    # full-scan DSBC field is not a trivial fixed point and the cleanup moves
    # it. Random main path (past-only neighbours) vs the fully-informed cleanup
    # guarantees a difference; the cleanup is sequential + full-scan DSBC, hence
    # start-independent and reproducible by the raster-order oracle.
    rng = np.random.default_rng(999)
    ti = gs.mps.TrainingImage(
        rng.random((7, 7)), categorical=False, distance="l2", n_neighbors=6
    )
    pos = (np.arange(10.0), np.arange(10.0))
    f0 = gs.mps.DirectSampling(gs.mps.MPSModel(ti, **_ORACLE_KW))(
        pos, seed=4, path="random"
    )
    f1 = gs.mps.DirectSampling(
        gs.mps.MPSModel(
            ti,
            post_processing=1,
            post_processing_path="sequential",
            **_ORACLE_KW,
        )
    )(pos, seed=4, path="random")
    oracle = _oracle_one_pass(ti, (10, 10), f0)
    np.testing.assert_allclose(f1, oracle)
    # The pass actually moved the field (not a vacuous fixed-point match).
    assert not np.array_equal(f0, f1)


def test_post_processing_zero_is_bit_identical():
    ti = _cont_ti()
    pos = (np.arange(8.0), np.arange(8.0))
    base = gs.mps.DirectSampling(gs.mps.MPSModel(ti, scan_fraction=0.4))(
        pos, seed=11
    )
    pp0 = gs.mps.DirectSampling(
        gs.mps.MPSModel(ti, scan_fraction=0.4, post_processing=0)
    )(pos, seed=11)
    np.testing.assert_array_equal(base, pp0)


def test_post_processing_is_reproducible():
    ti = _cont_ti()
    pos = (np.arange(8.0), np.arange(8.0))

    def run():
        return gs.mps.DirectSampling(
            gs.mps.MPSModel(ti, scan_fraction=0.4, post_processing=3)
        )(pos, seed=9)

    np.testing.assert_array_equal(run(), run())


def test_post_processing_preserves_hard_data():
    ti = gs.mps.TrainingImage(np.tile([0.0, 1.0], (24, 12)), n_neighbors=12)
    pos = (np.arange(10.0), np.arange(10.0))
    ds = gs.mps.DirectSampling(
        gs.mps.MPSModel(
            ti, scan_fraction=0.5, threshold=0.0, post_processing=3
        )
    )
    # Condition three grid nodes to fixed categories.
    cond_pos = [[0.0, 3.0, 9.0], [0.0, 7.0, 2.0]]
    cond_val = np.array([1.0, 0.0, 1.0])
    ds.set_condition(cond_pos, cond_val)
    field = ds(pos, seed=5)
    assert field[0, 0] == 1.0
    assert field[3, 7] == 0.0
    assert field[9, 2] == 1.0


def test_post_processing_mask_freezes_non_selected_cells():
    ti = _cont_ti()
    pos = (np.arange(8.0), np.arange(8.0))
    f0 = gs.mps.DirectSampling(
        gs.mps.MPSModel(ti, scan_fraction=0.4, post_processing=0)
    )(pos, seed=2)
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True  # only the interior block may be cleaned
    fm = gs.mps.DirectSampling(
        gs.mps.MPSModel(
            ti,
            scan_fraction=0.4,
            post_processing=2,
            post_processing_mask=mask,
        )
    )(pos, seed=2)
    # Cells outside the mask are never re-simulated -> identical to no cleanup.
    np.testing.assert_array_equal(fm[~mask], f0[~mask])
    # Cells inside the mask were refined (the pass did something).
    assert not np.array_equal(fm[mask], f0[mask])


@pytest.mark.parametrize("direction", ["l1_to_variation", "variation_to_l1"])
def test_post_processing_metric_override_both_directions(direction):
    rng = np.random.default_rng(7)
    base_distance = "l1" if direction == "l1_to_variation" else "variation"
    over_distance = "variation" if direction == "l1_to_variation" else "l1"
    ti = gs.mps.TrainingImage(
        rng.random((20, 20)),
        categorical=False,
        distance=base_distance,
        n_neighbors=12,
    )
    pos = (np.arange(10.0), np.arange(10.0))
    kw = dict(scan_fraction=0.4, threshold=0.1, post_processing=1)
    base = gs.mps.DirectSampling(gs.mps.MPSModel(ti, **kw))(pos, seed=3)
    override = ti.variable().replace(distance=over_distance)
    changed = gs.mps.DirectSampling(
        gs.mps.MPSModel(ti, post_processing_variables=[override], **kw)
    )(pos, seed=3)
    # The override flips the cleanup distance (and its paste rule: variation's
    # mean-shift on/off), so the cleaned field must differ.
    assert not np.allclose(base, changed)


@pytest.fixture
def capture_cleanup_threads():
    """Record the effective thread count of each cleanup pass."""
    seen = []
    _sim._MPS_CLEANUP_STATS_HOOK = lambda stats: seen.append(
        stats["used_threads"]
    )
    try:
        yield seen
    finally:
        _sim._MPS_CLEANUP_STATS_HOOK = None


def _big_ti():
    rng = np.random.default_rng(2024)
    # Categorical TI; sim grid below is 24x24 = 576 nodes (> the 512 cutoff).
    return gs.mps.TrainingImage(
        (rng.random((30, 30)) < 0.45).astype(float), n_neighbors=16
    )


_BIG_POS = (np.arange(24.0), np.arange(24.0))


def test_cleanup_serial_equals_parallel(capture_cleanup_threads):
    ti = _big_ti()
    kw = dict(scan_fraction=0.4, threshold=0.05, post_processing=2)
    f1 = gs.mps.DirectSampling(gs.mps.MPSModel(ti, **kw))(
        _BIG_POS, seed=7, num_threads=1
    )
    f4 = gs.mps.DirectSampling(gs.mps.MPSModel(ti, **kw))(
        _BIG_POS, seed=7, num_threads=4
    )
    np.testing.assert_array_equal(f1, f4)
    # The 576-node run with num_threads=4 must have actually gone parallel
    # (otherwise serial==parallel would be a vacuous assertion); the
    # num_threads=1 passes stay at 1.
    assert capture_cleanup_threads == [1, 1, 4, 4]


def test_cleanup_parallel_is_reproducible():
    ti = _big_ti()
    kw = dict(scan_fraction=0.4, threshold=0.05, post_processing=2)

    def run():
        return gs.mps.DirectSampling(gs.mps.MPSModel(ti, **kw))(
            _BIG_POS, seed=13, num_threads=4
        )

    np.testing.assert_array_equal(run(), run())


def test_cleanup_serial_equals_parallel_random_post_path():
    # A random post path still orders the DAG by (per-pass) visit position, so
    # thread-count invariance must hold there too.
    ti = _big_ti()
    kw = dict(
        scan_fraction=0.4,
        threshold=0.05,
        post_processing=2,
        post_processing_path="random",
    )
    f1 = gs.mps.DirectSampling(gs.mps.MPSModel(ti, **kw))(
        _BIG_POS, seed=21, num_threads=1
    )
    f8 = gs.mps.DirectSampling(gs.mps.MPSModel(ti, **kw))(
        _BIG_POS, seed=21, num_threads=8
    )
    np.testing.assert_array_equal(f1, f8)


def test_cleanup_serial_equals_parallel_with_zones(capture_cleanup_threads):
    primary = _big_ti()
    rng = np.random.default_rng(99)
    zone_ti = gs.mps.TrainingImage(
        (rng.random((30, 30)) < 0.7).astype(float), n_neighbors=16
    )
    region = np.zeros((24, 24), dtype=bool)
    region[:12] = True  # top half scans the zone TI
    zone = gs.mps.Zone(zone_ti, region)
    kw = dict(
        scan_fraction=0.4, threshold=0.05, post_processing=1, zones=[zone]
    )
    f1 = gs.mps.DirectSampling(gs.mps.MPSModel(primary, **kw))(
        _BIG_POS, seed=5, num_threads=1
    )
    f4 = gs.mps.DirectSampling(gs.mps.MPSModel(primary, **kw))(
        _BIG_POS, seed=5, num_threads=4
    )
    np.testing.assert_array_equal(f1, f4)
    assert capture_cleanup_threads == [1, 4]


def test_cleanup_configuration_keeps_sparse_search_overrides():
    image = gs.mps.TrainingImage(
        Variable("a", [0, 1], n_neighbors=8, max_radius=2)
    )
    overrides = {"a": {"n_neighbors": 4}}
    post = gs.mps.PostProcessing(1, factor=2, path="same", overrides=overrides)
    overrides["a"]["n_neighbors"] = 2
    model = gs.MPSModel(image, post_processing=post)
    image.variable().n_neighbors = 7
    resolved = _resolve_post_ti(
        image, model.post_processing_variables
    ).variable()
    assert model.post_processing == 1
    assert model.post_processing_factor == 2
    assert model.post_processing_path == "same"
    assert resolved.n_neighbors == 4
    assert resolved.max_radius == 2
    with pytest.raises(ValueError, match="unknown variables"):
        model.post_processing = gs.mps.PostProcessing(
            2, overrides={"missing": {}}
        )
    assert model.post_processing == 1
