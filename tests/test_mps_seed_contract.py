"""DS SeedSequence lifecycle and mutable-conditioning regressions."""

import importlib

import numpy as np
import pytest

from gstools.mps import DirectSampling, MPSModel, TrainingImage, Variable, Zone
from gstools.random.rng import RNG
from gstools.random.tools import MasterRNG


def _model(**kwargs):
    ti = TrainingImage(np.indices((8, 8)).sum(axis=0) % 3, n_neighbors=4)
    return MPSModel(ti, **kwargs)


@pytest.fixture
def capture_controls(monkeypatch):
    """Observe actual acquired streams, without relying on field differences."""
    captured = []

    def simulate(model, sim_shape, rng_path, rng_nodes, **kwargs):
        captured.append(
            (
                rng_path.uniform(size=32),
                rng_nodes.uniform(size=32),
            )
        )
        return {var.name: np.zeros(sim_shape) for var in model.ti.variables}

    module = importlib.import_module("gstools.mps.direct_sampling")
    monkeypatch.setattr(module, "_simulate_model", simulate)
    return captured


def test_rng_legacy_default_unchanged():
    """Independent legacy construction, including reseeding and child order."""
    for seed in [0, 329, 505, 19031977]:
        legacy = np.random.RandomState(seed)
        rng = RNG(seed)
        explicit = RNG(seed, seed_sequence=False)
        for _ in range(4):
            child = legacy.randint(1, 2**16)
            expected = np.random.RandomState(child).normal(size=16)
            np.testing.assert_array_equal(rng.random.normal(size=16), expected)
            np.testing.assert_array_equal(
                explicit.random.normal(size=16), expected
            )
        rng.seed = seed
        np.testing.assert_array_equal(
            rng.random.uniform(size=16), RNG(seed).random.uniform(size=16)
        )


@pytest.mark.parametrize("seed", [0, 329, 505, 2**63 + 1, 2**128 + 7])
def test_rng_seed_sequence_matches_reference_construction(seed):
    master = np.random.SeedSequence(seed)
    rng = RNG(seed, seed_sequence=True)
    for _ in range(4):
        expected = np.random.RandomState(np.random.MT19937(master.spawn(1)[0]))
        np.testing.assert_array_equal(
            rng.random.uniform(size=32), expected.uniform(size=32)
        )
    rng.seed = seed
    np.testing.assert_array_equal(
        rng.random.uniform(size=32),
        RNG(seed, seed_sequence=True).random.uniform(size=32),
    )


def test_seed_diversity_regression_and_prefix_fingerprints():
    assert MasterRNG(329)() == MasterRNG(505)() == 7050
    seeds = list(range(256)) + [329, 505, 2**63 + 1, 2**128 + 7]
    fingerprints = [
        RNG(seed, seed_sequence=True).random.bytes(64) for seed in seeds
    ]
    # A bounded regression against the old mapping, not proof of independence.
    assert len(set(fingerprints)) == len(seeds)


@pytest.mark.parametrize("control", ["path_seed", "node_seed"])
def test_explicit_controls_independent_of_master(capture_controls, control):
    ds = DirectSampling(_model(), seed=1)
    for seed in [1, 200, 2**64 + 1]:
        ds([np.arange(4)] * 2, seed=seed, **{control: 329})
    fixed_role = 0 if control == "path_seed" else 1
    for controls in capture_controls[1:]:
        np.testing.assert_array_equal(
            controls[fixed_role], capture_controls[0][fixed_role]
        )
    assert not np.array_equal(
        capture_controls[0][1 - fixed_role],
        capture_controls[1][1 - fixed_role],
    )


def test_explicit_role_controls_and_equal_seed_contract(capture_controls):
    ds = DirectSampling(_model(), seed=1)
    for path_seed, node_seed in [(7, 9), (7, 11), (13, 11), (7, 7)]:
        ds([np.arange(4)] * 2, path_seed=path_seed, node_seed=node_seed)
    np.testing.assert_array_equal(
        capture_controls[0][0], capture_controls[1][0]
    )
    np.testing.assert_array_equal(
        capture_controls[1][1], capture_controls[2][1]
    )
    np.testing.assert_array_equal(
        capture_controls[3][0], capture_controls[3][1]
    )
    assert ds.rng._master_rng._ss.n_children_spawned == 0


def test_master_continuation_reseeding_and_none_reset(capture_controls):
    ds = DirectSampling(_model(), seed=329)
    ds([np.arange(4)] * 2)
    first_master = ds.rng._master_rng
    ds(seed=np.nan)
    assert ds.rng._master_rng is first_master
    expected = RNG(329, seed_sequence=True)
    for controls in capture_controls:
        for role in controls:
            np.testing.assert_array_equal(
                role, expected.random.uniform(size=32)
            )
    ds(seed=329)
    for actual, first in zip(capture_controls[-1], capture_controls[0]):
        np.testing.assert_array_equal(actual, first)
    previous = ds.rng._master_rng
    ds(seed=None)
    assert ds.rng._master_rng is not previous
    assert ds.rng.seed is None
    assert ds.rng._master_rng._ss.n_children_spawned == 2


@pytest.mark.parametrize("unset", [None, np.nan])
def test_unset_role_seeds_derive_two_master_children(capture_controls, unset):
    ds = DirectSampling(_model(), seed=1)
    ds([np.arange(4)] * 2, path_seed=unset, node_seed=unset)
    assert ds.rng._master_rng._ss.n_children_spawned == 2


@pytest.mark.parametrize("scan_path", ["random", "sequential"])
@pytest.mark.parametrize("cleanup", [0, 1])
@pytest.mark.parametrize("path", ["random", "sequential", "explicit"])
def test_seeded_raw_fields_reproduce_with_cleanup_and_threads(
    scan_path, cleanup, path
):
    model = _model(
        scan_path=scan_path,
        post_processing=cleanup,
        post_processing_path="random",
    )
    where = np.zeros((24, 24), dtype=bool)
    where[12:] = True
    model.zones = [Zone(model.ti, where=where)]
    ds = DirectSampling(model, seed=7)
    ds.set_condition([[0], [0]], [1])
    order = path
    if path == "explicit":
        order = np.indices((24, 24)).reshape(2, -1).T
    outputs = [
        ds(
            [np.arange(24)] * 2,
            seed=13,
            path_seed=29,
            node_seed=505,
            path=order,
            num_threads=threads,
            post_process=False,
            store=False,
        )
        for threads in [1, 4]
    ]
    np.testing.assert_array_equal(*outputs)
    assert outputs[0][0, 0] == 1


def test_omitted_cleanup_matches_explicit_zero_and_draw_state():
    one = DirectSampling(_model(), seed=7)
    two = DirectSampling(_model(post_processing=0), seed=7)
    np.testing.assert_array_equal(
        one([np.arange(8)] * 2, post_process=False),
        two([np.arange(8)] * 2, post_process=False),
    )
    np.testing.assert_array_equal(
        one.rng.random.uniform(size=32), two.rng.random.uniform(size=32)
    )


@pytest.mark.parametrize("code", [-1, 0.5, 2])
def test_mutated_cleanup_penalty_rechecked_before_rng(code):
    # Use binary data so a 2x2 matrix is valid for the TI, but not the hard code.
    ti = TrainingImage(np.indices((4, 4)).sum(axis=0) % 2, n_neighbors=2)
    model = MPSModel(ti)
    ds = DirectSampling(model, seed=7)
    ds.set_condition([[0], [0]], [code])
    model.post_processing_variables = [
        ti.variables[0].replace(penalty_matrix=[[0, 1], [1, 0]])
    ]
    model.post_processing = 1
    state = ds.rng._master_rng
    original = ds._cond_val.copy()
    with pytest.raises(ValueError, match="conditioning code"):
        ds([np.arange(4)] * 2)
    assert ds.rng._master_rng is state
    assert state._ss.n_children_spawned == 0
    np.testing.assert_array_equal(ds._cond_val, original)
    model.post_processing = 0
    field = ds(post_process=False)
    assert field[0, 0] == code


def test_mutated_cleanup_penalty_multivariate_missingness():
    data = np.indices((4, 4)).sum(axis=0) % 2
    ti = TrainingImage([Variable("a", data), Variable("b", data)])
    model = MPSModel(ti)
    ds = DirectSampling(model, seed=7)
    ds.set_condition([[0, 1], [0, 1]], {"a": [2, np.nan], "b": [np.nan, 1]})
    model.post_processing_variables = [
        ti.variable("a").replace(penalty_matrix=[[0, 1], [1, 0]])
    ]
    model.post_processing = 1
    with pytest.raises(ValueError, match="conditioning code"):
        ds([np.arange(4)] * 2)
    assert ds.rng._master_rng._ss.n_children_spawned == 0
    ds.set_condition([[0, 1], [0, 1]], {"a": [1, np.nan], "b": [np.nan, 1]})
    result = ds(post_process=False)
    assert result["a"][0, 0] == 1
    assert result["b"][1, 1] == 1


def test_parallel_seed_contract_zones_mask_partial_conditioning(monkeypatch):
    module = importlib.import_module("gstools.mps.simulate")
    main, cleanup = [], []
    monkeypatch.setattr(
        module,
        "_MPS_RUST_ENGINE_STATS_HOOK",
        lambda stats: main.append(stats["used_threads"]),
    )
    monkeypatch.setattr(
        module,
        "_MPS_CLEANUP_STATS_HOOK",
        lambda stats: cleanup.append(stats["used_threads"]),
    )
    data = np.indices((30, 30)).sum(axis=0) % 2
    ti = TrainingImage(
        [
            Variable("facies", data, n_neighbors=4),
            Variable("value", data * 2.0, categorical=False, n_neighbors=4),
        ]
    )
    mask = np.ones((26, 26), dtype=bool)
    mask[0] = False
    zone = np.indices((26, 26))[0] > 12
    model = MPSModel(
        ti,
        zones=[Zone(ti, zone)],
        post_processing=1,
        post_processing_mask=mask,
        post_processing_path="random",
    )
    ds = DirectSampling(model)
    ds.set_condition(
        [[0, 1], [0, 1]], {"facies": [1, np.nan], "value": [np.nan, 2]}
    )
    outputs = [
        ds(
            [np.arange(26)] * 2,
            seed=7,
            path_seed=19,
            node_seed=505,
            num_threads=threads,
            post_process=False,
            store=False,
        )
        for threads in [1, 4]
    ]
    for name in outputs[0]:
        np.testing.assert_array_equal(outputs[0][name], outputs[1][name])
    assert outputs[0]["facies"][0, 0] == 1
    assert outputs[0]["value"][1, 1] == 2
    assert main == [1, 4]
    assert cleanup == [1, 4]


@pytest.mark.parametrize("multivariate", [False, True])
def test_infinite_condition_rejected_transactionally(multivariate):
    ti = (
        TrainingImage(
            [
                Variable("a", np.zeros((3, 3))),
                Variable("b", np.ones((3, 3))),
            ]
        )
        if multivariate
        else TrainingImage(np.zeros((3, 3)))
    )
    ds = DirectSampling(MPSModel(ti), seed=7)
    valid = {"a": [1], "b": [np.nan]} if multivariate else [1]
    ds.set_condition([[0], [0]], valid)
    previous = ds._cond_val
    invalid = {"a": [np.inf], "b": [np.nan]} if multivariate else [np.inf]
    with pytest.raises(ValueError, match="infinity"):
        ds.set_condition([[1], [1]], invalid)
    assert ds._cond_val is previous


@pytest.mark.parametrize("categorical", [False, True])
def test_lossy_integer_condition_rejected(categorical):
    ti = TrainingImage(np.zeros((3, 3)), categorical=categorical)
    ds = DirectSampling(MPSModel(ti))
    with pytest.raises(ValueError, match="not exactly representable"):
        ds.set_condition([[0], [0]], np.array([2**60 + 3], dtype=np.int64))


@pytest.mark.parametrize(
    "axes, message",
    [
        ([np.array([0, 1, 3]), np.arange(3)], "not uniformly spaced"),
        ([np.arange(3), np.arange(3) * 2], "unequal spacing"),
    ],
)
def test_index_space_warning(axes, message):
    with pytest.warns(UserWarning, match=message):
        DirectSampling(_model(), seed=7)(axes, store=False)


@pytest.mark.parametrize("value", [-1, 1.25, np.inf, "7", [7], 1 + 0j])
def test_seed_controls_validate_before_acquiring_rng(value):
    model = _model()
    with pytest.raises(ValueError, match="seed"):
        DirectSampling(model, seed=value)
    ds = DirectSampling(model, seed=7)
    control = DirectSampling(model, seed=7)
    pos = [np.arange(4)] * 2
    for role in ("seed", "path_seed", "node_seed"):
        with pytest.raises(ValueError, match=role):
            ds(pos, **{"seed": 99, role: value}, store=False)
    np.testing.assert_array_equal(
        ds(pos, store=False), control(pos, store=False)
    )
