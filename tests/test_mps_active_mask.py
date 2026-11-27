"""True active-domain contracts: inactive nodes are neither targets nor data."""

import numpy as np
import pytest

from gstools.mps import DirectSampling, MPSModel, TrainingImage, Variable, Zone
from gstools.mps import simulate as engine


def ti():
    y, x = np.indices((16, 16))
    return TrainingImage(
        ((x // 3 + y // 4) % 3).astype(float),
        categorical=True,
        n_neighbors=8,
        max_radius=4,
    )


def simulate(
    mask, threads=1, cleanup=0, path="random", rotation=None, zones=None
):
    ds = DirectSampling(
        MPSModel(
            ti(),
            threshold=0.05,
            scan_fraction=0.3,
            active_mask=mask,
            post_processing=cleanup,
            rotation=rotation,
            zones=zones,
        ),
        seed=182,
    )
    return ds(
        [np.arange(float(n)) for n in mask.shape],
        num_threads=threads,
        path=path,
        store=False,
    )


@pytest.mark.parametrize("cleanup", [0, 1, 2])
def test_irregular_mask_serial_parallel_and_cleanup(cleanup):
    mask = np.ones((30, 30), bool)
    mask[8:17, 9:18] = False
    mask[:4, :] = False
    a = simulate(mask, 1, cleanup)
    b = simulate(mask, 4, cleanup)
    np.testing.assert_array_equal(a, b)
    assert np.isnan(a[~mask]).all()
    assert np.isin(a[mask], [0, 1, 2]).all()


@pytest.mark.parametrize("cleanup", [0, 1])
def test_all_true_mask_is_exactly_legacy(cleanup):
    mask = np.ones((26, 26), bool)
    a = simulate(mask, 4, cleanup)
    ds = DirectSampling(
        MPSModel(
            ti(), threshold=0.05, scan_fraction=0.3, post_processing=cleanup
        ),
        seed=182,
    )
    b = ds([np.arange(26.0), np.arange(26.0)], num_threads=4, store=False)
    np.testing.assert_array_equal(a, b)


def test_masked_rectangle_equals_cropped_reference():
    # The same path draws and active geometry must give the same result when
    # surrounded by inactive padding; erasing simulated cells afterward fails.
    mask = np.zeros((12, 14), bool)
    mask[2:10, 3:11] = True
    padded = simulate(mask, path="sequential")
    cropped = simulate(np.ones((8, 8), bool), path="sequential")
    np.testing.assert_array_equal(padded[2:10, 3:11], cropped)


def test_empty_mask_is_all_nan_with_cleanup():
    assert np.isnan(simulate(np.zeros((8, 8), bool), 4, 2)).all()


def test_condition_outside_mask_fails_before_rng_acquisition():
    mask = np.ones((10, 10), bool)
    mask[1, 1] = False
    ds = DirectSampling(MPSModel(ti(), active_mask=mask), seed=10)
    ds.set_condition([[1.0], [1.0]], [2.0])
    with pytest.raises(ValueError, match="outside active_mask"):
        ds([np.arange(10.0), np.arange(10.0)], store=False)
    ds.set_condition([[2.0], [2.0]], [1.0])
    a = ds([np.arange(10.0), np.arange(10.0)], store=False)
    other = DirectSampling(MPSModel(ti(), active_mask=mask), seed=10)
    other.set_condition([[2.0], [2.0]], [1.0])
    np.testing.assert_array_equal(
        a, other([np.arange(10.0), np.arange(10.0)], store=False)
    )
    assert a[2, 2] == 1


def test_invalid_masks_and_mutation_are_checked():
    with pytest.raises(TypeError, match="boolean"):
        MPSModel(ti(), active_mask=np.ones((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError, match="shape"):
        ds = DirectSampling(MPSModel(ti(), active_mask=np.ones((2, 2), bool)))
        ds([np.arange(10.0), np.arange(10.0)])
    source = np.ones((10, 10), bool)
    model = MPSModel(ti(), active_mask=source)
    source[:] = False
    assert model.active_mask.all()  # caller array is snapshotted
    model.active_mask = np.zeros((10, 10), bool)
    assert not model.active_mask.any()


def test_multivariate_partial_conditions_zones_and_rotation(monkeypatch):
    base = ti().data
    image = TrainingImage(
        [
            Variable("a", base, categorical=True, n_neighbors=6),
            Variable("b", 2 * base, categorical=False, n_neighbors=4),
        ]
    )
    mask = np.ones((24, 24), bool)
    mask[8:12, 8:12] = False
    zone_mask = np.zeros((24, 24), bool)
    zone_mask[:, 12:] = True
    model = MPSModel(
        image,
        active_mask=mask,
        rotation=0.15,
        zones=[Zone(image, zone_mask)],
        post_processing=1,
        scan_fraction=0.2,
    )
    ds = DirectSampling(model, seed=5)
    main, cleanup = [], []
    monkeypatch.setattr(
        engine,
        "_MPS_RUST_ENGINE_STATS_HOOK",
        lambda stats: main.append(stats["used_threads"]),
    )
    monkeypatch.setattr(
        engine,
        "_MPS_CLEANUP_STATS_HOOK",
        lambda stats: cleanup.append(stats["used_threads"]),
    )
    ds.set_condition([[2.0], [3.0]], {"a": [2.0], "b": [np.nan]})
    a = ds(
        [np.arange(24.0), np.arange(24.0)], num_threads=1, seed=5, store=False
    )
    b = ds(
        [np.arange(24.0), np.arange(24.0)], num_threads=4, seed=5, store=False
    )
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])
        assert np.isnan(a[name][~mask]).all()
    assert a["a"][2, 3] == 2
    assert main == cleanup == [1, 4]


def test_explicit_path_cannot_omit_active_nodes_and_drops_inactive_extras():
    mask = np.zeros((5, 5), bool)
    mask[1:4, 1:4] = True
    whole = np.indices(mask.shape).reshape(2, -1).T
    a = simulate(mask, path=whole)
    np.testing.assert_array_equal(a, simulate(mask, path=whole[mask.ravel()]))
    with pytest.raises(ValueError, match="missing"):
        simulate(mask, path=whole[mask.ravel()][:-1])


def test_native_mask_validation_without_interpreter_abort(monkeypatch):
    from gstools.mps import simulate as engine

    native = engine._mps_simulate_gsc
    for mask in (
        np.array([2], np.uint8),
        np.array([], np.uint8),
        np.array([0], np.uint8),
    ):

        def invalid_native(*args, **kwargs):
            kwargs["active_mask"] = mask
            return native(*args, **kwargs)

        monkeypatch.setattr(engine, "_mps_simulate_gsc", invalid_native)
        with pytest.raises(ValueError, match="active|inactive"):
            DirectSampling(MPSModel(ti()), seed=1)(
                [np.arange(1.0), np.arange(1.0)], store=False
            )


def test_3d_mask_and_disconnected_islands():
    coords = np.indices((6, 6, 6))
    image = TrainingImage(
        (coords.sum(axis=0) % 2).astype(float),
        categorical=True,
        n_neighbors=6,
        max_radius=2,
    )
    mask = np.zeros((12, 12, 12), bool)
    mask[:5, :5, :5] = True
    mask[7:, 7:, 7:] = True
    model = MPSModel(
        image, active_mask=mask, post_processing=1, scan_fraction=0.2
    )
    ds = DirectSampling(model, seed=31)
    a = ds([np.arange(12.0)] * 3, num_threads=1, seed=31, store=False)
    b = ds([np.arange(12.0)] * 3, num_threads=4, seed=31, store=False)
    np.testing.assert_array_equal(a, b)
    assert np.isnan(a[~mask]).all() and np.isfinite(a[mask]).all()


@pytest.mark.parametrize(
    "distance", ["l1", "l2", "l3.5", "variation", "variation1.5"]
)
def test_continuous_masked_cleanup_uses_parallel_workers(
    distance, monkeypatch
):
    rng = np.random.default_rng(100)
    mask = np.ones((26, 26), dtype=bool)
    mask[0, :] = mask[:, 0] = False
    image = TrainingImage(
        rng.random((18, 18)),
        categorical=False,
        distance=distance,
        n_neighbors=8,
        max_radius=4,
    )
    ds = DirectSampling(
        MPSModel(
            image,
            scan_fraction=0.35,
            post_processing=1,
            active_mask=mask,
        )
    )
    ds.set_condition([[1], [1]], [0.7])
    main, cleanup = [], []
    monkeypatch.setattr(
        engine,
        "_MPS_RUST_ENGINE_STATS_HOOK",
        lambda stats: main.append(stats["used_threads"]),
    )
    monkeypatch.setattr(
        engine,
        "_MPS_CLEANUP_STATS_HOOK",
        lambda stats: cleanup.append(stats["used_threads"]),
    )
    serial, parallel = [
        ds([np.arange(26)] * 2, seed=12, num_threads=t, store=False)
        for t in (1, 4)
    ]
    assert main == cleanup == [1, 4]
    np.testing.assert_array_equal(serial, parallel)
    assert np.isnan(serial[~mask]).all()
    assert serial[1, 1] == 0.7


def test_all_missing_conditions_outside_mask_are_ignored():
    image = TrainingImage(
        [
            Variable("a", ti().data),
            Variable("b", ti().data),
        ]
    )
    mask = np.ones((3, 3), dtype=bool)
    mask[0, 0] = False
    ds = DirectSampling(MPSModel(image, active_mask=mask))
    ds.set_condition(
        [[0, 1], [0, 1]], {"a": [np.nan, 9.0], "b": [np.nan, np.nan]}
    )
    result = ds([np.arange(3)] * 2, seed=12, store=False)
    assert all(np.isnan(field[0, 0]) for field in result.values())
    assert result["a"][1, 1] == 9.0
    assert np.isfinite(result["b"][1, 1])
    ds.set_condition([[0], [0]], {"a": [np.nan], "b": [9.0]})
    with pytest.raises(ValueError, match="outside active_mask"):
        ds([np.arange(3)] * 2, seed=12, store=False)
