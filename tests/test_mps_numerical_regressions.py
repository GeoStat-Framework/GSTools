"""Regression witnesses for Direct Sampling numerical edge cases."""

import numpy as np
import pytest

from gstools.mps import DirectSampling, MPSModel, TrainingImage, Variable, Zone


# --- DS02: continuous range must not overflow its input dtype ---------------
@pytest.mark.parametrize(
    "data",
    [
        np.array([-128, 127], np.int8),
        np.array([-32768, 32767], np.int16),
        np.array([-3e38, 3e38], np.float32),
    ],
)
def test_endpoint_distance_is_one(data):
    ti = TrainingImage(data, categorical=False, distance="l1")
    # The normalized L1 distance between the two extrema is 1.0 regardless of
    # the storage dtype (the range is computed in float64).
    assert ti.vec_distance_var(
        None, [float(data[0])], [[float(data[-1])]]
    ) == [1.0]


def test_range_too_large_for_float64_errors():
    with pytest.raises(ValueError, match="not finite"):
        TrainingImage(
            np.array([-1e308, 1e308]), categorical=False, distance="l1"
        ).variable().d_max


# --- DS03: zero-weight variables must not alter geometry or poison scores ---
def test_zero_weight_does_not_change_match():
    ti = TrainingImage(
        [
            Variable(
                "a", np.array([1.0, 0.0, 1.0, 0.0]), weight=1.0, n_neighbors=1
            ),
            Variable(
                "b",
                np.array([1.0, np.nan, np.nan, np.nan]),
                weight=0.0,
                n_neighbors=1,
            ),
        ]
    )
    for post_processing in (0, 1):
        ds = DirectSampling(
            MPSModel(
                ti,
                threshold=0.0,
                scan_fraction=1.0,
                post_processing=post_processing,
            )
        )
        ds.set_condition([[0.0, 1.0]], {"a": [np.nan, 1.0], "b": [1.0, 0.0]})
        field = ds(
            [np.arange(2.0)],
            seed=5,
            node_seed=1,
            path="sequential",
            num_threads=1,
        )
        # The exact A match pastes 0 at node 0; the all-undefined zero-weight B
        # must not turn the score into NaN (0*inf) and flip it to 1.
        assert field["a"].tolist() == [0.0, 1.0]
        assert field["b"].tolist() == [1.0, 0.0]


# --- DS04: variation paste shifts over the common finite support ------------
@pytest.mark.parametrize("post_processing", [0, 1])
def test_variation_common_support_paste(post_processing):
    ti = TrainingImage(
        np.array([10.0, 1.0, np.nan]),
        categorical=False,
        distance="variation",
        n_neighbors=2,
    )
    ds = DirectSampling(
        MPSModel(
            ti,
            threshold=0.0,
            scan_fraction=1.0,
            post_processing=post_processing,
        )
    )
    ds.set_condition([[1.0, 2.0]], np.array([100.0, 200.0]))
    field = ds(
        [np.arange(3.0)], seed=5, node_seed=1, path="sequential", num_threads=1
    )
    # 10 - 1 + 100 = 109 over the single common-support neighbour, not the
    # 159 that the full-SG-mean shift produced.
    assert field.tolist() == [109.0, 100.0, 200.0]


# --- DS05: invalid native input raises instead of aborting ------------------
def test_out_of_bounds_path_raises():
    import gstools_core as core

    base = dict(
        ti=np.array([[0.0, 1.0]]),
        ti_shape=np.array([2], np.int64),
        initial_fields=np.array([[np.nan]]),
        conditioned=np.array([[0]], np.uint8),
        sim_shape=np.array([1], np.int64),
        path=np.array([[5]], np.int64),  # out of bounds
        lag_matrices=np.empty((0, 1, 1)),
        u_start=np.array([0.0]),
        u_fallback=np.array([[0.0]]),
        offsets=np.array([[-1], [1]], np.int64),
        n_neighbors=np.array([2], np.int64),
        max_radius=np.array([np.nan]),
        variable_weights=np.array([1.0]),
        metric_kinds=np.array([0], np.int64),
        has_nan=np.array([0], np.uint8),
        d_max=np.array([1.0]),
        p_norm=np.array([1.0]),
        penalty_flat=np.empty(0),
        penalty_offsets=np.zeros(1, np.int64),
        penalty_ncat=np.zeros(1, np.int64),
        threshold=0.0,
        scan_fraction=1.0,
        random_scan=False,
        distance_power=np.zeros(1),
        cond_weight=np.ones(1),
        partial_boundary=False,
        num_threads=1,
        zone_ti=np.empty((1, 0)),
        zone_offsets=np.array([0], np.int64),
        zone_shapes=np.empty((0, 1), np.int64),
        zone_d_max=np.empty((0, 1)),
        node_domain=np.zeros(1, np.int64),
    )
    with pytest.raises(ValueError):
        core.mps_simulate(**base)


def test_fallback_uniform_one_raises():
    import gstools_core as core

    base = dict(
        ti=np.array([[0.0, 1.0]]),
        ti_shape=np.array([2], np.int64),
        initial_fields=np.array([[np.nan]]),
        conditioned=np.array([[0]], np.uint8),
        sim_shape=np.array([1], np.int64),
        path=np.array([[0]], np.int64),
        lag_matrices=np.empty((0, 1, 1)),
        u_start=np.array([0.0]),
        u_fallback=np.array([[1.0]]),  # 1.0 is outside [0, 1)
        offsets=np.array([[-1], [1]], np.int64),
        n_neighbors=np.array([2], np.int64),
        max_radius=np.array([np.nan]),
        variable_weights=np.array([1.0]),
        metric_kinds=np.array([0], np.int64),
        has_nan=np.array([0], np.uint8),
        d_max=np.array([1.0]),
        p_norm=np.array([1.0]),
        penalty_flat=np.empty(0),
        penalty_offsets=np.zeros(1, np.int64),
        penalty_ncat=np.zeros(1, np.int64),
        threshold=0.0,
        scan_fraction=1.0,
        random_scan=False,
        distance_power=np.zeros(1),
        cond_weight=np.ones(1),
        partial_boundary=False,
        num_threads=1,
        zone_ti=np.empty((1, 0)),
        zone_offsets=np.array([0], np.int64),
        zone_shapes=np.empty((0, 1), np.int64),
        zone_d_max=np.empty((0, 1)),
        node_domain=np.zeros(1, np.int64),
    )
    with pytest.raises(ValueError):
        core.mps_simulate(**base)


def test_dist_block_negative_index_raises():
    import gstools_core as core

    with pytest.raises(ValueError):
        core.mps_dist_block_cat(
            np.array([0.0]),
            np.array([0.0]),
            np.array([-1], np.int64),
            np.array([0], np.int64),
            np.array([1.0]),
        )


# --- DS06: envelope covers all effective radii, bounded per axis ------------
def test_unlimited_radius_matches_full_grid_finite():
    fields = {}
    for radius in (None, 8):
        ti = TrainingImage(
            [
                Variable(
                    "a", np.zeros(10, dtype=int), max_radius=1, n_neighbors=1
                ),
                Variable(
                    "b",
                    np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1]),
                    max_radius=radius,
                    n_neighbors=1,
                ),
            ]
        )
        ds = DirectSampling(MPSModel(ti, scan_fraction=1.0))
        ds.set_condition(
            [np.arange(9.0)],
            {"a": np.zeros(9), "b": np.r_[np.full(8, np.nan), 1.0]},
        )
        fields[radius] = ds(
            [np.arange(9.0)],
            path="sequential",
            node_seed=1,
            post_process=False,
        )["b"]
    np.testing.assert_array_equal(fields[None], fields[8])


def test_thin_grid_does_not_overallocate():
    ti = TrainingImage((np.arange(8) % 2).reshape(2, 2, 2), n_neighbors=1)
    field = DirectSampling(MPSModel(ti), seed=3)(
        [np.arange(1000.0), np.arange(2.0), np.arange(2.0)],
        path="sequential",
        post_process=False,
        store=False,
    )
    assert field.shape == (1000, 2, 2)
    assert np.isfinite(field).all()


# --- DS07: identity transform must equal no transform -----------------------
def test_identity_scale_equals_no_transform():
    data = np.array([[0, 1], [1, 0], [0, 1], [0, 1], [0, 1]])
    base = np.argwhere(np.ones((5, 5), dtype=bool))
    path = np.vstack([[2, 2], base[np.any(base != [2, 2], axis=1)]])
    fields = {}
    for scale in (None, 1.0):
        ti = TrainingImage(data, n_neighbors=3)
        ds = DirectSampling(MPSModel(ti, scan_fraction=1.0, scale=scale))
        ds.set_condition([[1.0, 1.0, 4.0], [1.0, 3.0, 2.0]], [1.0, 0.0, 1.0])
        fields[scale] = ds(
            [np.arange(5.0), np.arange(5.0)],
            path=path,
            node_seed=0,
            num_threads=1,
            post_process=False,
        )
    np.testing.assert_array_equal(fields[None], fields[1.0])


# --- DS08: conditioning updates are transactional ---------------------------
def test_failed_update_preserves_prior_state():
    ti = TrainingImage(
        [
            Variable("a", np.array([[0.0, 1.0], [1.0, 0.0]]), n_neighbors=2),
            Variable("b", np.array([[1.0, 0.0], [0.0, 1.0]]), n_neighbors=2),
        ]
    )
    ds = DirectSampling(MPSModel(ti), seed=7)
    ds.set_condition([[0.0], [0.0]], {"a": [1.0], "b": [0.0]})
    with pytest.raises(ValueError):
        ds.set_condition([[1.0], [1.0]], {"a": ["invalid"], "b": [0.0]})
    np.testing.assert_array_equal(ds._cond_pos, [[0.0], [0.0]])
    assert ds._cond_val["a"].tolist() == [1.0]


def test_2d_value_array_rejected():
    ti = TrainingImage(
        [
            Variable("a", np.array([[0.0, 1.0], [1.0, 0.0]]), n_neighbors=2),
            Variable("b", np.array([[1.0, 0.0], [0.0, 1.0]]), n_neighbors=2),
        ]
    )
    ds = DirectSampling(MPSModel(ti), seed=7)
    with pytest.raises(ValueError, match="1-D"):
        ds.set_condition(
            [[0.0, 1.0], [0.0, 1.0]],
            {"a": [[0.0, 1.0], [1.0, 0.0]], "b": [0.0, 1.0]},
        )


def test_univariate_nonfinite_position_rollback():
    ds = DirectSampling(
        MPSModel(TrainingImage(np.array([[0, 1], [1, 0]]), n_neighbors=2)),
        seed=7,
    )
    ds.set_condition([[0.0], [0.0]], [0.0])
    with pytest.raises(ValueError):
        ds.set_condition([[np.nan], [1.0]], [1.0])
    assert np.isfinite(ds._cond_pos).all()


# --- DS09: penalty category screening on conditioning -----------------------
def test_invalid_hard_category_rejected():
    ti = TrainingImage(
        Variable(
            None,
            np.array([[0, 1], [1, 0]]),
            n_neighbors=2,
            penalty_matrix=[[0.0, 1.0], [1.0, 0.0]],
        )
    )
    ds = DirectSampling(MPSModel(ti), seed=7)
    with pytest.raises(ValueError):
        ds.set_condition(
            [[0.0], [0.0]], [2.0]
        )  # code 2 absent from the 2x2 matrix


# --- DS11: cleanup mask must match the grid exactly -------------------------
@pytest.mark.parametrize(
    "mask",
    [np.array([True, False]), np.array([[True], [False]]), np.array(True)],
)
def test_wrong_shaped_cleanup_mask_rejected(mask):
    ti = TrainingImage(np.array([[0, 1], [1, 0]]), n_neighbors=2)
    with pytest.raises(ValueError, match="post_processing_mask shape"):
        DirectSampling(
            MPSModel(ti, post_processing=1, post_processing_mask=mask), seed=1
        )([np.arange(2.0), np.arange(2.0)], store=False)


# --- DS12: named univariate TI conveniences work ----------------------------
def test_named_univariate_accessors():
    ti = TrainingImage(
        Variable("facies", np.array([[0, 1], [1, 0]]), n_neighbors=2)
    )
    assert ti.data is not None
    assert ti.categorical is True
    assert isinstance(ti.distance_type, str)
    assert isinstance(repr(ti), str)


# --- DS13: threshold None / None seeds / cleanup setters --------------------
def test_threshold_none_is_dsbc():
    assert (
        MPSModel(
            TrainingImage(np.array([[0, 1], [1, 0]]), n_neighbors=2),
            threshold=None,
        ).threshold
        == 0.0
    )


def test_none_seeds_accepted():
    ds = DirectSampling(
        MPSModel(TrainingImage(np.array([[0, 1], [1, 0]]), n_neighbors=2)),
        seed=None,
    )
    out = ds(
        [np.arange(2.0), np.arange(2.0)],
        seed=None,
        path_seed=None,
        node_seed=None,
        store=False,
    )
    assert out.shape == (2, 2)


def test_post_processing_setters():
    m = MPSModel(TrainingImage(np.array([[0, 1], [1, 0]]), n_neighbors=2))
    m.post_processing = 2
    m.post_processing_factor = 2.0
    m.post_processing_path = "same"
    assert (
        m.post_processing,
        m.post_processing_factor,
        m.post_processing_path,
    ) == (2, 2.0, "same")


# --- DS14: zones setter enforces the cleanup-penalty invariant --------------
def test_zone_setter_validates_cleanup_penalty():
    primary = TrainingImage(
        Variable("facies", np.array([[0, 1], [1, 0]]), n_neighbors=2)
    )
    override = primary.variable().replace(
        penalty_matrix=[[0.0, 1.0], [1.0, 0.0]]
    )
    zone = Zone(
        TrainingImage(
            Variable("facies", np.array([[0, 2], [2, 0]]), n_neighbors=2)
        ),
        np.ones((2, 2), bool),
    )
    model = MPSModel(
        primary, post_processing=1, post_processing_variables=[override]
    )
    with pytest.raises(ValueError):
        model.zones = [
            zone
        ]  # code 2 invalid under the cleanup override's 2x2 matrix


# --- DS15: normalization + fractional counts + nonfinite axes ---------------
def test_large_weights_normalize_to_half():
    w = TrainingImage(
        [
            Variable("a", np.array([0, 1]), weight=1e308),
            Variable("b", np.array([1, 0]), weight=1e308),
        ]
    ).weights
    assert w == {"a": 0.5, "b": 0.5}


@pytest.mark.parametrize("bad", [0.9, 1.9, -0.5])
def test_fractional_post_processing_rejected(bad):
    with pytest.raises(ValueError, match="integer"):
        MPSModel(
            TrainingImage(np.array([[0, 1], [1, 0]]), n_neighbors=2),
            post_processing=bad,
        )


@pytest.mark.parametrize("axis", [[0.0, np.nan, 2.0], [0.0, 1.0, np.inf]])
def test_nonfinite_grid_axis_rejected(axis):
    ds = DirectSampling(
        MPSModel(TrainingImage(np.array([[0, 1], [1, 0]]), n_neighbors=2)),
        seed=1,
    )
    with pytest.raises(ValueError, match="finite"):
        ds([axis, np.arange(3.0)], store=False)


# --- DS17: per-zone missingness flags ---------------------------------------
def test_whole_grid_zone_matches_standalone_masked_ti():
    data = np.ascontiguousarray(
        [[0, np.nan, 0, 1, 1], [10, 20, 30, 40, 50]], float
    )
    finite = data.copy()
    finite[0, 1] = 0.0

    def ti_from(v):
        return TrainingImage(
            [
                Variable("a", v[0], weight=1.0, n_neighbors=1),
                Variable("b", v[1], weight=0.0, n_neighbors=1),
            ]
        )

    results = []
    for primary, zones in [
        (ti_from(data), []),
        (ti_from(finite), [Zone(ti_from(data), np.ones(2, bool))]),
    ]:
        ds = DirectSampling(
            MPSModel(primary, zones=zones, threshold=0.0, scan_fraction=1.0)
        )
        ds.set_condition([[0.0, 1.0]], {"a": [0.0, 0.0], "b": [10.0, np.nan]})
        results.append(
            ds(
                [np.arange(2.0)],
                path="sequential",
                node_seed=1,
                post_process=False,
                store=False,
            )["b"].tolist()
        )
    # The zone now scans with its own (masked) flags, so a whole-grid identical
    # masked zone matches the standalone masked TI (both [10, 20]).
    assert results[0] == results[1] == [10.0, 20.0]


# --- DS18: native random scan prefix is unbiased ----------------------------
def test_random_scan_reaches_all_categories():
    ti = TrainingImage(
        [
            Variable("guide", np.zeros(4), weight=1.0, n_neighbors=1),
            Variable("value", np.arange(4.0), weight=0.0, n_neighbors=1),
        ]
    )
    seen = set()
    for seed in range(64):
        ds = DirectSampling(
            MPSModel(ti, scan_fraction=1.0, threshold=0.0, scan_path="random")
        )
        ds.set_condition([[0.0]], {"guide": [0.0]})
        field = ds(
            (np.arange(1.0),),
            path="sequential",
            node_seed=seed,
            post_process=False,
            store=False,
        )
        seen.add(int(field["value"][0]))
    # The biased LCG could only return 1 or 2 mod 4 first; every category is now
    # reachable on a budget-1 exact-match scan.
    assert seen == {0, 1, 2, 3}
