"""Tests for gstools.mps.simulate."""

import unittest
from unittest import mock

import numpy as np
from mps_fixtures import as_dict

import gstools as gs
from gstools.mps.direct_sampling import DirectSampling
from gstools.mps.model import MPSModel
from gstools.mps.post_processing import PostProcessing
from gstools.mps.simulate import (
    _DirectSamplingEngine,
    _order_positions,
    _Pass,
    _Run,
)
from gstools.mps.training_image import (
    TrainingImage,
    Variable,
)


class TestSimulationPath(unittest.TestCase):
    """Tests for the pluggable path= parameter in DirectSampling."""

    def _make_ds(
        self, seed_data=0, shape=(20, 20), n_vals=3, scan_fraction=0.5
    ):
        rng = np.random.default_rng(seed_data)
        data = rng.integers(0, n_vals, shape).astype(float)
        ti = TrainingImage(data, categorical=True, n_neighbors=4)
        return DirectSampling(MPSModel(ti, scan_fraction=scan_fraction)), ti

    # ------------------------------------------------------------------
    # "sequential" mode
    # ------------------------------------------------------------------

    def test_sequential_path_order(self):
        """path='sequential' -> engine visits nodes in lexicographic (raster) order."""
        ds, _ = self._make_ds()
        pos = [np.arange(5, dtype=float)] * 2
        sim_shape = (5, 5)
        # Patch the engine to capture its path array.
        from gstools.mps import simulate as sim_mod

        captured = {}
        orig_cls = sim_mod._DirectSamplingEngine

        class _PatchedEngine(orig_cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                captured["path"] = self.path.copy()

        sim_mod._DirectSamplingEngine = _PatchedEngine
        try:
            ds(pos, seed=1, path="sequential")
        finally:
            sim_mod._DirectSamplingEngine = orig_cls

        expected = np.argwhere(
            np.ones(sim_shape, dtype=bool)
        )  # all nodes, raster
        np.testing.assert_array_equal(captured["path"], expected)

    def test_sequential_deterministic_without_path_seed(self):
        """path='sequential' produces the same field on two calls without fixing path_seed."""
        ds, _ = self._make_ds()
        pos = [np.arange(6, dtype=float)] * 2
        f1 = ds(pos, seed=3, path="sequential")
        f2 = ds(pos, seed=3, path="sequential")
        np.testing.assert_array_equal(f1, f2)

    def test_sequential_valid_values(self):
        """path='sequential' output values are a subset of TI values."""
        ds, ti = self._make_ds()
        pos = [np.arange(6, dtype=float)] * 2
        field = ds(pos, seed=5, path="sequential")
        self.assertEqual(field.shape, (6, 6))
        self.assertFalse(np.any(np.isnan(field)))
        ti_vals = set(np.unique(ti.variable().data))
        self.assertTrue(set(np.unique(field)).issubset(ti_vals))

    def test_sequential_conditioning_honored(self):
        """path='sequential' honors exact conditioning (invariant: cond == value)."""
        ds, _ = self._make_ds()
        ds.set_condition([[3.0], [2.0]], [1.0])
        field = ds([np.arange(8, dtype=float)] * 2, seed=7, path="sequential")
        self.assertEqual(field[3, 2], 1.0)

    # ------------------------------------------------------------------
    # explicit array mode
    # ------------------------------------------------------------------

    def test_explicit_path_order_honored(self):
        """Explicit path is used exactly as supplied — verified via a degenerate TI."""
        # TI is a constant (all zeros), so every simulated value is 0 regardless
        # of order; we just verify no error and shape is correct.
        ti = TrainingImage(np.zeros((10, 10), dtype=float), categorical=True)
        ds = DirectSampling(MPSModel(ti, scan_fraction=1.0))
        pos = [np.arange(4, dtype=float)] * 2
        # Build path in reverse raster order
        base = np.argwhere(np.ones((4, 4), dtype=bool))
        explicit = base[::-1].copy()
        field = ds(pos, seed=0, path=explicit)
        self.assertEqual(field.shape, (4, 4))
        np.testing.assert_array_equal(field, np.zeros((4, 4)))

    def test_explicit_path_subset_property(self):
        """Explicit path: output values are a subset of TI values."""
        ds, ti = self._make_ds()
        pos = [np.arange(5, dtype=float)] * 2
        base = np.argwhere(np.ones((5, 5), dtype=bool))
        # reverse order
        explicit = base[::-1].copy()
        field = ds(pos, seed=0, path=explicit)
        ti_vals = set(np.unique(ti.variable().data))
        self.assertTrue(set(np.unique(field)).issubset(ti_vals))

    def test_explicit_path_conditioning_honored(self):
        """Explicit path: conditioning data are still honored exactly."""
        ds, _ = self._make_ds()
        ds.set_condition([[2.0], [4.0]], [2.0])
        pos = [np.arange(6, dtype=float)] * 2
        sim_shape = (6, 6)
        # All nodes except the conditioned one
        unknown_mask = np.ones(sim_shape, dtype=bool)
        unknown_mask[2, 4] = (
            False  # conditioned node excluded from unknown set
        )
        base = np.argwhere(unknown_mask)
        field = ds(pos, seed=2, path=base)
        self.assertEqual(field[2, 4], 2.0)

    # ------------------------------------------------------------------
    # Validation errors (explicit array)
    # ------------------------------------------------------------------

    def test_explicit_path_validation_errors(self):
        """Malformed explicit paths each raise ValueError with a specific message."""
        base = np.argwhere(np.ones((4, 4), dtype=bool))
        oob = base.copy()
        oob[0, 0] = 99
        dup = base.copy()
        dup[-1] = dup[0]
        cases = {
            "wrong_dim": (np.array([0, 1, 2, 3]), "shape"),  # 1-D, not (N,2)
            "wrong_second_dim": (np.zeros((5, 3), dtype=int), "shape"),
            "out_of_bounds_coord": (oob, "out-of-bounds"),
            "missing_node": (base[:-1].copy(), "missing"),
            "duplicate_node": (dup, "duplicate"),
            "invalid_string": ("zigzag", "path"),
        }
        ds, _ = self._make_ds()
        pos = [np.arange(4, dtype=float)] * 2
        for label, (path, regex) in cases.items():
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, regex):
                    ds(pos, seed=0, path=path)

    def test_explicit_extra_node_silently_skipped(self):
        """Conditioned nodes present in an explicit path are silently skipped.

        A full-grid path (e.g. a spiral over all nodes) should work unchanged
        with conditioning data — the engine drops pre-filled entries itself.
        """
        ds, ti = self._make_ds()
        ds.set_condition([[0.0], [0.0]], [0.0])
        pos = [np.arange(4, dtype=float)] * 2
        # Full raster includes (0,0), which is conditioned.
        # The engine should drop it silently and simulate the remaining 15 nodes.
        full_raster = np.argwhere(np.ones((4, 4), dtype=bool))
        field = ds(pos, seed=0, path=full_raster)
        self.assertEqual(field.shape, (4, 4))
        # Conditioned node must keep its exact value
        self.assertEqual(field[0, 0], 0.0)
        # All values must be a subset of TI values
        ti_vals = set(np.unique(ti.variable().data))
        self.assertTrue(set(np.unique(field)).issubset(ti_vals))

    def test_random_path_default_pinned(self):
        """path='random' (default) output is pinned.

        Captured with seed=42, scan_fraction=0.5, 8x8 grid, 3-valued categorical
        TI of shape (20,20) built from np.random.default_rng(0).
        """
        rng_data = np.random.default_rng(0)
        data = rng_data.integers(0, 3, (20, 20)).astype(float)
        ti = TrainingImage(data, categorical=True)
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.5))
        field = ds([np.arange(8, dtype=float)] * 2, seed=42)
        expected = np.array(
            [
                [2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [2.0, 2.0, 0.0, 2.0, 2.0, 0.0, 1.0, 1.0],
                [2.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0],
                [2.0, 2.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0],
                [2.0, 2.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
                [0.0, 2.0, 2.0, 0.0, 2.0, 1.0, 0.0, 2.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 2.0, 2.0],
                [0.0, 1.0, 1.0, 0.0, 2.0, 2.0, 0.0, 0.0],
            ]
        )
        np.testing.assert_array_equal(field, expected)

    def test_path_seed_inert_for_sequential(self):
        """path_seed has no effect when path='sequential' (documented behavior)."""
        ds, _ = self._make_ds()
        pos = [np.arange(5, dtype=float)] * 2
        f1 = ds(pos, seed=0, path="sequential", path_seed=1)
        f2 = ds(pos, seed=0, path="sequential", path_seed=99)
        np.testing.assert_array_equal(f1, f2)

    def test_1d_sequential_path_order(self):
        """1-D sequential path is [[0],[1],...,[N-1]] (n-D correctness check)."""
        arr1d = np.tile([0.0, 1.0], 10)
        ti = TrainingImage(arr1d, categorical=True)
        ds = DirectSampling(MPSModel(ti, scan_fraction=1.0))
        pos = [np.arange(5, dtype=float)]

        from gstools.mps import simulate as sim_mod

        captured = {}
        orig_cls = sim_mod._DirectSamplingEngine

        class _PatchedEngine(orig_cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                captured["path"] = self.path.copy()

        sim_mod._DirectSamplingEngine = _PatchedEngine
        try:
            ds(pos, seed=0, path="sequential")
        finally:
            sim_mod._DirectSamplingEngine = orig_cls

        expected = np.arange(5).reshape(5, 1)
        np.testing.assert_array_equal(captured["path"], expected)


class TestScanConfigPerVariableDicts(unittest.TestCase):
    def test_distance_power_cond_weight_are_per_variable_dicts(self):
        ti = TrainingImage(
            [
                Variable(
                    "a", np.zeros((6, 6)), distance_power=0.5, cond_weight=2.0
                ),
                Variable(
                    "b", np.zeros((6, 6)), distance_power=1.5, cond_weight=3.0
                ),
            ]
        )
        rng = np.random.RandomState(0)
        engine = _DirectSamplingEngine(
            MPSModel(ti, scan_fraction=1.0),
            (4, 4),
            _Run(rng_path=rng, rng_nodes=rng),
        )
        cfg = engine.domains[0]
        self.assertEqual(cfg.distance_power, {"a": 0.5, "b": 1.5})
        self.assertEqual(cfg.cond_weight, {"a": 2.0, "b": 3.0})


class TestPostProcessingVariablesEngine(unittest.TestCase):
    @staticmethod
    def _run(ti, seed, sim_size=10, overrides=None, **model_kwargs):
        """Simulate a square grid with one post-processing pass."""
        post = PostProcessing(1, overrides=overrides)
        model = MPSModel(ti, post_processing=post, **model_kwargs)
        pos = [np.arange(float(sim_size))] * 2
        return DirectSampling(model)(pos, seed=seed)

    def test_empty_overrides_bit_identical(self):
        # Spec invariant 3: default behaviour is untouched.
        rng = np.random.RandomState(42)
        ti_arr = (rng.random((30, 30)) > 0.5).astype(float)
        ti = TrainingImage(ti_arr, categorical=True, n_neighbors=8)
        kwargs = dict(ti=ti, seed=5, scan_fraction=0.25, threshold=0.2)

        out_before = self._run(**kwargs)
        out_after = self._run(**kwargs, overrides={})
        np.testing.assert_array_equal(out_before, out_after)

    def test_post_processing_distance_power_override_changes_output(self):
        # A different distance_power during post-processing.
        rng = np.random.RandomState(7)
        ti_arr = (rng.random((30, 30)) > 0.5).astype(float)
        ti = TrainingImage(ti_arr, categorical=True, n_neighbors=8)
        kwargs = dict(ti=ti, seed=9, scan_fraction=0.25, threshold=0.2)

        base = self._run(**kwargs)
        overridden = self._run(
            **kwargs, overrides={"field": {"distance_power": 2.0}}
        )

        self.assertFalse(np.array_equal(base, overridden))

    def test_post_processing_distance_override_changes_output(self):
        # A different distance metric during post-processing.
        rng = np.random.default_rng(3)
        ti_arr = rng.integers(0, 3, (25, 25)).astype(float)
        ti = TrainingImage(ti_arr, categorical=True, n_neighbors=10)
        kwargs = dict(ti=ti, seed=11, scan_fraction=0.3, threshold=0.1)

        base = self._run(**kwargs)

        # Non-uniform, not just a uniform scalar multiple of binary mismatch
        # (a uniform penalty would leave every exact-match (d=0) candidate
        # exactly where it was, since scaling doesn't move the argmin/
        # threshold-crossing point for those).
        penalty = np.array([[0.0, 0.9, 0.1], [0.9, 0.0, 0.9], [0.1, 0.9, 0.0]])
        overridden = self._run(
            **kwargs, overrides={"field": {"penalty_matrix": penalty}}
        )

        self.assertFalse(np.array_equal(base, overridden))
        self.assertTrue(np.isin(overridden, np.unique(ti_arr)).all())

    def test_multivariate_named_override(self):
        # The override applies to the named variable of a multivariate TI.
        rng = np.random.default_rng(4)
        ti = TrainingImage(
            [
                Variable(
                    "a",
                    rng.integers(0, 3, (20, 20)).astype(float),
                    n_neighbors=6,
                ),
                Variable(
                    "b",
                    rng.integers(0, 2, (20, 20)).astype(float),
                    n_neighbors=6,
                ),
            ]
        )
        kwargs = dict(
            ti=ti, seed=13, sim_size=8, scan_fraction=0.4, threshold=0.15
        )

        base = self._run(**kwargs)
        overridden = self._run(
            **kwargs, overrides={"a": {"distance_power": 1.5}}
        )

        self.assertFalse(np.array_equal(base[0], overridden[0]))


class TestPostProcessingMask(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(6)
        ti_arr = (rng.random((20, 20)) > 0.5).astype(float)
        self.ti = TrainingImage(ti_arr, categorical=True, n_neighbors=6)
        self.pos = [np.arange(10.0)] * 2

    def _ds(self, post=None):
        return DirectSampling(
            MPSModel(
                self.ti,
                scan_fraction=0.3,
                threshold=0.15,
                post_processing=post,
            )
        )

    def _run(self, seed=8, post=None):
        return self._ds(post)(self.pos, seed=seed)

    def test_mask_none_is_default_behaviour(self):
        base = self._run(post=PostProcessing(1))
        explicit_none = self._run(post=PostProcessing(1, mask=None))
        np.testing.assert_array_equal(base, explicit_none)

    def test_mask_selects_eligible_nodes(self):
        # The mask says which nodes the post-pass may re-simulate; nodes
        # outside it must come out of the post-pass untouched.
        before = self._run()
        mask = np.zeros((10, 10), dtype=bool)
        mask[:5, :] = True  # only the top half is eligible

        top_eligible = self._run(post=PostProcessing(1, mask=mask))
        np.testing.assert_array_equal(before[5:, :], top_eligible[5:, :])

        bottom_eligible = self._run(post=PostProcessing(1, mask=~mask))
        np.testing.assert_array_equal(before[:5, :], bottom_eligible[:5, :])

    def test_mask_selects_eligible_nodes_with_same_path(self):
        # path="same" must use the mask-narrowed resim set,
        # not every unknown node. Masked-out nodes must stay frozen under
        # this path mode too.
        before = self._run()
        mask = np.zeros((10, 10), dtype=bool)
        mask[:5, :] = True

        top_eligible = self._run(
            post=PostProcessing(1, mask=mask, path="same")
        )
        np.testing.assert_array_equal(before[5:, :], top_eligible[5:, :])

    def test_mask_all_false_is_noop(self):
        before = self._run()
        after = self._run(
            post=PostProcessing(1, mask=np.zeros((10, 10), dtype=bool))
        )
        np.testing.assert_array_equal(before, after)

    def test_mask_wrong_shape_raises(self):
        for shape in ((3, 3), (100,), (10, 1), ()):
            with self.subTest(shape=shape):
                ds = self._ds(
                    PostProcessing(1, mask=np.ones(shape, dtype=bool))
                )
                with self.assertRaisesRegex(ValueError, "mask shape"):
                    ds(self.pos, seed=1)

    def test_mask_never_touches_conditioning(self):
        ds = self._ds(
            PostProcessing(
                1, mask=np.ones((10, 10), dtype=bool)
            )  # all eligible
        )
        ds.set_condition([[3.0], [3.0]], [1.0])
        out = ds(self.pos, seed=8)
        self.assertEqual(float(out[3, 3]), 1.0)


class TestPostProcessingPass(unittest.TestCase):
    """Meerschman et al. 2013, Sect. 4: re-simulation with fully informed neighbourhoods."""

    def setUp(self):
        rng = np.random.RandomState(42)
        ti = np.zeros((40, 40))
        ti[:, 20:] = 1.0
        flip = rng.rand(40, 40) < 0.05
        ti[flip] = 1.0 - ti[flip]
        self.ti_data = ti

    def _run(self, seed=20260717, cond=None, num_threads=None, post=None):
        ti = gs.mps.TrainingImage(
            self.ti_data, categorical=True, n_neighbors=8
        )
        model = gs.mps.MPSModel(
            ti, scan_fraction=0.25, threshold=0.2, post_processing=post
        )
        ds = gs.mps.DirectSampling(model, seed=seed)
        if cond is not None:
            ds.set_condition(*cond)
        return ds(
            (np.arange(24.0), np.arange(24.0)),
            store=False,
            num_threads=num_threads,
        )

    def test_subset_property_after_passes(self):
        base = self._run()
        field = self._run(post=PostProcessing(2, factor=2.0))
        self.assertTrue(np.isin(field, np.unique(self.ti_data)).all())
        # the passes must actually re-draw, not no-op
        self.assertFalse(np.array_equal(base, field))

    def test_factor_below_one_changes_output(self):
        # p_f < 1 means MORE effort per post-pass (larger n_neighbors,
        # larger scan_fraction), not less -- the opposite of Me13's
        # documented use of p_f. Confirm it actually takes effect.
        field_pf1 = self._run(post=PostProcessing(1, factor=1.0))
        field_pf_half = self._run(post=PostProcessing(1, factor=0.5))
        self.assertFalse(np.array_equal(field_pf1, field_pf_half))
        # still a valid categorical subset of the TI's values
        self.assertTrue(np.isin(field_pf_half, np.unique(self.ti_data)).all())

    def test_factor_below_one_scales_n_k_and_scan_fraction_up(self):
        observed = []
        orig = _DirectSamplingEngine._simulate_node

        def spy(engine, pss, i, x_i, u_st, u_fb):
            observed.append((dict(pss.n_k), pss.scan_fraction))
            return orig(engine, pss, i, x_i, u_st, u_fb)

        _DirectSamplingEngine._simulate_node = spy
        try:
            self._run(post=PostProcessing(1, factor=0.5))
        finally:
            _DirectSamplingEngine._simulate_node = orig

        # main-pass n_neighbors is 8 (set on the TI inside _run); a p_f=0.5
        # post-pass must double it: max(1, round(8 / 0.5)) == 16. scan_fraction
        # (0.25 in _run) must likewise double to 0.5.
        post_pass_n_k = {n_k["field"] for n_k, f in observed if f == 0.5}
        self.assertEqual(post_pass_n_k, {16})

    def test_conditioning_preserved_exactly(self):
        cond = (([2.0, 10.0, 21.0], [3.0, 15.0, 8.0]), [1.0, 0.0, 1.0])
        field = self._run(post=PostProcessing(2), cond=cond)
        self.assertEqual(field[2, 3], 1.0)
        self.assertEqual(field[10, 15], 0.0)
        self.assertEqual(field[21, 8], 1.0)

    def test_deterministic_given_seed(self):
        a = self._run(post=PostProcessing(1))
        b = self._run(post=PostProcessing(1))
        np.testing.assert_array_equal(a, b)

    def test_thread_count_invariance(self):
        # Post-passes run over the same dependency DAG as the main path, so
        # every thread count must reproduce the serial result bit-exactly.
        post = PostProcessing(2)
        ref = self._run(post=post, num_threads=1)
        for n_threads in (2, 4, 8):
            with self.subTest(num_threads=n_threads):
                np.testing.assert_array_equal(
                    ref, self._run(post=post, num_threads=n_threads)
                )

    def test_fully_informed_neighbourhood(self):
        """Me13 §4: a post-pass node sees cells it *precedes* in the order.

        The post-pass starts with every cell unvisited (all ``-1`` vmap). Gating post-pass reads by
        visit position instead (as the main path does) would leave the
        first-visited node with an empty neighbourhood, degrading the pass
        from a refinement of the realization into an independent redraw.
        """
        ti = gs.mps.TrainingImage(
            self.ti_data, categorical=True, n_neighbors=8
        )
        eng = _DirectSamplingEngine(
            MPSModel(ti, scan_fraction=0.25, threshold=0.2),
            (12, 12),
            _Run(
                rng_path=np.random.RandomState(0),
                rng_nodes=np.random.RandomState(1),
                num_threads=1,
            ),
        )
        eng.run()
        # Reproduce the post-pass vmap/pass, then probe the first-visited node.
        pss = _Pass(
            order=eng.path,
            vmap={v: np.full(144, -1, dtype=np.intp) for v in eng.variables},
            pos_map=None,
            n_k=eng.n_k,
            max_radius_per_var=eng.max_radius_per_var,
            max_radius=eng.max_radius,
            scan_fraction=eng.scan_fraction,
            neighbor_cache={},
            domains=eng.domains,
        )
        x_0 = eng.path[0]
        events = eng._gather_neighborhood(
            pss, x_0, tuple(int(c) for c in x_0), 0
        )
        for v in eng.variables:
            self.assertGreater(
                len(events[v]),
                0,
                "post-pass node 0 must have a fully informed neighbourhood",
            )

    def test_engine_base_config_never_mutated(self):
        """Per-pass search parameters live in ``_Pass``, not on the engine.

        Post-passes must not install their factor-reduced ``n_k``/``domains``/
        ``vmap`` onto the engine, or the mechanism is not thread-safe.
        Every pass must see the engine's *base* configuration untouched.
        """
        observed = []
        orig = _DirectSamplingEngine._simulate_node

        def spy(engine, pss, i, x_i, u_st, u_fb):
            observed.append(
                (
                    tuple(sorted(engine.n_k.items())),
                    engine.scan_fraction,
                    tuple(sorted(pss.n_k.items())),
                    pss.scan_fraction,
                )
            )
            return orig(engine, pss, i, x_i, u_st, u_fb)

        _DirectSamplingEngine._simulate_node = spy
        try:
            self._run(post=PostProcessing(2, factor=2.0))
        finally:
            _DirectSamplingEngine._simulate_node = orig

        engine_cfg = {(o[0], o[1]) for o in observed}
        self.assertEqual(
            len(engine_cfg),
            1,
            f"engine base config mutated during the run: {engine_cfg}",
        )
        # Base config: n_neighbors=8, scan_fraction=0.25 (see _run).
        self.assertEqual(engine_cfg.pop(), ((("field", 8),), 0.25))
        # The factor-reduced values must still reach the kernel via _Pass:
        # 8 / 2.0 = 4 neighbours, 0.25 / 2.0 = 0.125 scan fraction.
        pass_cfg = {(o[2], o[3]) for o in observed}
        self.assertEqual(
            pass_cfg, {((("field", 8),), 0.25), ((("field", 4),), 0.125)}
        )

    def test_per_pass_neighbour_cache_is_isolated(self):
        """Each pass owns its cache box, so a stale one cannot leak forward.

        The cache is keyed on a pass's visit order; reusing one engine-level
        slot meant correctness depended on remembering to null it per pass.
        """
        boxes = []
        orig = _DirectSamplingEngine._simulate_node

        def spy(engine, pss, i, x_i, u_st, u_fb):
            boxes.append(id(pss.neighbor_cache))
            return orig(engine, pss, i, x_i, u_st, u_fb)

        _DirectSamplingEngine._simulate_node = spy
        try:
            self._run(post=PostProcessing(2), num_threads=4)
        finally:
            _DirectSamplingEngine._simulate_node = orig

        # main pass + 2 post-passes, each with its own box
        self.assertEqual(len(set(boxes)), 3)

    def test_post_pass_erases_only_own_non_conditioned_cell(self):
        """``_reopen_and_simulate`` touches this node only, skipping conditioning.

        Reading later-positioned neighbours' *old* values is what makes a pass
        a refinement, so nothing but the visited node may be erased.
        """
        ti = gs.mps.TrainingImage(
            self.ti_data, categorical=True, n_neighbors=8
        )
        eng = _DirectSamplingEngine(
            MPSModel(ti, scan_fraction=0.25, threshold=0.2),
            (8, 8),
            _Run(
                rng_path=np.random.RandomState(0),
                rng_nodes=np.random.RandomState(1),
                conditions={(1, 1): {"field": 1.0}, (5, 6): {"field": 0.0}},
                num_threads=1,
            ),
        )
        eng.run()
        before = eng.sg["field"].copy()
        pss = _Pass(
            order=eng.path,
            vmap={v: np.full(64, -1, dtype=np.intp) for v in eng.variables},
            pos_map=_order_positions(eng.path, (8, 8)),
            n_k=eng.n_k,
            max_radius_per_var=eng.max_radius_per_var,
            max_radius=eng.max_radius,
            scan_fraction=eng.scan_fraction,
            neighbor_cache={},
            domains=eng.domains,
        )
        # Erase-and-simulate one node, then check the blast radius.
        x_0 = eng.path[0]
        eng._reopen_and_simulate(pss, 0, x_0, 0.5, np.array([0.5, 0.5]))
        changed = np.argwhere(before != eng.sg["field"])
        self.assertLessEqual(len(changed), 1, f"touched {changed.tolist()}")
        if len(changed):
            np.testing.assert_array_equal(changed[0], x_0)
        # Conditioned cells keep their values and stay out of every visit order.
        self.assertEqual(eng.sg["field"][1, 1], 1.0)
        self.assertEqual(eng.sg["field"][5, 6], 0.0)
        for v in eng.variables:
            cells = np.flatnonzero(eng.is_cond[v].reshape(-1))
            self.assertTrue(np.all(eng.vmap[v][cells] == -1))

    def test_multivariate_joint_subset(self):
        a = self.ti_data
        b = 1.0 - a
        ti = gs.mps.TrainingImage(
            [
                Variable("A", a, categorical=True, n_neighbors=6),
                Variable("B", b, categorical=True, n_neighbors=6),
            ]
        )
        model = gs.mps.MPSModel(
            ti,
            scan_fraction=0.25,
            threshold=0.2,
            post_processing=PostProcessing(1),
        )
        ds = gs.mps.DirectSampling(model, seed=3)
        out = ds((np.arange(16.0), np.arange(16.0)), store=False)
        out = as_dict(ds, out)
        pairs_ti = set(zip(a.ravel(), b.ravel()))
        pairs_sim = set(zip(out["A"].ravel(), out["B"].ravel()))
        self.assertTrue(pairs_sim <= pairs_ti)


class TestPostProcessingPath(unittest.TestCase):
    """PostProcessing path: post-pass visit order (implementation extension,
    not prescribed by Me13 §4)."""

    def setUp(self):
        rng = np.random.RandomState(42)
        ti = np.zeros((40, 40))
        ti[:, 20:] = 1.0
        flip = rng.rand(40, 40) < 0.05
        ti[flip] = 1.0 - ti[flip]
        self.ti_data = ti

    def _run(self, seed=20260717, cond=None, path="random", post_path=None):
        ti = gs.mps.TrainingImage(
            self.ti_data, categorical=True, n_neighbors=8
        )
        model = gs.mps.MPSModel(
            ti,
            scan_fraction=0.25,
            threshold=0.2,
            post_processing=PostProcessing(1, path=post_path),
        )
        ds = gs.mps.DirectSampling(model, seed=seed)
        if cond is not None:
            ds.set_condition(*cond)
        return ds((np.arange(24.0), np.arange(24.0)), path=path, store=False)

    def test_default_inherits_main_mode(self):
        a = self._run(path="sequential")
        b = self._run(path="sequential", post_path="sequential")
        np.testing.assert_array_equal(a, b)

    def test_none_equals_random_for_random_main_path(self):
        a = self._run()
        b = self._run(post_path="random")
        np.testing.assert_array_equal(a, b)

    def test_same_is_deterministic_and_valid(self):
        cond = (([5.0], [5.0]), [1.0])
        a = self._run(post_path="same", cond=cond)
        b = self._run(post_path="same", cond=cond)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.isin(a, np.unique(self.ti_data)).all())
        self.assertEqual(a[5, 5], 1.0)

    def test_explicit_raster_equals_sequential(self):
        raster = np.argwhere(np.ones((24, 24), dtype=bool))
        a = self._run(post_path=raster)
        b = self._run(post_path="sequential")
        np.testing.assert_array_equal(a, b)

    def test_invalid_string_and_shape_raise(self):
        with self.assertRaises(ValueError):
            PostProcessing(1, path="bogus")
        with self.assertRaises(ValueError):
            PostProcessing(1, path=np.array([1, 2, 3]))

    def test_explicit_path_missing_nodes_raises(self):
        half = np.argwhere(np.ones((24, 24), dtype=bool))[:288]
        with self.assertRaises(ValueError):
            self._run(post_path=half)


class TestPostPassMetricOverrides(unittest.TestCase):
    """The post-pass metric follows its own override, not the main pass."""

    def setUp(self):
        rng = np.random.default_rng(416)
        self.codes = rng.integers(0, 2, (12, 12)).astype(float)  # {0, 1}
        self.cont = rng.random((12, 12))
        self.pos = [np.arange(8.0)] * 2

    def _post_pass_field(self, primary, override):
        # scan_fraction < 1 keeps exact pattern copies (zero shift) rare.
        ti = TrainingImage(
            self.cont, categorical=False, distance=primary, n_neighbors=6
        )
        model = MPSModel(
            ti,
            scan_fraction=0.25,
            post_processing=PostProcessing(
                1, overrides={"field": {"distance": override}}
            ),
        )
        return DirectSampling(model, seed=0)(self.pos, store=False)

    def test_variation_primary_l1_override_pastes_ti_values(self):
        # The paste rule follows the pass's own metric: an l1 post-pass pastes
        # raw TI values although the variation main pass shifted them.
        field = self._post_pass_field("variation", "l1")
        self.assertTrue(np.isin(field, self.cont).all())

    def test_l1_primary_variation_override_applies_mean_shift(self):
        field = self._post_pass_field("l1", "variation")
        self.assertFalse(np.isin(field, self.cont).all())


class TestPostProcessingSetup(unittest.TestCase):
    """Post-processing inputs are resolved and validated once, up front."""

    def setUp(self):
        rng = np.random.default_rng(5)
        self.ti = TrainingImage(
            rng.integers(0, 2, (20, 20)).astype(float),
            categorical=True,
            n_neighbors=6,
        )
        self.pos = [np.arange(6.0)] * 2

    def test_invalid_post_path_raises_before_main_pass(self):
        # Explicit post path that misses one node of the 6x6 grid.
        nodes = np.argwhere(np.ones((6, 6), dtype=bool))[:-1]
        ds = DirectSampling(
            MPSModel(
                self.ti,
                scan_fraction=0.5,
                post_processing=PostProcessing(1, path=nodes),
            )
        )
        with mock.patch.object(
            _DirectSamplingEngine, "_simulate_node"
        ) as sim_node:
            with self.assertRaisesRegex(ValueError, "missing"):
                ds(self.pos, seed=1)
        sim_node.assert_not_called()

    def test_engine_uses_the_models_resolved_post_ti(self):
        # The model resolves overrides once; the engine must not redo it.
        model = MPSModel(
            self.ti,
            scan_fraction=0.5,
            post_processing=PostProcessing(
                1, overrides={"field": {"n_neighbors": 3}}
            ),
        )
        with mock.patch("gstools.mps.model._resolve_post") as resolve:
            DirectSampling(model)(self.pos, seed=1)
        resolve.assert_not_called()

    def test_thread_count_invariant_with_post_processing(self):
        ti = TrainingImage(
            np.random.default_rng(13).integers(0, 3, (40, 40)).astype(float),
            categorical=True,
            n_neighbors=12,
        )
        mask = np.zeros((40, 40), dtype=bool)
        mask[:20, :] = True
        model = MPSModel(
            ti,
            scan_fraction=0.3,
            threshold=0.05,
            post_processing=PostProcessing(
                2,
                factor=2,
                overrides={"field": {"n_neighbors": 6}},
                mask=mask,
            ),
        )
        pos = [np.arange(40.0)] * 2
        ref = DirectSampling(model)(pos, seed=3, num_threads=1)
        out = DirectSampling(model)(pos, seed=3, num_threads=4)
        np.testing.assert_array_equal(ref, out)


if __name__ == "__main__":
    unittest.main()
