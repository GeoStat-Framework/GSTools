"""Tests for gstools.mps.zone."""

import unittest

import numpy as np
from mps_fixtures import ignoring

import gstools as gs
from gstools.mps.model import MPSModel
from gstools.mps.simulate import _DirectSamplingEngine, _Run
from gstools.mps.training_image import Variable


class TestZoneConstruction(unittest.TestCase):
    def _ti(self):
        return gs.mps.TrainingImage(
            np.random.default_rng(0).integers(0, 2, (10, 10))
        )

    def test_zone_holds_ti_and_where(self):
        ti = self._ti()
        mask = np.zeros((5, 5), dtype=bool)
        z = gs.mps.Zone(ti, where=mask)
        self.assertIs(z.ti, ti)
        np.testing.assert_array_equal(z.where, mask)

    def test_non_ti_raises(self):
        with self.assertRaises(TypeError):
            gs.mps.Zone(np.zeros((5, 5)), where=np.ones((5, 5), dtype=bool))

    def test_non_boolean_where_raises(self):
        # No silent != 0 coercion of a non-boolean mask.
        with self.assertRaisesRegex(ValueError, "bool"):
            gs.mps.Zone(self._ti(), where=np.ones((5, 5), dtype=int))

    def test_callable_where_accepted(self):
        z = gs.mps.Zone(self._ti(), where=lambda x, y: x > 2)
        self.assertTrue(callable(z.where))


class TestZonatedSimulation(unittest.TestCase):
    """Zonated DS: per-zone TI substitution (M10 para [40])."""

    def _two_zone_setup(self, sg=10):
        rng = np.random.default_rng(0)
        ti_a = gs.mps.TrainingImage(
            rng.integers(0, 3, (15, 15)), n_neighbors=4
        )
        ti_b = gs.mps.TrainingImage(
            rng.integers(10, 13, (15, 15)), n_neighbors=4
        )
        gx, _ = np.meshgrid(np.arange(sg), np.arange(sg), indexing="ij")
        zone = gs.mps.Zone(ti_b, where=(gx >= sg // 2))
        model = MPSModel(ti_a, scan_fraction=0.5, zones=[zone])
        pos = [np.arange(sg, dtype=float)] * 2
        return model, pos, sg

    def test_per_zone_subset_property(self):
        # Disjoint TI value sets -> values in zone z must be a subset of TI_z
        # values. Also covers the fallback path: fallback draws must come from
        # the selected zone TI, or rare fallbacks would leak primary values.
        model, pos, sg = self._two_zone_setup()
        field = gs.mps.DirectSampling(model)(pos, seed=0)
        self.assertTrue(np.all(np.isin(field[: sg // 2], [0, 1, 2])))
        self.assertTrue(np.all(np.isin(field[sg // 2 :], [10, 11, 12])))

    def test_zone_thread_count_determinism(self):
        # Zones + rotation, varying num_threads, must produce identical output.
        rng = np.random.default_rng(1)
        ti_a = gs.mps.TrainingImage(
            rng.integers(0, 3, (15, 15)), n_neighbors=4
        )
        ti_b = gs.mps.TrainingImage(
            rng.integers(10, 13, (15, 15)), n_neighbors=4
        )
        gx, _ = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
        model = MPSModel(
            ti_a,
            scan_fraction=0.5,
            rotation=np.pi / 6,
            zones=[gs.mps.Zone(ti_b, where=(gx >= 4))],
        )
        pos = [np.arange(8, dtype=float)] * 2
        with ignoring(RuntimeWarning):  # lag collapse under the rotation
            f1 = gs.mps.DirectSampling(model)(pos, seed=5, num_threads=1)
            f4 = gs.mps.DirectSampling(model)(pos, seed=5, num_threads=4)
        np.testing.assert_array_equal(f1, f4)

    def test_overlapping_zones_raise_at_call(self):
        # Overlap must be detected at resolve time, inside __call__.
        rng = np.random.default_rng(0)
        ti = gs.mps.TrainingImage(rng.integers(0, 2, (10, 10)))
        m1 = np.zeros((6, 6), dtype=bool)
        m1[:3] = True
        m2 = np.zeros((6, 6), dtype=bool)
        m2[2:] = True
        model = MPSModel(
            ti, zones=[gs.mps.Zone(ti, where=m1), gs.mps.Zone(ti, where=m2)]
        )
        ds = gs.mps.DirectSampling(model)
        with self.assertRaisesRegex(ValueError, "overlap"):
            ds([np.arange(6, dtype=float)] * 2, seed=0)

    def test_per_zone_d_max(self):
        # Distances in each zone must be normalized by that zone TI's d_max.
        # White-box: the zone domain's scan config carries the zone's range.
        from gstools.random.rng import RNG

        rng = np.random.default_rng(0)
        ti_narrow = gs.mps.TrainingImage(
            rng.random((10, 10)), categorical=False
        )  # range ~1
        ti_wide = gs.mps.TrainingImage(
            rng.random((10, 10)) * 100.0, categorical=False
        )  # range ~100
        sel = np.zeros((6, 6), dtype=np.intp)
        sel[3:] = 1
        engine = _DirectSamplingEngine(
            MPSModel(
                ti_narrow,
                threshold=0.1,
                scan_fraction=1.0,
                zones=[gs.mps.Zone(ti_wide, sel == 1)],
            ),
            (6, 6),
            _Run(
                rng_path=RNG(0).random,
                rng_nodes=RNG(1).random,
                zone_selector=sel,
            ),
        )
        d0 = engine.domains[0].d_max["field"]
        d1 = engine.domains[1].d_max["field"]
        self.assertAlmostEqual(d0, ti_narrow.variable().d_max)
        self.assertAlmostEqual(d1, ti_wide.variable().d_max)
        self.assertGreater(d1, 10 * d0)

    def test_window_zone_d_max_from_slice(self):
        # Sub-region zone: d_max comes from the slice, not the parent TI.
        data = np.zeros((10, 10))
        data[0, 0] = 100.0
        data[5:, 5:] = np.arange(25).reshape(5, 5)
        ti = gs.mps.TrainingImage(data, categorical=False)
        from gstools.random.rng import RNG

        sel = np.zeros((4, 4), dtype=np.intp)
        sel[2:] = 1
        engine = _DirectSamplingEngine(
            MPSModel(
                ti,
                threshold=0.1,
                scan_fraction=1.0,
                zones=[gs.mps.Zone(ti.window(np.s_[5:10, 5:10]), sel == 1)],
            ),
            (4, 4),
            _Run(
                rng_path=RNG(0).random,
                rng_nodes=RNG(1).random,
                zone_selector=sel,
            ),
        )
        self.assertEqual(engine.domains[1].d_max["field"], 24.0)

    def test_composite_zone_ti_joint_copy(self):
        # Composite multivariate zone TI with a swapped A-channel.
        # A-values in the zone ⊆ composite A values; B-values ⊆ composite B
        # values; every zone (a, b) pair exists co-located in the composite
        # (the whole TI-cell vector is copied — joint-scan invariant).
        rng = np.random.default_rng(3)
        prim_a = rng.integers(0, 3, (15, 15))
        prim_b = rng.integers(0, 2, (15, 15))
        comp_a = rng.integers(5, 8, (15, 15))
        comp_b = rng.integers(2, 4, (15, 15))
        primary = gs.mps.TrainingImage(
            [
                Variable("a", prim_a, n_neighbors=4),
                Variable("b", prim_b, n_neighbors=4),
            ]
        )
        composite = gs.mps.TrainingImage(
            [
                Variable("a", comp_a, n_neighbors=4),
                Variable("b", comp_b, n_neighbors=4),
            ]
        )
        gx, _ = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
        model = MPSModel(
            primary,
            scan_fraction=0.5,
            zones=[gs.mps.Zone(composite, where=(gx >= 4))],
        )
        out = gs.mps.DirectSampling(model)(
            [np.arange(8, dtype=float)] * 2, seed=0
        )
        za, zb = out[0][4:], out[1][4:]
        self.assertTrue(np.all(np.isin(za, [5, 6, 7])))
        self.assertTrue(np.all(np.isin(zb, [2, 3])))
        zone_pairs = set(zip(za.ravel().tolist(), zb.ravel().tolist()))
        ti_pairs = set(zip(comp_a.ravel().tolist(), comp_b.ravel().tolist()))
        self.assertTrue(zone_pairs <= ti_pairs)
        self.assertTrue(np.all(np.isin(out[0][:4], [0, 1, 2])))

    def test_zone_with_conditioning(self):
        # Conditioning honored exactly inside a zone.
        model, pos, sg = self._two_zone_setup()
        ds = gs.mps.DirectSampling(model)
        ds.set_condition([[7.0], [7.0]], [11])
        field = ds(pos, seed=0)
        self.assertEqual(int(field[7, 7]), 11)

    def test_no_zones_unchanged(self):
        # zones=[] must be bit-identical to the pre-zonation engine.
        rng = np.random.default_rng(0)
        ti = gs.mps.TrainingImage(rng.integers(0, 3, (15, 15)), n_neighbors=4)
        pos = [np.arange(8, dtype=float)] * 2
        f_plain = gs.mps.DirectSampling(MPSModel(ti, scan_fraction=0.5))(
            pos, seed=2
        )
        f_empty = gs.mps.DirectSampling(
            MPSModel(ti, scan_fraction=0.5, zones=[])
        )(pos, seed=2)
        np.testing.assert_array_equal(f_plain, f_empty)


if __name__ == "__main__":
    unittest.main()
