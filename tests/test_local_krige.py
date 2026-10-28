"""
This is the unittest of the LocalKrige prototype (moving-neighborhood
kriging, see gstools.krige.local). It mirrors test_krige.py, using the
Local* counterparts of the Krige convenience classes.
"""

import unittest

import numpy as np

import gstools as gs


def trend(*xyz):
    return xyz[0]


class TestLocalKrige(unittest.TestCase):
    def setUp(self):
        # GSTools-Core's local kriging currently only supports these
        # covariance models (see LocalKrige._cov_model_to_json / the
        # Rust CovModelSpec) -- unlike test_krige.py, no Spherical here
        self.cov_models = [gs.Gaussian, gs.Exponential]
        self.dims = range(1, 4)
        self.data = np.array(
            [
                [0.3, 1.2, 0.5, 0.47],
                [1.9, 0.6, 1.0, 0.56],
                [1.1, 3.2, 1.5, 0.74],
                [3.3, 4.4, 2.0, 1.47],
                [4.7, 3.8, 2.5, 1.74],
            ]
        )
        # x, y, z components for the condition position
        self.cond_pos = (self.data[:, 0], self.data[:, 1], self.data[:, 2])
        # condition values
        self.cond_val = self.data[:, 3]
        self.cond_err = np.array([0.01, 0.0, 0.1, 0.05, 0])
        # the arithmetic mean of the conditions
        self.mean = np.mean(self.cond_val)
        # a moderate grid: unlike global kriging, local kriging solves one
        # small system per target point, and (for now) does so
        # sequentially (see GSTools-Core's local_krige.rs) -- the
        # 51x61x71 grid test_krige.py uses would take tens of seconds
        # per call here, so this is deliberately much smaller
        self.x = np.linspace(0, 5, 11)
        self.y = np.linspace(0, 6, 13)
        self.z = np.linspace(0, 7, 15)
        self.pos = (self.x, self.y, self.z)
        self.grids = [self.x]
        self.grids.append(np.meshgrid(self.x, self.y, indexing="ij"))
        self.grids.append(np.meshgrid(self.x, self.y, self.z, indexing="ij"))
        self.grid_shape = [11, 13, 15]
        # a radius spanning the whole domain: every conditioning point is
        # then a "neighbor" of every target point, so local kriging
        # should reproduce the corresponding global kriging result (up
        # to solver round-off, since the local solver does a direct LU
        # solve instead of a pseudo-inverse)
        self.big_radius = 1e3

    def _check(self, local_krige, global_krige, dim):
        """Shared checks for a Local* class against its global counterpart."""
        # structured and unstructured evaluation have to agree
        field_1, __ = local_krige.unstructured(self.grids[dim - 1])
        field_1 = field_1.reshape(self.grid_shape[:dim])
        field_2, __ = local_krige.structured(self.pos[:dim])
        self.assertAlmostEqual(
            np.max(np.abs(field_1 - field_2)), 0.0, places=2
        )
        # with a radius spanning the whole domain, local kriging has to
        # match global kriging
        field_g, __ = global_krige.unstructured(self.grids[dim - 1])
        field_g = field_g.reshape(self.grid_shape[:dim])
        np.testing.assert_allclose(field_1, field_g, atol=1e-6)
        # both reproduce the conditioning values themselves
        field_cond, __ = local_krige.unstructured(self.cond_pos[:dim])
        for i, val in enumerate(self.cond_val):
            self.assertAlmostEqual(field_cond[i], val, places=2)

    def test_simple(self):
        for Model in self.cov_models:
            for dim in self.dims:
                model = Model(
                    dim=dim,
                    var=2,
                    len_scale=2,
                    anis=[0.9, 0.8],
                    angles=[2, 1, 0.5],
                )
                simple = gs.krige.Simple(
                    model, self.cond_pos[:dim], self.cond_val, self.mean
                )
                local_simple = gs.krige.LocalSimple(
                    model,
                    self.cond_pos[:dim],
                    self.cond_val,
                    self.big_radius,
                    self.mean,
                )
                self._check(local_simple, simple, dim)

    def test_ordinary(self):
        for trend_func in [None, trend]:
            for Model in self.cov_models:
                for dim in self.dims:
                    model = Model(
                        dim=dim,
                        var=5,
                        len_scale=10,
                        anis=[0.9, 0.8],
                        angles=[2, 1, 0.5],
                    )
                    ordinary = gs.krige.Ordinary(
                        model,
                        self.cond_pos[:dim],
                        self.cond_val,
                        trend=trend_func,
                    )
                    local_ordinary = gs.krige.LocalOrdinary(
                        model,
                        self.cond_pos[:dim],
                        self.cond_val,
                        self.big_radius,
                        trend=trend_func,
                    )
                    self._check(local_ordinary, ordinary, dim)

    def test_universal(self):
        # "quadratic" -> too few conditioning points
        for drift in ["linear", 0, 1, trend]:
            for Model in self.cov_models:
                for dim in self.dims:
                    model = Model(
                        dim=dim,
                        var=2,
                        len_scale=10,
                        anis=[0.9, 0.8],
                        angles=[2, 1, 0.5],
                    )
                    universal = gs.krige.Universal(
                        model, self.cond_pos[:dim], self.cond_val, drift
                    )
                    local_universal = gs.krige.LocalUniversal(
                        model,
                        self.cond_pos[:dim],
                        self.cond_val,
                        self.big_radius,
                        drift,
                    )
                    self._check(local_universal, universal, dim)

    def test_detrended(self):
        for Model in self.cov_models:
            for dim in self.dims:
                model = Model(
                    dim=dim,
                    var=2,
                    len_scale=10,
                    anis=[0.5, 0.2],
                    angles=[0.4, 0.2, 0.1],
                )
                detrended = gs.krige.Detrended(
                    model, self.cond_pos[:dim], self.cond_val, trend
                )
                local_detrended = gs.krige.LocalDetrended(
                    model,
                    self.cond_pos[:dim],
                    self.cond_val,
                    self.big_radius,
                    trend,
                )
                self._check(local_detrended, detrended, dim)

    def test_extdrift(self):
        rng = np.random.RandomState(42)
        for dim in self.dims:
            cond_drift = rng.normal(size=len(self.cond_val))
            target_drift = rng.normal(size=int(np.prod(self.grid_shape[:dim])))
            for Model in self.cov_models:
                model = Model(
                    dim=dim,
                    var=2,
                    len_scale=10,
                    anis=[0.5, 0.2],
                    angles=[0.4, 0.2, 0.1],
                )
                extdrift = gs.krige.ExtDrift(
                    model, self.cond_pos[:dim], self.cond_val, cond_drift
                )
                local_extdrift = gs.krige.LocalExtDrift(
                    model,
                    self.cond_pos[:dim],
                    self.cond_val,
                    self.big_radius,
                    cond_drift,
                )
                field_1, __ = local_extdrift.unstructured(
                    self.grids[dim - 1], ext_drift=target_drift
                )
                field_1 = field_1.reshape(self.grid_shape[:dim])
                field_2, __ = local_extdrift.structured(
                    self.pos[:dim], ext_drift=target_drift
                )
                self.assertAlmostEqual(
                    np.max(np.abs(field_1 - field_2)), 0.0, places=2
                )
                field_g, __ = extdrift.unstructured(
                    self.grids[dim - 1], ext_drift=target_drift
                )
                field_g = field_g.reshape(self.grid_shape[:dim])
                np.testing.assert_allclose(field_1, field_g, atol=1e-6)
                field_cond, __ = local_extdrift.unstructured(
                    self.cond_pos[:dim], ext_drift=cond_drift
                )
                for i, val in enumerate(self.cond_val):
                    self.assertAlmostEqual(field_cond[i], val, places=2)

    def test_error(self):
        for Model in self.cov_models:
            for dim in self.dims:
                model = Model(
                    dim=dim,
                    var=5,
                    len_scale=10,
                    nugget=0.1,
                    anis=[0.9, 0.8],
                    angles=[2, 1, 0.5],
                )
                ordinary = gs.krige.LocalOrdinary(
                    model,
                    self.cond_pos[:dim],
                    self.cond_val,
                    self.big_radius,
                    exact=False,
                    cond_err=self.cond_err,
                )
                __, err = ordinary(self.cond_pos[:dim])
                # when the given measurement error is 0, the kriging-var
                # should equal the nugget of the model
                self.assertAlmostEqual(err[1], model.nugget, places=2)
                self.assertAlmostEqual(err[4], model.nugget, places=2)

    def test_raise(self):
        # no cond_pos/cond_val given
        self.assertRaises(
            ValueError, gs.krige.LocalKrige, gs.Stable(), None, None, 1.0
        )
        # local_radius has to be strictly positive
        self.assertRaises(
            ValueError,
            gs.krige.LocalKrige,
            gs.Gaussian(),
            self.cond_pos,
            self.cond_val,
            0,
        )
        self.assertRaises(
            ValueError,
            gs.krige.LocalKrige,
            gs.Gaussian(),
            self.cond_pos,
            self.cond_val,
            -1.0,
        )

    def test_no_neighbors_raises(self):
        # a target point far outside of local_radius has no neighbors
        # at all -- this has to be reported, not silently mis-evaluated
        model = gs.Gaussian(dim=2, var=1, len_scale=1)
        local = gs.krige.LocalOrdinary(
            model, self.cond_pos[:2], self.cond_val, 1e-3
        )
        self.assertRaises(ValueError, local, ([100.0], [100.0]))

    def test_smaller_radius_gives_similar_result(self):
        # a denser, random conditioning setup: with a small radius only a
        # few neighbors enter each local system, so the result differs
        # from the global one -- but that difference should shrink
        # monotonically as the radius grows, and vanish once the radius
        # spans the whole domain
        rng = np.random.RandomState(2)
        cond_pos = (rng.uniform(0, 20, 60), rng.uniform(0, 20, 60))
        cond_val = rng.normal(size=60) + 0.05 * cond_pos[0]
        target_pos = (rng.uniform(2, 18, 25), rng.uniform(2, 18, 25))

        model = gs.Gaussian(dim=2, var=1.5, len_scale=4, nugget=0.05)
        ordinary = gs.krige.Ordinary(model, cond_pos, cond_val)
        field_g, __ = ordinary.unstructured(target_pos)

        prev_diff = np.inf
        for radius in [5.0, 8.0, 12.0, self.big_radius]:
            local = gs.krige.LocalOrdinary(model, cond_pos, cond_val, radius)
            field_l, __ = local.unstructured(target_pos)
            diff = np.max(np.abs(field_l - field_g))
            corr = np.corrcoef(field_l, field_g)[0, 1]
            # results should get closer to the global one (or at least
            # not get worse) as the radius grows
            self.assertLessEqual(diff, prev_diff + 1e-8)
            self.assertGreater(corr, 0.9)
            prev_diff = diff
        # with a radius spanning the whole domain, it should match closely
        self.assertAlmostEqual(prev_diff, 0.0, places=4)

    def test_non_finite_cond_pos_raises(self):
        model = gs.Gaussian(dim=2, var=1, len_scale=1)
        cond_pos = (self.cond_pos[0].copy(), self.cond_pos[1])
        cond_pos[0][2] = np.nan
        self.assertRaises(
            ValueError,
            gs.krige.LocalOrdinary,
            model,
            cond_pos,
            self.cond_val,
            1.0,
        )


class TestLocalKrigeMulti(unittest.TestCase):
    """Multiple fields at once: 2D cond_val with shape (T, cond_no)."""

    def setUp(self):
        rng = np.random.RandomState(5)
        self.n, self.t = 40, 6
        self.cond_pos = tuple(rng.uniform(0, 10, (2, self.n)))
        self.cond_val = rng.normal(size=(self.t, self.n)) + 2.0
        self.cond_val[rng.uniform(size=self.cond_val.shape) < 0.2] = np.nan
        self.cond_drift = rng.normal(size=self.n)
        self.grid = (np.linspace(1, 9, 6), np.linspace(1, 9, 5))
        self.grid_drift = rng.normal(size=(6, 5))
        self.points = tuple(rng.uniform(1, 9, (2, 15)))
        # every target has neighbors in every field, so the 1D reference
        # (which raises for targets without neighbors) always works
        self.radius = 6.0
        self.model = gs.Exponential(dim=2, var=1.5, len_scale=3, nugget=0.1)

    def _krige(self, cls, cond_val, **kwargs):
        # 1D cond_val drops points with NaN values, the external drift at
        # the conditions has to match the kept points
        drift = self.cond_drift
        if np.ndim(cond_val) == 1:
            drift = drift[np.isfinite(cond_val)]
        args = {
            gs.krige.LocalUniversal: ("linear",),
            gs.krige.LocalExtDrift: (drift,),
            gs.krige.LocalDetrended: (trend,),
        }.get(cls, ())
        return cls(
            self.model, self.cond_pos, cond_val, self.radius, *args, **kwargs
        )

    def _assert_matches_single(self, cls, **kwargs):
        multi = self._krige(cls, self.cond_val, **kwargs)
        self.assertTrue(multi.is_multi)
        self.assertEqual(multi.cond_no, self.n)
        call_kw = (
            {"ext_drift": self.grid_drift}
            if cls is gs.krige.LocalExtDrift
            else {}
        )
        field, var = multi.structured(self.grid, **call_kw)
        self.assertEqual(field.shape, (self.t, 6, 5))
        self.assertEqual(var.shape, (self.t, 6, 5))
        self.assertEqual(multi.field.shape, (self.t, 6, 5))
        self.assertEqual(multi.n_used.shape, (self.t, 6, 5))
        for t in range(self.t):
            single = self._krige(cls, self.cond_val[t], **kwargs)
            self.assertFalse(single.is_multi)
            field_1, var_1 = single.structured(self.grid, **call_kw)
            np.testing.assert_allclose(field[t], field_1, atol=1e-10)
            np.testing.assert_allclose(var[t], var_1, atol=1e-10)

    def test_matches_single_field_calls(self):
        for cls in (
            gs.krige.LocalOrdinary,
            gs.krige.LocalUniversal,
            gs.krige.LocalExtDrift,
            gs.krige.LocalDetrended,
        ):
            with self.subTest(cls=cls.__name__):
                self._assert_matches_single(cls)
        with self.subTest(cls="LocalSimple"):
            self._assert_matches_single(gs.krige.LocalSimple, mean=2.0)
        with self.subTest(normalizer="LogNormal"):
            self.cond_val = np.exp(self.cond_val)  # LogNormal needs values > 0
            self._assert_matches_single(
                gs.krige.LocalOrdinary, normalizer=gs.normalizer.LogNormal()
            )

    def test_unstructured_and_single_row(self):
        multi = self._krige(gs.krige.LocalOrdinary, self.cond_val)
        field = multi.unstructured(self.points, return_var=False)
        self.assertEqual(field.shape, (self.t, 15))
        # T = 1 is still the multi path and equals the 1D call
        one = self._krige(gs.krige.LocalOrdinary, self.cond_val[2:3])
        single = self._krige(gs.krige.LocalOrdinary, self.cond_val[2])
        field_one = one.unstructured(self.points, return_var=False)
        field_single = single.unstructured(self.points, return_var=False)
        self.assertEqual(field_one.shape, (1, 15))
        np.testing.assert_allclose(field_one[0], field_single, atol=1e-10)
        np.testing.assert_allclose(field[2], field_single, atol=1e-10)

    def test_n_used_and_missing_neighbors(self):
        multi = self._krige(gs.krige.LocalOrdinary, self.cond_val)
        targets = (
            np.append(self.points[0], 100.0),
            np.append(self.points[1], 100.0),
        )
        field = multi.unstructured(targets, return_var=False)
        # brute force count of available neighbors
        dist = np.hypot(
            targets[0][:, None] - self.cond_pos[0],
            targets[1][:, None] - self.cond_pos[1],
        )
        avail = np.isfinite(self.cond_val)
        expected = ((dist <= self.radius)[None] & avail[:, None]).sum(axis=2)
        np.testing.assert_array_equal(multi.n_used, expected)
        # the far away target: NaN instead of an error
        self.assertTrue(np.all(np.isnan(field[:, -1])))
        self.assertTrue(np.all(np.isfinite(field[:, :-1])))
        # min_neighbors: too few -> NaN
        min_nb = int(np.median(expected))
        field = multi.unstructured(
            targets, return_var=False, min_neighbors=min_nb
        )
        np.testing.assert_array_equal(np.isnan(field), expected < min_nb)

    def test_cache_modes_give_same_result(self):
        multi = self._krige(gs.krige.LocalExtDrift, self.cond_val)
        res = {}
        for cache in ("none", "last", "hash"):
            res[cache] = multi.structured(
                self.grid, ext_drift=self.grid_drift, cache=cache
            )
            self.assertIn("n_solves", multi.multi_stats)
        for cache in ("last", "hash"):
            np.testing.assert_array_equal(res[cache][0], res["none"][0])
            np.testing.assert_array_equal(res[cache][1], res["none"][1])

    def test_set_condition_switches_between_1d_and_2d(self):
        krige = self._krige(gs.krige.LocalOrdinary, self.cond_val)
        krige.set_condition(self.cond_pos, self.cond_val[0])
        self.assertFalse(krige.is_multi)
        self.assertEqual(krige.unstructured(self.points)[0].shape, (15,))
        self.assertIsNone(krige.n_used)
        krige.set_condition(self.cond_pos, self.cond_val)
        self.assertTrue(krige.is_multi)
        self.assertEqual(
            krige.unstructured(self.points)[0].shape, (self.t, 15)
        )

    def test_raise(self):
        cls = gs.krige.LocalOrdinary
        # wrong orientation (cond_no, T) and too many dimensions
        self.assertRaises(ValueError, self._krige, cls, self.cond_val.T)
        self.assertRaises(ValueError, self._krige, cls, self.cond_val[None])
        # non-finite positions
        cond_pos = (self.cond_pos[0].copy(), self.cond_pos[1])
        cond_pos[0][0] = np.inf
        self.assertRaises(
            ValueError, cls, self.model, cond_pos, self.cond_val, self.radius
        )
        # fitting is not available for multiple fields
        self.assertRaises(
            NotImplementedError,
            self._krige,
            cls,
            self.cond_val,
            fit_variogram=True,
        )
        self.assertRaises(
            NotImplementedError,
            self._krige,
            cls,
            self.cond_val,
            fit_normalizer=True,
        )
        # CondSRF needs a single field
        srf = gs.CondSRF(self._krige(cls, self.cond_val))
        self.assertRaises(ValueError, srf, self.points)


if __name__ == "__main__":
    unittest.main()
