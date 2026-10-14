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


if __name__ == "__main__":
    unittest.main()
