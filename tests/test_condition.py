"""This is the unittest of CondSRF class."""

import unittest
from copy import copy

import numpy as np

import gstools as gs


class TestCondition(unittest.TestCase):
    def setUp(self):
        self.cov_models = [
            gs.Gaussian,
            gs.Exponential,
        ]
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
        self.cond_pos = (self.data[:, 0], self.data[:, 1], self.data[:, 2])
        self.cond_val = self.data[:, 3]
        self.mean = np.mean(self.cond_val)
        grid = np.linspace(5, 20, 10)
        self.grid_x = np.concatenate((self.cond_pos[0], grid))
        self.grid_y = np.concatenate((self.cond_pos[1], grid))
        self.grid_z = np.concatenate((self.cond_pos[2], grid))
        self.pos = (self.grid_x, self.grid_y, self.grid_z)

    def test_simple(self):
        for Model in self.cov_models:
            model = Model(
                dim=1, var=0.5, len_scale=2, anis=[0.1, 1], angles=[0.5, 0, 0]
            )
            krige = gs.krige.Simple(
                model, self.cond_pos[0], self.cond_val, self.mean
            )
            crf = gs.CondSRF(krige, seed=19970221)
            field_1 = crf.unstructured(self.pos[0])
            field_2 = crf.structured(self.pos[0])
            for i, val in enumerate(self.cond_val):
                self.assertAlmostEqual(val, field_1[i], places=2)
                self.assertAlmostEqual(val, field_2[(i,)], places=2)

            for dim in self.dims[1:]:
                model = Model(
                    dim=dim,
                    var=0.5,
                    len_scale=2,
                    anis=[0.1, 1],
                    angles=[0.5, 0, 0],
                )
                krige = gs.krige.Simple(
                    model, self.cond_pos[:dim], self.cond_val, self.mean
                )
                crf = gs.CondSRF(krige, seed=19970221)
                field_1 = crf.unstructured(self.pos[:dim])
                field_2 = crf.structured(self.pos[:dim])
                # check reuse
                raw_kr2 = copy(crf["raw_krige"])
                crf(seed=19970222)
                self.assertTrue(np.allclose(raw_kr2, crf["raw_krige"]))
                for i, val in enumerate(self.cond_val):
                    self.assertAlmostEqual(val, field_1[i], places=2)
                    self.assertAlmostEqual(val, field_2[dim * (i,)], places=2)

    def test_ordinary(self):
        for Model in self.cov_models:
            model = Model(
                dim=1, var=0.5, len_scale=2, anis=[0.1, 1], angles=[0.5, 0, 0]
            )
            krige = gs.krige.Ordinary(model, self.cond_pos[0], self.cond_val)
            crf = gs.CondSRF(krige, seed=19970221)
            field_1 = crf.unstructured(self.pos[0])
            field_2 = crf.structured(self.pos[0])
            for i, val in enumerate(self.cond_val):
                self.assertAlmostEqual(val, field_1[i], places=2)
                self.assertAlmostEqual(val, field_2[(i,)], places=2)

            for dim in self.dims[1:]:
                model = Model(
                    dim=dim,
                    var=0.5,
                    len_scale=2,
                    anis=[0.1, 1],
                    angles=[0.5, 0, 0],
                )
                krige = gs.krige.Ordinary(
                    model, self.cond_pos[:dim], self.cond_val
                )
                crf = gs.CondSRF(krige, seed=19970221)
                field_1 = crf.unstructured(self.pos[:dim])
                field_2 = crf.structured(self.pos[:dim])
                for i, val in enumerate(self.cond_val):
                    self.assertAlmostEqual(val, field_1[i], places=2)
                    self.assertAlmostEqual(val, field_2[dim * (i,)], places=2)

    def test_raise_error(self):
        self.assertRaises(ValueError, gs.CondSRF, gs.Gaussian())
        krige = gs.krige.Ordinary(gs.Stable(), self.cond_pos, self.cond_val)
        self.assertRaises(ValueError, gs.CondSRF, krige, generator="unknown")

    def test_nugget(self):
        model = gs.Gaussian(
            nugget=0.01,
            var=0.5,
            len_scale=2,
            anis=[0.1, 1],
            angles=[0.5, 0, 0],
        )
        krige = gs.krige.Ordinary(
            model, self.cond_pos, self.cond_val, exact=True
        )
        crf = gs.CondSRF(krige, seed=19970221)
        field_1 = crf.unstructured(self.pos)
        field_2 = crf.structured(self.pos)
        for i, val in enumerate(self.cond_val):
            self.assertAlmostEqual(val, field_1[i], places=2)
            self.assertAlmostEqual(val, field_2[3 * (i,)], places=2)

    def test_setter(self):
        krige1 = gs.krige.Krige(gs.Exponential(), self.cond_pos, self.cond_val)
        krige2 = gs.krige.Krige(
            gs.Gaussian(var=2),
            self.cond_pos,
            self.cond_val,
            mean=-1,
            trend=-2,
            normalizer=gs.normalizer.YeoJohnson(),
        )
        crf1 = gs.CondSRF(krige1)
        crf2 = gs.CondSRF(krige2, seed=19970221)
        # update settings
        crf1.model = gs.Gaussian(var=2)
        crf1.mean = -1
        crf1.trend = -2
        # also checking correctly setting uninitialized normalizer
        crf1.normalizer = gs.normalizer.YeoJohnson
        # check if setting went right
        self.assertTrue(crf1.model == crf2.model)
        self.assertTrue(crf1.normalizer == crf2.normalizer)
        self.assertAlmostEqual(crf1.mean, crf2.mean)
        self.assertAlmostEqual(crf1.trend, crf2.trend)
        # reset kriging
        crf1.krige.set_condition()
        # compare fields
        field1 = crf1(self.pos, seed=19970221)
        field2 = crf2(self.pos)
        self.assertTrue(np.all(np.isclose(field1, field2)))

    def test_krige_raw(self):
        model = gs.Gaussian(dim=1, var=1.0, len_scale=2.0)
        for unbiased in (False, True):
            krige = gs.Krige(
                model, self.cond_pos[0], self.cond_val, unbiased=unbiased
            )
            ref = krige(self.pos[0], post_process=False, store=False)[0]
            raw = krige.krige_raw(krige._krige_cond[: krige.cond_no])
            np.testing.assert_allclose(raw, ref, rtol=1e-12, atol=1e-12)
        self.assertRaises(ValueError, krige.krige_raw, [1.0, 2.0])

    def test_cond_method_error_exact(self):
        model = gs.Gaussian(dim=3, var=0.5, len_scale=2.0)
        for unbiased in (False, True):
            krige = gs.Krige(
                model,
                self.cond_pos,
                self.cond_val,
                mean=self.mean,
                unbiased=unbiased,
            )
            srf = gs.CondSRF(krige, cond_method="error", seed=19)
            self.assertEqual(srf.cond_method, "error")
            field = srf(self.pos)
            np.testing.assert_allclose(field[:5], self.cond_val, atol=1e-10)
            # reuse of the stored kriging field gives the same result
            np.testing.assert_allclose(srf(), field)
        self.assertRaises(ValueError, gs.CondSRF, krige, cond_method="foo")

    def test_condition_update_invalidates_cached_fields(self):
        model = gs.Gaussian(dim=1)
        pos = [0.0, 1.0]
        for method in ("error", "rescale"):
            for krige_store in (True, False):
                with self.subTest(method=method, krige_store=krige_store):
                    krige = gs.Krige(model, pos, [1.0, 2.0], unbiased=False)
                    srf = gs.CondSRF(krige, cond_method=method, seed=1)
                    srf(pos, krige_store=krige_store)
                    krige.set_condition(cond_val=[10.0, 20.0])
                    self.assertEqual(krige.field_names, [])
                    # Re-populating kriging storage must not validate the
                    # conditioned field's old raw estimate.
                    krige(pos)
                    field = srf(krige_store=krige_store)
                    np.testing.assert_allclose(field, [10.0, 20.0], atol=1e-7)
                    np.testing.assert_allclose(srf(), field, atol=1e-7)

    def test_cond_method_error_covariance(self):
        model = gs.Gaussian(dim=1, var=1.0, len_scale=5.0)
        cond_pos = np.array([0.0, 3.0, 7.0, 12.0])
        krige = gs.Krige(
            model, cond_pos, [1, -0.5, 2, 0.3], mean=0.0, unbiased=False
        )
        pnt = np.array([1.5, 9.0])
        cov = model.covariance(np.abs(np.subtract.outer(cond_pos, cond_pos)))
        c_0 = model.covariance(np.abs(np.subtract.outer(cond_pos, pnt)))
        ref = model.covariance(np.abs(np.subtract.outer(pnt, pnt)))
        ref -= c_0.T @ np.linalg.solve(cov, c_0)
        emp = {}
        for method in ("error", "rescale"):
            srf = gs.CondSRF(krige, cond_method=method, mode_no=100)
            smp = [srf(pnt, seed=s, store=False) for s in range(1000)]
            emp[method] = np.cov(np.array(smp).T)
        np.testing.assert_allclose(emp["error"], ref, rtol=0.15)
        # "rescale" provably misses the conditional cross-covariance
        self.assertGreater(abs(emp["rescale"][0, 1] / ref[0, 1] - 1), 0.5)

    def test_cond_method_nugget_raises(self):
        model = gs.Gaussian(dim=3, var=0.5, len_scale=2.0, nugget=0.1)
        krige = gs.Krige(model, self.cond_pos, self.cond_val)
        srf = gs.CondSRF(krige, cond_method="error")
        self.assertRaises(NotImplementedError, srf, self.pos)


if __name__ == "__main__":
    unittest.main()
