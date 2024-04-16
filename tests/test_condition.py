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


if __name__ == "__main__":
    unittest.main()
