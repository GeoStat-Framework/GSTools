# -*- coding: utf-8 -*-
"""
This is the unittest of CovModel class.
"""

import numpy as np
import unittest
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


if __name__ == "__main__":
    unittest.main()
