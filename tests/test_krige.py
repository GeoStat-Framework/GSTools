# -*- coding: utf-8 -*-
"""
This is the unittest of the kriging module.
"""

import numpy as np
import unittest
import gstools as gs


def trend(*xyz):
    return xyz[0]


def mean_func(*xyz):
    return 2 * xyz[0]


class TestKrige(unittest.TestCase):
    def setUp(self):
        self.cov_models = [gs.Gaussian, gs.Exponential, gs.Spherical]
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
        # redundant data for pseudo-inverse
        self.p_data = np.zeros((3, 3))
        self.p_vals = np.array([1.0, 2.0, 6.0])
        self.p_meth = [1, 2, 3]  # method selector
        # indices for the date in the grid
        self.data_idx = tuple(np.array(self.data[:, :3] * 10, dtype=int).T)
        # x, y, z componentes for the conditon position
        self.cond_pos = (self.data[:, 0], self.data[:, 1], self.data[:, 2])
        # condition values
        self.cond_val = self.data[:, 3]
        self.cond_err = np.array([0.01, 0.0, 0.1, 0.05, 0])
        # the arithmetic mean of the conditions
        self.mean = np.mean(self.cond_val)
        # the grid
        self.x = np.linspace(0, 5, 51)
        self.y = np.linspace(0, 6, 61)
        self.z = np.linspace(0, 7, 71)
        self.pos = (self.x, self.y, self.z)
        self.grids = [self.x]
        self.grids.append(np.meshgrid(self.x, self.y, indexing="ij"))
        self.grids.append(np.meshgrid(self.x, self.y, self.z, indexing="ij"))
        self.grid_shape = [51, 61, 71]

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
                field_1, __ = simple.unstructured(self.grids[dim - 1])
                field_1 = field_1.reshape(self.grid_shape[:dim])
                field_2, __ = simple.structured(self.pos[:dim])
                self.assertAlmostEqual(
                    np.max(np.abs(field_1 - field_2)), 0.0, places=2
                )
                for i, val in enumerate(self.cond_val):
                    self.assertAlmostEqual(
                        field_1[self.data_idx[:dim]][i], val, places=2
                    )

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
                    field_1, __ = ordinary.unstructured(self.grids[dim - 1])
                    field_1 = field_1.reshape(self.grid_shape[:dim])
                    field_2, __ = ordinary.structured(self.pos[:dim])
                    self.assertAlmostEqual(
                        np.max(np.abs(field_1 - field_2)), 0.0, places=2
                    )
                    for i, val in enumerate(self.cond_val):
                        self.assertAlmostEqual(
                            field_1[self.data_idx[:dim]][i], val, places=2
                        )

    def test_universal(self):
        # "quad" -> to few conditional points
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
                    field_1, __ = universal.unstructured(self.grids[dim - 1])
                    field_1 = field_1.reshape(self.grid_shape[:dim])
                    field_2, __ = universal.structured(self.pos[:dim])
                    self.assertAlmostEqual(
                        np.max(np.abs(field_1 - field_2)), 0.0, places=2
                    )
                    for i, val in enumerate(self.cond_val):
                        self.assertAlmostEqual(
                            field_2[self.data_idx[:dim]][i], val, places=2
                        )

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
                field_1, __ = detrended.unstructured(self.grids[dim - 1])
                field_1 = field_1.reshape(self.grid_shape[:dim])
                field_2, __ = detrended.structured(self.pos[:dim])
                # detrended.plot()
                self.assertAlmostEqual(
                    np.max(np.abs(field_1 - field_2)), 0.0, places=2
                )
                for i, val in enumerate(self.cond_val):
                    self.assertAlmostEqual(
                        field_2[self.data_idx[:dim]][i], val, places=2
                    )

    def test_extdrift(self):
        ext_drift = []
        cond_drift = []
        for i, grid in enumerate(self.grids):
            dim = i + 1
            model = gs.Exponential(
                dim=dim,
                var=2,
                len_scale=10,
                anis=[0.9, 0.8],
                angles=[2, 1, 0.5],
            )
            srf = gs.SRF(model)
            field = srf(grid)
            ext_drift.append(field)
            field = field.reshape(self.grid_shape[:dim])
            cond_drift.append(field[self.data_idx[:dim]])

        for Model in self.cov_models:
            for dim in self.dims:
                model = Model(
                    dim=dim,
                    var=2,
                    len_scale=10,
                    anis=[0.5, 0.2],
                    angles=[0.4, 0.2, 0.1],
                )
                extdrift = gs.krige.ExtDrift(
                    model,
                    self.cond_pos[:dim],
                    self.cond_val,
                    cond_drift[dim - 1],
                )
                field_1, __ = extdrift.unstructured(
                    self.grids[dim - 1], ext_drift=ext_drift[dim - 1]
                )
                field_1 = field_1.reshape(self.grid_shape[:dim])
                field_2, __ = extdrift.structured(
                    self.pos[:dim], ext_drift=ext_drift[dim - 1]
                )
                # extdrift.plot()
                self.assertAlmostEqual(
                    np.max(np.abs(field_1 - field_2)), 0.0, places=2
                )
                for i, val in enumerate(self.cond_val):
                    self.assertAlmostEqual(
                        field_2[self.data_idx[:dim]][i], val, places=2
                    )

    def test_pseudo(self):
        for Model in self.cov_models:
            for dim in self.dims:
                model = Model(
                    dim=dim,
                    var=2,
                    len_scale=10,
                    anis=[0.5, 0.2],
                    angles=[0.4, 0.2, 0.1],
                )
                for meth in self.p_meth:
                    krig = gs.krige.Krige(
                        model, self.p_data[:dim], self.p_vals, unbiased=False
                    )
                    field, __ = krig([0, 0, 0][:dim])
                    # with the pseudo-inverse, the estimated value
                    # should be the mean of the 3 redundant input values
                    self.assertAlmostEqual(
                        field[0], np.mean(self.p_vals), places=2
                    )

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
                ordinary = gs.krige.Ordinary(
                    model,
                    self.cond_pos[:dim],
                    self.cond_val,
                    exact=False,
                    cond_err=self.cond_err,
                )
                field, err = ordinary(self.cond_pos[:dim])
                # when the given measurement error is 0, the kriging-var
                # should equal the nugget of the model
                self.assertAlmostEqual(err[1], model.nugget, places=2)
                self.assertAlmostEqual(err[4], model.nugget, places=2)

    def test_raise(self):
        # no cond_pos/cond_val given
        self.assertRaises(ValueError, gs.krige.Krige, gs.Stable(), None, None)

    def test_krige_mean(self):
        # check for constant mean (simple kriging)
        krige = gs.krige.Simple(gs.Gaussian(), self.cond_pos, self.cond_val)
        mean_f = krige.structured(self.pos, only_mean=True)
        self.assertTrue(np.all(np.isclose(mean_f, 0)))
        krige = gs.krige.Simple(
            gs.Gaussian(),
            self.cond_pos,
            self.cond_val,
            mean=mean_func,
            normalizer=gs.normalizer.YeoJohnson,
            trend=trend,
        )
        # check applying mean, norm, trend
        mean_f1 = krige.structured(self.pos, only_mean=True)
        mean_f2 = gs.normalizer.tools.apply_mean_norm_trend(
            self.pos,
            np.zeros(tuple(map(len, self.pos))),
            mean=mean_func,
            normalizer=gs.normalizer.YeoJohnson,
            trend=trend,
            mesh_type="structured",
        )
        self.assertTrue(np.all(np.isclose(mean_f1, mean_f2)))
        krige = gs.krige.Simple(gs.Gaussian(), self.cond_pos, self.cond_val)
        mean_f = krige.structured(self.pos, only_mean=True)
        self.assertTrue(np.all(np.isclose(mean_f, 0)))
        # check for constant mean (ordinary kriging)
        krige = gs.krige.Ordinary(gs.Gaussian(), self.cond_pos, self.cond_val)
        mean_f = krige.structured(self.pos, only_mean=True)
        self.assertTrue(np.all(np.isclose(mean_f, krige.get_mean())))


if __name__ == "__main__":
    unittest.main()
