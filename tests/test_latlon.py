# -*- coding: utf-8 -*-
"""
This is the unittest of CovModel class.
"""

import numpy as np
import unittest
import gstools as gs


def _rel_err(a, b):
    return np.abs(a / ((a + b) / 2) - 1)


class TestCondition(unittest.TestCase):
    def setUp(self):
        self.cov_model = gs.Gaussian(
            latlon=True, var=2, len_scale=777, rescale=gs.EARTH_RADIUS
        )
        self.lat = self.lon = range(-80, 81)

        self.data = np.array(
            [
                [52.9336, 8.237, 15.7],
                [48.6159, 13.0506, 13.9],
                [52.4853, 7.9126, 15.1],
                [50.7446, 9.345, 17.0],
                [52.9437, 12.8518, 21.9],
                [53.8633, 8.1275, 11.9],
                [47.8342, 10.8667, 11.4],
                [51.0881, 12.9326, 17.2],
                [48.406, 11.3117, 12.9],
                [49.7273, 8.1164, 17.2],
                [49.4691, 11.8546, 13.4],
                [48.0197, 12.2925, 13.9],
                [50.4237, 7.4202, 18.1],
                [53.0316, 13.9908, 21.3],
                [53.8412, 13.6846, 21.3],
                [54.6792, 13.4343, 17.4],
                [49.9694, 9.9114, 18.6],
                [51.3745, 11.292, 20.2],
                [47.8774, 11.3643, 12.7],
                [50.5908, 12.7139, 15.8],
            ]
        )

    def test_cov_model(self):
        self.assertAlmostEqual(
            self.cov_model.vario_yadrenko(1.234),
            self.cov_model.sill - self.cov_model.cov_yadrenko(1.234),
        )
        self.assertAlmostEqual(
            self.cov_model.cov_yadrenko(1.234),
            self.cov_model.var * self.cov_model.cor_yadrenko(1.234),
        )

    def test_vario_est(self):

        srf = gs.SRF(self.cov_model, seed=12345)
        field = srf.structured((self.lat, self.lon))

        bin_edges = [0.01 * i for i in range(30)]
        bin_center, emp_vario = gs.vario_estimate(
            *((self.lat, self.lon), field, bin_edges),
            latlon=True,
            mesh_type="structured",
            sampling_size=2000,
            sampling_seed=12345,
        )
        mod = gs.Gaussian(latlon=True, rescale=gs.EARTH_RADIUS)
        mod.fit_variogram(bin_center, emp_vario, nugget=False)
        # allow 10 percent relative error
        self.assertLess(_rel_err(mod.var, self.cov_model.var), 0.1)
        self.assertLess(_rel_err(mod.len_scale, self.cov_model.len_scale), 0.1)

    def test_krige(self):
        bin_max = np.deg2rad(8)
        bin_edges = np.linspace(0, bin_max, 5)
        emp_vario = gs.vario_estimate(
            (self.data[:, 0], self.data[:, 1]),
            self.data[:, 2],
            bin_edges,
            latlon=True,
        )
        mod = gs.Spherical(latlon=True, rescale=gs.EARTH_RADIUS)
        mod.fit_variogram(*emp_vario, nugget=False)
        kri = gs.krige.Ordinary(
            mod,
            (self.data[:, 0], self.data[:, 1]),
            self.data[:, 2],
        )
        field, var = kri((self.data[:, 0], self.data[:, 1]))
        for i, dat in enumerate(self.data[:, 2]):
            self.assertAlmostEqual(field[i], dat)

    def test_cond_srf(self):
        bin_max = np.deg2rad(8)
        bin_edges = np.linspace(0, bin_max, 5)
        emp_vario = gs.vario_estimate(
            (self.data[:, 0], self.data[:, 1]),
            self.data[:, 2],
            bin_edges,
            latlon=True,
        )
        mod = gs.Spherical(latlon=True, rescale=gs.EARTH_RADIUS)
        mod.fit_variogram(*emp_vario, nugget=False)
        srf = gs.SRF(mod)
        srf.set_condition((self.data[:, 0], self.data[:, 1]), self.data[:, 2])
        field = srf((self.data[:, 0], self.data[:, 1]))
        for i, dat in enumerate(self.data[:, 2]):
            self.assertAlmostEqual(field[i], dat)


if __name__ == "__main__":
    unittest.main()
