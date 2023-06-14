"""
This is the unittest for latlon related routines.
"""

import unittest

import numpy as np

import gstools as gs


def _rel_err(a, b):
    return np.abs(a / ((a + b) / 2) - 1)


class ErrMod(gs.CovModel):
    def cor(self, h):
        return np.exp(-(h**2))

    def fix_dim(self):
        return 2


class TestLatLon(unittest.TestCase):
    def setUp(self):
        self.cmod = gs.Gaussian(
            latlon=True, var=2, len_scale=777, geo_scale=gs.KM_SCALE
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

    def test_conv(self):
        p_ll = gs.tools.geometric.latlon2pos((self.lat, self.lon), 2.56)
        ll_p = gs.tools.geometric.pos2latlon(p_ll, 2.56)
        for i, v in enumerate(self.lat):
            self.assertAlmostEqual(v, ll_p[0, i])
            self.assertAlmostEqual(v, ll_p[1, i])
        self.assertAlmostEqual(
            8, self.cmod.anisometrize(self.cmod.isometrize((8, 6)))[0, 0]
        )
        self.assertAlmostEqual(
            6, self.cmod.anisometrize(self.cmod.isometrize((8, 6)))[1, 0]
        )
        self.assertAlmostEqual(
            gs.EARTH_RADIUS,
            self.cmod.isometrize(
                self.cmod.anisometrize((gs.EARTH_RADIUS, 0, 0))
            )[0, 0],
        )

    def test_cov_model(self):
        self.assertAlmostEqual(
            self.cmod.vario_yadrenko(1.234),
            self.cmod.sill - self.cmod.cov_yadrenko(1.234),
        )
        self.assertAlmostEqual(
            self.cmod.cov_yadrenko(1.234),
            self.cmod.var * self.cmod.cor_yadrenko(1.234),
        )
        # test if correctly handling tries to set anisotropy
        self.cmod.anis = [1, 2]
        self.cmod.angles = [1, 2, 3]
        self.assertAlmostEqual(self.cmod.anis[0], 1)
        self.assertAlmostEqual(self.cmod.anis[1], 1)
        self.assertAlmostEqual(self.cmod.angles[0], 0)
        self.assertAlmostEqual(self.cmod.angles[1], 0)
        self.assertAlmostEqual(self.cmod.angles[2], 0)

    def test_vario_est(self):
        srf = gs.SRF(self.cmod, seed=12345)
        field = srf.structured((self.lat, self.lon))

        bin_edges = np.linspace(0, 3 * 777, 30)
        bin_center, emp_vario = gs.vario_estimate(
            *((self.lat, self.lon), field, bin_edges),
            latlon=True,
            mesh_type="structured",
            sampling_size=2000,
            sampling_seed=12345,
            geo_scale=gs.KM_SCALE,
        )
        mod = gs.Gaussian(latlon=True, geo_scale=gs.KM_SCALE)
        mod.fit_variogram(bin_center, emp_vario, nugget=False)
        # allow 10 percent relative error
        self.assertLess(_rel_err(mod.var, self.cmod.var), 0.1)
        self.assertLess(_rel_err(mod.len_scale, self.cmod.len_scale), 0.1)

    def test_krige(self):
        bin_max = np.deg2rad(8)
        bin_edges = np.linspace(0, bin_max, 5)
        emp_vario = gs.vario_estimate(
            (self.data[:, 0], self.data[:, 1]),
            self.data[:, 2],
            bin_edges,
            latlon=True,
        )
        mod = gs.Spherical(latlon=True, geo_scale=gs.KM_SCALE)
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
        mod = gs.Spherical(latlon=True, geo_scale=gs.KM_SCALE)
        mod.fit_variogram(*emp_vario, nugget=False)
        krige = gs.krige.Ordinary(
            mod, (self.data[:, 0], self.data[:, 1]), self.data[:, 2]
        )
        crf = gs.CondSRF(krige)
        field = crf((self.data[:, 0], self.data[:, 1]))
        for i, dat in enumerate(self.data[:, 2]):
            self.assertAlmostEqual(field[i], dat, 3)

    def test_error(self):
        # try fitting directional variogram
        mod = gs.Gaussian(latlon=True)
        with self.assertRaises(ValueError):
            mod.fit_variogram([0, 1], [[0, 1], [0, 1], [0, 1]])
        # try to use fixed dim=2 with latlon
        with self.assertRaises(ValueError):
            ErrMod(latlon=True)
        # try to estimate latlon vario on wrong dim
        with self.assertRaises(ValueError):
            gs.vario_estimate([[1], [1], [1]], [1], [0, 1], latlon=True)
        # try to estimate directional vario with latlon
        with self.assertRaises(ValueError):
            gs.vario_estimate([[1], [1]], [1], [0, 1], latlon=True, angles=1)
        # try to create a vector field with latlon
        with self.assertRaises(ValueError):
            srf = gs.SRF(mod, generator="VectorField", mode_no=2)
            srf([1, 2])


if __name__ == "__main__":
    unittest.main()
