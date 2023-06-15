"""
This is the unittest for temporal related routines.
"""

import unittest

import numpy as np

import gstools as gs


class TestTemporal(unittest.TestCase):
    def setUp(self):
        self.mod = gs.Gaussian(
            latlon=True,
            temporal=True,
            len_scale=1000,
            anis=0.5,
            geo_scale=gs.KM_SCALE,
        )

    def test_latlon(self):
        mod = gs.Gaussian(
            latlon=True, temporal=True, angles=[1, 2, 3, 4, 5, 6]
        )
        self.assertEqual(mod.dim, 4)
        self.assertEqual(mod.field_dim, 3)
        self.assertEqual(mod.spatial_dim, 2)
        self.assertTrue(np.allclose(mod.angles, 0))

        mod1 = gs.Gaussian(latlon=True, temporal=True, len_scale=[10, 5])
        mod2 = gs.Gaussian(latlon=True, temporal=True, len_scale=10, anis=0.5)

        self.assertTrue(np.allclose(mod1.anis, mod2.anis))
        self.assertAlmostEqual(mod1.len_scale, mod2.len_scale)

    def test_latlon2pos(self):
        self.assertAlmostEqual(
            8, self.mod.anisometrize(self.mod.isometrize((8, 6, 9)))[0, 0]
        )
        self.assertAlmostEqual(
            6, self.mod.anisometrize(self.mod.isometrize((8, 6, 9)))[1, 0]
        )
        self.assertAlmostEqual(
            9, self.mod.anisometrize(self.mod.isometrize((8, 6, 9)))[2, 0]
        )
        self.assertAlmostEqual(
            gs.EARTH_RADIUS,
            self.mod.isometrize(
                self.mod.anisometrize((gs.EARTH_RADIUS, 0, 0, 10))
            )[0, 0],
        )
        self.assertAlmostEqual(
            10,
            self.mod.isometrize(
                self.mod.anisometrize((gs.EARTH_RADIUS, 0, 0, 10))
            )[3, 0],
        )

    def test_rotation(self):
        mod = gs.Gaussian(
            spatial_dim=3, temporal=True, angles=[1, 2, 3, 4, 5, 6]
        )
        self.assertTrue(np.allclose(mod.angles, [1, 2, 3, 0, 0, 0]))
        self.assertEqual(mod.dim, 4)

    def test_krige(self):
        # auto-fitting latlon-temporal model in kriging not possible
        with self.assertRaises(ValueError):
            kri = gs.Krige(self.mod, 3 * [[1, 2]], [1, 2], fit_variogram=True)

    def test_field(self):
        srf = gs.SRF(self.mod)
        self.assertTrue(srf.temporal)


if __name__ == "__main__":
    unittest.main()
