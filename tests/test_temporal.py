"""
This is the unittest for temporal related routines.
"""

import unittest

import numpy as np

import gstools as gs


class TestTemporal(unittest.TestCase):
    def setUp(self):
        ...

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

    def test_rotation(self):
        mod = gs.Gaussian(dim=3, temporal=True, angles=[1, 2, 3, 4, 5, 6])
        self.assertTrue(np.allclose(mod.angles, [1, 2, 3, 0, 0, 0]))

    def test_krige(self):
        mod = gs.Gaussian(latlon=True, temporal=True)
        # auto-fitting latlon-temporal model in kriging not possible
        with self.assertRaises(ValueError):
            kri = gs.Krige(mod, 3 * [[1, 2, 3]], [1, 2, 3], fit_variogram=True)


if __name__ == "__main__":
    unittest.main()
