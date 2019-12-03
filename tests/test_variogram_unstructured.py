# -*- coding: utf-8 -*-
"""
This is a unittest of the variogram module.
"""

import unittest
import numpy as np
from gstools import vario_estimate_unstructured


class TestVariogramUnstructured(unittest.TestCase):
    def setUp(self):
        pass

    def test_doubles(self):
        x = np.arange(1, 11, 1, dtype=np.double)
        z = np.array(
            (41.2, 40.2, 39.7, 39.2, 40.1, 38.3, 39.1, 40.0, 41.1, 40.3),
            dtype=np.double,
        )
        bins = np.arange(1, 11, 1, dtype=np.double)
        bin_centres, gamma = vario_estimate_unstructured([x], z, bins)
        self.assertAlmostEqual(gamma[0], 0.4917, places=4)

    def test_ints(self):
        x = np.arange(1, 5, 1, dtype=int)
        z = np.array((10, 20, 30, 40), dtype=int)
        bins = np.arange(1, 11, 1, dtype=int)
        bin_centres, gamma = vario_estimate_unstructured([x], z, bins)
        self.assertAlmostEqual(gamma[0], 50.0, places=4)

    def test_np_int(self):
        x = np.arange(1, 5, 1, dtype=np.int)
        z = np.array((10, 20, 30, 40), dtype=np.int)
        bins = np.arange(1, 11, 1, dtype=np.int)
        bin_centres, gamma = vario_estimate_unstructured([x], z, bins)
        self.assertAlmostEqual(gamma[0], 50.0, places=4)

    def test_mixed(self):
        x = np.arange(1, 11, 1, dtype=np.double)
        z = np.array(
            (41.2, 40.2, 39.7, 39.2, 40.1, 38.3, 39.1, 40.0, 41.1, 40.3),
            dtype=np.double,
        )
        bins = np.arange(1, 11, 1, dtype=int)
        bin_centres, gamma = vario_estimate_unstructured([x], z, bins)
        self.assertAlmostEqual(gamma[0], 0.4917, places=4)

        x = np.arange(1, 5, 1, dtype=np.double)
        z = np.array((10, 20, 30, 40), dtype=int)
        bins = np.arange(1, 11, 1, dtype=int)
        bin_centres, gamma = vario_estimate_unstructured([x], z, bins)
        self.assertAlmostEqual(gamma[0], 50.0, places=4)

        x = np.arange(1, 5, 1, dtype=np.double)
        z = np.array((10, 20, 30, 40), dtype=int)
        bins = np.arange(1, 11, 1, dtype=np.double)
        bin_centres, gamma = vario_estimate_unstructured([x], z, bins)
        self.assertAlmostEqual(gamma[0], 50.0, places=4)

    def test_list(self):
        x = np.arange(1, 11, 1, dtype=np.double)
        z = [41.2, 40.2, 39.7, 39.2, 40.1, 38.3, 39.1, 40.0, 41.1, 40.3]
        bins = np.arange(1, 11, 1, dtype=np.double)
        bin_centres, gamma = vario_estimate_unstructured([x], z, bins)
        self.assertAlmostEqual(gamma[1], 0.7625, places=4)

    def test_1d(self):
        x = np.arange(1, 11, 1, dtype=np.double)
        # literature values
        z = np.array(
            (41.2, 40.2, 39.7, 39.2, 40.1, 38.3, 39.1, 40.0, 41.1, 40.3),
            dtype=np.double,
        )
        bins = np.arange(1, 11, 1, dtype=np.double)
        bin_centres, gamma = vario_estimate_unstructured([x], z, bins)
        self.assertAlmostEqual(gamma[0], 0.4917, places=4)
        self.assertAlmostEqual(gamma[1], 0.7625, places=4)

    def test_uncorrelated_2d(self):
        x_c = np.linspace(0.0, 100.0, 60)
        y_c = np.linspace(0.0, 100.0, 60)
        x, y = np.meshgrid(x_c, y_c)
        x = np.reshape(x, len(x_c) * len(y_c))
        y = np.reshape(y, len(x_c) * len(y_c))

        rng = np.random.RandomState(1479373475)
        field = rng.rand(len(x))

        bins = np.arange(0, 100, 10)

        bin_centres, gamma = vario_estimate_unstructured((x, y), field, bins)

        var = 1.0 / 12.0
        self.assertAlmostEqual(gamma[0], var, places=2)
        self.assertAlmostEqual(gamma[len(gamma) // 2], var, places=2)
        self.assertAlmostEqual(gamma[-1], var, places=2)

    def test_uncorrelated_3d(self):
        x_c = np.linspace(0.0, 100.0, 15)
        y_c = np.linspace(0.0, 100.0, 15)
        z_c = np.linspace(0.0, 100.0, 15)
        x, y, z = np.meshgrid(x_c, y_c, z_c)
        x = np.reshape(x, len(x_c) * len(y_c) * len(z_c))
        y = np.reshape(y, len(x_c) * len(y_c) * len(z_c))
        z = np.reshape(z, len(x_c) * len(y_c) * len(z_c))

        rng = np.random.RandomState(1479373475)
        field = rng.rand(len(x))

        bins = np.arange(0, 100, 10)

        bin_centres, gamma = vario_estimate_unstructured(
            (x, y, z), field, bins
        )

        var = 1.0 / 12.0
        self.assertAlmostEqual(gamma[0], var, places=2)
        self.assertAlmostEqual(gamma[len(gamma) // 2], var, places=2)
        self.assertAlmostEqual(gamma[-1], var, places=2)

    def test_sampling_1d(self):
        x = np.linspace(0.0, 100.0, 21000)

        rng = np.random.RandomState(1479373475)
        field = rng.rand(len(x))

        bins = np.arange(0, 100, 10)

        bin_centres, gamma = vario_estimate_unstructured(
            [x], field, bins, sampling_size=5000, sampling_seed=1479373475
        )

        var = 1.0 / 12.0
        self.assertAlmostEqual(gamma[0], var, places=2)
        self.assertAlmostEqual(gamma[len(gamma) // 2], var, places=2)
        self.assertAlmostEqual(gamma[-1], var, places=2)

    def test_sampling_2d(self):
        x_c = np.linspace(0.0, 100.0, 600)
        y_c = np.linspace(0.0, 100.0, 600)
        x, y = np.meshgrid(x_c, y_c)
        x = np.reshape(x, len(x_c) * len(y_c))
        y = np.reshape(y, len(x_c) * len(y_c))

        rng = np.random.RandomState(1479373475)
        field = rng.rand(len(x))

        bins = np.arange(0, 100, 10)

        bin_centres, gamma = vario_estimate_unstructured(
            (x, y), field, bins, sampling_size=2000, sampling_seed=1479373475
        )

        var = 1.0 / 12.0
        self.assertAlmostEqual(gamma[0], var, places=2)
        self.assertAlmostEqual(gamma[len(gamma) // 2], var, places=2)
        self.assertAlmostEqual(gamma[-1], var, places=2)

    def test_sampling_3d(self):
        x_c = np.linspace(0.0, 100.0, 100)
        y_c = np.linspace(0.0, 100.0, 100)
        z_c = np.linspace(0.0, 100.0, 100)
        x, y, z = np.meshgrid(x_c, y_c, z_c)
        x = np.reshape(x, len(x_c) * len(y_c) * len(z_c))
        y = np.reshape(y, len(x_c) * len(y_c) * len(z_c))
        z = np.reshape(z, len(x_c) * len(y_c) * len(z_c))

        rng = np.random.RandomState(1479373475)
        field = rng.rand(len(x))

        bins = np.arange(0, 100, 10)

        bin_centres, gamma = vario_estimate_unstructured(
            (x, y, z),
            field,
            bins,
            sampling_size=2000,
            sampling_seed=1479373475,
        )
        var = 1.0 / 12.0
        self.assertAlmostEqual(gamma[0], var, places=2)
        self.assertAlmostEqual(gamma[len(gamma) // 2], var, places=2)
        self.assertAlmostEqual(gamma[-1], var, places=2)

    def test_assertions(self):
        x = np.arange(0, 10)
        x_e = np.arange(0, 11)
        y = np.arange(0, 11)
        y_e = np.arange(0, 12)
        z = np.arange(0, 12)
        z_e = np.arange(0, 15)
        bins = np.arange(0, 3)
        #        bins_e = np.arange(0, 1)
        field = np.arange(0, 10)
        field_e = np.arange(0, 9)

        self.assertRaises(
            ValueError, vario_estimate_unstructured, [x_e], field, bins
        )
        self.assertRaises(
            ValueError, vario_estimate_unstructured, (x, y_e), field, bins
        )
        self.assertRaises(
            ValueError, vario_estimate_unstructured, (x, y_e, z), field, bins
        )
        self.assertRaises(
            ValueError, vario_estimate_unstructured, (x, y, z_e), field, bins
        )
        self.assertRaises(
            ValueError, vario_estimate_unstructured, (x_e, y, z), field, bins
        )
        self.assertRaises(
            ValueError, vario_estimate_unstructured, (x, y, z), field_e, bins
        )
        self.assertRaises(
            ValueError, vario_estimate_unstructured, [x], field_e, bins
        )


if __name__ == "__main__":
    unittest.main()
