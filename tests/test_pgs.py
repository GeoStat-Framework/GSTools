"""
This is the unittest of the PGS class.
"""

import unittest

import numpy as np

import gstools as gs


class TestPGS(unittest.TestCase):
    def test_struct_1d(self):
        n = 100
        x = np.arange(n)
        model = gs.Gaussian(dim=1, var=2, len_scale=15)
        srf = gs.SRF(model, seed=436239)
        field = srf.structured((x,))

        m = 10
        L = np.zeros(n)
        L[n // 3 - m // 2 : n // 3 + m // 2] = 1
        L[n // 3 - 2 * m : n // 3 - m // 2] = 2
        L[4 * n // 5 - m // 2 : 4 * n // 5 + m // 2] = 3

        pgs = gs.PGS(1, field, L)
        self.assertAlmostEqual(pgs.P[n // 2], 0.0)
        self.assertAlmostEqual(pgs.P[0], 0.0)
        self.assertAlmostEqual(pgs.P[-1], 1.0)
        self.assertAlmostEqual(pgs.P[-20], 3.0)

    def test_struct_2d(self):
        n1 = 100
        n2 = 100
        pos = [np.arange(n1), np.arange(n2)]

        model1 = gs.Gaussian(dim=2, var=1, len_scale=10)
        srf1 = gs.SRF(model1, seed=20170519)
        field1 = srf1.structured(pos)

        model2 = gs.Gaussian(dim=2, var=5, len_scale=20)
        srf2 = gs.SRF(model2, seed=20160519)
        field2 = srf2.structured(pos)

        # create rectangle in middle of L
        m1 = 16
        m2 = 16

        L = np.zeros((n1, n2))
        L[
            n1 // 2 - m1 // 2 : n1 // 2 + m1 // 2,
            n2 // 2 - m2 // 2 : n2 // 2 + m2 // 2,
        ] = 1
        L[
            n1 // 2 - m1 // 2 + m1 : n1 // 2 + m1 // 2 + m1,
            n2 // 2 - m2 // 2 : n2 // 2 + m2 // 2,
        ] = 2
        L[
            n1 // 2 - m1 // 2 + m1 : n1 // 2 + m1 // 2 + m1,
            n2 // 2 - m2 // 2 + m2 : n2 // 2 + m2 // 2 + m2,
        ] = 3
        L[
            n1 // 3 - m1 // 2 : n1 // 3 + m1 // 2,
            n2 // 3 - m2 // 2 : n2 // 3 + m2 // 2,
        ] = 4

        pgs = gs.PGS(2, [field1, field2], L)

        self.assertAlmostEqual(pgs.P[n1 // 2, n2 // 2], 2.0)
        self.assertAlmostEqual(pgs.P[0, 0], 1.0)
        self.assertAlmostEqual(pgs.P[-1, -1], 1.0)
        self.assertAlmostEqual(pgs.P[0, -1], 0.0)
        self.assertAlmostEqual(pgs.P[-1, 0], 1.0)

    def test_struct_3d(self):
        n1 = 30
        n2 = 30
        n3 = 30
        pos = [np.arange(n1), np.arange(n2), np.arange(n3)]

        model1 = gs.Gaussian(dim=3, var=1, len_scale=10)
        srf1 = gs.SRF(model1, seed=20170519)
        field1 = srf1.structured(pos)

        model2 = gs.Gaussian(dim=3, var=5, len_scale=20)
        srf2 = gs.SRF(model2, seed=20160519)
        field2 = srf2.structured(pos)

        model3 = gs.Gaussian(dim=3, var=0.1, len_scale=5)
        srf3 = gs.SRF(model3, seed=20191219)
        field3 = srf3.structured(pos)

        # create rectangle in middle of L
        m1 = 10
        m2 = 10
        m3 = 10

        L = np.zeros((n1, n2, n3))
        L[
            n1 // 2 - m1 // 2 : n1 // 2 + m1 // 2,
            n2 // 2 - m2 // 2 : n2 // 2 + m2 // 2,
            n3 // 2 - m3 // 2 : n3 // 2 + m3 // 2,
        ] = 1

        pgs = gs.PGS(3, [field1, field2, field3], L)

        self.assertAlmostEqual(pgs.P[n1 // 2, n2 // 2, n3 // 2], 1.0)
        self.assertAlmostEqual(pgs.P[n1 // 3, n2 // 3, n3 // 3], 1.0)
        self.assertAlmostEqual(
            pgs.P[2 * n1 // 3, 2 * n2 // 3, 2 * n3 // 3], 1.0
        )
        self.assertAlmostEqual(pgs.P[0, 0, 0], 1.0)
        self.assertAlmostEqual(pgs.P[-1, -1, -1], 0.0)
        self.assertAlmostEqual(pgs.P[-1, 0, 0], 1.0)
        self.assertAlmostEqual(pgs.P[0, -1, 0], 1.0)
        self.assertAlmostEqual(pgs.P[0, 0, -1], 1.0)
        self.assertAlmostEqual(pgs.P[0, -1, -1], 1.0)
        self.assertAlmostEqual(pgs.P[-1, 0, -1], 0.0)
        self.assertAlmostEqual(pgs.P[-1, -1, 0], 1.0)

    def test_struct_4d(self):
        n1 = 20
        n2 = 20
        n3 = 20
        n4 = 20
        pos = [np.arange(n1), np.arange(n2), np.arange(n4), np.arange(n4)]

        model1 = gs.Gaussian(dim=4, var=1, len_scale=10)
        srf1 = gs.SRF(model1, seed=20170519)
        field1 = srf1.structured(pos)

        model2 = gs.Gaussian(dim=4, var=5, len_scale=20)
        srf2 = gs.SRF(model2, seed=20160519)
        field2 = srf2.structured(pos)

        model3 = gs.Gaussian(dim=4, var=0.1, len_scale=5)
        srf3 = gs.SRF(model3, seed=20191219)
        field3 = srf3.structured(pos)

        model4 = gs.Exponential(dim=4, var=0.5, len_scale=12)
        srf3 = gs.SRF(model4, seed=20191219)
        field4 = srf3.structured(pos)

        # create rectangle in middle of L
        m1 = 5
        m2 = 5
        m3 = 5
        m4 = 5

        L = np.zeros((n1, n2, n3, n4))
        L[
            n1 // 2 - m1 // 2 : n1 // 2 + m1 // 2,
            n2 // 2 - m2 // 2 : n2 // 2 + m2 // 2,
            n3 // 2 - m3 // 2 : n3 // 2 + m3 // 2,
            n4 // 2 - m4 // 2 : n4 // 2 + m4 // 2,
        ] = 1

        pgs = gs.PGS(4, [field1, field2, field3, field4], L)

        self.assertAlmostEqual(pgs.P[n1 // 2, n2 // 2, n3 // 2, n4 // 2], 1.0)
        self.assertAlmostEqual(pgs.P[0, 0, 0, 0], 0.0)
        self.assertAlmostEqual(pgs.P[-1, -1, -1, -1], 0.0)

    def test_unstruct_2d(self):
        n1 = 10
        n2 = 8
        rng = np.random.RandomState(seed=438430)
        x_unstruct = rng.randint(0, n1, size=n1 * n2)
        y_unstruct = rng.randint(0, n2, size=n1 * n2)

        pos_unstruct = [x_unstruct, y_unstruct]

        model1 = gs.Gaussian(dim=2, var=1, len_scale=10)
        srf1 = gs.SRF(model1, seed=20170519)
        field1_unstruct = srf1.unstructured(pos_unstruct)

        model2 = gs.Gaussian(dim=2, var=5, len_scale=20)
        srf2 = gs.SRF(model2, seed=20160519)
        field2_unstruct = srf2.unstructured(pos_unstruct)

        # create rectangle in middle of L
        m1 = 4
        m2 = 4

        L_struct = np.zeros((n1, n2))
        L_struct[
            n1 // 2 - m1 // 2 : n1 // 2 + m1 // 2,
            n2 // 2 - m2 // 2 : n2 // 2 + m2 // 2,
        ] = 1
        L_struct[
            n1 // 2 - m1 // 2 + m1 : n1 // 2 + m1 // 2 + m1,
            n2 // 2 - m2 // 2 : n2 // 2 + m2 // 2,
        ] = 2
        L_struct[
            n1 // 2 - m1 // 2 + m1 : n1 // 2 + m1 // 2 + m1,
            n2 // 2 - m2 // 2 + m2 : n2 // 2 + m2 // 2 + m2,
        ] = 3
        L_struct[
            n1 // 3 - m1 // 2 : n1 // 3 + m1 // 2,
            n2 // 3 - m2 // 2 : n2 // 3 + m2 // 2,
        ] = 4

        pgs = gs.PGS(2, [field1_unstruct, field2_unstruct], L_struct)

        self.assertAlmostEqual(pgs.P[0], 4.0)
        self.assertAlmostEqual(pgs.P[-1], 1.0)
        self.assertAlmostEqual(pgs.P[n1 * n2 // 2], 1.0)

    def test_assertions(self):
        n = 30
        pos = [np.arange(n), np.arange(n), np.arange(n)]
        L_2d = np.empty((n, n))
        field1 = np.empty((n, n))
        field2 = np.empty((n - 1, n - 1))
        self.assertRaises(ValueError, gs.PGS, 3, [0, 1], None)
        self.assertRaises(ValueError, gs.PGS, 3, pos, L_2d)
        self.assertRaises(ValueError, gs.PGS, 2, [field1, field2], L_2d)