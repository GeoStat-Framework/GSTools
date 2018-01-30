#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This is the unittest of the RNG class.
"""
from __future__ import division, absolute_import, print_function

import unittest
import numpy as np
from scipy.stats import (skew, kurtosis)
from scipy.stats import normaltest
from gstools.field import (RNG, RandMeth)


class TestRNG(unittest.TestCase):
    def setUp(self):
        self.rngs = [RNG(1, 19031977), RNG(2, 19031977), RNG(3, 19031977)]
        self.many_modes = 1000000
        self.few_modes = 100
        self.len_scale = 20.

    def test_rng_consistency_1d(self):
        rng = RNG(1, 21021997)
        l = 2.5
        Z_refs = [-0.04100185,
                   3.33332132,
                   0.22111093,
                  -1.30856723,
                   2.40095964
                 ]
        k_refs = [-0.107329380,
                   0.407318334,
                   1.067490292,
                   1.256078044,
                   0.100332117
                 ]

        #explicit for better debugging, so the line is found immediately
        Z, k = rng('gau', l, mode_no=1)
        self.assertAlmostEqual(Z[0,0], Z_refs[0])
        self.assertAlmostEqual(k[0,0], k_refs[0])
        Z, k = rng('gau', l, mode_no=1)
        self.assertAlmostEqual(Z[0,0], Z_refs[1])
        self.assertAlmostEqual(k[0,0], k_refs[1])

    def test_rng_consistency_2d(self):
        rng = RNG(2, 21021997)
        l = 2.5
        Z_refs = [-0.04100185,
                   1.01829584,
                  -1.30856723,
                   2.17740051,
                  -0.32662374
                 ]
        k_refs = [-0.107329380,
                   0.088444371,
                  -4.047265921,
                   0.873734481,
                  -0.002127327
                 ]

        #explicit for better debugging, so the line is found immediately
        Z, k = rng('gau', l, mode_no=1)
        self.assertAlmostEqual(Z[0,0], Z_refs[0])
        self.assertAlmostEqual(k[0,0], k_refs[0])
        Z, k = rng('gau', l, mode_no=1)
        self.assertAlmostEqual(Z[0,0], Z_refs[1])
        self.assertAlmostEqual(k[0,0], k_refs[1])

    def test_rng_consistency_3d(self):
        rng = RNG(3, 21021997)
        l = 2.5
        Z_refs = [-0.04100185,
                   0.22111093,
                   2.40095964,
                  -0.32578687,
                  -1.96961056
                 ]
        k_refs = [-0.107329380,
                   0.092119696,
                  -0.216183123,
                   0.538085014,
                  -0.723404118
                 ]

        #explicit for better debugging, so the line is found immediately
        Z, k = rng('gau', l, mode_no=1)
        self.assertAlmostEqual(Z[0,0], Z_refs[0])
        self.assertAlmostEqual(k[0,0], k_refs[0])
        Z, k = rng('gau', l, mode_no=1)
        self.assertAlmostEqual(Z[0,0], Z_refs[1])
        self.assertAlmostEqual(k[0,0], k_refs[1])

    def test_call(self):
        for d in range(len(self.rngs)):
            Z, k = self.rngs[d]('gau', self.len_scale, self.many_modes)
            self.assertEqual(Z.shape, (2, self.many_modes))
            self.assertAlmostEqual(np.mean(Z), 0., places=2)
            self.assertAlmostEqual(np.std(Z), 1., places=2)

    def test_gau(self):
        for d in range(len(self.rngs)):
            Z, k = self.rngs[d]('gau', self.len_scale, self.many_modes)
            self.assertEqual(k.shape, (d+1, self.many_modes))
            self.assertAlmostEqual(np.mean(k), 0., places=2)
            self.assertAlmostEqual(np.std(k), 1/self.len_scale, places=2)
            self.assertAlmostEqual(skew(k[0,:]), 0., places=2)
            self.assertAlmostEqual(kurtosis(k[0,:]), 0., places=1)
            self.assertLess(normaltest(k[0,:])[1], 0.05)


if __name__ == '__main__':
    unittest.main()
