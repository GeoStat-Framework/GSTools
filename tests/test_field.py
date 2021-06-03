#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This is the unittest of SRF class.
"""

import unittest
import numpy as np
import gstools as gs


class TestField(unittest.TestCase):
    def setUp(self):
        self.cov_model = gs.Gaussian(dim=2, var=1.5, len_scale=4.0)
        rng = np.random.RandomState(123018)
        x = rng.uniform(0.0, 10, 100)
        y = rng.uniform(0.0, 10, 100)
        self.field = rng.uniform(0.0, 10, 100)
        self.pos = np.array([x, y])

    def test_standalone(self):
        fld = gs.field.Field(dim=2)
        fld_cov = gs.field.Field(model=self.cov_model)
        field1 = fld(self.pos, self.field)
        field2 = fld_cov(self.pos, self.field)
        self.assertTrue(np.all(np.isclose(field1, field2)))
        self.assertTrue(np.all(np.isclose(field1, self.field)))

    def test_raise(self):
        # vector field on latlon
        fld = gs.field.Field(gs.Gaussian(latlon=True), value_type="vector")
        self.assertRaises(ValueError, fld, [1, 2], [1, 2])
        # no pos tuple present
        fld = gs.field.Field(dim=2)
        self.assertRaises(ValueError, fld.post_field, [1, 2])
        # wrong model type
        with self.assertRaises(ValueError):
            gs.field.Field(model=3.1415)
        # no model and no dim given
        with self.assertRaises(ValueError):
            gs.field.Field()
        # wrong value type
        with self.assertRaises(ValueError):
            gs.field.Field(dim=2, value_type="complex")
        # wrong mean shape
        with self.assertRaises(ValueError):
            gs.field.Field(dim=3, mean=[1, 2])


if __name__ == "__main__":
    unittest.main()
