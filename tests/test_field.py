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

    def test_pos_compare(self):
        fld = gs.field.Field(dim=1)
        fld.set_pos([1, 2])
        fld._dim = 2
        info = fld.set_pos([[1], [2]], info=True)
        self.assertTrue(info["deleted"])
        info = fld.set_pos([[2], [3]], info=True)
        self.assertTrue(info["deleted"])

    def test_magic(self):
        fld = gs.field.Field(dim=1)
        f1 = np.array([0, 0], dtype=np.double)
        f2 = np.array([2, 3], dtype=np.double)
        fld([1, 2], store="f1")  # default field with zeros
        fld([1, 2], f2, store="f2")
        fields1 = fld[:]
        fields2 = fld[[0, 1]]
        fields3 = fld[["f1", "f2"]]
        fields4 = fld.all_fields
        self.assertTrue(np.allclose([f1, f2], fields1))
        self.assertTrue(np.allclose([f1, f2], fields2))
        self.assertTrue(np.allclose([f1, f2], fields3))
        self.assertTrue(np.allclose([f1, f2], fields4))
        self.assertEqual(len(fld), 2)
        self.assertTrue("f1" in fld)
        self.assertTrue("f2" in fld)
        self.assertFalse("f3" in fld)
        # subscription
        with self.assertRaises(KeyError):
            fld["f3"]
        with self.assertRaises(KeyError):
            del fld["f3"]
        with self.assertRaises(KeyError):
            del fld[["f3"]]
        del fld["f1"]
        self.assertFalse("f1" in fld)
        fld([1, 2], f1, store="f1")
        del fld[-1]
        self.assertFalse("f1" in fld)
        fld([1, 2], f1, store="f1")
        del fld[:]
        self.assertEqual(len(fld), 0)
        fld([1, 2], f1, store="f1")
        del fld.field_names
        self.assertEqual(len(fld), 0)
        # store config (missing check)
        name, save = fld.get_store_config(store="fld", fld_cnt=1)
        self.assertEqual(name, ["fld"])
        self.assertTrue(save[0])

    def test_reuse(self):
        fld = gs.field.Field(dim=1)
        # no pos tuple
        with self.assertRaises(ValueError):
            fld()
        # no field shape
        with self.assertRaises(ValueError):
            fld.post_field([1, 2])
        # bad name
        fld.set_pos([1, 2])
        with self.assertRaises(ValueError):
            fld.post_field([1, 2], process=False, name=0)
        # incompatible reuse
        with self.assertRaises(ValueError):
            fld.structured()
        fld.set_pos([1, 2], "structured")
        with self.assertRaises(ValueError):
            fld.unstructured()


if __name__ == "__main__":
    unittest.main()
