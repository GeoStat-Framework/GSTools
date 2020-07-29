#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This is the unittest of Mesh class.
"""
import unittest
import numpy as np

from gstools import Mesh
from gstools.field.mesh import convert_points


class TestMesh(unittest.TestCase):
    def setUp(self):
        self.x_grid = np.linspace(0.0, 12.0, 48)
        self.y_grid = np.linspace(0.0, 10.0, 46)
        self.z_grid = np.linspace(0.0, 10.0, 40)

        self.f1_grid = self.x_grid
        self.f2_grid = self.x_grid.reshape(
            (len(self.x_grid), 1)
        ) * self.y_grid.reshape((1, len(self.y_grid)))
        self.f3_grid = (
            self.x_grid.reshape((len(self.x_grid), 1, 1))
            * self.y_grid.reshape((1, len(self.y_grid), 1))
            * self.z_grid.reshape((1, 1, len(self.z_grid)))
        )

        self.rng = np.random.RandomState(123018)
        self.x_tuple = self.rng.uniform(0.0, 10, 100)
        self.y_tuple = self.rng.uniform(0.0, 10, 100)
        self.z_tuple = self.rng.uniform(0.0, 10, 100)

        self.f1_tuple = self.x_tuple
        self.f2_tuple = self.x_tuple * self.y_tuple
        self.f3_tuple = self.x_tuple * self.y_tuple * self.z_tuple

        self.m1_grid = Mesh(1, (self.x_grid,), mesh_type="structured")
        self.m2_grid = Mesh(
            2, (self.x_grid, self.y_grid), mesh_type="structured"
        )
        self.m3_grid = Mesh(
            3, (self.x_grid, self.y_grid, self.z_grid), mesh_type="structured"
        )
        self.m1_tuple = Mesh(1, (self.x_tuple,))
        self.m2_tuple = Mesh(2, (self.x_tuple, self.y_tuple))
        self.m3_tuple = Mesh(3, (self.x_tuple, self.y_tuple, self.z_tuple))

    def test_convert_points(self):
        pass
        # TODO figure out what this is all about
        #1d
        #p1_target = np.array((1, 2, 3, 4)).reshape(4, 1)
        #print(p1_target.shape)
        #p1_is = convert_points(1, p1_target)
        #np.testing.assert_array_equal(p1_is, p1_target)

        #p2 = [1, 2, 3, 4, 5]
        #p2_target = np.array(p2)
        #p2_is = convert_points(1, p2)
        #np.testing.assert_array_equal(p2_is, p2_target)

        #p3 = [[1], [2], [3], [4], [5]]
        #p3_target = np.array(p3).reshape(-1)
        #p3_is = convert_points(1, p3)
        #np.testing.assert_array_equal(p3_is, p3_target)

        ##2d
        #p4 = [[1, 2], [3, 4], [5, 6]]
        #p4_target = np.array(p4)
        #print(p4_target.shape)


    def test_item_getter_setter(self):
        self.m3_grid.add_field(256.0 * self.f3_grid, name="2nd")
        self.m3_grid.add_field(512.0 * self.f3_grid, name="3rd")
        self.assertEqual(
            self.m3_grid["2nd"].all(), (256.0 * self.f3_grid).all()
        )
        self.assertEqual(
            self.m3_grid["3rd"].all(), (512.0 * self.f3_grid).all()
        )

        self.m3_tuple["tmp_field"] = 2.0 * self.f3_tuple
        self.assertEqual(
            self.m3_tuple["tmp_field"].all(), (2.0 * self.f3_tuple).all()
        )

    def test_points_getter(self):
        self.assertEqual(self.m1_grid.points, (self.x_grid,))
        self.assertEqual(self.m1_tuple.points, (self.x_tuple,))
        self.assertEqual(self.m2_grid.points, (self.x_grid, self.y_grid))
        self.assertEqual(
            self.m3_grid.points, (self.x_grid, self.y_grid, self.z_grid)
        )
        self.assertEqual(
            self.m3_tuple.points, (self.x_tuple, self.y_tuple, self.z_tuple)
        )

    def test_default_value_type(self):
        # value_type getter with no field set
        self.assertEqual(self.m1_tuple.value_type, None)
        self.assertEqual(self.m2_tuple.value_type, None)
        self.assertEqual(self.m3_tuple.value_type, None)
        self.assertEqual(self.m1_grid.value_type, None)
        self.assertEqual(self.m2_grid.value_type, None)
        self.assertEqual(self.m3_grid.value_type, None)

    def test_field_data_setter(self):
        # attribute creation by adding field_data
        self.m2_tuple.set_field_data(3.14, "mean")
        self.assertEqual(self.m2_tuple.field_data["mean"], 3.14)
        self.assertEqual(self.m2_tuple.mean, 3.14)

    def test_new_points(self):
        # set new points (which causes reset)
        x_tuple2 = self.rng.uniform(0.0, 10, 100)
        y_tuple2 = self.rng.uniform(0.0, 10, 100)
        self.m2_tuple.add_field(self.f2_tuple)

        self.m2_tuple.points = (x_tuple2, y_tuple2)

        self.assertEqual(self.m2_tuple.points, (x_tuple2, y_tuple2))

        # previous field has to be deleted
        self.assertEqual(self.m2_tuple.field, None)

    def test_add_field(self):
        # structured
        self.m1_grid.add_field(self.f1_grid)
        self.assertEqual(self.m1_grid.field.all(), self.f1_grid.all())
        self.m2_grid.add_field(self.f2_grid)
        self.assertEqual(self.m2_grid.field.all(), self.f2_grid.all())
        self.m3_grid.add_field(self.f3_grid)
        self.assertEqual(self.m3_grid.field.all(), self.f3_grid.all())

        # unstructured
        self.m1_tuple.add_field(self.f1_tuple)
        self.assertEqual(self.m1_tuple.field.all(), self.f1_tuple.all())
        self.m2_tuple.add_field(self.f2_tuple)
        self.assertEqual(self.m2_tuple.field.all(), self.f2_tuple.all())
        self.m3_tuple.add_field(self.f3_tuple)
        self.assertEqual(self.m3_tuple.field.all(), self.f3_tuple.all())

        # multiple fields
        new_field = 10.0 * self.f1_grid
        self.m1_grid.add_field(new_field, name="2nd")
        # default field
        self.assertEqual(self.m1_grid.field.all(), self.f1_grid.all())
        self.assertEqual(self.m1_grid["2nd"].all(), new_field.all())
        # overwrite default field
        newer_field = 100.0 * self.f1_grid
        self.m1_grid.add_field(newer_field, name="3rd", is_default_field=True)
        self.assertEqual(self.m1_grid.field.all(), newer_field.all())

    def test_default_field(self):
        # first field added, should be default_field
        f2_grid = 5.0 * self.f1_grid
        self.m1_grid.add_field(self.f1_grid, name="test_field1")
        self.m1_grid.add_field(f2_grid, name="test_field2")
        self.assertEqual(self.m1_grid.default_field, "test_field1")
        self.m1_grid.default_field = "test_field2"
        self.assertEqual(self.m1_grid.default_field, "test_field2")
        self.assertEqual(self.m1_grid.field[5], self.m1_grid["test_field2"][5])

    def test_reset(self):
        self.m3_tuple.set_field_data("TestLoc", "location")
        self.m3_tuple.del_field_data()
        self.assertEqual(self.m3_tuple.field_data["default_field"], "field")
        self.assertEqual(self.m3_tuple.field_data["mesh_type"], "unstructured")
        self.assertEqual(self.m3_tuple.default_field, "field")
        self.assertEqual(self.m3_tuple.mesh_type, "unstructured")

    def test_point_data_check(self):
        self.assertRaises(ValueError, self.m1_tuple.add_field, self.f1_grid)
        self.assertRaises(ValueError, self.m1_tuple.add_field, self.f2_grid)
        self.assertRaises(ValueError, self.m1_tuple.add_field, self.f3_grid)
        self.assertRaises(ValueError, self.m2_tuple.add_field, self.f1_grid)
        self.assertRaises(ValueError, self.m2_tuple.add_field, self.f2_grid)
        self.assertRaises(ValueError, self.m2_tuple.add_field, self.f3_grid)
        self.assertRaises(ValueError, self.m3_tuple.add_field, self.f1_grid)
        self.assertRaises(ValueError, self.m3_tuple.add_field, self.f2_grid)
        self.assertRaises(ValueError, self.m3_tuple.add_field, self.f3_grid)

        self.assertRaises(ValueError, self.m1_grid.add_field, self.f2_grid)
        self.assertRaises(ValueError, self.m1_grid.add_field, self.f3_grid)
        self.assertRaises(ValueError, self.m2_grid.add_field, self.f1_grid)
        self.assertRaises(ValueError, self.m2_grid.add_field, self.f3_grid)
        self.assertRaises(ValueError, self.m3_grid.add_field, self.f1_grid)
        self.assertRaises(ValueError, self.m3_grid.add_field, self.f2_grid)
        self.assertRaises(ValueError, self.m1_grid.add_field, self.f1_tuple)
        self.assertRaises(ValueError, self.m1_grid.add_field, self.f2_tuple)
        self.assertRaises(ValueError, self.m1_grid.add_field, self.f3_tuple)
        self.assertRaises(ValueError, self.m2_grid.add_field, self.f1_tuple)
        self.assertRaises(ValueError, self.m2_grid.add_field, self.f2_tuple)
        self.assertRaises(ValueError, self.m2_grid.add_field, self.f3_tuple)
        self.assertRaises(ValueError, self.m3_grid.add_field, self.f1_tuple)
        self.assertRaises(ValueError, self.m3_grid.add_field, self.f2_tuple)
        self.assertRaises(ValueError, self.m3_grid.add_field, self.f3_tuple)

        x_tuple2 = self.rng.uniform(0.0, 10, 100)
        y_tuple2 = self.rng.uniform(0.0, 10, 100)
        f_tuple2 = np.vstack((x_tuple2, y_tuple2))
        x_tuple3 = self.rng.uniform(0.0, 10, (3, 100))
        y_tuple3 = self.rng.uniform(0.0, 10, (3, 100))
        z_tuple3 = self.rng.uniform(0.0, 10, (3, 100))
        f_tuple3 = np.vstack((x_tuple2, y_tuple2, z_tuple3))

        m2_tuple = Mesh(2, (x_tuple2, y_tuple2))
        m3_tuple = Mesh(3, (x_tuple3, y_tuple3, z_tuple3))

        self.assertRaises(ValueError, m2_tuple.add_field, f_tuple3)
        self.assertRaises(ValueError, m3_tuple.add_field, f_tuple2)

        f_grid2 = np.zeros((2, len(self.x_grid), len(self.y_grid)))
        f_grid3 = np.zeros(
            (3, len(self.x_grid), len(self.y_grid), len(self.z_grid))
        )

        self.assertRaises(ValueError, self.m2_grid.add_field, f_grid3)
        self.assertRaises(ValueError, self.m3_grid.add_field, f_grid2)


if __name__ == "__main__":
    unittest.main()
