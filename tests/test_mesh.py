import unittest
from typing import Dict, Tuple
import numpy as np


def value_type(mesh_type, shape):
    """Determine the value type ("scalar" or "vector")"""
    r = "scalar"
    if mesh_type == "unstructured":
        # TODO is this the right place for doing these checks?!
        if len(shape) == 2 and 2 <= shape[0] <= 3:
            r = "vector"
    else:
        # for very small (2-3 points) meshes, this could break
        # a 100% solution would require dim. information.
        if len(shape) == shape[0] + 1:
            r = "vector"
    return r


class Mesh:
    def __init__(
        self,
        pos=None,
        name: str = "field",
        values: np.ndarray = None,
        *,
        mesh_type: str = "unstructured",
    ) -> None:
        # the pos/ points of the mesh
        self._pos = pos

        # data stored at each pos/ point, the "fields"
        self.point_data: Dict[str, np.ndarray] = {name: values}

        # data valid for the global field
        self.field_data = {}

        self.set_field_data("default_field", name)

        # mesh_type needs a special setter, therefore, `set_field_data` is not
        # used here
        self.mesh_type = mesh_type
        self.field_data["mesh_type"] = mesh_type

    def set_field_data(self, name: str, value) -> None:
        """Add an attribute to this instance and add it the `field_data`

        This helper method is used to create attributes for easy access
        while using this class, but it also adds an entry to the dictionary
        `field_data`, which is used for exporting the data.
        """
        setattr(self, name, value)
        self.field_data[name] = value

    def add_field(
        self,
        values: np.ndarray,
        name: str = "field",
        *,
        is_default_field: bool = False,
    ) -> None:
        """Add a field (point data) to the mesh

        .. note::
            If no field has existed before calling this method,
            the given field will be set to the default field.

        .. warning::
            If point data with the same `name` already exists, it will be
            overwritten.

        Parameters
        ----------
        values : :class:`numpy.ndarray`
            the point data, has to be the same shape as the mesh
        name : :class:`str`, optional
            the name of the point data
        is_default_field : :class:`bool`, optional
            is this the default field of the mesh?

        """

        values = np.array(values)
        self._check_point_data(values)
        self.point_data[name] = values
        # set the default_field to the first field added
        if len(self.point_data) == 1 or is_default_field:
            self._default_field = name

    def __getitem__(self, key: str) -> np.ndarray:
        """:any:`numpy.ndarray`: The values of the field."""
        return self.point_data[key]

    def __setitem__(self, key: str, value):
        self.point_data[key] = value

    @property
    def default_field(self) -> str:
        """:class:`str`: the name of the default field."""
        return self._default_field

    @default_field.setter
    def default_field(self, value: str):
        self._default_field = value

    @property
    def pos(self) -> Tuple[np.ndarray]:
        """:any:`numpy.ndarray`: The pos. on which the field is defined."""
        return self._pos

    @pos.setter
    def pos(self, value: Tuple[np.ndarray]):
        """
        Warning: setting new positions deletes all previously stored fields.
        """
        self.point_data = {self.default_field: None}
        self._pos = value

    @property
    def field(self) -> np.ndarray:
        """:class:`numpy.ndarray`: The point data of the default field."""
        return self.point_data[self.default_field]

    @field.setter
    def field(self, values: np.ndarray):
        self._check_point_data(values)
        self.point_data[self.default_field] = values

    @property
    def value_type(self, field="field") -> str:
        """:any:`str`: The value type of the default field."""
        if self.point_data[field] is None:
            r = None
        else:
            r = value_type(self.mesh_type, self.point_data[field].shape)
        return r

    @property
    def mesh_type(self) -> str:
        """:any:`str`: The mesh type of the fields."""
        return self._mesh_type

    @mesh_type.setter
    def mesh_type(self, value: str):
        """
        Warning: setting a new mesh type deletes all previously stored fields.
        """
        self._check_mesh_type(value)
        self.point_data = {self.default_field: None}
        self._mesh_type = value

    def _check_mesh_type(self, mesh_type: str) -> None:
        if mesh_type != "unstructured" and mesh_type != "structured":
            raise ValueError("Unknown 'mesh_type': {}".format(mesh_type))

    def _check_point_data(self, values: np.ndarray):
        """Compare field shape to pos shape.

        Parameters
        ----------
        values : :class:`numpy.ndarray`
            the values of the field to be checked
        """
        err = True
        if self.mesh_type == "unstructured":
            # scalar
            if len(values.shape) == 1:
                if values.shape[0] == len(self.pos[0]):
                    err = False
            # vector
            elif len(values.shape) == 2:
                if (
                    values.shape[1] == len(self.pos[0])
                    and 2 <= values.shape[0] <= 3
                ):
                    err = False
            if err:
                raise ValueError(
                    "Wrong field shape: {0} does not match mesh shape ({1},)".format(
                        values.shape, len(self.pos[0])
                    )
                )
        else:
            # scalar
            if len(values.shape) == len(self.pos):
                if all(
                    [
                        values.shape[i] == len(self.pos[i])
                        for i in range(len(self.pos))
                    ]
                ):
                    err = False
            # vector
            elif len(values.shape) == len(self.pos) + 1:
                if all(
                    [
                        values.shape[i + 1] == len(self.pos[i])
                        for i in range(len(self.pos))
                    ]
                ) and values.shape[0] == len(self.pos):
                    err = False
            if err:
                raise ValueError(
                    "Wrong field shape: {0} does not match mesh shape [0/2/3]{1}".format(
                        list(values.shape),
                        [len(self.pos[i]) for i in range(len(self.pos))],
                    )
                )


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

        self.m1_grid = Mesh((self.x_grid,), mesh_type="structured")
        self.m2_grid = Mesh((self.x_grid, self.y_grid), mesh_type="structured")
        self.m3_grid = Mesh(
            (self.x_grid, self.y_grid, self.z_grid), mesh_type="structured"
        )
        self.m1_tuple = Mesh((self.x_tuple,))
        self.m2_tuple = Mesh((self.x_tuple, self.y_tuple))
        self.m3_tuple = Mesh((self.x_tuple, self.y_tuple, self.z_tuple))

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

    def test_pos_getter(self):
        self.assertEqual(self.m1_grid.pos, (self.x_grid,))
        self.assertEqual(self.m1_tuple.pos, (self.x_tuple,))
        self.assertEqual(self.m2_grid.pos, (self.x_grid, self.y_grid))
        self.assertEqual(
            self.m3_grid.pos, (self.x_grid, self.y_grid, self.z_grid)
        )
        self.assertEqual(
            self.m3_tuple.pos, (self.x_tuple, self.y_tuple, self.z_tuple)
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
        self.m2_tuple.set_field_data("mean", 3.14)
        self.assertEqual(self.m2_tuple.field_data["mean"], 3.14)
        self.assertEqual(self.m2_tuple.mean, 3.14)

    def test_new_pos(self):
        # set new pos. (which causes reset)
        x_tuple2 = self.rng.uniform(0.0, 10, 100)
        y_tuple2 = self.rng.uniform(0.0, 10, 100)
        self.m2_tuple.add_field(self.f2_tuple)

        self.m2_tuple.pos = (x_tuple2, y_tuple2)

        self.assertEqual(self.m2_tuple.pos, (x_tuple2, y_tuple2))

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

        m2_tuple = Mesh((x_tuple2, y_tuple2))
        m3_tuple = Mesh((x_tuple3, y_tuple3, z_tuple3))

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
