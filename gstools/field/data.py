# -*- coding: utf-8 -*-
"""
GStools subpackage providing a data class for spatial fields.

.. currentmodule:: gstools.field.data

The following classes are provided

.. autosummary::
   FieldData
"""
# pylint: disable=C0103

__all__ = ["FieldData"]


from typing import List, Dict
import numpy as np


class Data:
    """A data class mainly storing the specific values of an individual field.
    """

    def __init__(
        self, values: np.ndarray, mean: float = 0.0, value_type: str = "scalar"
    ):
        self.values = values
        self.mean = mean
        self.value_type = value_type


class FieldData:
    """A base class encapsulating field data.

    It holds a position array, which define the spatial locations of the
    field values.
    It can hold multiple fields in the :any:`self.values` list. This assumes
    that each field is defined on the same positions.
    The mesh type must also be specified.

    Parameters
    ----------
    pos : :class:`numpy.ndarray`, optional
        positions of the field values
    values : :any:`list`, optional
        a list of the values of the fields

    Examples
    --------
    >>> import numpy as np
    >>> pos = np.random.random((100, 100))
    >>> z = np.random.random((100, 100))
    >>> z2 = np.random.random((100, 100))
    >>> field_data = FieldData(pos)
    >>> field_data.add_field("test_field1", z)
    >>> field_data.add_field("test_field2", z2)
    >>> field_data.set_default_field("test_field2")
    >>> print(field.values)

    """

    def __init__(
        self, pos: np.ndarray = None, mesh_type: str = "unstructured",
    ):
        # initialize attributes
        self.pos = pos
        self.fields: Dict[str, np.ndarray] = {}
        if mesh_type != "unstructured" and mesh_type != "structured":
            raise ValueError("Unknown 'mesh_type': {}".format(mesh_type))
        self.mesh_type = mesh_type

    def add_field(
        self,
        name: str,
        values: np.ndarray,
        mean: float = 0.0,
        *,
        value_type: str = "scalar",
        default_field: bool = False,
    ):
        values = np.array(values)
        self._check_field(values)
        self.fields[name] = Data(values, mean, value_type)
        # set the default_field to the first field added
        if len(self.fields) == 1 or default_field:
            self.default_field = name

    def get_data(self, key):
        """:class:`Data`: The field data class."""
        return self.fields[key]

    def set_default_field(self, default_field):
        self.default_field = default_field

    def __setitem__(self, key, value):
        self.fields[key].values = value

    def __getitem__(self, key):
        """:any:`numpy.ndarray`: The values of the field."""
        return self.fields[key].values

    @property
    def field(self):
        return self.fields[self.default_field]

    @property
    def values(self):
        return self.fields[self.default_field].values

    @values.setter
    def values(self, values):
        self.fields[self.default_field].values = values

    @property
    def value_type(self):
        return self.fields[self.default_field].value_type

    @property
    def mean(self):
        return self.fields[self.default_field].mean

    @mean.setter
    def mean(self, value):
        self.fields[self.default_field].mean = value

    def _check_field(self, values: np.ndarray):
        # TODO
        if self.mesh_type == "unstructured":
            pass
        elif self.mesh_type == "structured":
            pass
        else:
            raise ValueError("Unknown 'mesh_type': {}".format(mesh_type)
