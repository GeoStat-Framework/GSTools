# -*- coding: utf-8 -*-
"""
GStools subpackage providing a base class for kriging.

.. currentmodule:: gstools.krige.base

The following classes are provided

.. autosummary::
   Krige
"""
# pylint: disable=C0103

import numpy as np

# from scipy.linalg import inv
from scipy.spatial.distance import cdist
from gstools.field.tools import reshape_field_from_unstruct_to_struct
from gstools.field.base import Field
from gstools.krige.krigesum import krigesum
from gstools.krige.tools import set_condition

__all__ = ["Krige"]


class Krige(Field):
    """
    A base class for kriging.

    Parameters
    ----------
    model : :any:`CovModel`
        Covariance Model used for kriging.
    cond_pos : :class:`list`
        tuple, containing the given condition positions (x, [y, z])
    cond_val : :class:`numpy.ndarray`
        the values of the conditions
    mean : :class:`float`, optional
        mean value of the kriging field
    """

    def __init__(self, model, cond_pos, cond_val, mean=0.0):
        super().__init__(model, mean)
        self.krige_var = None
        # initialize private attributes
        self._value_type = "scalar"
        self._cond_pos = None
        self._cond_val = None
        self._krige_mat = None
        self._krige_cond = None
        self.set_condition(cond_pos, cond_val)

    def __call__(
        self, pos, mesh_type="unstructured", ext_drift=None, chunk_size=None
    ):
        """
        Generate the kriging field.

        The field is saved as `self.field` and is also returned.
        The error variance is saved as `self.krige_var` and is also returned.

        Parameters
        ----------
        pos : :class:`list`
            the position tuple, containing main direction and transversal
            directions (x, [y, z])
        mesh_type : :class:`str`, optional
            'structured' / 'unstructured'
        ext_drift : :class:`numpy.ndarray` or :any:`None`, optional
            the external drift values at the given positions (only for EDK)

        Returns
        -------
        field : :class:`numpy.ndarray`
            the kriged field
        krige_var : :class:`numpy.ndarray`
            the kriging error variance
        """
        self.mesh_type = mesh_type
        # internal conversation
        x, __, __, self.pos, __, mt_changed, ax_lens = self.pre_pos(
            pos, mesh_type, make_unstruct=True
        )
        point_no = len(x)
        # set chunk size
        chunk_size = point_no if chunk_size is None else int(chunk_size)
        chunk_no = int(np.ceil(point_no / chunk_size))
        field = np.empty_like(x)
        krige_var = np.empty_like(x)
        ext_drift = self.pre_ext_drift(point_no, ext_drift)
        # iterate of chunks
        for i in range(chunk_no):
            # get chunk slice for actual chunk
            chunk_slice = (i * chunk_size, (i + 1) * chunk_size)
            c_slice = slice(*chunk_slice)
            # get RHS of the kriging system (access pos via self.pos)
            k_vec = self.krige_vecs(chunk_slice, ext_drift)
            # generate the raw kriging field and error variance
            field[c_slice], krige_var[c_slice] = krigesum(
                self.krige_mat, k_vec, self.krige_cond
            )
        # reshape field if we got a structured mesh
        if mt_changed:
            field = reshape_field_from_unstruct_to_struct(
                self.model.dim, field, ax_lens
            )
            krige_var = reshape_field_from_unstruct_to_struct(
                self.model.dim, krige_var, ax_lens
            )
        self.post_field(field, krige_var)
        return self.field, self.krige_var

    def pre_ext_drift(self, point_no, ext_drift=None):
        """
        Preprocessor for external drift terms.

        Parameters
        ----------
        point_no : :class:`numpy.ndarray`
            Number of points of the mesh.
        ext_drift : :class:`numpy.ndarray` or :any:`None`, optional
            the external drift values at the given positions (only for EDK)
            For multiple external drifts, the first dimension
            should be the index of the drift term.

        Returns
        -------
        ext_drift : :class:`numpy.ndarray` or :any:`None`
            the drift values at the given positions
        """
        if ext_drift is not None:
            shape = (self.drift_no, point_no)
            return np.array(ext_drift, dtype=np.double).reshape(shape)
        return None

    def get_dists(self, pos1, pos2, pos2_slice=(0, None)):
        """
        Calculate pairwise distances.

        Parameters
        ----------
        pos1 : :class:`tuple` of :class:`numpy.ndarray`
            the first position tuple
        pos2 : :class:`tuple` of :class:`numpy.ndarray`
            the second position tuple
        pos2_slice : :class:`tuple` of :class:`int`, optional
            Start and stop of slice for the pos2 array. Default: all values.

        Returns
        -------
        :class:`numpy.ndarray`
            Matrix containing the pairwise distances.
        """
        return cdist(
            np.column_stack(pos1[: self.model.dim]),
            np.column_stack(pos2[: self.model.dim])[slice(*pos2_slice), ...],
        )

    def krige_vecs(self, chunk_slice=(0, None), ext_drift=None):
        """Calculate the RHS of the kriging equation."""
        return None

    def post_field(self, field, krige_var):
        """
        Postprocessing and saving of kriging field and error variance.

        Parameters
        ----------
        field : :class:`numpy.ndarray`
            Raw kriging field.
        krige_var : :class:`numpy.ndarray`
            Raw kriging error variance.
        """
        self.field = field
        self.krige_var = krige_var

    def set_condition(self, cond_pos, cond_val):
        """Set the conditions for kriging.

        Parameters
        ----------
        cond_pos : :class:`list`
            the position tuple of the conditions (x, [y, z])
        cond_val : :class:`numpy.ndarray`
            the values of the conditions
        """
        self._cond_pos, self._cond_val = set_condition(
            cond_pos, cond_val, self.model.dim
        )
        self.update_model()

    @property
    def krige_mat(self):
        """:class:`numpy.ndarray`: The kriging matrix."""
        return self._krige_mat

    @property
    def krige_cond(self):
        """:class:`numpy.ndarray`: The prepared kriging conditions."""
        return np.pad(self.cond_val, (0, self.drift_no), constant_values=0)

    @property
    def drift_no(self):
        """:class:`int`: Number of drift values per point."""
        return 0

    @property
    def cond_pos(self):
        """:class:`list`: The position tuple of the conditions."""
        return self._cond_pos

    @property
    def cond_val(self):
        """:class:`list`: The values of the conditions."""
        return self._cond_val


if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod()
