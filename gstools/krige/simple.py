# -*- coding: utf-8 -*-
"""
GStools subpackage providing a class for simple kriging.

.. currentmodule:: gstools.krige.simple

The following classes are provided

.. autosummary::
   Simple
"""
# pylint: disable=C0103

from scipy.linalg import inv
from gstools.krige.base import Krige

__all__ = ["Simple"]


class Simple(Krige):
    """
    A class for simple kriging.

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

    def get_krige_mat(self):
        """Update the kriging model settings."""
        return inv(self.model.cov_nugget(self.get_dists(self.krige_pos)))

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
        # add the given mean
        self.field = field + self.mean
        self.krige_var = self.model.sill - krige_var

    def krige_vecs(self, pos, chunk_slice=(0, None), ext_drift=None):
        """Calculate the RHS of the kriging equation."""
        return self.model.cov_nugget(
            self.get_dists(self.krige_pos, pos, chunk_slice)
        )

    @property
    def krige_cond(self):
        """:class:`numpy.ndarray`: The prepared kriging conditions."""
        return self.cond_val - self.mean

    def __repr__(self):
        """Return String representation."""
        return "Simple(model={0}, mean={1}, cond_pos={2}, cond_val={3}".format(
            self.model, self.mean, self.cond_pos, self.cond_val
        )


if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod()
