"""
GStools subpackage providing plurigaussian simulation (PGS)

.. currentmodule:: gstools.field.pgs

The following classes are provided

.. autosummary::
   :toctree:

   PGS
"""

# pylint: disable=C0103
import numpy as np

# very clunky way of supporting both np 1.x and 2.x exceptions
try:
    np.AxisError = np.exceptions.AxisError
except AttributeError:
    ...


class PGS:
    """A class to generate plurigaussian field simulations (PGS).

    See e.g. [Ricketts2023]_ and [Armstrong2011]_ for more details.

    Parameters
    ----------
    dim : :class:`int`
        dimension of the field
    fields : :class:`list` or :class:`numpy.ndarray`
        For `dim > 1` a list of spatial random fields (SRFs), with
        `len(fields) == dim`. For `dim == 1`, the SRF can be directly given,
        instead of a list. This class supports structured and unstructured meshes.
        All fields must have the same shapes.

    Notes
    -----
    Using plurigaussian fields for conditioning fields is still a beta feature.

    References
    ----------
    .. [Ricketts2023] Ricketts, E.J., Freeman, B.L., Cleall, P.J. et al.
        A Statistical Finite Element Method Integrating a Plurigaussian Random
        Field Generator for Multi-scale Modelling of Solute Transport in
        Concrete. Transp Porous Med 148, 95–121 (2023)
        https://doi.org/10.1007/s11242-023-01930-8
    .. [Armstrong2011] Armstrong, Margaret, et al.
        Plurigaussian simulations in geosciences.
        Springer Science & Business Media, 2011.
        https://doi.org/10.1007/978-3-642-19607-2
    """

    def __init__(self, dim, fields):
        # hard to test for 1d case
        if dim > 1:
            if dim != len(fields):
                raise ValueError(
                    "PGS: Mismatch between dim. and no. of fields."
                )
        for d in range(1, dim):
            if not fields[0].shape == fields[d].shape:
                raise ValueError("PGS: Not all fields have the same shape.")
        self._dim = dim
        self._fields = np.array(fields)
        self._lithotypes = None
        self._pos_lith = None
        try:
            self._mapping = np.stack(self._fields, axis=1)
        except np.AxisError:
            # if dim==1, `fields` is prob. a raw field & not a 1-tuple or
            # equivalent
            if self._dim == 1:
                self._fields = np.array([self._fields])
                self._mapping = np.stack(self._fields, axis=1)
            else:
                raise

    def __call__(self, lithotypes):
        """Generate the plurigaussian field.

        Parameters
        ----------
        lithotypes : :class:`numpy.ndarray`
            A `dim` dimensional structured field, whose values are mapped to the PGS.
            It does not have to have the same shape as the `fields`, as the indices are
            automatically scaled.
        Returns
        -------
        pgs : :class:`numpy.ndarray`
            the plurigaussian field
        """
        self._lithotypes = np.array(lithotypes)
        if len(self._lithotypes.shape) != self._dim:
            raise ValueError("PGS: Mismatch between dim. and facies shape.")
        self._pos_lith = self.calc_lithotype_axes(self._lithotypes.shape)

        P_dig = []
        for d in range(self._dim):
            P_dig.append(np.digitize(self._mapping[:, d], self._pos_lith[d]))

        # once Py3.11 has reached its EOL, we can drop the 1-tuple :-)
        return self._lithotypes[(*P_dig,)]

    def calc_lithotype_axes(self, lithotypes_shape):
        """Calculate the axes on which the lithorypes are defined.

        With the centroid of the correlations of the SRFs at the center,
        the axes are calculated, which hold all correlations.
        These axes are used for the lithotypes field.

        Parameters
        ----------
        lithotypes_shape : :class:`tuple`
            The shape of the lithotypes field.

        Returns
        -------
        pos_lith : :class:`numpy.ndarray`
            the axes holding all field correlations
        """
        pos_lith = []
        try:
            # structured grid
            centroid = self._fields.mean(axis=tuple(range(1, self._dim + 1)))
        except np.AxisError:
            # unstructured grid
            centroid = self._fields.mean(axis=1)
        for d in range(self._dim):
            l = np.floor(self._fields[d].min()) - 1
            h = np.ceil(self._fields[d].max()) + 1
            m = (h + l) / 2.0
            dist = max(np.abs(h - m), np.abs(l - m))
            pos_lith.append(
                np.linspace(
                    centroid[d] - dist,
                    centroid[d] + dist,
                    lithotypes_shape[d],
                )
            )
        return pos_lith

    def transform_coords(self, lithotypes_shape, pos):
        """Transform position from correlation coords to L indices.

        This is a helper method to get the lithoty pes indices for given
        correlated field values.

        Parameters
        ----------
        lithotypes_shape : :class:`tuple`
            The shape of the lithotypes field.
        pos : :class:`numpy.ndarray`
            The position in field coordinates, which will be transformed.

        Returns
        -------
        pos_trans : :class:`list`
            the transformed position tuple
        """
        pos_trans = []
        pos_lith = self.calc_lithotype_axes(lithotypes_shape)
        for d in range(self._dim):
            pos_trans.append(
                int(
                    (pos[d] - pos_lith[d][0])
                    / (pos_lith[d][-1] - pos_lith[d][0])
                    * lithotypes_shape[d]
                )
            )
        return pos_trans
