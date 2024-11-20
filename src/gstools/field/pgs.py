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


class PGS:
    """A simple class to generate plurigaussian field simulations (PGS).

    See [Ricketts2023]_ for more details.

    Parameters
    ----------
    dim : :class:`int`
        dimension of the field
    fields : :class:`list` or :class:`numpy.ndarray`
        For `dim > 1` a list of spatial random fields (SRFs), with
        `len(fields) == dim`. For `dim == 1`, the SRF can be directly given,
        instead of a list. This class supports structured and unstructured meshes.
        All fields must have the same shapes.
    facies : :class:`numpy.ndarray`
        A `dim` dimensional structured field, whose values are mapped to the PGS.
        It does not have to have the same shape as the `fields`, as the indices are
        automatically scaled.

    References
    ----------
    .. [Ricketts2023] Ricketts, E.J., Freeman, B.L., Cleall, P.J. et al.
        A Statistical Finite Element Method Integrating a Plurigaussian Random
        Field Generator for Multi-scale Modelling of Solute Transport in
        Concrete. Transp Porous Med 148, 95–121 (2023)
        https://doi.org/10.1007/s11242-023-01930-8
    """

    def __init__(self, dim, fields, facies):
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
        self._Zs = fields
        self._L = np.array(facies)
        if len(self._L.shape) != dim:
            raise ValueError("PGS: Mismatch between dim. and facies shape.")
        self._P = self.calc_pgs()

    def calc_pgs(self):
        """Generate the plurigaussian field.

        The PGS is saved as `self.P` and is also returned.

        Returns
        -------
        pgs : :class:`numpy.ndarray`
            the plurigaussian field
        """
        try:
            mapping = np.stack(self._Zs, axis=1)
        except np.AxisError:
            # if dim==1, `fields` is prob. a raw field & not a 1-tuple or
            # equivalent
            if self._dim == 1:
                self._Zs = [self._Zs]
                mapping = np.stack(self._Zs, axis=1)
            else:
                raise
        pos_l = []
        for d in range(self._dim):
            pos_l.append(
                np.linspace(
                    np.floor(self._Zs[d].min()) - 1,
                    np.ceil(self._Zs[d].max()) + 1,
                    self._L.shape[d],
                )
            )

        P_dig = []
        for d in range(self._dim):
            P_dig.append(np.digitize(mapping[:, d], pos_l[d]))

        # once Py3.11 has reached its EOL, we can drop the 1-tuple :-)
        return self._L[(*P_dig,)]

    @property
    def P(self):
        """:class:`numpy.ndarray`: The plurigaussian field"""
        return self._P
