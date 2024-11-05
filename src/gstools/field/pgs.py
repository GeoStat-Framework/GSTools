"""
GStools subpackage providing plurigaussian simulation (PGS)

.. currentmodule:: gstools.field.pgs

The following classes are provided

.. autosummary::
   :toctree:

   PGS
"""

import numpy as np


class PGS:
    def __init__(self, dim, fields, facies):
        # TODO check that dimensions, domain size, ... are the same for all
        # fields
        for d in range(1, dim):
            if not fields[0].shape == fields[d].shape:
                raise ValueError("PGS: Not all fields have the same shape.")
        self._dim = dim
        self._Zs = fields
        self._L = facies
        self._P = self.calc_pgs()

    def calc_pgs(self):
        try:
            mapping = np.stack(self._Zs, axis=1)
        except np.exceptions.AxisError:
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
                    np.floor(self._Zs[d].min()),
                    np.ceil(self._Zs[d].max()),
                    self._L.shape[d],
                )
            )

        P_dig = []
        for d in range(self._dim):
            P_dig.append(np.digitize(mapping[:, d], pos_l[d]))

        return self._L[*P_dig]

    @property
    def P(self):
        """:class:`str`: Plurigaussian field"""
        return self._P
