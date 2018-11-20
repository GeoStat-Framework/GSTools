# -*- coding: utf-8 -*-
"""
GStools subpackage providing generators for spatial random fields.

.. currentmodule:: gstools.field.generator

The following classes are provided

.. autosummary::
   RandMeth
"""
from __future__ import division, absolute_import, print_function

import numpy as np
from gstools.random.rng import RNG

__all__ = ["RandMeth"]


class RandMeth(object):
    """Randomization method for calculating isotropic spatial random fields.

    Examples
    --------
    """

    def __init__(self, model, mode_no=1000, seed=None, **kwargs):
        """Initialize the randomization method

        Parameters
        ----------
            model : :class:`gstools.CovModel`
                covariance model
            mode_no : :class:`int`, optional
                number of Fourier modes. Default: 1000
            seed : :class:`int`, optional
                the seed of the random number generator.
                If "None", a random seed is used. Default: None
        """
        self._seed = None
        self._rng = None
        self._Z = None
        self._k = None
        self.reset(model, mode_no, seed, **kwargs)

    def reset(self, model, mode_no=1000, seed=None, **kwargs):
        """Reset the random amplitudes and wave numbers with a new seed.

        Parameters
        ----------
            model : :class:`gstools.CovModel`
                covariance model
            mode_no : :class:`int`, optional
                number of Fourier modes. Default: 1000
            seed : :class:`int`, optional
                the seed of the random number generator.
                If "None", a random seed is used. Default: None
        """
        self._model = model
        self._mode_no = mode_no
        self._kwargs = kwargs
        self._seed = np.nan
        self.seed = seed

    def __call__(self, x, y=None, z=None):
        """Calculates the random modes for the randomization method.

        Parameters
        ----------
            x : :class:`float`, :class:`numpy.ndarray`
                the x components of the position tuple, the shape has to be
                (len(x), 1, 1) for 3d and accordingly shorter for lower
                dimensions
            y : :class:`float`, :class:`numpy.ndarray`, optional
                the y components of the pos. tupls
            z : :class:`float`, :class:`numpy.ndarray`, optional
                the z components of the pos. tuple

        Returns
        -------
            :class:`numpy.ndarray`
                the random modes
        """
        self._Z, self._k = self._rng(
            self._model,
            self._len_scale,
            mode_no=self._mode_no,
            kwargs=self._kwargs,
        )
        # preshape for unstructured grid
        for dim_i in range(self._dim):
            self._k[dim_i] = np.squeeze(self._k[dim_i])
            self._k[dim_i] = np.reshape(
                self._k[dim_i], (1, len(self._k[dim_i]))
            )

        summed_modes = np.broadcast(x, y, z)
        summed_modes = np.squeeze(np.zeros(summed_modes.shape))

        # Test to see if enough memory is available.
        # In case there isn't, divide Fourier modes into smaller chunks
        # TODO: make a better guess fo the chunk_no according to the input
        chunk_no_exp = 0
        while True:
            try:
                chunk_no = 2 ** chunk_no_exp
                chunk_len = int(np.ceil(self._mode_no / chunk_no))
                for chunk in range(chunk_no):
                    a = chunk * chunk_len
                    # In case k[d,a:e] with e >= len(k[d,:]) causes errors in
                    # numpy, use the commented min-function below
                    # e = min((chunk + 1) * chunk_len, self.mode_no-1)
                    e = (chunk + 1) * chunk_len

                    if self._dim == 1:
                        phase = self._k[0, a:e] * x
                    elif self._dim == 2:
                        phase = self._k[0, a:e] * x + self._k[1, a:e] * y
                    else:
                        phase = (
                            self._k[0, a:e] * x
                            + self._k[1, a:e] * y
                            + self._k[2, a:e] * z
                        )

                    # no factor 2*pi needed
                    summed_modes += np.squeeze(
                        np.sum(
                            self._Z[0, a:e] * np.cos(phase)
                            + self._Z[1, a:e] * np.sin(phase),
                            axis=-1,
                        )
                    )
            except MemoryError:
                chunk_no_exp += 1
                print(
                    "Not enough memory. Dividing Fourier modes into {} "
                    "chunks.".format(chunk_no)
                )
            else:
                break

        return np.sqrt(1.0 / self._mode_no) * summed_modes

    @property
    def seed(self):
        """:class:`int`: the seed of the master RNG

        Notes
        -----
        If a new seed is given, the setter property not only saves the
        new seed, but also creates new random modes with the new seed.
        """
        return self._seed

    @seed.setter
    def seed(self, new_seed=None):
        if new_seed is not self._seed:
            self._seed = new_seed
            self._rng = RNG(self._seed)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return (
            "RandMeth(dim={0}, model={1}, len_scale={2}, mode_no={3}, "
            "seed={4})".format(
                self._dim,
                self._model,
                self._len_scale,
                self._mode_no,
                self.seed,
            )
        )


if __name__ == "__main__":
    import doctest

    doctest.testmod()
