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
    r"""Randomization method for calculating isotropic spatial random fields.

    Notes
    -----
    The Randomization method is used to generate isotropic
    spatial random fields characterized by a given covarance model.
    The calculation looks like:

    .. math::
       u\left(x\right)=
       \sqrt{\frac{\sigma^{2}}{N}}\cdot
       \sum_{i=1}^{N}\left(
       Z_{1,i}\cdot\cos\left(\left\langle k_{i},x\right\rangle \right)+
       Z_{2,i}\cdot\sin\left(\left\langle k_{i},x\right\rangle \right)
       \right)

    where:

        * :math:`N` : fourier mode number
        * :math:`Z_{j,i}` : random samples from a normal distribution
        * :math:`k_i` : samples from the spectral density distribution of
          the covariance model
    """

    def __init__(self, model, mode_no=1000, seed=None):
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
        self._Z_1 = None
        self._Z_2 = None
        self._cov_sample = None
        self.reset(model, mode_no, seed)

    def reset(self, model, mode_no=1000, seed=None):
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
        self._seed = np.nan
        self.seed = seed

    def __call__(self, x, y=None, z=None, chunk_tmp_size=1e6):
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
            chunk_tmp_size : :class:`int`, optional
                Number of temporarily stored points for an initial guess
                of the chunk number. Default: 1e6
        Returns
        -------
            :class:`numpy.ndarray`
                the random modes
        """
        summed_modes = np.broadcast(x, y, z)
        summed_modes = np.squeeze(np.zeros(summed_modes.shape))
        # Test to see if enough memory is available.
        # In case there isn't, divide Fourier modes into smaller chunks
        # make a better guess fo the chunk_no according to the input
        tmp_pnt = 0
        if x is not None:
            tmp_pnt += len(x)
        if y is not None:
            tmp_pnt += len(y)
        if z is not None:
            tmp_pnt += len(z)
        chunk_no_exp = int(max(0, np.ceil(np.log2(tmp_pnt / chunk_tmp_size))))
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

                    if self._model.dim == 1:
                        phase = self._cov_sample[0, a:e] * x
                    elif self._model.dim == 2:
                        phase = (
                            self._cov_sample[0, a:e] * x
                            + self._cov_sample[1, a:e] * y
                        )
                    else:
                        phase = (
                            self._cov_sample[0, a:e] * x
                            + self._cov_sample[1, a:e] * y
                            + self._cov_sample[2, a:e] * z
                        )
                    summed_modes += np.squeeze(
                        np.sum(
                            self._Z_1[a:e] * np.cos(phase)
                            + self._Z_2[a:e] * np.sin(phase),
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

        if self._model.nugget > 0:
            nugget = np.sqrt(self._model.nugget) * self._rng.random.normal(
                size=summed_modes.shape
            )
        else:
            nugget = 0.0

        return np.sqrt(self._model.var / self._mode_no) * summed_modes + nugget

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
            self._Z_1 = self._rng.random.normal(size=self._mode_no)
            self._Z_2 = self._rng.random.normal(size=self._mode_no)
            # sample uniform on a sphere
            sphere_coord = self._rng.sample_sphere(
                self._model.dim, self._mode_no
            )
            # sample radii acording to radial spectral density of the model
            if self._model.has_ppf:
                rad = self._rng.sample_dist(
                    size=self._mode_no,
                    pdf=self._model.spectral_rad_pdf,
                    cdf=self._model.spectral_rad_cdf,
                    ppf=self._model.spectral_rad_ppf,
                    a=0,
                )
            else:
                rad = self._rng.sample_ln_pdf(
                    ln_pdf=self._model.ln_spectral_rad_pdf, size=self._mode_no
                )
            # get fully spatial samples by multiplying sphere samples and radii
            self._cov_sample = rad * sphere_coord

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "RandMeth(model={0}, mode_no={1}, seed={2})".format(
            repr(self._model), self._mode_no, self.seed
        )


if __name__ == "__main__":
    import doctest

    doctest.testmod()
