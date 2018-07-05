# -*- coding: utf-8 -*-
"""
GStools subpackage providing the core of the spatial random field generation.

.. currentmodule:: gstools.field.rng

The following classes are provided

.. autosummary::
   RNG
"""
from __future__ import division, absolute_import, print_function

import numpy as np
import numpy.random as rand

__all__ = [
    "RNG",
]


class RNG(object):
    """
    A random number generator for different distributions and multiple streams.

    It only generates isotropic fields, as anisotropic fields can be created by
    coordinate transformations.

    Examples
    --------
        >>> r = RNG(dim = 2, seed=76386181534)
        >>> Z, k = r(model='gau', len_scale=10., mode_no=100)
    """
    def __init__(self, dim, seed=None):
        '''Initialize a random number generator

        Parameters
        ----------
            dim : :class:`int`
                spatial dimension
            seed : :class:`int`, optional
                set the seed of the master RNG, if "None",
                a random seed is used
        '''
        if dim < 1 or dim > 3:
            raise ValueError('Only dimensions of 1 <= d <= 3 are supported.')
        self._dim = dim
        # set seed
        self._seed = None
        self._master_RNG_fct = None
        self._master_RNG = None
        self.seed = seed

        self._trans_1d = lambda len_scale, r: r / len_scale
        self._trans_2d = [lambda len_scale, r, phi:
                          r*np.cos(phi) / len_scale,
                          lambda len_scale, r, phi:
                          r*np.sin(phi) / len_scale]
        self._trans_3d = [lambda len_scale, r, theta, phi:
                          r * np.sin(theta) * np.cos(phi) / len_scale,
                          lambda len_scale, r, theta, phi:
                          r * np.sin(theta) * np.sin(phi) / len_scale,
                          lambda len_scale, r, theta, phi=None:
                          r * np.cos(theta) / len_scale]

    def __call__(self, model, len_scale, mode_no=1000, **kwargs):
        """ A standardized interface for the different covariance models.

        Parameters
        ----------
            model : :class:`str`
                the covariance model (gau, exp, mat, tfg, and tfe exist)
            len_scale : :class:`float`
                the length scale
            mode_no : :class:`int`, optional
                number of Fourier modes
            **kwargs
                keyword arguments are passed on to the chosen spectrum method

        Returns
        -------
           Z : :class:`numpy.ndarray`
               2 Gaussian distr. arrays
           k : :class:`numpy.ndarray`
               the Fourier samples
        """
        Z = self._create_normal_dists(mode_no)
        try:
            k = getattr(self, model.lower())(len_scale, mode_no, **kwargs)
        except AttributeError:
            raise ValueError('Unknown covariance model type {0}'.format(model))
        return Z, k

    def _get_random_stream(self):
        """ Returns a new RNG stream. DO NOT ACCESS FROM OUTSIDE.
        """
        return rand.RandomState(self._master_RNG())

    def create_empty_k(self, mode_no=None):
        """ Create empty mode array with the correct shape.

        Parameters
        ----------
            mode_no : :class:`int`
                number of the fourier modes

        Returns
        -------
            :class:`numpy.ndarray`
                the empty mode array
        """
        if mode_no is None:
            k = np.empty(self.dim)
        else:
            k = np.empty((self.dim, mode_no))
        return k

    def _create_normal_dists(self, mode_no=None):
        """ Create 2 arrays of normal random variables.

        Parameters
        ----------
            mode_no : :class:`int`, optional
                number of the Fourier modes

        Returns
        -------
            : :class:`numpy.ndarray`
                the normal distributed arrays
        """
        if mode_no is None:
            Z = np.empty(2)
        else:
            Z = np.empty((2, mode_no))
        rng = self._get_random_stream()
        for i in range(2):
            Z[i] = rng.normal(size=mode_no)
        return Z

    def _prepare_random_numbers(self, mode_no):
        """Create and partly fill some standard arrays.

        Parameters
        ----------
            mode_no : :class:`int`, optional
                number of the Fourier modes

        Returns
        -------
            k : :class:`numpy.ndarray`
                the empty mode array
            iid : :class:`numpy.ndarray`
                uniformly distributed variables
        """
        k = self.create_empty_k(mode_no)
        iid = np.empty_like(k)

        for dim_i in range(self.dim):
            rng = self._get_random_stream()
            iid[dim_i] = rng.uniform(0., 1., mode_no)
        return k, iid

    def gau(self, len_scale, mode_no=1000, **kwargs):
        """Compute a Gaussian spectrum.

        Computes the spectral density of following covariance

        .. math:: C(r) = \\exp\\left(r^2 / \\lambda^2\\right),

        with :math:`\\lambda =` len_scale

        Parameters
        ----------
            len_scale : :class:`float`
                the length scale of the distribution
            mode_no : :class:`int`, optional
                number of the Fourier modes
            **kwargs
                not used

        Returns
        -------
            :class:`numpy.ndarray`
                the modes
        """
        k = self.create_empty_k(mode_no)
        for dim_i in range(self.dim):
            rng = self._get_random_stream()
            k[dim_i] = rng.normal(0., 1./len_scale, mode_no)
        return k

    def exp(self, len_scale, mode_no=1000, **kwargs):
        """ Compute an exponential spectrum.

        Computes the spectral density of following covariance

        .. math:: C(r) = \\exp\\left(r / \\lambda\\right),

        with :math:`\\lambda =` len_scale

        Parameters
        ----------
            len_scale : :class:`float`
                the length scale of the distribution
            mode_no : :class:`int`, optional
                number of the Fourier modes
            **kwargs
                not used

        Returns
        -------
            :class:`numpy.ndarray`
                the modes
        """
        raise NotImplementedError('exp covariance model not yet implemented.')

    def mat(self, len_scale, mode_no=1000, **kwargs):
        """ Compute a Matérn spectrum.

        Parameters
        ----------
            len_scale : :class:`float`
                the length scale of the distribution
            mode_no : :class:`int`, optional
                number of the Fourier modes
            **kwargs
                kappa : :class:`float`, optional
                    the kappa coefficient

        Returns
        -------
            :class:`numpy.ndarray`
                the modes
        """
        raise NotImplementedError('mat covariance model not yet implemented.')

    def tfg(self, len_scale, mode_no=1000, **kwargs):
        """ Compute a truncated fractal Gaussian spectrum.

        Parameters
        ----------
            len_scale : :class:`float`
                the upper cutoff scale of the distribution
            mode_no : :class:`int`, optional
                number of the Fourier modes
            **kwargs
                not used

        Returns:
            :class:`numpy.ndarray`
                the modes
        """
        raise NotImplementedError('tfg covariance model not yet implemented.')

    def tfe(self, len_scale, mode_no=None, **kwargs):
        """ Compute a truncated fractal exponential spectrum.

        Args:
            len_scale : :class:`float`
                the upper cutoff scale of the distribution
            mode_no : :class:`int`, optional
                number of the Fourier modes
            **kwargs
                H : :class:`float`
                    the Hurst coefficient

        Returns
        -------
            :class:`numpy.ndarray`
                the modes
        """
        raise NotImplementedError('tfe covariance model not yet implemented.')

    @property
    def dim(self):
        """:class:`int`: The dimension of the spatial random field.
        """
        return self._dim

    @property
    def seed(self):
        """:class:`int`: the seed of the master RNG

        The setter property not only saves the new seed, but also creates
        a new master RNG function with the new seed.
        """
        return self._seed

    @seed.setter
    def seed(self, new_seed=None):
        self._seed = new_seed
        self._master_RNG_fct = rand.RandomState(new_seed)
        self._master_RNG = (lambda:
                            self._master_RNG_fct.random_integers(2**16 - 1))

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return 'RNG(dim={0:d}, seed={1})'.format(self.dim, self.seed)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
