#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This is the core of the spatial random field generation.
"""
from __future__ import division, absolute_import, print_function

import numpy as np
import numpy.random as rand


class RNG(object):
    """
    A random number generator for different distributions and multiple streams.

    It only generates isotropic fields, as anisotropic fields can be created by
    coordinate transformations.

    Args:
        dim (int): spatial dimension
        seed (int, opt.): set the seed of the master RNG, if "None",
            a random seed is used

    Examples:
        >>> r = RNG(dim = 2, seed=76386181534)
        >>> Z, k = r(model='gau', len_scale=10., mode_no=100)
    """
    def __init__(self, dim, seed=None):
        if dim < 1 or dim > 3:
            raise ValueError('Only dimensions of 1 <= d <= 3 are supported.')
        self._dim = dim
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

        Args:
            model (str): the covariance model (gau, exp, mat, tfg, and tfe exist)
            len_scale (float): the length scale
            mode_no (int, opt.): number of Fourier modes

        Keyword Args:
            kwargs: they are passed on to the chosen spectrum method

        Returns:
           Z (ndarray): 2 Gaussian distr. arrays
           k (ndarray): the Fourier samples
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

        Args:
            mode_no (int): number of the fourier modes

        Returns:
            the empty mode array
        """
        if mode_no is None:
            k = np.empty(self.dim)
        else:
            k = np.empty((self.dim, mode_no))
        return k

    def _create_normal_dists(self, mode_no=None):
        """ Create 2 arrays of normal random variables.

        Args:
            mode_no (int, opt.): number of the Fourier modes

        Returns:
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
        """ Create and partly fill some standard arrays.

        Args:
            mode_no (int, opt.): number of the Fourier modes

        Returns:
            k (ndarray): the empty mode array
            iid (ndarray): uniformly distributed variables
        """
        k = self.create_empty_k(mode_no)
        iid = np.empty_like(k)

        for d in range(self.dim):
            rng = self._get_random_stream()
            iid[d] = rng.uniform(0., 1., mode_no)
        return k, iid

    def gau(self, len_scale, mode_no=1000, **kwargs):
        """ Compute a Gaussian spectrum.

        Computes the spectral density of following covariance

        .. math:: C(r) = \\exp\\left(r^2 / \\lambda^2\\right),

        with :math:`\\lambda =` len_scale

        Args:
            len_scale (float): the length scale of the distribution
            mode_no (int, opt.): number of the Fourier modes
            **kwargs: not used

        Returns:
            the modes
        """
        k = self.create_empty_k(mode_no)
        for d in range(self.dim):
            rng = self._get_random_stream()
            k[d] = rng.normal(0., 1./len_scale, mode_no)
        return k

    def exp(self, len_scale, mode_no=1000, **kwargs):
        """ Compute an exponential spectrum.

        Computes the spectral density of following covariance

        .. math:: C(r) = \\exp\\left(r / \\lambda\\right),

        with :math:`\\lambda =` len_scale

        Args:
            len_scale (float): the length scale of the distribution
            mode_no (int, opt.): number of the Fourier modes
            **kwargs: not used

        Returns:
            the modes
        """
        raise NotImplementedError('exp covariance model not yet implemented.')

    def mat(self, len_scale, mode_no=1000, **kwargs):
        """ Compute a Matérn spectrum.

        Args:
            len_scale (float): the length scale of the distribution
            mode_no (int, opt.): number of the Fourier modes
            **kwargs: see below

        Keyword Args:
            kappa (float, opt.): the kappa coefficient

        Returns:
            the modes
        """
        raise NotImplementedError('mat covariance model not yet implemented.')

    def tfg(self, len_scale, mode_no=1000, **kwargs):
        """ Compute a truncated fractal Gaussian spectrum.

        Args:
            len_scale (float): the upper cutoff scale of the distribution
            mode_no (int, opt.): number of the Fourier modes
            **kwargs: not used

        Returns:
            the modes
        """
        raise NotImplementedError('tfg covariance model not yet implemented.')

    def tfe(self, len_scale, mode_no=None, **kwargs):
        """ Compute a truncated fractal exponential spectrum.

        Args:
            len_scale (float): the upper cutoff scale of the distribution
            mode_no (int, opt.): number of the Fourier modes
            **kwargs: see below

        Keyword Args:
            H (float): the Hurst coefficient

        Returns:
            the modes
        """
        raise NotImplementedError('tfe covariance model not yet implemented.')

    @property
    def dim(self):
        """ The dimension of the spatial random field.
        """
        return self._dim

    @property
    def seed(self):
        """ seed (int): the seed of the master RNG

        The setter property not only saves the new seed, but also creates
        a new master RNG function with the new seed.
        """
        return self._seed

    @seed.setter
    def seed(self, new_seed=None):
        self._seed = new_seed
        self._master_RNG_fct = rand.RandomState(new_seed)
        self._master_RNG =  (lambda:
                             self._master_RNG_fct.random_integers(2**16 - 1))


if __name__ == '__main__':
    import doctest
    doctest.testmod()
