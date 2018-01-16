#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A generator for standard spatial random fields.
"""
from __future__ import division, absolute_import, print_function

import sys
import numpy as np
from gstools.field import RNG


class RandMeth(object):
    """Randomization method for calculating isotropic spatial random fields.

    Args:
        dim (int): spatial dimension
        cov_model (dict): covariance model
        mode_no (int, optional): number of Fourier modes
        seed (int, optional): the seed of the master RNG, if "None",
            a random seed is used

    Examples:
        >>> x_tuple = np.array([ 4, 0, 3])
        >>> y_tuple = np.array([-1, 0, 1])
        >>> x_tuple = np.reshape(x_tuple, (len(x_tuple), 1))
        >>> y_tuple = np.reshape(y_tuple, (len(y_tuple), 1))
        >>> cov_model = {'model' : 'gau', 'scale' : 6.}
        >>> rm = RandMeth(2, cov_model, 100, seed=12091986)
        >>> rm(x_tuple, y_tuple)
    """
    def __init__(self, dim, cov_model, mode_no=1000, seed=None, **kwargs):
        self.reset(dim, cov_model, mode_no, seed, kwargs=kwargs)

    def reset(self, dim, cov_model, mode_no, seed=None, **kwargs):
        """Reset the random amplitudes and wave numbers with a new seed.

        Args:
            dim (int): spatial dimension
            cov_model (dict): covariance model
            mode_no (int, optional): number of Fourier modes
            seed (int, optional): the seed of the master RNG, if "None",
                a random seed is used
        """
        self._dim = dim
        self._cov_model = cov_model
        self._mode_no = mode_no
        self._rng = RNG(self.dim, seed)
        model = self.cov_model['model']
        len_scale = self.cov_model['len_scale']
        self._Z, self._k = self._rng(model, len_scale, mode_no=self.mode_no, kwargs=kwargs)
        #preshape for unstructured grid
        for d in range(self.dim):
            self._k[d] = np.squeeze(self._k[d])
            self._k[d] = np.reshape(self._k[d], (1, len(self._k[d])))

    def __call__(self, x, y=None, z=None):
        """Calculates the random modes for the randomization method.
        
        Args:
            x (float, ndarray): the x components of the position tuple,
                the shape has to be (len(x), 1, 1) for 3d and accordingly
                shorter for lower dimensions
            y (float, ndarray, optional): the y components of the pos. tupls
            z (float, ndarray, optional): the z components of the pos. tuple

        Returns:
            the random modes
        """
        summed_modes = np.broadcast(x, y, z)
        summed_modes = np.squeeze(np.zeros(summed_modes.shape))

        #Test to see if enough memory is available.
        #In case there isn't, divide Fourier modes into smaller chunks
        chunk_no = 1
        chunk_no_exp = 0
        while True:
            try:
                chunk_len = int(np.ceil(self._mode_no / chunk_no))

                for chunk in range(chunk_no):
                    a = chunk * chunk_len
                    #In case k[d,a:e] with e >= len(k[d,:]) causes errors in
                    #numpy, use the commented min-function below
                    #e = min((chunk + 1) * chunk_len, self.mode_no-1)
                    e = (chunk + 1) * chunk_len

                    if self._dim == 1:
                        phase = self._k[0,a:e]*x
                    elif self._dim == 2:
                        phase = self._k[0,a:e]*x + self._k[1,a:e]*y
                    else:
                        phase = (self._k[0,a:e]*x + self._k[1,a:e]*y +
                                 self._k[2,a:e]*z)

                    summed_modes += np.squeeze(
                            np.sum(self._Z[0,a:e] * np.cos(2.*np.pi*phase) +
                                   self._Z[1,a:e] * np.sin(2.*np.pi*phase),
                                   axis=-1))
            except MemoryError:
                chunk_no += 2**chunk_no_exp
                chunk_no_exp += 1
                print('Not enough memory. Dividing Fourier modes into {} '
                      'chunks.'.format(chunk_no))
            else:
                break

        return np.sqrt(1. / self._mode_no) * summed_modes

    @property
    def dim(self):
        """ The dimension of the spatial random field.
        """
        return self._dim

    @property
    def cov_model(self):
        """ The covariance model.
        """
        return self._cov_model

    @property
    def mode_no(self):
        """ The number of the Fourier modes.
        """
        return self._mode_no


if __name__ == '__main__':
    import doctest
    doctest.testmod()
