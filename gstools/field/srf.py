#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A generator for standard spatial random fields.
"""
from __future__ import division, absolute_import, print_function

import numpy as np
from gstools.field import RNG


class SRF(object):
    """A class to generate a spatial random field (SRF).

    Args:
        dim (int): spatial dimension
        mean (float, opt.): mean value of the SRF
        var (float, opt.): variance of the SRF
        len_scale (float, opt.): the length scale of the SRF in x direction
        model (str, opt.): the covariance model ('gau', 'exp', ..., see RNG)
        anis (float/list, opt.): the anisotropy of length scales along
            the y- and z-directions
        angles (float/list, opt.): the rotation of the stretching, with the
            values corrisponding the yaw, pitch, and roll
        mode_no (int, opt.): number of Fourier modes

    Examples:
        >>> cov_model = {'dim': 2, 'mean': .0, 'var': 2.6, 'len_scale': 4.,
        >>>              'model': 'gau', 'anis': 5., 'rotate': np.pi/4.
        >>>              'mode_no': 100,}
        >>> x = np.arange(0, 10, 1)
        >>> y = np.arange(-5, 5, 0.5)
        >>> srf = SRF(**cov_model)
        >>> field = srf(x, y, seed=987654)
    """
    def __init__(self, dim, mean=0., var=1., len_scale=1., model='gau',
                 anis=1., angles=0., mode_no = 1000):
        if dim < 1 or dim > 3:
            raise ValueError('Only dimensions of 1 <= d <= 3 are supported.')
        self._dim = dim
        self._mean = mean
        self._var = var
        self._len_scale = len_scale
        self._model = model
        self._anis = np.atleast_1d(anis)
        if len(self._anis) < self._dim:
            #fill up the anisotropy array with ones, such that len() == dim
            self._anis = np.pad(self._anis, (0, self._dim-len(self._anis)-1),
                                  'constant', constant_values=1.)
        self._angles = np.atleast_1d(angles)
        #fill up the rotation angle array with zeros, such that len() == dim
        self._angles = np.pad(self._angles, (0, self._dim-len(self._angles)),
                              'constant', constant_values=0.)
        self._mode_no = mode_no
        self._do_rotation = not np.all(np.isclose(angles, 0.))
        self._randmeth = RandMeth(self._dim, self._model, self._len_scale,
                                  self._mode_no, seed=None)

    def __call__(self, x, y=None, z=None, seed=None, mesh_type='unstructured'):
        """Generate an SRF and return it without saving it internally.

        Args:
            x (ndarray): grid axis in x-direction if structured, or
                first components of position vectors if unstructured
            y (ndarray, opt.): analog to x
            z (ndarray, opt.): analog to x
            seed (int, opt.): seed for RNG
            mesh_type (str): 'structured' / 'unstructured'
        Returns:
            field (ndarray): the SRF
        """
        self._check_mesh(x, y, z, mesh_type)
        mesh_type_changed = False
        if self._do_rotation:
            if mesh_type == 'structured':
                mesh_type_changed = True
                mesh_type_old = mesh_type
                mesh_type = 'unstructured'
                x, y, z, axis_lens = \
                        self._reshape_axis_from_struct_to_unstruct(x, y, z)
            x, y, z = self._unrotate_mesh(x, y, z)

        y, z = self._make_isotropic(y, z)
        x, y, z = self._reshape_input(x, y, z, mesh_type)

        self._randmeth.seed = seed
        field = self._randmeth(x, y, z)

        if mesh_type_changed:
            mesh_type = mesh_type_old
            field = self._reshape_field_from_unstruct_to_struct(field, axis_lens)
        return self._mean + np.sqrt(self._var)*field

    def structured(self, x, y=None, z=None, seed=None):
        """Generate an SRF on a structured mesh without saving it internally.

        Args:
            x (ndarray): grid axis in x-direction if structured
            y (ndarray, opt.): analog to x
            z (ndarray, opt.): analog to x
            seed (int, opt.): seed for RNG
        Returns:
            field (ndarray): the SRF
        """
        return self(x, y, z, seed, 'structured')

    def unstructured(self, x, y=None, z=None, seed=None):
        """Generate an SRF on an unstructured mesh without saving it internally.

        Args:
            x (ndarray): first components of position vectors if unstructured
            y (ndarray, opt.): analog to x
            z (ndarray, opt.): analog to x
            seed (int, opt.): seed for RNG
        Returns:
            field (ndarray): the SRF
        """
        return self(x, y, z, seed)

    def generate(self, x, y=None, z=None, seed=None, mesh_type='unstructured'):
        """Generate an SRF and save it as an attribute self.field.

        Args:
            x (ndarray): grid axis in x-direction if structured, or
                first components of position vectors if unstructured
            y (ndarray, opt.): analog to x
            z (ndarray, opt.): analog to x
            seed (int, opt.): seed for RNG
            mesh_type (str): 'structured' / 'unstructured'
        Returns:
            field (ndarray): the SRF
        """
        self.field = self(x, y, z, seed, mesh_type)
        return self.field

    def _check_mesh(self, x, y, z, mesh_type):
        """Do a basic check of the shapes of the input arrays."""
        if self._dim >= 2:
            if y is None:
                raise ValueError('The y-component is missing for '
                                 '{0} dimensions'.format(self._dim))
        if self._dim == 3:
            if z is None:
                raise ValueError('The z-component is missing for '
                                 '{0} dimensions'.format(self._dim))
        if mesh_type == 'unstructured':
            if self._dim >= 2:
                try:
                    if len(x) != len(y):
                        raise ValueError('len(x) = {0} != len(y) = {1} '
                                         'for unstructured grids'.
                                         format(len(x), len(y)))
                except TypeError:
                    pass
                if self._dim == 3:
                    try:
                        if len(x) != len(z):
                            raise ValueError('len(x) = {0} != len(z) = {1} '
                                             'for unstructured grids'.
                                             format(len(x), len(z)))
                    except TypeError:
                        pass
        elif mesh_type == 'structured':
            pass
        else:
            raise ValueError('Unknown mesh type {0}'.
                              format(mesh_type))

    def _make_isotropic(self, y, z):
        """Stretch given axes in order to implement anisotropy."""
        if self._dim == 1:
            return y, z
        elif self._dim == 2:
            return y * self._anis[0], z
        elif self._dim == 3:
            return y * self._anis[0], z * self._anis[1]

    def R3d_x(self, theta):
        """Rotation matrix about x axis."""
        sin = np.sin(theta)
        cos = np.cos(theta)
        return np.array(((1.,  0.,   0.),
                         (0., cos, -sin),
                         (0., sin, cos)))
    def R3d_y(self, theta):
        """Rotation matrix about y axis."""
        sin = np.sin(theta)
        cos = np.cos(theta)
        return np.array((( cos, 0., sin),
                         (  0., 1.,  0.),
                         (-sin, 0., cos)))
    def R3d_z(self, theta):
        """Rotation matrix about z axis."""
        sin = np.sin(theta)
        cos = np.cos(theta)
        return np.array(((cos, -sin, 0.),
                         (sin,  cos, 0.),
                         ( 0.,   0., 1.)))

    def _unrotate_mesh(self, x, y, z):
        """Rotate axes in order to implement rotation.

        for 3d: yaw, pitch, and roll angles are alpha, beta, and gamma,
        of intrinsic rotation rotation whose Tait-Bryan angles are
        alpha, beta, gamma about axes x, y, z.
        """
        if self._dim == 1:
            return x, y, z
        elif self._dim == 2:
            #extract 2d rotation matrix
            R = self.R3d_z(self._angles[0])[0:2,0:2]
            tuple = np.vstack((x, y))
            tuple = np.vsplit(np.dot(R, tuple), 2)
            x = np.squeeze(tuple[0])
            y = np.squeeze(tuple[1])
            return x, y, z
        elif self._dim == 3:
            alpha = self._angles[0]
            beta = self._angles[1]
            gamma = self._angles[2]
            R = np.dot(np.dot(self.R3d_z(alpha),
                              self.R3d_y(beta)),
                              self.R3d_x(gamma))
            tuple = np.vstack((x, y, z))
            tuple = np.vsplit(np.dot(R, tuple), 3)
            x = np.squeeze(tuple[0])
            y = np.squeeze(tuple[1])
            z = np.squeeze(tuple[2])
            return x, y, z

    def _reshape_input(self, x, y=None, z=None, mesh_type='unstructured'):
        """Reshape given axes, depending on the mesh type."""
        if mesh_type == 'unstructured':
            x, y, z = self._reshape_input_axis_from_unstruct(x, y, z)
        elif mesh_type == 'structured':
            x, y, z = self._reshape_input_axis_from_struct(x, y, z)
        return x, y, z

    def _reshape_input_axis_from_unstruct(self, x, y=None, z=None):
        """Reshape given axes for vectorisation on unstructured grid."""
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        z = np.atleast_1d(z)
        x = np.reshape(x, (len(x), 1))
        y = np.reshape(y, (len(y), 1))
        z = np.reshape(z, (len(z), 1))
        return (x, y, z)

    def _reshape_input_axis_from_struct(self, x, y=None, z=None):
        """Reshape given axes for vectorisation on unstructured grid."""
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        z = np.atleast_1d(z)
        x = np.reshape(x, (len(x), 1, 1, 1))
        y = np.reshape(y, (1, len(y), 1, 1))
        z = np.reshape(z, (1, 1, len(z), 1))
        return (x, y, z)

    def _reshape_axis_from_struct_to_unstruct(self, x, y=None, z=None):
        """Reshape given axes from struct to unstruct for rotation."""
        if self._dim == 1:
            return x, y, z, (len(x),)
        elif self._dim == 2:
            xu, yu = np.meshgrid(x, y, indexing='ij')
            len_unstruct = len(x) * len(y)
            xu = np.reshape(xu, len_unstruct)
            yu = np.reshape(yu, len_unstruct)
            return xu, yu, z, (len(x), len(y))
        elif self._dim == 3:
            xu, yu, zu = np.meshgrid(x, y, z, indexing='ij')
            len_unstruct = len(x) * len(y) * len(z)
            xu = np.reshape(xu, len_unstruct)
            yu = np.reshape(yu, len_unstruct)
            zu = np.reshape(zu, len_unstruct)
            return xu, yu, zu, (len(x), len(y), len(z))

    def _reshape_field_from_unstruct_to_struct(self, f, axis_lens):
        """Reshape the rotated field back to struct."""
        if self._dim == 1:
            return f
        elif self._dim == 2:
            f = np.reshape(f, axis_lens)
            return f
        elif self._dim == 3:
            f = np.reshape(f, axis_lens)
            return f

    @property
    def dim(self):
        """ The dimension of the spatial random field.
        """
        return self._dim
    @property
    def mean(self):
        """ The mean of the spatial random field.
        """
        return self._mean
    @property
    def var(self):
        """ The variance of the spatial random field.
        """
        return self._var
    @property
    def len_scale(self):
        """ The length scale of the spatial random field.
        """
        return self._len_scale
    @property
    def model(self):
        """ The length scale of the spatial random field.
        """
        return self._model
    @property
    def anis(self):
        """ The anisotropy factors of the spatial random field.
        """
        return self._anis
    @property
    def angles(self):
        """ The rotation angles (in rad) of the spatial random field.
        """
        return self._angles


class RandMeth(object):
    """Randomization method for calculating isotropic spatial random fields.

    Args:
        dim (int): spatial dimension
        model (dict): covariance model
        mode_no (int, opt.): number of Fourier modes
        seed (int, opt.): the seed of the master RNG, if "None",
            a random seed is used

    Examples:
        >>> x_tuple = np.array([ 4, 0, 3])
        >>> y_tuple = np.array([-1, 0, 1])
        >>> x_tuple = np.reshape(x_tuple, (len(x_tuple), 1))
        >>> y_tuple = np.reshape(y_tuple, (len(y_tuple), 1))
        >>> model = 'gau'
        >>> len_scale = 6.
        >>> rm = RandMeth(2, model, len_scale, 100, seed=12091986)
        >>> rm(x_tuple, y_tuple)
    """
    def __init__(self, dim, model, len_scale, mode_no=1000, seed=None,
                 **kwargs):
        self.reset(dim, model, len_scale, mode_no, seed, kwargs=kwargs)

    def reset(self, dim, model, len_scale, mode_no=1000, seed=None, **kwargs):
        """Reset the random amplitudes and wave numbers with a new seed.

        Args:
            dim (int): spatial dimension
            model (str): covariance model
            len_scale (float): length scale
            mode_no (int, opt.): number of Fourier modes
            seed (int, opt.): the seed of the master RNG, if "None",
                a random seed is used
        """
        self._dim = dim
        self._model = model
        self._len_scale = len_scale
        self._mode_no = mode_no
        self._kwargs = kwargs
        self._seed = np.nan
        self.seed = seed

    def __call__(self, x, y=None, z=None):
        """Calculates the random modes for the randomization method.

        Args:
            x (float, ndarray): the x components of the position tuple,
                the shape has to be (len(x), 1, 1) for 3d and accordingly
                shorter for lower dimensions
            y (float, ndarray, opt.): the y components of the pos. tupls
            z (float, ndarray, opt.): the z components of the pos. tuple

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
    def seed(self):
        """ seed (int): the seed of the master RNG

        If a new seed is given, the setter property not only saves the
        new seed, but also creates new random modes with the new seed.
        """
        return self._seed

    @seed.setter
    def seed(self, new_seed=None):
        if new_seed is not self._seed:
            self._seed = new_seed
            self._rng = RNG(self._dim, self._seed)
            self._Z, self._k = self._rng(self._model, self._len_scale,
                                         mode_no=self._mode_no,
                                         kwargs=self._kwargs)
            #preshape for unstructured grid
            for d in range(self._dim):
                self._k[d] = np.squeeze(self._k[d])
                self._k[d] = np.reshape(self._k[d], (1, len(self._k[d])))


if __name__ == '__main__':
    import doctest
    doctest.testmod()
