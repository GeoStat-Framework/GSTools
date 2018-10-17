# -*- coding: utf-8 -*-
"""
GStools subpackage providing a generator for standard spatial random fields.

.. currentmodule:: gstools.field.srf

The following classes and functions are provided

.. autosummary::
   SRF
   RandMeth
   r3d_x
   r3d_y
   r3d_z
"""
from __future__ import division, absolute_import, print_function

import numpy as np
from gstools.field.rng import RNG

__all__ = [
    "SRF",
    "RandMeth",
    "r3d_x",
    "r3d_y",
    "r3d_z",
]


class SRF(object):
    """A class to generate a spatial random field (SRF).

    Examples
    --------
        >>> cov_model = {'dim': 2, 'mean': .0, 'var': 2.6, 'len_scale': 4.,
        >>>              'model': 'gau', 'anis': 5., 'angles': np.pi/4.
        >>>              'mode_no': 100,}
        >>> x = np.arange(0, 10, 1)
        >>> y = np.arange(-5, 5, 0.5)
        >>> srf = SRF(**cov_model)
        >>> field = srf(x, y, seed=987654)
    """
    def __init__(self, dim, mean=0., var=1., len_scale=1., model='gau',
                 anis=1., angles=0., mode_no=1000):
        """Initialize a spatial random field

        Parameters
        ----------
            dim : :class:`int`
                spatial dimension
            mean : :class:`float`, optional
                mean value of the SRF
            var : :class:`float`, optional
                variance of the SRF
            len_scale : :class:`float`, optional
                the length scale of the SRF in x direction
            model : :class:`str`, optional
                the covariance model ('gau', 'exp', ..., see RNG)
            anis : :class:`float`/list, optional
                the anisotropy of length scales along the y- and z-directions
            angles : :class:`float`/list, optional
                the rotation of the stretching, with the values corrisponding
                the yaw, pitch, and roll
            mode_no : :class:`int`, optional
                number of Fourier modes
        """
        if dim < 1 or dim > 3:
            raise ValueError('Only dimensions of 1 <= d <= 3 are supported.')
        self._dim = dim
        self._mean = mean
        self._var = var
        self._len_scale = len_scale
        self._model = model
        self._anis = np.atleast_1d(anis)
        if len(self._anis) < self._dim:
            # fill up the anisotropy array with ones, such that len() == dim
            self._anis = np.pad(self._anis, (0, self._dim-len(self._anis)-1),
                                'constant', constant_values=1.)
        self._angles = np.atleast_1d(angles)
        # fill up the rotation angle array with zeros, such that len() == dim
        self._angles = np.pad(self._angles, (0, self._dim-len(self._angles)),
                              'constant', constant_values=0.)
        self._mode_no = mode_no
        self._do_rotation = not np.all(np.isclose(angles, 0.))
        self._randmeth = RandMeth(self._dim, self._model, self._len_scale,
                                  self._mode_no, seed=None)
        # initialize attributes
        self.field = None

    def __call__(self, x, y=None, z=None, seed=None, mesh_type='unstructured',
                 force_moments=False):
        """Generate an SRF and return it without saving it internally.

        Parameters
        ----------
            x : :class:`numpy.ndarray`
                grid axis in x-direction if structured, or first components of
                position vectors if unstructured
            y : :class:`numpy.ndarray`, optional
                analog to x
            z : :class:`numpy.ndarray`, optional
                analog to x
            seed : :class:`int`, optional
                seed for RNG
            mesh_type : :class:`str`
                'structured' / 'unstructured'
            force_moments : :class:`bool`
                Force the generator to exactly match mean and variance.
                Default: False
        Returns
        -------
            field : :class:`numpy.ndarray`
                the SRF
        """
        self._check_mesh(x, y, z, mesh_type)
        mesh_type_changed = False
        if self._do_rotation:
            if mesh_type == 'structured':
                mesh_type_changed = True
                mesh_type_old = mesh_type
                mesh_type = 'unstructured'
                x, y, z, axis_lens = (
                    self._reshape_axis_from_struct_to_unstruct(x, y, z))
            x, y, z = self._unrotate_mesh(x, y, z)

        y, z = self._make_isotropic(y, z)
        x, y, z = _reshape_input(x, y, z, mesh_type)

        self._randmeth.seed = seed
        field = self._randmeth(x, y, z)

        if mesh_type_changed:
            mesh_type = mesh_type_old
            field = self._reshape_field_from_unstruct_to_struct(field,
                                                                axis_lens)
        if force_moments:
            var_in = np.var(field)
            mean_in = np.mean(field)
            scale = np.sqrt(self.var/var_in)
            self.field = scale*(field - mean_in) + self.mean
        else:
            self.field = field

        return self.field

    def structured(self, x, y=None, z=None, seed=None):
        """Generate an SRF on a structured mesh without saving it internally.

        Parameters
        ----------
            x : :class:`numpy.ndarray`
                grid axis in x-direction if structured
            y : :class:`numpy.ndarray`, optional
                analog to x
            z : :class:`numpy.ndarray`, optional
                analog to x
            seed : :class:`int`, optional
                seed for RNG
        Returns
        -------
            field : :class:`numpy.ndarray`
                the SRF
        """
        return self(x, y, z, seed, 'structured')

    def unstructured(self, x, y=None, z=None, seed=None):
        """Generate an SRF on an unstructured mesh
        without saving it internally.

        Parameters
        ----------
            x : :class:`numpy.ndarray`
                first components of position vectors if unstructured
            y : :class:`numpy.ndarray`, optional
                analog to x
            z : :class:`numpy.ndarray`, optional
                analog to x
            seed : :class:`int`, optional
                seed for RNG
        Returns
        -------
            field : :class:`numpy.ndarray`
                the SRF
        """
        return self(x, y, z, seed)

    def generate(self, x, y=None, z=None, seed=None, mesh_type='unstructured',
                 force_moments=False):
        """Generate an SRF and save it as an attribute self.field.

        Parameters
        ----------
            x : :class:`numpy.ndarray`
                grid axis in x-direction if structured, or first components of
                position vectors if unstructured
            y : :class:`numpy.ndarray`, optional
                analog to x
            z : :class:`numpy.ndarray`, optional
                analog to x
            seed : :class:`int`, optional
                seed for RNG
            mesh_type : :class:`str`
                'structured' / 'unstructured'
            force_moments : :class:`bool`
                Force the generator to exactly match mean and variance.
                Default: False
        Returns
        -------
            field : :class:`numpy.ndarray`
                the SRF
        """
        return self(x, y, z, seed, mesh_type, force_moments)

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
        return None

    def _unrotate_mesh(self, x, y, z):
        """Rotate axes in order to implement rotation.

        for 3d: yaw, pitch, and roll angles are alpha, beta, and gamma,
        of intrinsic rotation rotation whose Tait-Bryan angles are
        alpha, beta, gamma about axes x, y, z.
        """
        if self._dim == 1:
            return x, y, z
        elif self._dim == 2:
            # extract 2d rotation matrix
            rot_mat = r3d_z(self._angles[0])[0:2, 0:2]
            pos_tuple = np.vstack((x, y))
            pos_tuple = np.vsplit(np.dot(rot_mat, pos_tuple), 2)
            x = np.squeeze(pos_tuple[0])
            y = np.squeeze(pos_tuple[1])
            return x, y, z
        elif self._dim == 3:
            alpha = self._angles[0]
            beta = self._angles[1]
            gamma = self._angles[2]
            rot_mat = np.dot(np.dot(r3d_z(alpha),
                                    r3d_y(beta)),
                             r3d_x(gamma))
            pos_tuple = np.vstack((x, y, z))
            pos_tuple = np.vsplit(np.dot(rot_mat, pos_tuple), 3)
            x = np.squeeze(pos_tuple[0])
            y = np.squeeze(pos_tuple[1])
            z = np.squeeze(pos_tuple[2])
            return x, y, z
        return None

    def _reshape_axis_from_struct_to_unstruct(self, x, y=None, z=None):
        """Reshape given axes from struct to unstruct for rotation."""
        if self._dim == 1:
            return x, y, z, (len(x),)
        elif self._dim == 2:
            x_u, y_u = np.meshgrid(x, y, indexing='ij')
            len_unstruct = len(x) * len(y)
            x_u = np.reshape(x_u, len_unstruct)
            y_u = np.reshape(y_u, len_unstruct)
            return x_u, y_u, z, (len(x), len(y))
        elif self._dim == 3:
            x_u, y_u, z_u = np.meshgrid(x, y, z, indexing='ij')
            len_unstruct = len(x) * len(y) * len(z)
            x_u = np.reshape(x_u, len_unstruct)
            y_u = np.reshape(y_u, len_unstruct)
            z_u = np.reshape(z_u, len_unstruct)
            return x_u, y_u, z_u, (len(x), len(y), len(z))
        return None

    def _reshape_field_from_unstruct_to_struct(self, field, axis_lens):
        """Reshape the rotated field back to struct."""
        if self._dim == 1:
            return field
        elif self._dim == 2:
            field = np.reshape(field, axis_lens)
            return field
        elif self._dim == 3:
            field = np.reshape(field, axis_lens)
            return field
        return None

    @property
    def dim(self):
        """ The dimension of the spatial random field."""
        return self._dim

    @property
    def mean(self):
        """ The mean of the spatial random field."""
        return self._mean

    @property
    def var(self):
        """ The variance of the spatial random field."""
        return self._var

    @property
    def len_scale(self):
        """ The length scale of the spatial random field."""
        return self._len_scale

    @property
    def model(self):
        """ The length scale of the spatial random field."""
        return self._model

    @property
    def anis(self):
        """ The anisotropy factors of the spatial random field."""
        return self._anis

    @property
    def angles(self):
        """ The rotation angles (in rad) of the spatial random field."""
        return self._angles

    @property
    def mode_no(self):
        """ number of Fourier modes"""
        return self._mode_no

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return ('SRF(dim={0}, mean={1}, var={2}, len_scale={3}, model={4}, '
                'anis={5}, angles={6}, mode_no={7})'.
                format(self.dim, self.mean, self.var, self.len_scale,
                       self.model, np.squeeze(self.anis),
                       np.squeeze(self.angles), self.mode_no))


class RandMeth(object):
    """Randomization method for calculating isotropic spatial random fields.

    Examples
    --------
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
        """Initialize the randomization method

        Parameters
        ----------
            dim : :class:`int`
                spatial dimension
            model : :class:`dict`
                covariance model
            mode_no : :class:`int`, optional
                number of Fourier modes
            seed : :class:`int`, optional
                the seed of the master RNG, if "None", a random seed is used
        """
        self._seed = None
        self._rng = None
        self._Z = None
        self._k = None
        self.reset(dim, model, len_scale, mode_no, seed, kwargs=kwargs)

    def reset(self, dim, model, len_scale, mode_no=1000, seed=None, **kwargs):
        """Reset the random amplitudes and wave numbers with a new seed.

        Parameters
        ----------
            dim : :class:`int`
                spatial dimension
            model : :class:`str`
                covariance model
            len_scale : :class:`float`
                length scale
            mode_no : :class:`int`, optional
                number of Fourier modes
            seed : :class:`int`, optional
                the seed of the master RNG, if "None", a random seed is used
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
        summed_modes = np.broadcast(x, y, z)
        summed_modes = np.squeeze(np.zeros(summed_modes.shape))

        # Test to see if enough memory is available.
        # In case there isn't, divide Fourier modes into smaller chunks
        chunk_no = 1
        chunk_no_exp = 0
        while True:
            try:
                chunk_len = int(np.ceil(self._mode_no / chunk_no))

                for chunk in range(chunk_no):
                    a = chunk * chunk_len
                    # In case k[d,a:e] with e >= len(k[d,:]) causes errors in
                    # numpy, use the commented min-function below
                    # e = min((chunk + 1) * chunk_len, self.mode_no-1)
                    e = (chunk + 1) * chunk_len

                    if self._dim == 1:
                        phase = self._k[0, a:e]*x
                    elif self._dim == 2:
                        phase = self._k[0, a:e]*x + self._k[1, a:e]*y
                    else:
                        phase = (self._k[0, a:e]*x + self._k[1, a:e]*y +
                                 self._k[2, a:e]*z)

                    # no factor 2*pi needed
                    summed_modes += np.squeeze(
                        np.sum(self._Z[0, a:e] * np.cos(phase) +
                               self._Z[1, a:e] * np.sin(phase),
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
            self._rng = RNG(self._dim, self._seed)
            self._Z, self._k = self._rng(self._model, self._len_scale,
                                         mode_no=self._mode_no,
                                         kwargs=self._kwargs)
            # preshape for unstructured grid
            for dim_i in range(self._dim):
                self._k[dim_i] = np.squeeze(self._k[dim_i])
                self._k[dim_i] = np.reshape(self._k[dim_i],
                                            (1, len(self._k[dim_i])))

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return ('RandMeth(dim={0}, model={1}, len_scale={2}, mode_no={3}, '
                'seed={4})'.format(self._dim, self._model, self._len_scale,
                                   self._mode_no, self.seed))


def r3d_x(theta):
    """Rotation matrix about x axis.

    Parameters
    ----------
    theta : :class:`float`
        Rotation angle

    Returns
    -------
        :class:`numpy.ndarray`
            Rotation matrix.
    """
    sin = np.sin(theta)
    cos = np.cos(theta)
    return np.array(((1., +0., +0.0),
                     (0., cos, -sin),
                     (0., sin, cos)))


def r3d_y(theta):
    """Rotation matrix about y axis.

    Parameters
    ----------
    theta : :class:`float`
        Rotation angle

    Returns
    -------
        :class:`numpy.ndarray`
            Rotation matrix.
    """
    sin = np.sin(theta)
    cos = np.cos(theta)
    return np.array(((+cos, 0., sin),
                     (+0.0, 1., +0.),
                     (-sin, 0., cos)))


def r3d_z(theta):
    """Rotation matrix about z axis.

    Parameters
    ----------
    theta : :class:`float`
        Rotation angle

    Returns
    -------
        :class:`numpy.ndarray`
            Rotation matrix.
    """
    sin = np.sin(theta)
    cos = np.cos(theta)
    return np.array(((cos, -sin, 0.),
                     (sin, +cos, 0.),
                     (+0., +0.0, 1.)))


def _reshape_input(x, y=None, z=None, mesh_type='unstructured'):
    """Reshape given axes, depending on the mesh type."""
    if mesh_type == 'unstructured':
        x, y, z = _reshape_input_axis_from_unstruct(x, y, z)
    elif mesh_type == 'structured':
        x, y, z = _reshape_input_axis_from_struct(x, y, z)
    return x, y, z


def _reshape_input_axis_from_unstruct(x, y=None, z=None):
    """Reshape given axes for vectorisation on unstructured grid."""
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    x = np.reshape(x, (len(x), 1))
    y = np.reshape(y, (len(y), 1))
    z = np.reshape(z, (len(z), 1))
    return (x, y, z)


def _reshape_input_axis_from_struct(x, y=None, z=None):
    """Reshape given axes for vectorisation on unstructured grid."""
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    x = np.reshape(x, (len(x), 1, 1, 1))
    y = np.reshape(y, (1, len(y), 1, 1))
    z = np.reshape(z, (1, 1, len(z), 1))
    return (x, y, z)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
