# -*- coding: utf-8 -*-
"""
GStools subpackage providing generators for spatial random fields.

.. currentmodule:: gstools.field.generator

The following classes are provided

.. autosummary::
   RandMeth
   IncomprRandMeth
"""
# pylint: disable=C0103

from copy import deepcopy as dcp
import numpy as np
from gstools.covmodel.base import CovModel
from gstools.random.rng import RNG
from gstools.field.summator import (
    summate_unstruct,
    summate_struct,
    summate_incompr_unstruct,
    summate_incompr_struct,
)

__all__ = ["RandMeth", "IncomprRandMeth"]


class RandMeth:
    r"""Randomization method for calculating isotropic spatial random fields.

    Parameters
    ----------
    model : :any:`CovModel`
        Covariance model
    mode_no : :class:`int`, optional
        Number of Fourier modes. Default: ``1000``
    seed : :class:`int` or :any:`None`, optional
        The seed of the random number generator.
        If "None", a random seed is used. Default: :any:`None`
    verbose : :class:`bool`, optional
        Be chatty during the generation.
        Default: :any:`False`
    **kwargs
        Placeholder for keyword-args

    Notes
    -----
    The Randomization method is used to generate isotropic
    spatial random fields characterized by a given covariance model.
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

    def __init__(
        self, model, mode_no=1000, seed=None, verbose=False, **kwargs
    ):
        if kwargs:
            print("gstools.RandMeth: **kwargs are ignored")
        # initialize atributes
        self._mode_no = int(mode_no)
        self._verbose = bool(verbose)
        # initialize private atributes
        self._model = None
        self._seed = None
        self._rng = None
        self._z_1 = None
        self._z_2 = None
        self._cov_sample = None
        self._value_type = "scalar"
        # set model and seed
        self.update(model, seed)

    def __call__(self, x, y=None, z=None, mesh_type="unstructured"):
        """Calculate the random modes for the randomization method.

        This method  calls the `summate_*` Cython methods, which are the
        heart of the randomization method.

        Parameters
        ----------
        x : :class:`float`, :class:`numpy.ndarray`
            The x components of the pos. tuple.
        y : :class:`float`, :class:`numpy.ndarray`, optional
            The y components of the pos. tuple.
        z : :class:`float`, :class:`numpy.ndarray`, optional
            The z components of the pos. tuple.
        mesh_type : :class:`str`, optional
            'structured' / 'unstructured'

        Returns
        -------
        :class:`numpy.ndarray`
            the random modes
        """
        if mesh_type == "unstructured":
            pos = _reshape_pos(x, y, z, dtype=np.double)

            summed_modes = summate_unstruct(
                self._cov_sample, self._z_1, self._z_2, pos
            )
        else:
            x, y, z = _set_dtype(x, y, z, dtype=np.double)
            summed_modes = summate_struct(
                self._cov_sample, self._z_1, self._z_2, x, y, z
            )

        nugget = self._set_nugget(summed_modes.shape)

        return np.sqrt(self.model.var / self._mode_no) * summed_modes + nugget

    def _set_nugget(self, shape):
        """
        Generate normal distributed values for the nugget simulation.

        Parameters
        ----------
        shape : :class:`tuple`
            the shape of the summed modes
        Returns
        -------
        nugget : :class:`numpy.ndarray`
            the nugget in the same shape as the summed modes
        """
        if self.model.nugget > 0:
            nugget = np.sqrt(self.model.nugget) * self._rng.random.normal(
                size=shape
            )
        else:
            nugget = 0.0
        return nugget

    def update(self, model=None, seed=np.nan):
        """Update the model and the seed.

        If model and seed are not different, nothing will be done.

        Parameters
        ----------
        model : :any:`CovModel` or :any:`None`, optional
            covariance model. Default: :any:`None`
        seed : :class:`int` or :any:`None` or :any:`numpy.nan`, optional
            the seed of the random number generator.
            If :any:`None`, a random seed is used. If :any:`numpy.nan`,
            the actual seed will be kept. Default: :any:`numpy.nan`
        """
        # check if a new model is given
        if isinstance(model, CovModel):
            if self.model != model:
                self._model = dcp(model)
                if seed is None or not np.isnan(seed):
                    self.reset_seed(seed)
                else:
                    self.reset_seed(self._seed)
            # just update the seed, if its a new one
            elif seed is None or not np.isnan(seed):
                self.seed = seed
        # or just update the seed, when no model is given
        elif model is None and (seed is None or not np.isnan(seed)):
            if isinstance(self._model, CovModel):
                self.seed = seed
            else:
                raise ValueError(
                    "gstools.field.generator.RandMeth: no 'model' given"
                )
        # if the user tries to trick us, we beat him!
        elif model is None and np.isnan(seed):
            if (
                isinstance(self._model, CovModel)
                and self._z_1 is not None
                and self._z_2 is not None
                and self._cov_sample is not None
            ):
                if self.verbose:
                    print("RandMeth.update: Nothing will be done...")
            else:
                raise ValueError(
                    "gstools.field.generator.RandMeth: "
                    + "neither 'model' nor 'seed' given!"
                )
        # wrong model type
        else:
            raise ValueError(
                "gstools.field.generator.RandMeth: 'model' is not an "
                + "instance of 'gstools.CovModel'"
            )

    def reset_seed(self, seed=np.nan):
        """
        Recalculate the random amplitudes and wave numbers with the given seed.

        Parameters
        ----------
        seed : :class:`int` or :any:`None` or :any:`numpy.nan`, optional
            the seed of the random number generator.
            If :any:`None`, a random seed is used. If :any:`numpy.nan`,
            the actual seed will be kept. Default: :any:`numpy.nan`

        Notes
        -----
        Even if the given seed is the present one, modes will be recalculated.
        """
        if seed is None or not np.isnan(seed):
            self._seed = seed
        self._rng = RNG(self._seed)
        # normal distributed samples for randmeth
        self._z_1 = self._rng.random.normal(size=self._mode_no)
        self._z_2 = self._rng.random.normal(size=self._mode_no)
        # sample uniform on a sphere
        sphere_coord = self._rng.sample_sphere(self.model.dim, self._mode_no)
        # sample radii acording to radial spectral density of the model
        if self.model.has_ppf:
            pdf, cdf, ppf = self.model.dist_func
            rad = self._rng.sample_dist(
                size=self._mode_no, pdf=pdf, cdf=cdf, ppf=ppf, a=0
            )
        else:
            rad = self._rng.sample_ln_pdf(
                ln_pdf=self.model.ln_spectral_rad_pdf,
                size=self._mode_no,
                sample_around=1.0 / self.model.len_scale,
            )
        # get fully spatial samples by multiplying sphere samples and radii
        self._cov_sample = rad * sphere_coord

    @property
    def seed(self):
        """:class:`int`: Seed of the master RNG.

        Notes
        -----
        If a new seed is given, the setter property not only saves the
        new seed, but also creates new random modes with the new seed.
        """
        return self._seed

    @seed.setter
    def seed(self, new_seed):
        if new_seed is not self._seed:
            self.reset_seed(new_seed)

    @property
    def model(self):
        """:any:`CovModel`: Covariance model of the spatial random field."""
        return self._model

    @model.setter
    def model(self, model):
        self.update(model)

    @property
    def mode_no(self):
        """:class:`int`: Number of modes in the randomization method."""
        return self._mode_no

    @mode_no.setter
    def mode_no(self, mode_no):
        if int(mode_no) != self._mode_no:
            self._mode_no = int(mode_no)
            self.reset_seed(self._seed)

    @property
    def verbose(self):
        """:class:`bool`: Verbosity of the generator."""
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        self._verbose = bool(verbose)

    @property
    def name(self):
        """:class:`str`: Name of the generator."""
        return self.__class__.__name__

    @property
    def value_type(self):
        """:class:`str`: Type of the field values (scalar, vector)."""
        return self._value_type

    def __str__(self):
        """Return String representation."""
        return self.__repr__()

    def __repr__(self):
        """Return String representation."""
        return "RandMeth(model={0}, mode_no={1}, seed={2})".format(
            repr(self.model), self._mode_no, self.seed
        )


class IncomprRandMeth(RandMeth):
    r"""RandMeth for incompressible random vector fields.

    Parameters
    ----------
    model : :any:`CovModel`
        covariance model
    mean_velocity : :class:`float`, optional
        the mean velocity in x-direction
    mode_no : :class:`int`, optional
        number of Fourier modes. Default: ``1000``
    seed : :class:`int` or :any:`None`, optional
        the seed of the random number generator.
        If "None", a random seed is used. Default: :any:`None`
    verbose : :class:`bool`, optional
        State if there should be output during the generation.
        Default: :any:`False`
    **kwargs
        Placeholder for keyword-args

    Notes
    -----
    The Randomization method is used to generate isotropic
    spatial incompressible random vector fields characterized
    by a given covariance model. The equation is:

    .. math::
       u_i\left(x\right)= \bar{u_i} \delta_{i1} +
       \bar{u_i}\sqrt{\frac{\sigma^{2}}{N}}\cdot
       \sum_{j=1}^{N}p_i(k_{j})\left(
       Z_{1,j}\cdot\cos\left(\left\langle k_{j},x\right\rangle \right)+
       Z_{2,j}\cdot\sin\left(\left\langle k_{j},x\right\rangle \right)
       \right)

    where:

        * :math:`\bar u` : mean velocity in :math:`e_1` direction
        * :math:`N` : fourier mode number
        * :math:`Z_{k,j}` : random samples from a normal distribution
        * :math:`k_j` : samples from the spectral density distribution of
          the covariance model
        * :math:`p_i(k_j) = e_1 - \frac{k_i k_1}{k^2}` : the projector
          ensuring the incompressibility
    """

    def __init__(
        self,
        model,
        mean_velocity=1.0,
        mode_no=1000,
        seed=None,
        verbose=False,
        **kwargs
    ):
        if model.dim < 2:
            raise ValueError(
                "Only 2- and 3-dimensional incompressible fields "
                + "can be generated."
            )
        super().__init__(model, mode_no, seed, verbose, **kwargs)

        self.mean_u = mean_velocity
        self._value_type = "vector"

    def __call__(self, x, y=None, z=None, mesh_type="unstructured"):
        """Calculate the random modes for the randomization method.

        This method  calls the `summate_incompr_*` Cython methods,
        which are the heart of the randomization method.
        In this class the method contains a projector to
        ensure the incompressibility of the vector field.

        Parameters
        ----------
        x : :class:`float`, :class:`numpy.ndarray`
            the x components of the position tuple, the shape has to be
            (len(x), 1, 1) for 3d and accordingly shorter for lower
            dimensions
        y : :class:`float`, :class:`numpy.ndarray`, optional
            the y components of the pos. tuples. Default: ``None``
        z : :class:`float`, :class:`numpy.ndarray`, optional
            the z components of the pos. tuple. Default: ``None``
        mesh_type : :class:`str`, optional
            'structured' / 'unstructured'

        Returns
        -------
        :class:`numpy.ndarray`
            the random modes
        """
        if mesh_type == "unstructured":
            pos = _reshape_pos(x, y, z, dtype=np.double)

            summed_modes = summate_incompr_unstruct(
                self._cov_sample, self._z_1, self._z_2, pos
            )
        else:
            x, y, z = _set_dtype(x, y, z, dtype=np.double)
            summed_modes = summate_incompr_struct(
                self._cov_sample, self._z_1, self._z_2, x, y, z
            )

        nugget = self._set_nugget(summed_modes.shape)

        e1 = self._create_unit_vector(summed_modes.shape)

        return (
            self.mean_u * e1
            + self.mean_u
            * np.sqrt(self.model.var / self._mode_no)
            * summed_modes
            + nugget
        )

    def _create_unit_vector(self, broadcast_shape, axis=0):
        """Create a unit vector.

        Can be multiplied with a vector of shape broadcast_shape

        Parameters
        ----------
        broadcast_shape : :class:`tuple`
            the shape of the array with which
            the unit vector is to be multiplied
        axis : :class:`int`, optional
            the direction of the unit vector. Default: ``0``

        Returns
        -------
        :class:`numpy.ndarray`
            the unit vector
        """
        shape = np.ones(len(broadcast_shape), dtype=int)
        shape[0] = self.model.dim

        e1 = np.zeros(shape)
        e1[axis] = 1.0
        return e1


def _reshape_pos(x, y=None, z=None, dtype=np.double):
    """
    Reshape the 1d x, y, z positions to a 2d position array.

    Parameters
    ----------
    x : :class:`float`, :class:`numpy.ndarray`
        the x components of the position tuple, the shape has to be
        (len(x), 1, 1) for 3d and accordingly shorter for lower
        dimensions
    y : :class:`float`, :class:`numpy.ndarray`, optional
        the y components of the pos. tuple
    z : :class:`float`, :class:`numpy.ndarray`, optional
        the z components of the pos. tuple
    dtype : :class:`numpy.dtype`, optional
        the numpy dtype to which the elements should be converted

    Returns
    -------
    :class:`numpy.ndarray`
        the positions in one convinient data structure
    """
    if y is None and z is None:
        pos = np.array(x.reshape(1, len(x)), dtype=dtype)
    elif z is None:
        pos = np.array(np.vstack((x, y)), dtype=dtype)
    else:
        pos = np.array(np.vstack((x, y, z)), dtype=dtype)
    return pos


def _set_dtype(x, y=None, z=None, dtype=np.double):
    """
    Convert the dtypes of the input arrays to given dtype.

    Parameters
    ----------
    x : :class:`float`, :class:`numpy.ndarray`
        The array to be converted.
    y : :class:`float`, :class:`numpy.ndarray`, optional
        The array to be converted.
    z : :class:`float`, :class:`numpy.ndarray`, optional
        The array to be converted.
    dtype : :class:`numpy.dtype`, optional
        The numpy dtype to which the elements should be converted.

    Returns
    -------
    :class:`numpy.ndarray`
        The input lists/ arrays as numpy arrays with given dtype.
    """
    x = x.astype(dtype, copy=False)
    if y is not None:
        y = y.astype(dtype, copy=False)
    if z is not None:
        z = z.astype(dtype, copy=False)
    return x, y, z


if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod()
