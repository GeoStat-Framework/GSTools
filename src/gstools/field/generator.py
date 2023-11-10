"""
GStools subpackage providing generators for spatial random fields.

.. currentmodule:: gstools.field.generator

The following classes are provided

.. autosummary::
   :toctree:

   Generator
   RandMeth
   IncomprRandMeth
   Fourier
"""
# pylint: disable=C0103, W0222, C0412, W0231
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy as dcp

import numpy as np

from gstools import config
from gstools.covmodel.base import CovModel
from gstools.random.rng import RNG
from gstools.tools.geometric import generate_grid

if config.USE_RUST:  # pragma: no cover
    # pylint: disable=E0401
    from gstools_core import summate, summate_incompr
else:
    from gstools.field.summator import (
        summate,
        summate_incompr,
        summate_fourier,
    )

__all__ = ["Generator", "RandMeth", "IncomprRandMeth", "Fourier"]


SAMPLING = ["auto", "inversion", "mcmc"]


class Generator(ABC):
    """
    Abstract generator class.

    Parameters
    ----------
    model : :any:`CovModel`
        Covariance model
    **kwargs
        Placeholder for keyword-args
    """

    @abstractmethod
    def __init__(self, model, **kwargs):
        pass

    @abstractmethod
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

    @abstractmethod
    def get_nugget(self, shape):
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

    @abstractmethod
    def __call__(self, pos, add_nugget=True):
        """
        Generate the field.

        Parameters
        ----------
        pos : (d, n), :class:`numpy.ndarray`
            the position tuple with d dimensions and n points.
        add_nugget : :class:`bool`
            Whether to add nugget noise to the field.

        Returns
        -------
        :class:`numpy.ndarray`
            the random modes
        """

    @property
    @abstractmethod
    def value_type(self):
        """:class:`str`: Type of the field values (scalar, vector)."""

    @property
    def name(self):
        """:class:`str`: Name of the generator."""
        return self.__class__.__name__


class RandMeth(Generator):
    r"""Randomization method for calculating isotropic random fields.

    Parameters
    ----------
    model : :any:`CovModel`
        Covariance model
    mode_no : :class:`int`, optional
        Number of Fourier modes. Default: ``1000``
    seed : :class:`int` or :any:`None`, optional
        The seed of the random number generator.
        If "None", a random seed is used. Default: :any:`None`
    sampling : :class:`str`, optional
        Sampling strategy. Either

            * "auto": select best strategy depending on given model
            * "inversion": use inversion method
            * "mcmc": use mcmc sampling

    **kwargs
        Placeholder for keyword-args

    Notes
    -----
    The Randomization method is used to generate isotropic
    spatial random fields characterized by a given covariance model.
    The calculation looks like [Hesse2014]_:

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

    References
    ----------
    .. [Hesse2014] Heße, F., Prykhodko, V., Schlüter, S., and Attinger, S.,
           "Generating random fields with a truncated power-law variogram:
           A comparison of several numerical methods",
           Environmental Modelling & Software, 55, 32-48., (2014)
    """

    def __init__(
        self,
        model,
        *,
        mode_no=1000,
        seed=None,
        sampling="auto",
        **kwargs,
    ):
        if kwargs:
            warnings.warn("gstools.RandMeth: **kwargs are ignored")
        # initialize attributes
        self._mode_no = int(mode_no)
        # initialize private attributes
        self._model = None
        self._seed = None
        self._rng = None
        self._z_1 = None
        self._z_2 = None
        self._cov_sample = None
        self._value_type = "scalar"
        # set sampling strategy
        self._sampling = None
        self.sampling = sampling
        # set model and seed
        self.update(model, seed)

    def __call__(self, pos, add_nugget=True):
        """Calculate the random modes for the randomization method.

        This method  calls the `summate_*` Cython methods, which are the
        heart of the randomization method.

        Parameters
        ----------
        pos : (d, n), :class:`numpy.ndarray`
            the position tuple with d dimensions and n points.
        add_nugget : :class:`bool`
            Whether to add nugget noise to the field.

        Returns
        -------
        :class:`numpy.ndarray`
            the random modes
        """
        pos = np.asarray(pos, dtype=np.double)
        summed_modes = summate(self._cov_sample, self._z_1, self._z_2, pos)
        nugget = self.get_nugget(summed_modes.shape) if add_nugget else 0.0
        return np.sqrt(self.model.var / self._mode_no) * summed_modes + nugget

    def get_nugget(self, shape):
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
            if not (
                isinstance(self._model, CovModel)
                and self._z_1 is not None
                and self._z_2 is not None
                and self._cov_sample is not None
            ):
                raise ValueError(
                    "gstools.field.generator.RandMeth: "
                    "neither 'model' nor 'seed' given!"
                )
        # wrong model type
        else:
            raise ValueError(
                "gstools.field.generator.RandMeth: 'model' is not an "
                "instance of 'gstools.CovModel'"
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
        # sample radii according to radial spectral density of the model
        if self.sampling == "inversion" or (
            self.sampling == "auto" and self.model.has_ppf
        ):
            pdf, cdf, ppf = self.model.dist_func
            rad = self._rng.sample_dist(
                size=self._mode_no, pdf=pdf, cdf=cdf, ppf=ppf, a=0
            )
        else:
            rad = self._rng.sample_ln_pdf(
                ln_pdf=self.model.ln_spectral_rad_pdf,
                size=self._mode_no,
                sample_around=1.0 / self.model.len_rescaled,
            )
        # get fully spatial samples by multiplying sphere samples and radii
        self._cov_sample = rad * sphere_coord

    @property
    def sampling(self):
        """:class:`str`: Sampling strategy."""
        return self._sampling

    @sampling.setter
    def sampling(self, sampling):
        if sampling not in ["auto", "inversion", "mcmc"]:
            raise ValueError(f"RandMeth: sampling not in {SAMPLING}.")
        self._sampling = sampling

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
    def value_type(self):
        """:class:`str`: Type of the field values (scalar, vector)."""
        return self._value_type

    def __repr__(self):
        """Return String representation."""
        return (
            f"{self.name}(model={self.model}, "
            f"mode_no={self._mode_no}, seed={self.seed})"
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
    sampling : :class:`str`, optional
        Sampling strategy. Either

            * "auto": select best strategy depending on given model
            * "inversion": use inversion method
            * "mcmc": use mcmc sampling

    **kwargs
        Placeholder for keyword-args

    Notes
    -----
    The Randomization method is used to generate isotropic
    spatial incompressible random vector fields characterized
    by a given covariance model. The equation is [Kraichnan1970]_:

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

    References
    ----------
    .. [Kraichnan1970] Kraichnan, R. H.,
           "Diffusion by a random velocity field.",
           The physics of fluids, 13(1), 22-31., (1970)
    """

    def __init__(
        self,
        model,
        *,
        mean_velocity=1.0,
        mode_no=1000,
        seed=None,
        sampling="auto",
        **kwargs,
    ):
        if model.dim < 2 or model.dim > 3:
            raise ValueError(
                "Only 2D and 3D incompressible fields can be generated."
            )
        super().__init__(
            model=model,
            mode_no=mode_no,
            seed=seed,
            sampling=sampling,
            **kwargs,
        )

        self.mean_u = mean_velocity
        self._value_type = "vector"

    def __call__(self, pos, add_nugget=True):
        """Calculate the random modes for the randomization method.

        This method  calls the `summate_incompr_*` Cython methods,
        which are the heart of the randomization method.
        In this class the method contains a projector to
        ensure the incompressibility of the vector field.

        Parameters
        ----------
        pos : (d, n), :class:`numpy.ndarray`
            the position tuple with d dimensions and n points.
        add_nugget : :class:`bool`
            Whether to add nugget noise to the field.

        Returns
        -------
        :class:`numpy.ndarray`
            the random modes
        """
        pos = np.asarray(pos, dtype=np.double)
        summed_modes = summate_incompr(
            self._cov_sample, self._z_1, self._z_2, pos
        )
        nugget = self.get_nugget(summed_modes.shape) if add_nugget else 0.0
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


class Fourier(Generator):
    r"""Fourier method for calculating isotropic random fields.

    Parameters
    ----------
    model : :any:`CovModel`
        Covariance model
    mode_no : :class:`list`
        Number of Fourier modes per dimension.
    mode_truncation : :class:`list`
        Cut-off values of the Fourier modes.
    period_len : :class:`float` or :class:`list`, optional
        Period length of the field in each dim as a factor of the domain size.
    seed : :class:`int`, optional
        The seed of the random number generator.
        If "None", a random seed is used. Default: :any:`None`
    verbose : :class:`bool`, optional
        Be chatty during the generation.
        Default: :any:`False`

    **kwargs
        Placeholder for keyword-args

    Notes
    -----
    The Fourier method is used to generate isotropic
    spatial random fields characterized by a given covariance model.
    The calculation looks like [Hesse2014]_: # TODO check different source

    .. math::
       u\left(x\right)=
       \sqrt{2\sigma^{2}}\cdot
       \sum_{i=1}^{N}\sqrt{E(k_{i})}\left(
       Z_{1,i}\cdot\cos\left(2\pi\left\langle k_{i},x\right\rangle \right)+
       Z_{2,i}\cdot\sin\left(2\pi\left\langle k_{i},x\right\rangle \right)
       \right) \sqrt{\Delta k}

    where:

        * :math:`N` : fourier mode number
        * :math:`Z_{j,i}` : random samples from a normal distribution
        * :math:`k_i` : the equidistant spectral density of the covariance model

    References
    ----------
    .. [Hesse2014] Heße, F., Prykhodko, V., Schlüter, S., and Attinger, S.,
           "Generating random fields with a truncated power-law variogram:
           A comparison of several numerical methods",
           Environmental Modelling & Software, 55, 32-48., (2014)
    """

    def __init__(
        self,
        model,
        modes_no,
        modes_truncation,
        period_len=None,
        seed=None,
        verbose=False,
        **kwargs,
    ):
        if kwargs:
            warnings.warn("gstools.Fourier: **kwargs are ignored")
        # initialize attributes
        self._modes_truncation = self._fill_to_dim(
            model.dim, modes_truncation, np.double
        )
        self._modes_no = self._fill_to_dim(model.dim, modes_no, int)
        self._modes = []
        [
            self._modes.append(
                np.linspace(
                    -self._modes_truncation[d] / 2,
                    self._modes_truncation[d] / 2,
                    self._modes_no[d],
                    endpoint=False,
                ).T
            )
            for d in range(model.dim)
        ]

        self._period_len = self._fill_to_dim(
            model.dim, period_len, np.double, 1.0
        )
        self._verbose = bool(verbose)
        # initialize private attributes
        self._model = None
        self._seed = None
        self._rng = None
        self._z_1 = None
        self._z_2 = None
        self._spectral_density_sqrt = None
        self._value_type = "scalar"
        # set model and seed
        self.update(model, seed)

    def __call__(self, pos, add_nugget=True):
        """Calculate the modes for the Fourier method.

        This method  calls the `summate_*` Cython methods, which are the
        heart of the randomization method.

        Parameters
        ----------
        pos : (d, n), :class:`numpy.ndarray`
            the position tuple with d dimensions and n points.
        add_nugget : :class:`bool`
            Whether to add nugget noise to the field.

        Returns
        -------
        :class:`numpy.ndarray`
            the random modes
        """
        pos = np.asarray(pos, dtype=np.double)
        domain_size = pos.max(axis=1) - pos.min(axis=1)
        self._modes = [
            self._modes[d] / domain_size[d] * self._period_len[d]
            for d in range(self._model.dim)
        ]

        self._modes = generate_grid(self._modes)

        # pre calc. the spectral density for all wave numbers
        # they are handed over to Cython
        k_norm = np.linalg.norm(self._modes, axis=0)
        self._spectral_density_sqrt = np.sqrt(
            self._model.spectral_density(k_norm)
        )
        summed_modes = summate_fourier(
            self._spectral_density_sqrt,
            self._modes,
            self._z_1,
            self._z_2,
            pos,
        )
        nugget = self.get_nugget(summed_modes.shape) if add_nugget else 0.0
        return (
            np.sqrt(2.0 * self.model.var / np.prod(domain_size)) * summed_modes
            + nugget
        )

    def get_nugget(self, shape):
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
                and self._spectral_density_sqrt is not None
            ):
                if self.verbose:
                    print("RandMeth.update: Nothing will be done...")
            else:
                raise ValueError(
                    "gstools.field.generator.RandMeth: "
                    "neither 'model' nor 'seed' given!"
                )
        # wrong model type
        else:
            raise ValueError(
                "gstools.field.generator.RandMeth: 'model' is not an "
                "instance of 'gstools.CovModel'"
            )

    def reset_seed(self, seed=np.nan):
        """
        Recalculate the random values with the given seed.

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
        self._z_1 = self._rng.random.normal(size=np.prod(self._modes_no))
        self._z_2 = self._rng.random.normal(size=np.prod(self._modes_no))

    def _fill_to_dim(self, dim, values, dtype, default_value=None):
        """Fill an array with last element up to len(dim)."""
        r = values
        if values is None:
            if default_value is None:
                raise ValueError(f"Fourier: Value has to be provided")
            r = default_value
        r = np.array(r, dtype=dtype)
        r = np.atleast_1d(r)[:dim]
        if len(r) > dim:
            raise ValueError(f"Fourier: len(values) <= {dim=} not fulfilled")
        # fill up values with values[-1], such that len()==dim
        if len(r) < dim:
            r = np.pad(r, (0, dim - len(r)), "edge")
        return r

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
    def modes_truncation(self):
        """:class:`list`: Cut-off values of the Fourier modes."""
        return self._modes_truncation

    @modes_truncation.setter
    def modes_truncation(self, modes_truncation):
        self._modes_truncation = modes_truncation

    @property
    def period_len(self):
        """:class:`list`: Period length of the field in each dim."""
        return self._period_len

    @period_len.setter
    def period_len(self, period_len):
        self._period_len = self._fill_to_dim(
            self._model.dim, period_len, np.double, 1.0
        )

    @property
    def verbose(self):
        """:class:`bool`: Verbosity of the generator."""
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        self._verbose = bool(verbose)

    @property
    def value_type(self):
        """:class:`str`: Type of the field values (scalar, vector)."""
        return self._value_type

    def __repr__(self):
        """Return String representation."""
        return f"{self.name}(model={self.model}, seed={self.seed})"
