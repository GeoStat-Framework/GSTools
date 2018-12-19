# -*- coding: utf-8 -*-
"""
GStools subpackage providing generators for spatial random fields.

.. currentmodule:: gstools.field.generator

The following classes are provided

.. autosummary::
   RandMeth
"""
# pylint: disable=C0103
from __future__ import division, absolute_import, print_function

from copy import deepcopy as dcp
import numpy as np
from gstools.covmodel.base import CovModel
from gstools.random.rng import RNG

__all__ = ["RandMeth"]


class RandMeth(object):
    r"""Randomization method for calculating isotropic spatial random fields.

    Parameters
    ----------
    model : :any:`CovModel`
        covariance model
    mode_no : :class:`int`, optional
        number of Fourier modes. Default: ``1000``
    seed : :class:`int` or :any:`None`, optional
        the seed of the random number generator.
        If "None", a random seed is used. Default: :any:`None`
    chunk_tmp_size : :class:`int`, optional
        Number of points (number of coordinates * mode_no)
        to be handled by one chunk while creating the fild.
        This is used to prevent memory overflows while
        generating the field. Default: ``1e7``
    verbose : :class:`bool`, optional
        State if there should be output during the generation.
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
        self,
        model,
        mode_no=1000,
        seed=None,
        chunk_tmp_size=1e7,
        verbose=False,
        **kwargs
    ):
        if kwargs:
            print("gstools.RandMeth: **kwargs are ignored")
        # initialize atributes
        self._mode_no = int(mode_no)
        self._chunk_tmp_size = int(chunk_tmp_size)
        self._verbose = bool(verbose)
        # initialize private atributes
        self._model = None
        self._seed = None
        self._rng = None
        self._z_1 = None
        self._z_2 = None
        self._cov_sample = None
        # set model and seed
        self.update(model, seed)

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
        # make a guess fo the chunk_no according to the input
        tmp_pnt = np.prod(summed_modes.shape) * self._mode_no
        chunk_no_exp = int(
            max(0, np.ceil(np.log2(tmp_pnt / self.chunk_tmp_size)))
        )
        # Test to see if enough memory is available.
        # In case there isn't, divide Fourier modes into 2 smaller chunks
        while True:
            try:
                chunk_no = 2 ** chunk_no_exp
                chunk_len = int(np.ceil(self._mode_no / chunk_no))
                if self.verbose:
                    print(
                        "RandMeth: Generating field with "
                        + str(chunk_no)
                        + " chunks"
                    )
                    print("(chunk length " + str(chunk_len) + ")")
                for chunk in range(chunk_no):
                    if self.verbose:
                        print(
                            "chunk " + str(chunk + 1) + " of " + str(chunk_no)
                        )
                    ch_start = chunk * chunk_len
                    # In case k[d,ch_start:ch_stop] with
                    # ch_stop >= len(k[d,:]) causes errors in
                    # numpy, use the commented min-function below
                    # ch_stop = min((chunk + 1) * chunk_len, self._mode_no-1)
                    ch_stop = (chunk + 1) * chunk_len

                    if self.model.dim == 1:
                        phase = self._cov_sample[0, ch_start:ch_stop] * x
                    elif self.model.dim == 2:
                        phase = (
                            self._cov_sample[0, ch_start:ch_stop] * x
                            + self._cov_sample[1, ch_start:ch_stop] * y
                        )
                    else:
                        phase = (
                            self._cov_sample[0, ch_start:ch_stop] * x
                            + self._cov_sample[1, ch_start:ch_stop] * y
                            + self._cov_sample[2, ch_start:ch_stop] * z
                        )
                    summed_modes += np.squeeze(
                        np.sum(
                            self._z_1[ch_start:ch_stop] * np.cos(phase)
                            + self._z_2[ch_start:ch_stop] * np.sin(phase),
                            axis=-1,
                        )
                    )
            except MemoryError:
                chunk_no_exp += 1
                print(
                    "Not enough memory. Dividing Fourier modes into {} "
                    "chunks.".format(2 ** chunk_no_exp)
                )
            else:
                # we break out of the endless loop if we don't get MemoryError
                break

        # generate normal distributed values for the nugget simulation
        if self.model.nugget > 0:
            nugget = np.sqrt(self.model.nugget) * self._rng.random.normal(
                size=summed_modes.shape
            )
        else:
            nugget = 0.0

        return np.sqrt(self.model.var / self._mode_no) * summed_modes + nugget

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
        Even if the given seed is the present one, modes will be racalculated.
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
        """:class:`int`: the seed of the master RNG

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
        """:any:`CovModel`: The covariance model of the spatial random field.
        """
        return self._model

    @model.setter
    def model(self, model):
        self.update(model)

    @property
    def mode_no(self):
        """:class:`int`: The number of modes in the randomization method."""
        return self._mode_no

    @mode_no.setter
    def mode_no(self, mode_no):
        if int(mode_no) != self._mode_no:
            self._mode_no = int(mode_no)
            self.reset_seed(self._seed)

    @property
    def chunk_tmp_size(self):
        """:class:`int`: temporary chunk size"""
        return self._chunk_tmp_size

    @chunk_tmp_size.setter
    def chunk_tmp_size(self, size):
        self._chunk_tmp_size = int(size)

    @property
    def verbose(self):
        """:class:`bool`: verbosity of the generator"""
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        self._verbose = bool(verbose)

    @property
    def name(self):
        """:class:`str`: The name of the generator"""
        return self.__class__.__name__

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "RandMeth(model={0}, mode_no={1}, seed={2})".format(
            repr(self.model), self._mode_no, self.seed
        )


if __name__ == "__main__": # pragma: no cover
    import doctest

    doctest.testmod()
