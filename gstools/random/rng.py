# -*- coding: utf-8 -*-
"""
GStools subpackage providing the core of the spatial random field generation.

.. currentmodule:: gstools.random.rng

The following classes are provided

.. autosummary::
   RNG
"""
# pylint: disable=E1101
import numpy as np
import numpy.random as rand
import emcee as mc
from emcee.state import State
from gstools.random.tools import MasterRNG, dist_gen

__all__ = ["RNG"]


class RNG:
    """
    A random number generator for different distributions and multiple streams.

    Parameters
    ----------
    seed : :class:`int` or :any:`None`, optional
        The seed of the master RNG, if ``None``,
        a random seed is used. Default: ``None``
    """

    def __init__(self, seed=None):
        # set seed
        self._master_rng = None
        self.seed = seed

    def sample_ln_pdf(
        self,
        ln_pdf,
        size=None,
        sample_around=1.0,
        nwalkers=50,
        burn_in=20,
        oversampling_factor=10,
    ):
        """Sample from a distribution given by ln(pdf).

        This algorithm uses the :class:`emcee.EnsembleSampler`

        Parameters
        ----------
        ln_pdf : :any:`callable`
            The logarithm of the Probability density function
            of the given distribution, that takes a single argument
        size : :class:`int` or :any:`None`, optional
            sample size. Default: None
        sample_around : :class:`float`, optional
            Starting point for initial guess Default: 1.
        nwalkers : :class:`int`, optional
            The number of walkers in the mcmc sampler. Used for the
            emcee.EnsembleSampler class.
            Default: 50
        burn_in : :class:`int`, optional
            Number of burn-in runs in the mcmc algorithm.
            Default: 20
        oversampling_factor : :class:`int`, optional
            To guess the sample number needed for proper results, we use a
            factor for oversampling. The intern used sample-size is
            calculated by

            ``sample_size = max(burn_in, (size/nwalkers)*oversampling_factor)``

            So at least, as much as the burn-in runs.
            Default: 10
        """
        if size is None:  # pragma: no cover
            sample_size = burn_in
        else:
            sample_size = max(burn_in, (size / nwalkers) * oversampling_factor)
        # sample_size needs to be integer for emcee >= 3.1
        sample_size = int(sample_size)
        # initial guess
        init_guess = (
            self.random.rand(nwalkers).reshape((nwalkers, 1)) * sample_around
        )
        # initialize the sampler
        sampler = mc.EnsembleSampler(nwalkers, 1, ln_pdf)
        # burn in phase with saving of last position
        initial_state = State(init_guess, copy=True)
        initial_state.random_state = self.random.get_state()
        burn_in_state = sampler.run_mcmc(
            initial_state=initial_state, nsteps=burn_in
        )
        # reset after burn_in
        sampler.reset()
        # actual sampling
        initial_state = State(burn_in_state, copy=True)
        initial_state.random_state = self.random.get_state()
        sampler.run_mcmc(initial_state=initial_state, nsteps=sample_size)
        samples = sampler.get_chain(flat=True)[:, 0]

        # choose samples according to size
        return self.random.choice(samples, size)

    def sample_dist(self, pdf=None, cdf=None, ppf=None, size=None, **kwargs):
        """Sample from a distribution given by pdf, cdf and/or ppf.

        Parameters
        ----------
        pdf : :any:`callable` or :any:`None`, optional
            Probability density function of the given distribution,
            that takes a single argument
            Default: ``None``
        cdf : :any:`callable` or :any:`None`, optional
            Cumulative distribution function of the given distribution, that
            takes a single argument
            Default: ``None``
        ppf : :any:`callable` or :any:`None`, optional
            Percent point function of the given distribution, that
            takes a single argument
            Default: ``None``
        size : :class:`int` or :any:`None`, optional
            sample size. Default: None
        **kwargs
            Keyword-arguments that are forwarded to
            :any:`scipy.stats.rv_continuous`.

        Returns
        -------
        samples : :class:`float` or :class:`numpy.ndarray`
            the samples from the given distribution

        Notes
        -----
        At least pdf or cdf needs to be given.
        """
        kwargs["seed"] = self.random
        dist = dist_gen(pdf_in=pdf, cdf_in=cdf, ppf_in=ppf, **kwargs)
        return dist.rvs(size=size)

    def sample_sphere(self, dim, size=None):
        """Uniform sampling on a d-dimensional sphere.

        Parameters
        ----------
        dim : :class:`int`
            Dimension of the sphere. Just 1, 2, and 3 supported.
        size : :class:`int`, optional
            sample size

        Returns
        -------
        coord : :class:`numpy.ndarray`
            x[, y[, z]] coordinates on the sphere with shape (dim, size)
        """
        if size is None:  # pragma: no cover
            coord = np.empty((dim, 1), dtype=np.double)
        else:
            coord = np.empty(  # saver conversion of size to resulting shape
                (dim,) + tuple(np.atleast_1d(size)), dtype=np.double
            )
        if dim == 1:
            coord[0] = self.random.choice([-1, 1], size=size)
        elif dim == 2:
            ang1 = self.random.uniform(0.0, 2 * np.pi, size)
            coord[0] = np.cos(ang1)
            coord[1] = np.sin(ang1)
        elif dim == 3:
            ang1 = self.random.uniform(0.0, 2 * np.pi, size)
            ang2 = self.random.uniform(-1.0, 1.0, size)
            coord[0] = np.sqrt(1.0 - ang2 ** 2) * np.cos(ang1)
            coord[1] = np.sqrt(1.0 - ang2 ** 2) * np.sin(ang1)
            coord[2] = ang2
        else:  # pragma: no cover
            # http://corysimon.github.io/articles/uniformdistn-on-sphere/
            coord = self.random.normal(size=coord.shape)
            while True:  # loop until all norms are non-zero
                norm = np.linalg.norm(coord, axis=0)
                # check for zero norms
                zero_norms = np.isclose(norm, 0)
                # exit the loop if all norms are non-zero
                if not np.any(zero_norms):
                    break
                # transpose, since the next transpose reverses axis order
                zero_samples = zero_norms.T.nonzero()
                # need to transpose to have dim-axis last
                new_shape = coord.T[zero_samples].shape
                # resample the zero norm samples
                coord.T[zero_samples] = self.random.normal(size=new_shape)
            # project onto sphere
            coord = coord / norm
        return np.reshape(coord, dim) if size is None else coord

    @property
    def random(self):
        """:any:`numpy.random.RandomState`: Randomstate.

        Get a stream to the numpy Random number generator.
        You can use this, to call any provided distribution
        from :any:`numpy.random.RandomState`.
        """
        return rand.RandomState(self._master_rng())

    @property  # pragma: no cover
    def seed(self):
        """:class:`int`: Seed of the master RNG.

        The setter property not only saves the new seed, but also creates
        a new master RNG function with the new seed.
        """
        return self._master_rng.seed

    @seed.setter
    def seed(self, new_seed=None):
        self._master_rng = MasterRNG(new_seed)

    def __repr__(self):
        """Return String representation."""
        return f"RNG(seed={self.seed})"
