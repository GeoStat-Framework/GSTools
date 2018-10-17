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
from scipy import special as sps
from scipy.stats import rv_continuous
from hankel import SymmetricFourierTransform as SFT

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

    def _create_empty_k(self, mode_no=None):
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
        # TODO: Why rng called just once (in contrast to later use)
        rng = self._get_random_stream()
        for i in range(2):
            Z[i] = rng.normal(size=mode_no)
        return Z

    # TODO: What is the use of this method?
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
        k = self._create_empty_k(mode_no)
        iid = np.empty_like(k)

        for d in range(self.dim):
            # TODO: Why rng multiple called (in contrast to former use)
            rng = self._get_random_stream()
            iid[d] = rng.uniform(0., 1., mode_no)
        return k, iid

    def _sample_sphere(self, mode_no):
        """Uniform sampling on a d-dimensional sphere

        Parameters
        ----------
            mode_no : :class:`int`, optional
                number of the Fourier modes

        Returns
        -------
            coord : :class:`numpy.ndarray`
                x[, y[, z]] coordinates on the sphere with shape (dim, mode_no)
        """
        coord = self._create_empty_k(mode_no)
        if self.dim == 1:
            rng = self._get_random_stream()
            ang1 = rng.random_sample(mode_no)
            coord[0] = 2*np.around(ang1, decimals=0) - 1
        elif self.dim == 2:
            rng = self._get_random_stream()
            ang1 = rng.uniform(0.0, 2*np.pi, mode_no)
            coord[0] = np.cos(ang1)
            coord[1] = np.sin(ang1)
        elif self.dim == 3:
            rng = self._get_random_stream()
            ang1 = rng.uniform(0.0, 2*np.pi, mode_no)
            rng = self._get_random_stream()
            ang2 = rng.uniform(-1.0, 1.0, mode_no)
            coord[0] = np.sqrt(1.0-ang2**2)*np.cos(ang1)
            coord[1] = np.sqrt(1.0-ang2**2)*np.sin(ang1)
            coord[2] = ang2
        return coord

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
        if self.dim == 1:
            k = self._create_empty_k(mode_no)
            rng = self._get_random_stream()
            k[0] = rng.normal(0., np.pi/2.0/len_scale**2, mode_no)
        elif self.dim == 2:
            coord = self._sample_sphere(mode_no)
            rng = self._get_random_stream()
            rad_u = rng.random_sample(mode_no)
            # weibull distribution sampling
            rad = np.sqrt(np.pi)/len_scale*np.sqrt(-np.log(rad_u))
            k = rad*coord
        elif self.dim == 3:
            coord = self._sample_sphere(mode_no)
            rng = self._get_random_stream()

            def pdf(r):
                res = (4*r**2*len_scale**3/np.pi**2 *
                       np.exp(-(r*len_scale)**2/np.pi))
                return res

            def cdf(r):
                res = (sps.erf(r*len_scale/np.sqrt(np.pi)) -
                       2*r*len_scale/np.pi*np.exp(-(r*len_scale)**2/np.pi))
                return res

            dist = dist_gen(pdf_in=pdf, cdf_in=cdf, a=0, seed=rng)
            rad = dist.rvs(size=mode_no)
            k = rad*coord

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
        if self.dim == 1:
            k = self._create_empty_k(mode_no)
            rng = self._get_random_stream()
            k_u = rng.rng.uniform(-np.pi/2.0, np.pi/2.0, mode_no)
            k[0] = np.tan(k_u)/len_scale
        elif self.dim == 2:
            coord = self._sample_sphere(mode_no)
            rng = self._get_random_stream()
            rad_u = rng.random_sample(mode_no)
            # sampling with ppf
            rad = np.sqrt(1.0/rad_u**2 - 1.0)/len_scale
            k = rad*coord
        elif self.dim == 3:
            coord = self._sample_sphere(mode_no)
            rng = self._get_random_stream()

            def pdf(r):
                res = (4*r**2*len_scale**3/np.pi /
                       (1.0 + (r*len_scale)**2)**2)
                return res

            def cdf(r):
                res = 2.0/np.pi*(np.arctan(r*len_scale) -
                                 r*len_scale/(1.0 + (r*len_scale)**2))
                return res

            dist = dist_gen(pdf_in=pdf, cdf_in=cdf, a=0, seed=rng)
            rad = dist.rvs(size=mode_no)
            k = rad*coord

        return k

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

        if "kappa" in kwargs:
            kappa = kwargs["kappa"]
        else:
            kappa = 5.

        def cov_norm(r):
            """Normalized covariance function of the Matérn spectrum"""
            return (2**(1.-kappa)/sps.gamma(kappa) *
                    (np.sqrt(2*kappa)*r/len_scale)**kappa *
                    sps.kv(kappa, np.sqrt(2*kappa)*r/len_scale))

        pdf = gen_ft_func(cov_norm, self.dim, spec_dens=True)

        def spec_dens(r):
            r = np.abs(r)
            if self.dim == 1:
                mult = 1.0
            elif self.dim == 2:
                mult = 2*np.pi*r
            elif self.dim == 3:
                mult = 4*np.pi*r**2
            sub1 = (r*len_scale*sps.beta(kappa, 0.5)/np.pi)**(-2)
            return (mult*np.pi**(-self.dim/2.0)/r**self.dim *
                    sub1**kappa/(1+sub1)**(kappa+self.dim/2.0) *
                    sps.gamma(self.dim/2.0)/sps.beta(kappa, self.dim/2.0))

        if self.dim == 1:
            k = self._create_empty_k(mode_no)
            rng = self._get_random_stream()
            dist = dist_gen(pdf_in=spec_dens, seed=rng)
            k[0] = dist.rvs(size=mode_no)
        else:
            coord = self._sample_sphere(mode_no)
            rng = self._get_random_stream()
            dist = dist_gen(pdf_in=spec_dens, a=0, seed=rng)
            rad = dist.rvs(size=mode_no)
            k = rad*coord

        return k

    def sph(self, len_scale, mode_no=1000, **kwargs):
        """ Compute a spherical covariance model.

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
        def cov_norm(r):
            """Normalized covariance function of the spherical model"""
            r = np.abs(r)
            res = 1. - 9./16.*r/len_scale + 27./1024.*(r/len_scale)**3
            res[r > 8./3.*len_scale] = 0.
            return res

        pdf = gen_ft_func(cov_norm, self.dim, spec_dens=True)

        if self.dim == 1:
            k = self._create_empty_k(mode_no)
            rng = self._get_random_stream()
            dist = dist_gen(pdf_in=pdf, seed=rng)
            k[0] = dist.rvs(size=mode_no)
        else:
            coord = self._sample_sphere(mode_no)
            rng = self._get_random_stream()
            dist = dist_gen(pdf_in=pdf, a=0, seed=rng)
            rad = dist.rvs(size=mode_no)
            k = rad*coord

        return k

    def user(self, cov_norm, mode_no=1000):
        """User specified covariance model

        Parameters
        ----------
            len_scale : :class:`float`
                the length scale of the distribution
            cov_norm : :any:`callable`
                user specified normalized covariance model
                (divided by variance)

        Returns
        -------
            :class:`numpy.ndarray`
                the modes
        """
        pdf = gen_ft_func(cov_norm, self.dim, spec_dens=True)

        if self.dim == 1:
            k = self._create_empty_k(mode_no)
            rng = self._get_random_stream()
            dist = dist_gen(pdf_in=pdf, seed=rng)
            k[0] = dist.rvs(size=mode_no)
        else:
            coord = self._sample_sphere(mode_no)
            rng = self._get_random_stream()
            dist = dist_gen(pdf_in=pdf, a=0, seed=rng)
            rad = dist.rvs(size=mode_no)
            k = rad*coord

        return k

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


def dist_gen(pdf_in=None, cdf_in=None, ppf_in=None, **kwargs):
    '''Distribution Factory'''
    if ppf_in is None:
        if cdf_in is None:
            if pdf_in is None:
                raise ValueError("Either pdf or cdf must be given")
            else:
                return dist_gen_pdf(pdf_in, **kwargs)
        else:
            if pdf_in is None:
                return dist_gen_cdf(cdf_in, **kwargs)
            else:
                return dist_gen_pdf_cdf(pdf_in, cdf_in, **kwargs)
    else:
        if pdf_in is None or cdf_in is None:
            raise ValueError("pdf and cdf must be given along with the ppf")
        else:
            return dist_gen_pdf_cdf_ppf(pdf_in, cdf_in, ppf_in, **kwargs)


class dist_gen_pdf(rv_continuous):
    "Generate distribution from pdf"
    def __init__(self, pdf_in, **kwargs):
        self.pdf_in = pdf_in
        super(dist_gen_pdf, self).__init__(**kwargs)

    def _pdf(self, x):
        return self.pdf_in(x)


class dist_gen_cdf(rv_continuous):
    "Generate distribution from cdf"
    def __init__(self, cdf_in, **kwargs):
        self.cdf_in = cdf_in
        super(dist_gen_cdf, self).__init__(**kwargs)

    def _cdf(self, x):
        return self.cdf_in(x)


class dist_gen_pdf_cdf(rv_continuous):
    "Generate distribution from pdf and cdf"
    def __init__(self, pdf_in, cdf_in, **kwargs):
        self.pdf_in = pdf_in
        self.cdf_in = cdf_in
        super(dist_gen_pdf_cdf, self).__init__(**kwargs)

    def _pdf(self, x):
        return self.pdf_in(x)

    def _cdf(self, x):
        return self.cdf_in(x)


class dist_gen_pdf_cdf_ppf(rv_continuous):
    "Generate distribution from pdf, cdf and ppf"
    def __init__(self, pdf_in, cdf_in, ppf_in, **kwargs):
        self.pdf_in = pdf_in
        self.cdf_in = cdf_in
        self.ppf_in = ppf_in
        super(dist_gen_pdf_cdf_ppf, self).__init__(**kwargs)

    def _pdf(self, x):
        return self.pdf_in(x)

    def _cdf(self, x):
        return self.cdf_in(x)

    def _ppf(self, x):
        return self.ppf_in(x)


def gen_ft_func(func_in, dim, a=-1, b=1, N=1000, h=0.001, spec_dens=True):
    "Generate a callable fourier-transformation of func_in"
    ft = SFT(ndim=dim, a=a, N=N, h=h)

    def func_out(r):
        if spec_dens:
            if dim == 1:
                mult = 1.0
            elif dim == 2:
                mult = 2*np.pi*r
            elif dim == 3:
                mult = 4*np.pi*r**2
        else:
            mult = 1.0
        return mult*ft.transform(func_in, r, ret_err=False)

    return func_out


if __name__ == '__main__':
    import doctest
    doctest.testmod()
