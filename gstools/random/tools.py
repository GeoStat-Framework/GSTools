# -*- coding: utf-8 -*-
"""
GStools subpackage providing tools for random sampling.

.. currentmodule:: gstools.random.tools

The following classes are provided

.. autosummary::
   MasterRNG
   dist_gen
"""

from scipy.stats import rv_continuous
import numpy.random as rand

__all__ = ["MasterRNG", "dist_gen"]


class MasterRNG:
    """Master random number generator for generating seeds.

    Parameters
    ----------
    seed : :class:`int` or :any:`None`, optional
        The seed of the master RNG, if ``None``,
        a random seed is used. Default: ``None``

    """

    def __init__(self, seed):
        self._seed = seed
        self._master_rng_fct = rand.RandomState(seed)
        self._master_rng = lambda: self._master_rng_fct.randint(1, 2 ** 16)

    def __call__(self):
        """Return a random seed."""
        return self._master_rng()

    @property  # pragma: no cover
    def seed(self):
        """:class:`int`: Seed of the master RNG.

        The setter property not only saves the new seed, but also creates
        a new master RNG function with the new seed.
        """
        return self._seed

    def __repr__(self):
        """Return String representation."""
        return f"MasterRNG(seed={self.seed})"


def dist_gen(pdf_in=None, cdf_in=None, ppf_in=None, **kwargs):
    """Distribution Factory.

    Parameters
    ----------
    pdf_in : :any:`callable` or :any:`None`, optional
        Proprobability distribution function of the given distribution, that
        takes a single argument
        Default: ``None``
    cdf_in : :any:`callable` or :any:`None`, optional
        Cumulative distribution function of the given distribution, that
        takes a single argument
        Default: ``None``
    ppf_in : :any:`callable` or :any:`None`, optional
        Percent point function of the given distribution, that
        takes a single argument
        Default: ``None``
    **kwargs
        Keyword-arguments forwarded to :any:`scipy.stats.rv_continuous`.

    Returns
    -------
    dist : :class:`scipy.stats.rv_continuous`
        The constructed distribution.

    Notes
    -----
    At least pdf or cdf needs to be given.
    """
    if ppf_in is None:
        if pdf_in is not None and cdf_in is None:
            return DistPdf(pdf_in, **kwargs)
        if pdf_in is None and cdf_in is not None:
            return DistCdf(cdf_in, **kwargs)
        if pdf_in is not None and cdf_in is not None:
            return DistPdfCdf(pdf_in, cdf_in, **kwargs)
        raise ValueError("Either pdf or cdf must be given")

    if pdf_in is not None and cdf_in is None:
        return DistPdfPpf(pdf_in, ppf_in, **kwargs)
    if pdf_in is None and cdf_in is not None:
        return DistCdfPpf(cdf_in, ppf_in, **kwargs)
    if pdf_in is not None and cdf_in is not None:
        return DistPdfCdfPpf(pdf_in, cdf_in, ppf_in, **kwargs)
    raise ValueError("pdf or cdf must be given along with the ppf")


class DistPdf(rv_continuous):
    """Generate distribution from pdf."""

    def __init__(self, pdf_in, **kwargs):
        self.pdf_in = pdf_in
        super().__init__(**kwargs)

    def _pdf(self, x, *args):
        return self.pdf_in(x)


class DistCdf(rv_continuous):
    """Generate distribution from cdf."""

    def __init__(self, cdf_in, **kwargs):
        self.cdf_in = cdf_in
        super().__init__(**kwargs)

    def _cdf(self, x, *args):
        return self.cdf_in(x)


class DistPdfCdf(rv_continuous):
    """Generate distribution from pdf and cdf."""

    def __init__(self, pdf_in, cdf_in, **kwargs):
        self.pdf_in = pdf_in
        self.cdf_in = cdf_in
        super().__init__(**kwargs)

    def _pdf(self, x, *args):
        return self.pdf_in(x)

    def _cdf(self, x, *args):
        return self.cdf_in(x)


class DistPdfPpf(rv_continuous):
    """Generate distribution from pdf and ppf."""

    def __init__(self, pdf_in, ppf_in, **kwargs):
        self.pdf_in = pdf_in
        self.ppf_in = ppf_in
        super().__init__(**kwargs)

    def _pdf(self, x, *args):
        return self.pdf_in(x)

    def _ppf(self, q, *args):
        return self.ppf_in(q)


class DistCdfPpf(rv_continuous):
    """Generate distribution from cdf and ppf."""

    def __init__(self, cdf_in, ppf_in, **kwargs):
        self.cdf_in = cdf_in
        self.ppf_in = ppf_in
        super().__init__(**kwargs)

    def _cdf(self, x, *args):
        return self.cdf_in(x)

    def _ppf(self, q, *args):
        return self.ppf_in(q)


class DistPdfCdfPpf(rv_continuous):
    """Generate distribution from pdf, cdf and ppf."""

    def __init__(self, pdf_in, cdf_in, ppf_in, **kwargs):
        self.pdf_in = pdf_in
        self.cdf_in = cdf_in
        self.ppf_in = ppf_in
        super().__init__(**kwargs)

    def _pdf(self, x, *args):
        return self.pdf_in(x)

    def _cdf(self, x, *args):
        return self.cdf_in(x)

    def _ppf(self, q, *args):
        return self.ppf_in(q)
