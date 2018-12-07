# -*- coding: utf-8 -*-
"""
GStools subpackage providing tools for random sampling.

.. currentmodule:: gstools.random.tools

The following classes are provided

.. autosummary::
   dist_gen
"""
from __future__ import division, absolute_import, print_function

from scipy.stats import rv_continuous

__all__ = ["dist_gen"]


def dist_gen(pdf_in=None, cdf_in=None, ppf_in=None, **kwargs):
    """Distribution Factory

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
        Keyword-arguments that are forwarded to ``scipy.stats.rv_continuous``.

    Note
    ----
    At least pdf or cdf needs to given.
    """
    if ppf_in is None:
        if pdf_in is not None and cdf_in is None:
            return DistPdf(pdf_in, **kwargs)
        if pdf_in is None and cdf_in is not None:
            return DistCdf(cdf_in, **kwargs)
        if pdf_in is not None and cdf_in is not None:
            return DistPdfCdf(pdf_in, cdf_in, **kwargs)
        raise ValueError("Either pdf or cdf must be given")
    else:
        if pdf_in is not None and cdf_in is None:
            return DistPdfPpf(pdf_in, ppf_in, **kwargs)
        if pdf_in is None and cdf_in is not None:
            return DistCdfPpf(cdf_in, ppf_in, **kwargs)
        if pdf_in is not None and cdf_in is not None:
            return DistPdfCdfPpf(pdf_in, cdf_in, ppf_in, **kwargs)
        raise ValueError("pdf or cdf must be given along with the ppf")


class DistPdf(rv_continuous):
    "Generate distribution from pdf"

    def __init__(self, pdf_in, **kwargs):
        self.pdf_in = pdf_in
        super(DistPdf, self).__init__(**kwargs)

    def _pdf(self, x, *args):
        return self.pdf_in(x)


class DistCdf(rv_continuous):
    "Generate distribution from cdf"

    def __init__(self, cdf_in, **kwargs):
        self.cdf_in = cdf_in
        super(DistCdf, self).__init__(**kwargs)

    def _cdf(self, x, *args):
        return self.cdf_in(x)


class DistPdfCdf(rv_continuous):
    "Generate distribution from pdf and cdf"

    def __init__(self, pdf_in, cdf_in, **kwargs):
        self.pdf_in = pdf_in
        self.cdf_in = cdf_in
        super(DistPdfCdf, self).__init__(**kwargs)

    def _pdf(self, x, *args):
        return self.pdf_in(x)

    def _cdf(self, x, *args):
        return self.cdf_in(x)


class DistPdfPpf(rv_continuous):
    "Generate distribution from pdf and ppf"

    def __init__(self, pdf_in, ppf_in, **kwargs):
        self.pdf_in = pdf_in
        self.ppf_in = ppf_in
        super(DistPdfPpf, self).__init__(**kwargs)

    def _pdf(self, x, *args):
        return self.pdf_in(x)

    def _ppf(self, q, *args):
        return self.ppf_in(q)


class DistCdfPpf(rv_continuous):
    "Generate distribution from cdf and ppf"

    def __init__(self, cdf_in, ppf_in, **kwargs):
        self.cdf_in = cdf_in
        self.ppf_in = ppf_in
        super(DistCdfPpf, self).__init__(**kwargs)

    def _cdf(self, x, *args):
        return self.cdf_in(x)

    def _ppf(self, q, *args):
        return self.ppf_in(q)


class DistPdfCdfPpf(rv_continuous):
    "Generate distribution from pdf, cdf and ppf"

    def __init__(self, pdf_in, cdf_in, ppf_in, **kwargs):
        self.pdf_in = pdf_in
        self.cdf_in = cdf_in
        self.ppf_in = ppf_in
        super(DistPdfCdfPpf, self).__init__(**kwargs)

    def _pdf(self, x, *args):
        return self.pdf_in(x)

    def _cdf(self, x, *args):
        return self.cdf_in(x)

    def _ppf(self, q, *args):
        return self.ppf_in(q)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
