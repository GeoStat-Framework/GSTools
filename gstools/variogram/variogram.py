#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A collection of tools for estimating and fitting variograms.
"""
from __future__ import division, absolute_import, print_function

import numpy as np

from gstools.field import RNG
from gstools.variogram.estimator import unstructured


def estimate(field, bins, x, y=None, z=None, mesh_type='unstructured'):
    """Estimates the variogram of the input data.

    .. math:: \\gamma(r) = \\frac{1}{2 N} \\sum_{i=1}^N (z(\\mathbf x_i) - z(\\mathbf x_i'))^2

    Args:
        f (ndarray): the spatially distributed data
        bins (ndarray): the bins on which the variogram will be calculated
        x (ndarray): first components of position vectors if unstructured
        y (ndarray, opt.): analog to x
        z (ndarray, opt.): analog to x
        mesh_type (str, opt): 'structured' / 'unstructured' (for measurements, samples)
    Returns:
        the estimated variogram
    """
    if mesh_type == 'unstructured':
        return unstructured(field, bins, x, y, z)
    else:
        raise NotImplementedError('Variogram estimator for structured grids not yet implemented.')
