#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This is the variogram estimater, implemented in cython.
"""
from __future__ import division, absolute_import, print_function

import numpy as np

cimport cython
from libc.math cimport sqrt
cimport numpy as np


DTYPE = np.double
ctypedef np.double_t DTYPE_t

ctypedef fused scalar:
    short
    int
    long
    float
    double

ctypedef fused scalar_bins:
    short
    int
    long
    float
    double

ctypedef fused scalar_f:
    short
    int
    long
    float
    double


@cython.boundscheck(False)
cdef inline double _distance_1d(scalar[:] x, scalar[:] y, scalar[:] z,
                               int i, int j):
    return sqrt((x[i] - x[j]) * (x[i] - x[j]))

@cython.boundscheck(False)
cdef inline double _distance_2d(scalar[:] x, scalar[:] y, scalar[:] z,
                               int i, int j):
    return sqrt((x[i] - x[j]) * (x[i] - x[j]) + (y[i] - y[j]) * (y[i] - y[j]))

@cython.boundscheck(False)
cdef inline double _distance_3d(scalar[:] x, scalar[:] y, scalar[:] z,
                               int i, int j):
    return sqrt((x[i] - x[j]) * (x[i] - x[j]) +
                (y[i] - y[j]) * (y[i] - y[j]) +
                (z[i] - z[j]) * (z[i] - z[j]))

ctypedef double (*_dist_func)(scalar[:], scalar[:], scalar[:], int, int)

def unstructured(scalar_f[:] f, scalar_bins[:] bins, scalar[:] x, scalar[:] y=None, scalar[:] z=None):
    if x.shape[0] != f.shape[0]:
        raise ValueError('len(x) = {0} != len(f) = {1} '.
                         format(x.shape[0], f.shape[0]))
    if bins.shape[0] < 2:
        raise ValueError('len(bins) too small')

    cdef _dist_func distance
    #3d
    if z is not None:
        if z.shape[0] != f.shape[0]:
            raise ValueError('len(z) = {0} != len(f) = {1} '.
                             format(z.shape[0], f.shape[0]))
        distance = _distance_3d
    #2d
    elif y is not None:
        if y.shape[0] != f.shape[0]:
            raise ValueError('len(y) = {0} != len(f) = {1} '.
                             format(y.shape[0], f.shape[0]))
        distance = _distance_2d
    #1d
    else:
        distance = _distance_1d

    cdef int i_max = bins.shape[0] - 1
    cdef int j_max = x.shape[0] - 1
    cdef int k_max = x.shape[0]

    cdef double[:] variogram = np.zeros(len(bins)-1)
    cdef long[:] counts = np.zeros_like(variogram, dtype=np.int)
    cdef int i, j, k, d
    cdef DTYPE_t dist
    for i in range(i_max):
        for j in range(j_max):
            for k in range(j+1, k_max):
                dist = distance(x, y, z, k, j)
                if dist >= bins[i] and dist < bins[i+1]:
                    counts[i] += 1
                    variogram[i] += (f[k] - f[j])**2
    #avoid division by zero
    for i in range(i_max):
        if counts[i] == 0:
            counts[i] = 1
        variogram[i] /= (2. * counts[i])
    return np.asarray(variogram)
