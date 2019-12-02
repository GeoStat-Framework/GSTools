#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# -*- coding: utf-8 -*-
"""
This is the variogram estimater, implemented in cython.
"""

import numpy as np

cimport cython
from cython.parallel import prange
from libc.math cimport sqrt
cimport numpy as np


DTYPE = np.double
ctypedef np.double_t DTYPE_t


cdef inline double _distance_1d(
    const double[:] x,
    const double[:] y,
    const double[:] z,
    const int i,
    const int j
) nogil:
    return sqrt((x[i] - x[j]) * (x[i] - x[j]))

cdef inline double _distance_2d(
    const double[:] x,
    const double[:] y,
    const double[:] z,
    const int i,
    const int j
) nogil:
    return sqrt((x[i] - x[j]) * (x[i] - x[j]) + (y[i] - y[j]) * (y[i] - y[j]))

cdef inline double _distance_3d(
    const double[:] x,
    const double[:] y,
    const double[:] z,
    const int i,
    const int j
) nogil:
    return sqrt((x[i] - x[j]) * (x[i] - x[j]) +
                (y[i] - y[j]) * (y[i] - y[j]) +
                (z[i] - z[j]) * (z[i] - z[j]))

ctypedef double (*_dist_func)(
    const double[:],
    const double[:],
    const double[:],
    const int,
    const int
) nogil


def unstructured(
    const double[:] f,
    const double[:] bin_edges,
    const double[:] x,
    const double[:] y=None,
    const double[:] z=None
):
    if x.shape[0] != f.shape[0]:
        raise ValueError('len(x) = {0} != len(f) = {1} '.
                         format(x.shape[0], f.shape[0]))
    if bin_edges.shape[0] < 2:
        raise ValueError('len(bin_edges) too small')

    cdef _dist_func distance
    # 3d
    if z is not None:
        if z.shape[0] != f.shape[0]:
            raise ValueError('len(z) = {0} != len(f) = {1} '.
                             format(z.shape[0], f.shape[0]))
        distance = _distance_3d
    # 2d
    elif y is not None:
        if y.shape[0] != f.shape[0]:
            raise ValueError('len(y) = {0} != len(f) = {1} '.
                             format(y.shape[0], f.shape[0]))
        distance = _distance_2d
    # 1d
    else:
        distance = _distance_1d

    cdef int i_max = bin_edges.shape[0] - 1
    cdef int j_max = x.shape[0] - 1
    cdef int k_max = x.shape[0]

    cdef double[:] variogram = np.zeros(len(bin_edges)-1)
    cdef long[:] counts = np.zeros_like(variogram, dtype=np.int)
    cdef int i, j, k, d
    cdef DTYPE_t dist
    for i in prange(i_max, nogil=True):
        for j in range(j_max):
            for k in range(j+1, k_max):
                dist = distance(x, y, z, k, j)
                if dist >= bin_edges[i] and dist < bin_edges[i+1]:
                    counts[i] += 1
                    variogram[i] += (f[k] - f[j])**2
    for i in range(i_max):
        # avoid division by zero
        if counts[i] == 0:
            counts[i] = 1
        variogram[i] /= (2. * counts[i])
    return np.asarray(variogram)


def structured_3d(const double[:,:,:] f):
    cdef int i_max = f.shape[0] - 1
    cdef int j_max = f.shape[1]
    cdef int k_max = f.shape[2]
    cdef int l_max = i_max + 1

    cdef double[:] variogram = np.zeros(l_max)
    cdef long[:] counts = np.zeros_like(variogram, dtype=np.int)
    cdef int i, j, k, l

    for i in prange(i_max, nogil=True):
        for j in range(j_max):
            for k in range(k_max):
                for l in range(1, l_max-i):
                    counts[l] += 1
                    variogram[l] += (f[i,j,k] - f[i+l,j,k])**2
    for i in range(l_max):
        # avoid division by zero
        if counts[i] == 0:
            counts[i] = 1
        variogram[i] /= (2. * counts[i])
    return np.asarray(variogram)

def structured_2d(const double[:,:] f):
    cdef int i_max = f.shape[0] - 1
    cdef int j_max = f.shape[1]
    cdef int k_max = i_max + 1

    cdef double[:] variogram = np.zeros(k_max)
    cdef long[:] counts = np.zeros_like(variogram, dtype=np.int)
    cdef int i, j, k

    for i in prange(i_max, nogil=True):
        for j in range(j_max):
            for k in range(1, k_max-i):
                counts[k] += 1
                variogram[k] += (f[i,j] - f[i+k,j])**2
    for i in range(k_max):
        # avoid division by zero
        if counts[i] == 0:
            counts[i] = 1
        variogram[i] /= (2. * counts[i])
    return np.asarray(variogram)

def structured_1d(const double[:] f):
    cdef int i_max = f.shape[0] - 1
    cdef int j_max = i_max + 1

    cdef double[:] variogram = np.zeros(j_max)
    cdef long[:] counts = np.zeros_like(variogram, dtype=np.int)
    cdef int i, j

    for i in range(i_max):
        for j in range(1, j_max-i):
            counts[j] += 1
            variogram[j] += (f[i] - f[i+j])**2
    for i in range(j_max):
        # avoid division by zero
        if counts[i] == 0:
            counts[i] = 1
        variogram[i] /= (2. * counts[i])
    return np.asarray(variogram)

def ma_structured_3d(const double[:,:,:] f, const bint[:,:,:] mask):
    cdef int i_max = f.shape[0] - 1
    cdef int j_max = f.shape[1]
    cdef int k_max = f.shape[2]
    cdef int l_max = i_max + 1

    cdef double[:] variogram = np.zeros(l_max)
    cdef long[:] counts = np.zeros_like(variogram, dtype=np.int)
    cdef int i, j, k, l

    for i in prange(i_max, nogil=True):
        for j in range(j_max):
            for k in range(k_max):
                for l in range(1, l_max-i):
                    if not mask[i,j,k] and not mask[i+l,j,k]:
                        counts[l] += 1
                        variogram[l] += (f[i,j,k] - f[i+l,j,k])**2
    for i in range(l_max):
        # avoid division by zero
        if counts[i] == 0:
            counts[i] = 1
        variogram[i] /= (2. * counts[i])
    return np.asarray(variogram)

def ma_structured_2d(const double[:,:] f, const bint[:,:] mask):
    cdef int i_max = f.shape[0] - 1
    cdef int j_max = f.shape[1]
    cdef int k_max = i_max + 1

    cdef double[:] variogram = np.zeros(k_max)
    cdef long[:] counts = np.zeros_like(variogram, dtype=np.int)
    cdef int i, j, k

    for i in prange(i_max, nogil=True):
        for j in range(j_max):
            for k in range(1, k_max-i):
                if not mask[i,j] and not mask[i+k,j]:
                    counts[k] += 1
                    variogram[k] += (f[i,j] - f[i+k,j])**2
    for i in range(k_max):
        # avoid division by zero
        if counts[i] == 0:
            counts[i] = 1
        variogram[i] /= (2. * counts[i])
    return np.asarray(variogram)

def ma_structured_1d(const double[:] f, const bint[:] mask):
    cdef int i_max = f.shape[0] - 1
    cdef int j_max = i_max + 1

    cdef double[:] variogram = np.zeros(j_max)
    cdef long[:] counts = np.zeros_like(variogram, dtype=np.int)
    cdef int i, j

    for i in range(i_max):
        for j in range(1, j_max-i):
            if not mask[i] and not mask[j]:
                counts[j] += 1
                variogram[j] += (f[i] - f[i+j])**2
    for i in range(j_max):
        # avoid division by zero
        if counts[i] == 0:
            counts[i] = 1
        variogram[i] /= (2. * counts[i])
    return np.asarray(variogram)
