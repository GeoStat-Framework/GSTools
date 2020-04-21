#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: language = c++
# -*- coding: utf-8 -*-
"""
This is the variogram estimater, implemented in cython.
"""

import numpy as np

cimport cython
from cython.parallel import prange, parallel
from libcpp.vector cimport vector
from libc.math cimport fabs, sqrt
cimport numpy as np


DTYPE = np.double
ctypedef np.double_t DTYPE_t


cdef inline double distance(vector[double] pos1, vector[double] pos2) nogil:
    cdef int i = 0
    cdef double dist_squared = 0.0
    for i in range(pos1.size()):
        dist_squared += ((pos1[i] - pos2[i]) * (pos1[i] - pos2[i]))
    return sqrt(dist_squared)

cdef inline double estimator_matheron(const double f_diff) nogil:
    return f_diff * f_diff

cdef inline double estimator_cressie(const double f_diff) nogil:
    return sqrt(fabs(f_diff))
ctypedef double (*_estimator_func)(const double) nogil

cdef inline void normalization_matheron(
    vector[double]& variogram,
    vector[long]& counts
):
    cdef int i
    for i in range(variogram.size()):
        # avoid division by zero
        if counts[i] == 0:
            counts[i] = 1
        variogram[i] /= (2. * counts[i])

cdef inline void normalization_cressie(
    vector[double]& variogram,
    vector[long]& counts
):
    cdef int i
    for i in range(variogram.size()):
        # avoid division by zero
        if counts[i] == 0:
            counts[i] = 1
        variogram[i] = (
            0.5 * (1./counts[i] * variogram[i])**4 /
            (0.457 + 0.494 / counts[i] + 0.045 / counts[i]**2)
        )

ctypedef void (*_normalization_func)(
    vector[double]&,
    vector[long]&
)

cdef _estimator_func choose_estimator_func(str estimator_type):
    cdef _estimator_func estimator_func
    if estimator_type == 'm':
        estimator_func = estimator_matheron
    elif estimator_type == 'c':
        estimator_func = estimator_cressie
    return estimator_func

cdef _normalization_func choose_estimator_normalization(str estimator_type):
    cdef _normalization_func normalization_func
    if estimator_type == 'm':
        normalization_func = normalization_matheron
    elif estimator_type == 'c':
        normalization_func = normalization_cressie
    return normalization_func

ctypedef double (*_dist_func)(
    const double[:],
    const double[:],
    const double[:],
    const int,
    const int
) nogil


def unstructured(
    const int dim,
    const double[:] f,
    const double[:] bin_edges,
    const double[:,:] pos,
    str estimator_type='m'
):
    if bin_edges.shape[0] < 2:
        raise ValueError('len(bin_edges) too small')

    cdef _estimator_func estimator_func = choose_estimator_func(estimator_type)
    cdef _normalization_func normalization_func = (
        choose_estimator_normalization(estimator_type)
    )

    cdef int i_max = bin_edges.shape[0] - 1
    cdef int j_max = pos.shape[1] - 1
    cdef int k_max = pos.shape[1]

    cdef vector[double] variogram = vector[double](len(bin_edges)-1, 0.0)
    cdef vector[long] counts = vector[long](len(bin_edges)-1, 0)
    cdef vector[double] pos1 = vector[double](dim, 0.0)
    cdef vector[double] pos2 = vector[double](dim, 0.0)
    cdef int i, j, k, l
    cdef DTYPE_t dist
    #for i in prange(i_max, nogil=True):
    for i in range(i_max):
        for j in range(j_max):
            for k in range(j+1, k_max):
                for l in range(dim):
                    pos1[l] = pos[l, k]
                    pos2[l] = pos[l, j]
                dist = distance(pos1, pos2)
                if dist >= bin_edges[i] and dist < bin_edges[i+1]:
                    counts[i] += 1
                    variogram[i] += estimator_func(f[k] - f[j])

    normalization_func(variogram, counts)
    return np.asarray(variogram)


def structured(const double[:,:,:] f, str estimator_type='m'):
    cdef _estimator_func estimator_func = choose_estimator_func(estimator_type)
    cdef _normalization_func normalization_func = (
        choose_estimator_normalization(estimator_type)
    )

    cdef int i_max = f.shape[0] - 1
    cdef int j_max = f.shape[1]
    cdef int k_max = f.shape[2]
    cdef int l_max = i_max + 1

    cdef vector[double] variogram = vector[double](l_max, 0.0)
    cdef vector[long] counts = vector[long](l_max, 0)
    cdef int i, j, k, l

    with nogil, parallel():
        for i in range(i_max):
            for j in range(j_max):
                for k in range(k_max):
                    for l in prange(1, l_max-i):
                        counts[l] += 1
                        variogram[l] += estimator_func(f[i,j,k] - f[i+l,j,k])

    normalization_func(variogram, counts)
    return np.asarray(variogram)

def ma_structured(
    const double[:,:,:] f,
    const bint[:,:,:] mask,
    str estimator_type='m'
):
    cdef _estimator_func estimator_func = choose_estimator_func(estimator_type)
    cdef _normalization_func normalization_func = (
        choose_estimator_normalization(estimator_type)
    )

    cdef int i_max = f.shape[0] - 1
    cdef int j_max = f.shape[1]
    cdef int k_max = f.shape[2]
    cdef int l_max = i_max + 1

    cdef vector[double] variogram = vector[double](l_max, 0.0)
    cdef vector[long] counts = vector[long](l_max, 0)
    cdef int i, j, k, l

    with nogil, parallel():
        for i in range(i_max):
            for j in range(j_max):
                for k in range(k_max):
                    for l in prange(1, l_max-i):
                        if not mask[i,j,k] and not mask[i+l,j,k]:
                            counts[l] += 1
                            variogram[l] += estimator_func(f[i,j,k] - f[i+l,j,k])

    normalization_func(variogram, counts)
    return np.asarray(variogram)
