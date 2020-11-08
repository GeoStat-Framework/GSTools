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
from libc.math cimport fabs, sqrt, isnan, acos, M_PI
cimport numpy as np


DTYPE = np.double
ctypedef np.double_t DTYPE_t


cdef inline double distance(
    const int dim,
    const double[:,:] pos,
    const int i,
    const int j
) nogil:
    cdef int d
    cdef double dist_squared = 0.0
    for d in range(dim):
        dist_squared += ((pos[d,i] - pos[d,j]) * (pos[d,i] - pos[d,j]))
    return sqrt(dist_squared)


cdef inline bint dir_test(
    const int dim,
    const double[:,:] pos,
    const double[:,:] direction,
    const double angles_tol,
    const double bandwidth,
    const int i,
    const int j,
    const int d
) nogil:
    cdef double s_prod = 0.0
    cdef double p_norm = 0.0
    cdef double b_dist = 0.0
    cdef int k
    cdef bint in_band = True
    cdef bint in_angle

    # scalar-product calculation for bandwidth projection and angle calculation
    for k in range(dim):
        s_prod += (pos[k,i] - pos[k,j]) * direction[d,k]
        p_norm += (pos[k,i] - pos[k,j])**2
    p_norm = sqrt(p_norm)

    # calculate band-distance by projection of point-pair-vec to direction line
    if bandwidth > 0.0:
        for k in range(dim):
            b_dist += ((pos[k,i] - pos[k,j]) - s_prod * direction[d,k]) ** 2
        b_dist = sqrt(b_dist)
        in_band = b_dist < bandwidth

    # use smallest angle by taking absolut value for arccos angle formula
    in_angle = acos(fabs(s_prod) / p_norm) < angles_tol

    return in_band and in_angle


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
        variogram[i] /= (2. * max(counts[i], 1))

cdef inline void normalization_cressie(
    vector[double]& variogram,
    vector[long]& counts
):
    cdef int i
    cdef long cnt
    for i in range(variogram.size()):
        # avoid division by zero
        cnt = max(counts[i], 1)
        variogram[i] = (
            0.5 * (1./cnt * variogram[i])**4 /
            (0.457 + 0.494 / cnt + 0.045 / cnt**2)
        )

ctypedef void (*_normalization_func)(
    vector[double]&,
    vector[long]&
)

cdef inline void normalization_matheron_vec(
    double[:,:]& variogram,
    long[:,:]& counts
):
    cdef int d, i
    for d in range(variogram.shape[0]):
        for i in range(variogram.shape[1]):
            # avoid division by zero
            variogram[d, i] /= (2. * max(counts[d, i], 1))

cdef inline void normalization_cressie_vec(
    double[:,:]& variogram,
    long[:,:]& counts
):
    cdef int d, i
    cdef long cnt
    for d in range(variogram.shape[0]):
        for i in range(variogram.shape[1]):
            # avoid division by zero
            cnt = max(counts[d, i], 1)
            variogram[d, i] = (
                0.5 * (1./cnt * variogram[d, i])**4 /
                (0.457 + 0.494 / cnt + 0.045 / cnt**2)
            )

ctypedef void (*_normalization_func_vec)(
    double[:,:]&,
    long[:,:]&
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

cdef _normalization_func_vec choose_estimator_normalization_vec(str estimator_type):
    cdef _normalization_func_vec normalization_func_vec
    if estimator_type == 'm':
        normalization_func_vec = normalization_matheron_vec
    elif estimator_type == 'c':
        normalization_func_vec = normalization_cressie_vec
    return normalization_func_vec


def directional(
    const int dim,
    const double[:,:] f,
    const double[:] bin_edges,
    const double[:,:] pos,
    const double[:,:] direction,  # should be normed
    const double angles_tol=M_PI/8.0,
    const double bandwidth=-1.0,  # negative values to turn of bandwidth search
    str estimator_type='m'
):
    if pos.shape[1] != f.shape[1]:
        raise ValueError('len(pos) = {0} != len(f) = {1} '.
                         format(pos.shape[1], f.shape[1]))

    if bin_edges.shape[0] < 2:
        raise ValueError('len(bin_edges) too small')

    if angles_tol <= 0:
        raise ValueError('tolerance for angle search masks must be > 0')

    cdef _estimator_func estimator_func = choose_estimator_func(estimator_type)
    cdef _normalization_func_vec normalization_func_vec = (
        choose_estimator_normalization_vec(estimator_type)
    )

    cdef int d_max = direction.shape[0]
    cdef int i_max = bin_edges.shape[0] - 1
    cdef int j_max = pos.shape[1] - 1
    cdef int k_max = pos.shape[1]
    cdef int f_max = f.shape[0]

    cdef double[:,:] variogram = np.zeros((d_max, len(bin_edges)-1))
    cdef long[:,:] counts = np.zeros((d_max, len(bin_edges)-1), dtype=long)
    cdef vector[double] pos1 = vector[double](dim, 0.0)
    cdef vector[double] pos2 = vector[double](dim, 0.0)
    cdef int i, j, k, m, d
    cdef DTYPE_t dist

    for i in prange(i_max, nogil=True):
        for j in range(j_max):
            for k in range(j+1, k_max):
                dist = distance(dim, pos, j, k)
                if dist >= bin_edges[i] and dist < bin_edges[i+1]:
                    for d in range(d_max):
                        if dir_test(dim, pos, direction, angles_tol, bandwidth, k, j, d):
                            for m in range(f_max):
                                # skip no data values
                                if not (isnan(f[m,k]) or isnan(f[m,j])):
                                    counts[d, i] += 1
                                    variogram[d, i] += estimator_func(f[m,k] - f[m,j])

    normalization_func_vec(variogram, counts)
    return np.asarray(variogram), np.asarray(counts)

def unstructured(
    const int dim,
    const double[:,:] f,
    const double[:] bin_edges,
    const double[:,:] pos,
    str estimator_type='m'
):
    if pos.shape[1] != f.shape[1]:
        raise ValueError('len(pos) = {0} != len(f) = {1} '.
                         format(pos.shape[1], f.shape[1]))

    if bin_edges.shape[0] < 2:
        raise ValueError('len(bin_edges) too small')

    cdef _estimator_func estimator_func = choose_estimator_func(estimator_type)
    cdef _normalization_func normalization_func = (
        choose_estimator_normalization(estimator_type)
    )

    cdef int i_max = bin_edges.shape[0] - 1
    cdef int j_max = pos.shape[1] - 1
    cdef int k_max = pos.shape[1]
    cdef int f_max = f.shape[0]

    cdef vector[double] variogram = vector[double](len(bin_edges)-1, 0.0)
    cdef vector[long] counts = vector[long](len(bin_edges)-1, 0)
    cdef vector[double] pos1 = vector[double](dim, 0.0)
    cdef vector[double] pos2 = vector[double](dim, 0.0)
    cdef int i, j, k, m
    cdef DTYPE_t dist

    for i in prange(i_max, nogil=True):
        for j in range(j_max):
            for k in range(j+1, k_max):
                dist = distance(dim, pos, j, k)
                if dist >= bin_edges[i] and dist < bin_edges[i+1]:
                    for m in range(f_max):
                        # skip no data values
                        if not (isnan(f[m,k]) or isnan(f[m,j])):
                            counts[i] += 1
                            variogram[i] += estimator_func(f[m,k] - f[m,j])

    normalization_func(variogram, counts)
    return np.asarray(variogram), np.asarray(counts)


def structured(const double[:,:] f, str estimator_type='m'):
    cdef _estimator_func estimator_func = choose_estimator_func(estimator_type)
    cdef _normalization_func normalization_func = (
        choose_estimator_normalization(estimator_type)
    )

    cdef int i_max = f.shape[0] - 1
    cdef int j_max = f.shape[1]
    cdef int k_max = i_max + 1

    cdef vector[double] variogram = vector[double](k_max, 0.0)
    cdef vector[long] counts = vector[long](k_max, 0)
    cdef int i, j, k

    with nogil, parallel():
        for i in range(i_max):
            for j in range(j_max):
                for k in prange(1, k_max-i):
                    counts[k] += 1
                    variogram[k] += estimator_func(f[i,j] - f[i+k,j])

    normalization_func(variogram, counts)
    return np.asarray(variogram)


def ma_structured(
    const double[:,:] f,
    const bint[:,:] mask,
    str estimator_type='m'
):
    cdef _estimator_func estimator_func = choose_estimator_func(estimator_type)
    cdef _normalization_func normalization_func = (
        choose_estimator_normalization(estimator_type)
    )

    cdef int i_max = f.shape[0] - 1
    cdef int j_max = f.shape[1]
    cdef int k_max = i_max + 1

    cdef vector[double] variogram = vector[double](k_max, 0.0)
    cdef vector[long] counts = vector[long](k_max, 0)
    cdef int i, j, k

    with nogil, parallel():
        for i in range(i_max):
            for j in range(j_max):
                for k in prange(1, k_max-i):
                    if not mask[i,j] and not mask[i+k,j]:
                        counts[k] += 1
                        variogram[k] += estimator_func(f[i,j] - f[i+k,j])

    normalization_func(variogram, counts)
    return np.asarray(variogram)
