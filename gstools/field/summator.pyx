#!python
#cython: language_level=2
# -*- coding: utf-8 -*-
"""
This is the variogram estimater, implemented in cython.
"""
from __future__ import division, absolute_import, print_function

import numpy as np

cimport cython
from cython.parallel import prange
from libc.math cimport sin, cos
cimport numpy as np


DTYPE = np.double
ctypedef np.double_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def summate_unstruct(
    double[:,:] cov_samples,
    double[:] z_1,
    double[:] z_2,
    double[:,:] pos
    ):
    cdef int i, j, d, X_len, N
    cdef double phase
    cdef int dim
    dim = pos.shape[0]

    X_len = pos.shape[1]
    N = cov_samples.shape[1]

    cdef double[:] summed_modes = np.zeros(X_len, dtype=DTYPE)

    for i in prange(X_len, nogil=True):
        for j in range(N):
            phase = 0.
            for d in range(dim):
                phase += cov_samples[d,j] * pos[d,i]
            summed_modes[i] += z_1[j] * cos(phase) + z_2[j] * sin(phase)

    return np.asarray(summed_modes)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def summate_struct(
    double[:,:] cov_samples,
    double[:] z_1,
    double[:] z_2,
    double[:] x,
    double[:] y=None,
    double[:] z=None,
):
    if y == None and z == None:
        return summate_struct_1d(cov_samples, z_1, z_2, x)
    elif z == None:
        return summate_struct_2d(cov_samples, z_1, z_2, x, y)
    else:
        return summate_struct_3d(cov_samples, z_1, z_2, x, y, z)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def summate_struct_1d(
    double[:,:] cov_samples,
    double[:] z_1,
    double[:] z_2,
    double[:] x,
    ):

    cdef int i, j, X_len, N
    cdef double phase

    X_len = x.shape[0]
    N = cov_samples.shape[1]

    cdef double[:] summed_modes = np.zeros(X_len, dtype=DTYPE)

    for i in prange(X_len, nogil=True):
        for j in range(N):
            phase = cov_samples[0,j] * x[i]
            summed_modes[i] += z_1[j] * cos(phase) + z_2[j] * sin(phase)

    return np.asarray(summed_modes)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def summate_struct_2d(
    double[:,:] cov_samples,
    double[:] z_1,
    double[:] z_2,
    double[:] x,
    double[:] y,
    ):
    cdef int i, j, k, X_len, Y_len, N
    cdef double phase

    X_len = x.shape[0]
    Y_len = y.shape[0]
    N = cov_samples.shape[1]

    cdef double[:,:] summed_modes = np.zeros((X_len, Y_len), dtype=DTYPE)

    for i in prange(X_len, nogil=True):
        for j in range(Y_len):
            for k in range(N):
                phase = cov_samples[0,k] * x[i] + cov_samples[1,k] * y[j]
                summed_modes[i,j] += z_1[k] * cos(phase) + z_2[k] * sin(phase)

    return np.asarray(summed_modes)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def summate_struct_3d(
    double[:,:] cov_samples,
    double[:] z_1,
    double[:] z_2,
    double[:] x,
    double[:] y,
    double[:] z,
    ):
    cdef int i, j, k, l, X_len, Y_len, Z_len, N
    cdef double phase

    X_len = x.shape[0]
    Y_len = y.shape[0]
    Z_len = z.shape[0]
    N = cov_samples.shape[1]

    cdef double[:,:,:] summed_modes = np.zeros((X_len, Y_len, Z_len), dtype=DTYPE)

    for i in prange(X_len, nogil=True):
        for j in range(Y_len):
            for k in range(Z_len):
                for l in range(N):
                    phase = (
                        cov_samples[0,l] * x[i] +
                        cov_samples[1,l] * y[j] +
                        cov_samples[2,l] * z[k]
                    )
                    summed_modes[i,j,k] += z_1[l] * cos(phase) + z_2[l] * sin(phase)

    return np.asarray(summed_modes)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef (double) abs_square(double[:] vec) nogil:
    cdef int i
    cdef double r = 0.

    for i in range(vec.shape[0]):
        r += vec[i]**2

    return r

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def summate_incompr_unstruct(
    double[:,:] cov_samples,
    double[:] z_1,
    double[:] z_2,
    double[:,:] pos
    ):
    cdef int i, j, d, X_len, N
    cdef double phase
    cdef int dim
    cdef double k_2
    dim = pos.shape[0]

    cdef double[:] e1 = np.zeros(dim, dtype=DTYPE)
    e1[0] = 1.
    cdef double[:] proj = np.empty(dim, dtype=DTYPE)

    X_len = pos.shape[1]
    N = cov_samples.shape[1]

    cdef double[:,:] summed_modes = np.zeros((dim, X_len), dtype=DTYPE)

    for i in prange(X_len, nogil=True):
        for j in range(N):
            k_2 = abs_square(cov_samples[:,j])
            phase = 0.
            for d in range(dim):
                phase += cov_samples[d,j] * pos[d,i]
            for d in range(dim):
                proj[d] = e1[d] - cov_samples[d,j] * cov_samples[0,j] / k_2
                summed_modes[d,i] += proj[d] * (z_1[j] * cos(phase) + z_2[j] * sin(phase))

    return np.asarray(summed_modes)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def summate_incompr_struct(
    double[:,:] cov_samples,
    double[:] z_1,
    double[:] z_2,
    double[:] x,
    double[:] y=None,
    double[:] z=None,
):
    if z == None:
        return summate_incompr_struct_2d(cov_samples, z_1, z_2, x, y)
    else:
        return summate_incompr_struct_3d(cov_samples, z_1, z_2, x, y, z)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def summate_incompr_struct_2d(
    double[:,:] cov_samples,
    double[:] z_1,
    double[:] z_2,
    double[:] x,
    double[:] y,
    ):
    cdef int i, j, k, d, X_len, Y_len, N
    cdef double phase
    cdef int dim = 2
    cdef double k_2

    cdef double[:] e1 = np.zeros(dim, dtype=DTYPE)
    e1[0] = 1.
    cdef double[:] proj = np.empty(dim, dtype=DTYPE)

    X_len = x.shape[0]
    Y_len = y.shape[0]
    N = cov_samples.shape[1]

    cdef double[:,:,:] summed_modes = np.zeros((dim, X_len, Y_len), dtype=DTYPE)

    for i in prange(X_len, nogil=True):
        for j in range(Y_len):
            for k in range(N):
                k_2 = abs_square(cov_samples[:,k])
                phase = cov_samples[0,k] * x[i] + cov_samples[1,k] * y[j]
                for d in range(dim):
                    proj[d] = e1[d] - cov_samples[d,k] * cov_samples[0,k] / k_2
                    summed_modes[d,i,j] += proj[d] * (z_1[k] * cos(phase) + z_2[k] * sin(phase))

    return np.asarray(summed_modes)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def summate_incompr_struct_3d(
    double[:,:] cov_samples,
    double[:] z_1,
    double[:] z_2,
    double[:] x,
    double[:] y,
    double[:] z,
    ):
    cdef int i, j, k, l, d, X_len, Y_len, Z_len, N
    cdef double phase
    cdef int dim = 3
    cdef double k_2

    cdef double[:] e1 = np.zeros(dim, dtype=DTYPE)
    e1[0] = 1.
    cdef double[:] proj = np.empty(dim, dtype=DTYPE)

    X_len = x.shape[0]
    Y_len = y.shape[0]
    Z_len = z.shape[0]
    N = cov_samples.shape[1]

    cdef double[:,:,:,:] summed_modes = np.zeros((dim, X_len, Y_len, Z_len), dtype=DTYPE)

    for i in prange(X_len, nogil=True):
        for j in range(Y_len):
            for k in range(Z_len):
                for l in range(N):
                    k_2 = abs_square(cov_samples[:,l])
                    phase = (
                        cov_samples[0,l] * x[i] +
                        cov_samples[1,l] * y[j] +
                        cov_samples[2,l] * z[k]
                    )
                    for d in range(dim):
                        proj[d] = e1[d] - cov_samples[d,l] * cov_samples[0,l] / k_2
                        summed_modes[d,i,j,k] += proj[d] * (z_1[l] * cos(phase) + z_2[l] * sin(phase))

    return np.asarray(summed_modes)
