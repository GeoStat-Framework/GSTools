# cython: language_level=2
# -*- coding: utf-8 -*-
"""
This is the variogram estimater, implemented in cython.
"""
from __future__ import division, absolute_import, print_function

import numpy as np

cimport cython
from libc.math cimport sin, cos
cimport numpy as np


DTYPE = np.double
ctypedef np.double_t DTYPE_t


@cython.boundscheck(False)
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

    cdef double[:] summed_modes = np.zeros(X_len)

    for i in range(X_len):
        for j in range(N):
            phase = 0.
            for d in range(dim):
                phase += cov_samples[d,j] * pos[d,i]
            summed_modes[i] += z_1[j] * cos(phase) + z_2[j] * sin(phase)

    return np.asarray(summed_modes)

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

    cdef double[:] summed_modes = np.zeros(X_len)

    for i in range(X_len):
        for j in range(N):
            phase = cov_samples[0,j] * x[i]
            summed_modes[i] += z_1[j] * cos(phase) + z_2[j] * sin(phase)

    return np.asarray(summed_modes)

@cython.boundscheck(False)
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

    cdef double[:,:] summed_modes = np.zeros((X_len, Y_len))

    for i in range(X_len):
        for j in range(Y_len):
            for k in range(N):
                phase = cov_samples[0,k] * x[i] + cov_samples[1,k] * y[j]
                summed_modes[i,j] += z_1[k] * cos(phase) + z_2[k] * sin(phase)

    return np.asarray(summed_modes)

@cython.boundscheck(False)
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

    cdef double[:,:,:] summed_modes = np.zeros((X_len, Y_len, Z_len))

    for i in range(X_len):
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
