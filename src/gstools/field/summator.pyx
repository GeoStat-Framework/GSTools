#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# -*- coding: utf-8 -*-
"""
This is the randomization method summator, implemented in cython.
"""

import numpy as np

cimport cython

from cython.parallel import prange

cimport numpy as np
from libc.math cimport cos, sin


def summate(
    const double[:, :] cov_samples,
    const double[:] z_1,
    const double[:] z_2,
    const double[:, :] pos
    ):
    cdef int i, j, d
    cdef double phase
    cdef int dim = pos.shape[0]

    cdef int X_len = pos.shape[1]
    cdef int N = cov_samples.shape[1]

    cdef double[:] summed_modes = np.zeros(X_len, dtype=float)

    for i in prange(X_len, nogil=True):
        for j in range(N):
            phase = 0.
            for d in range(dim):
                phase += cov_samples[d, j] * pos[d, i]
            summed_modes[i] += z_1[j] * cos(phase) + z_2[j] * sin(phase)

    return np.asarray(summed_modes)


cdef (double) abs_square(const double[:] vec) nogil:
    cdef int i
    cdef double r = 0.

    for i in range(vec.shape[0]):
        r += vec[i]**2

    return r


def summate_incompr(
    const int vec_dim,
    const double[:, :] cov_samples,
    const double[:] z_1,
    const double[:] z_2,
    const double[:, :] pos
    ):
    cdef int i, j, d
    cdef double phase
    cdef double k_2
    cdef int field_dim = pos.shape[0]

    cdef double[:] e1 = np.zeros(vec_dim, dtype=float)
    e1[0] = 1.
    cdef double[:] proj = np.empty(vec_dim)

    cdef int X_len = pos.shape[1]
    cdef int N = cov_samples.shape[1]

    cdef double[:, :] summed_modes = np.zeros((vec_dim, X_len), dtype=float)

    for i in range(X_len):
        for j in range(N):
            k_2 = abs_square(cov_samples[:vec_dim, j])
            phase = 0.
            for d in range(field_dim):
                phase += cov_samples[d, j] * pos[d, i]
            for d in range(vec_dim):
                proj[d] = e1[d] - cov_samples[d, j] * cov_samples[0, j] / k_2
                summed_modes[d, i] += proj[d] * (z_1[j] * cos(phase) + z_2[j] * sin(phase))

    return np.asarray(summed_modes)


# function makes incompressible random vector field with zero velocity, unlike summate_incompr which makes the x-axis a preferential direction
# z_1 has shape (vec_dim,) such that z_1[i] for i =0,...,vec_dim-1 is a 1D array of length N = cov_samples.shape[1]
def summate_incompr_zero_vel(
    const int vec_dim,
    const double[:, :] cov_samples,
    const double[:,:] z_1,
    const double[:,:] z_2,
    const double[:, :] pos
    ):
    cdef int i, j, d
    cdef double phase
    cdef int field_dim = pos.shape[0]

    cdef int X_len = pos.shape[1]
    cdef int N = cov_samples.shape[1]

    cdef double[:, :] summed_modes = np.zeros((vec_dim, X_len), dtype=float)
    cdef double[:] w_1 = np.empty(vec_dim, dtype=float) # !!! Joshua hopefully initialized w, v correctly, this is just for memory allocation right?
    cdef double[:] w_2 = np.empty(vec_dim, dtype=float)

    for i in range(X_len):
        for j in range(N):
            # compute cross-product of wavevectors with random vectors z_1,z_2 of size vec_dim drawn from (vec_dim)-dimensional Gaussian with unit covariance matrix
            
            # !!! using the following two lines is cleaner, but code takes forever (type instability?) WHY?????
            
            #w_1 = np.cross(z_1[j,:], cov_samples[:vec_dim, j]) # 1D array of size vec_dim
            #w_2 = np.cross(z_2[j,:], cov_samples[:vec_dim, j])
            
            # Pending trying to get above two lines to work, I am hard coding the cross-product for vec_dim =3
            # !!! WARNING: this only works for vec_dim = 3 
            w_1[0] = z_1[j,1] * cov_samples[2, j] - z_1[j,2] * cov_samples[1, j]
            w_1[1] = z_1[j,2] * cov_samples[0, j] - z_1[j,0] * cov_samples[2, j]
            w_1[2] = z_1[j,0] * cov_samples[1, j] - z_1[j,1] * cov_samples[0, j]
            
            w_2[0] = z_2[j,1] * cov_samples[2, j] - z_2[j,2] * cov_samples[1, j]
            w_2[1] = z_2[j,2] * cov_samples[0, j] - z_2[j,0] * cov_samples[2, j]
            w_2[2] = z_2[j,0] * cov_samples[1, j] - z_2[j,1] * cov_samples[0, j]

            phase = 0.
            for d in range(field_dim):
                phase += cov_samples[d, j] * pos[d, i]
            for d in range(vec_dim):
                summed_modes[d, i] +=  w_1[d] * cos(phase) + w_2[d] * sin(phase) #1 * cos(phase) + 1 * sin(phase)

    return np.asarray(summed_modes)

def summate_generic_vector_field(
    const int vec_dim,
    const double[:, :] cov_samples,
    const double[:] z_1,
    const double[:] z_2,
    const double[:, :] pos
    ):
    cdef int i, j, d
    cdef double phase
    cdef double k_2
    cdef int field_dim = pos.shape[0]

    cdef double[:] e1 = np.zeros(vec_dim, dtype=float)
    e1[0] = 1.
    cdef double[:] proj = np.empty(vec_dim)

    cdef int X_len = pos.shape[1]
    cdef int N = cov_samples.shape[1]

    cdef double[:, :] summed_modes = np.zeros((vec_dim, X_len), dtype=float)

    for i in range(X_len):
        for j in range(N):
            k_2 = abs_square(cov_samples[:vec_dim, j])
            phase = 0.
            for d in range(field_dim):
                phase += cov_samples[d, j] * pos[d, i]
            for d in range(vec_dim):
                #proj[d] = e1[d] - cov_samples[d, j] * cov_samples[0, j] / k_2 #!!! don't want incompressibility projector  here
                summed_modes[d, i] += (z_1[j] * cos(phase) + z_2[j] * sin(phase))

    return np.asarray(summed_modes)
