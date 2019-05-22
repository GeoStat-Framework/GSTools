# cython: language_level=2
# -*- coding: utf-8 -*-
"""
This is a summator for the kriging routines
"""
from __future__ import division, absolute_import, print_function

import numpy as np

cimport cython
from cython.parallel import prange
cimport numpy as np


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def krigesum(double[:,:] krig_mat, double[:,:] krig_vecs, double[:] cond):

    cdef int mat_i = krig_mat.shape[0]
    cdef int res_i = krig_vecs.shape[1]

    cdef double[:] field = np.zeros(res_i)
    cdef double[:] error = np.zeros(res_i)
    cdef double[:] krig_facs = np.zeros(mat_i)

    cdef int i, j, k

    # krig_facs = cond * krig_mat
    for j in prange(mat_i, nogil=True):
        for i in range(mat_i):
            krig_facs[j] += cond[i] * krig_mat[i,j]

    # error = krig_vecs * krig_mat * krig_vecs
    # field = krig_facs * krig_vecs
    for k in prange(res_i, nogil=True):
        for i in range(mat_i):
            for j in range(mat_i):
                error[k] += krig_mat[i,j] * krig_vecs[i,k] * krig_vecs[j,k]
            field[k] += krig_facs[i] * krig_vecs[i,k]

    return np.asarray(field), np.asarray(error)
