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
    cdef double krig_fac

    cdef int i, j, k

    # error = krig_vecs * krig_mat * krig_vecs
    # field = krig_facs * krig_vecs
    for k in prange(res_i, nogil=True):
        for i in range(mat_i):
            krig_fac = 0.0
            for j in range(mat_i):
                krig_fac += krig_mat[i,j] * krig_vecs[j,k]
            error[k] += krig_vecs[i,k] * krig_fac
            field[k] += cond[i] * krig_fac

    return np.asarray(field), np.asarray(error)
