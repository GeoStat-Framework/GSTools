#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# -*- coding: utf-8 -*-
"""
This is a summator for the kriging routines
"""

import numpy as np

cimport cython
from cython.parallel import prange
cimport numpy as np


def calc_field_krige_and_variance(
    const double[:,:] krig_mat,
    const double[:,:] krig_vecs,
    const double[:] cond
):

    cdef int mat_i = krig_mat.shape[0]
    cdef int res_i = krig_vecs.shape[1]

    cdef double[:] field = np.zeros(res_i)
    cdef double[:] error = np.zeros(res_i)
    cdef double krig_fac

    cdef int i, j, k

    # error = krig_vecs * krig_mat * krig_vecs
    # field = cond * krig_mat * krig_vecs
    for k in prange(res_i, nogil=True):
        for i in range(mat_i):
            krig_fac = 0.0
            for j in range(mat_i):
                krig_fac += krig_mat[i,j] * krig_vecs[j,k]
            error[k] += krig_vecs[i,k] * krig_fac
            field[k] += cond[i] * krig_fac

    return np.asarray(field), np.asarray(error)


def calc_field_krige(
    const double[:,:] krig_mat,
    const double[:,:] krig_vecs,
    const double[:] cond
):

    cdef int mat_i = krig_mat.shape[0]
    cdef int res_i = krig_vecs.shape[1]

    cdef double[:] field = np.zeros(res_i)
    cdef double krig_fac

    cdef int i, j, k

    # field = cond * krig_mat * krig_vecs
    for k in prange(res_i, nogil=True):
        for i in range(mat_i):
            krig_fac = 0.0
            for j in range(mat_i):
                krig_fac += krig_mat[i,j] * krig_vecs[j,k]
            field[k] += cond[i] * krig_fac

    return np.asarray(field)
