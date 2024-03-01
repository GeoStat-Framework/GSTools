# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
This is a summator for the kriging routines
"""

import numpy as np
from cython.parallel import prange

IF OPENMP:
    cimport openmp

cimport numpy as np


def set_num_threads(num_threads):
    cdef int num_threads_c = 1
    if num_threads is None:
        # OPENMP set during setup
        IF OPENMP:
            num_threads_c = openmp.omp_get_num_procs()
        ELSE:
            ...
    else:
        num_threads_c = num_threads
    return num_threads_c


def calc_field_krige_and_variance(
    const double[:, :] krig_mat,
    const double[:, :] krig_vecs,
    const double[:] cond,
    num_threads=None,
):

    cdef int mat_i = krig_mat.shape[0]
    cdef int res_i = krig_vecs.shape[1]

    cdef double[:] field = np.zeros(res_i)
    cdef double[:] error = np.zeros(res_i)
    cdef double krig_fac

    cdef int i, j, k

    cdef int num_threads_c = set_num_threads(num_threads)

    # error = krig_vecs * krig_mat * krig_vecs
    # field = cond * krig_mat * krig_vecs
    for k in prange(res_i, nogil=True, num_threads=num_threads_c):
        for i in range(mat_i):
            krig_fac = 0.0
            for j in range(mat_i):
                krig_fac += krig_mat[i, j] * krig_vecs[j, k]
            error[k] += krig_vecs[i, k] * krig_fac
            field[k] += cond[i] * krig_fac

    return np.asarray(field), np.asarray(error)


def calc_field_krige(
    const double[:, :] krig_mat,
    const double[:, :] krig_vecs,
    const double[:] cond,
    const int num_threads=1,
):

    cdef int mat_i = krig_mat.shape[0]
    cdef int res_i = krig_vecs.shape[1]

    cdef double[:] field = np.zeros(res_i)
    cdef double krig_fac

    cdef int i, j, k

    cdef int num_threads_c = set_num_threads(num_threads)

    # field = cond * krig_mat * krig_vecs
    for k in prange(res_i, nogil=True, num_threads=num_threads_c):
        for i in range(mat_i):
            krig_fac = 0.0
            for j in range(mat_i):
                krig_fac += krig_mat[i, j] * krig_vecs[j, k]
            field[k] += cond[i] * krig_fac

    return np.asarray(field)
