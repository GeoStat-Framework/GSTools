# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: language = c++
"""
This is the variogram estimater, implemented in cython.
"""

import numpy as np
from cython.parallel import parallel, prange

IF OPENMP:
    cimport openmp

cimport numpy as np
from libc.math cimport M_PI, acos, atan2, cos, fabs, isnan, pow, sin, sqrt


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


cdef inline double dist_euclid(
    const int dim,
    const double[:, :] pos,
    const int i,
    const int j,
) nogil:
    cdef int d
    cdef double dist_squared = 0.0
    for d in range(dim):
        dist_squared += ((pos[d, i] - pos[d, j]) * (pos[d, i] - pos[d, j]))
    return sqrt(dist_squared)


cdef inline double dist_haversine(
    const int dim,
    const double[:, :] pos,
    const int i,
    const int j,
) nogil:
    # pos holds lat-lon in deg
    cdef double deg_2_rad = M_PI / 180.0
    cdef double diff_lat = (pos[0, j] - pos[0, i]) * deg_2_rad
    cdef double diff_lon = (pos[1, j] - pos[1, i]) * deg_2_rad
    cdef double arg = (
        pow(sin(diff_lat/2.0), 2) +
        cos(pos[0, i]*deg_2_rad) *
        cos(pos[0, j]*deg_2_rad) *
        pow(sin(diff_lon/2.0), 2)
    )
    return 2.0 * atan2(sqrt(arg), sqrt(1.0-arg))


ctypedef double (*_dist_func)(
    const int,
    const double[:, :],
    const int,
    const int,
) nogil


cdef inline bint dir_test(
    const int dim,
    const double[:, :] pos,
    const double dist,
    const double[:, :] direction,
    const double angles_tol,
    const double bandwidth,
    const int i,
    const int j,
    const int d,
) nogil:
    cdef double s_prod = 0.0  # scalar product
    cdef double b_dist = 0.0  # band-distance
    cdef double tmp  # temporary variable
    cdef int k
    cdef bint in_band = True
    cdef bint in_angle = True

    # scalar-product calculation for bandwidth projection and angle calculation
    for k in range(dim):
        s_prod += (pos[k, i] - pos[k, j]) * direction[d, k]

    # calculate band-distance by projection of point-pair-vec to direction line
    if bandwidth > 0.0:
        for k in range(dim):
            tmp = (pos[k, i] - pos[k, j]) - s_prod * direction[d, k]
            b_dist += tmp * tmp
        in_band = sqrt(b_dist) < bandwidth

    # allow repeating points (dist = 0)
    if dist > 0.0:
        # use smallest angle by taking absolute value for arccos angle formula
        tmp = fabs(s_prod) / dist
        if tmp < 1.0:  # else same direction (prevent numerical errors)
            in_angle = acos(tmp) < angles_tol

    return in_band and in_angle


cdef inline double estimator_matheron(const double f_diff) nogil:
    return f_diff * f_diff

cdef inline double estimator_cressie(const double f_diff) nogil:
    return sqrt(fabs(f_diff))

ctypedef double (*_estimator_func)(const double) nogil

cdef inline void normalization_matheron(
    double[:] variogram,
    np.int64_t[:] counts,
):
    cdef int i
    for i in range(variogram.shape[0]):
        # avoid division by zero
        variogram[i] /= (2. * max(counts[i], 1))

cdef inline void normalization_cressie(
    double[:] variogram,
    np.int64_t[:] counts,
):
    cdef int i
    cdef np.int64_t cnt
    for i in range(variogram.shape[0]):
        # avoid division by zero
        cnt = max(counts[i], 1)
        variogram[i] = (
            0.5 * (1./cnt * variogram[i])**4 /
            (0.457 + 0.494 / cnt + 0.045 / cnt**2)
        )

ctypedef void (*_normalization_func)(
    double[:],
    np.int64_t[:],
)

cdef inline void normalization_matheron_vec(
    double[:, :] variogram,
    np.int64_t[:, :] counts,
):
    cdef int d
    for d in range(variogram.shape[0]):
        normalization_matheron(variogram[d, :], counts[d, :])

cdef inline void normalization_cressie_vec(
    double[:, :] variogram,
    np.int64_t[:, :] counts,
):
    cdef int d
    for d in range(variogram.shape[0]):
        normalization_cressie(variogram[d, :], counts[d, :])

ctypedef void (*_normalization_func_vec)(
    double[:, :],
    np.int64_t[:, :],
)

cdef _estimator_func choose_estimator_func(str estimator_type):
    cdef _estimator_func estimator_func
    if estimator_type == 'm':
        estimator_func = estimator_matheron
    else:  # estimator_type == 'c'
        estimator_func = estimator_cressie
    return estimator_func

cdef _normalization_func choose_estimator_normalization(str estimator_type):
    cdef _normalization_func normalization_func
    if estimator_type == 'm':
        normalization_func = normalization_matheron
    else:  # estimator_type == 'c'
        normalization_func = normalization_cressie
    return normalization_func

cdef _normalization_func_vec choose_estimator_normalization_vec(str estimator_type):
    cdef _normalization_func_vec normalization_func_vec
    if estimator_type == 'm':
        normalization_func_vec = normalization_matheron_vec
    else:  # estimator_type == 'c'
        normalization_func_vec = normalization_cressie_vec
    return normalization_func_vec


def directional(
    const double[:, :] f,
    const double[:] bin_edges,
    const double[:, :] pos,
    const double[:, :] direction,  # should be normed
    const double angles_tol=M_PI/8.0,
    const double bandwidth=-1.0,  # negative values to turn of bandwidth search
    const bint separate_dirs=False,  # whether the direction bands don't overlap
    str estimator_type='m',
    num_threads=None,
):
    if pos.shape[1] != f.shape[1]:
        raise ValueError(f'len(pos) = {pos.shape[1]} != len(f) = {f.shape[1])}')

    if bin_edges.shape[0] < 2:
        raise ValueError('len(bin_edges) too small')

    if angles_tol <= 0:
        raise ValueError('tolerance for angle search masks must be > 0')

    cdef _estimator_func estimator_func = choose_estimator_func(estimator_type)
    cdef _normalization_func_vec normalization_func_vec = (
        choose_estimator_normalization_vec(estimator_type)
    )

    cdef int dim = pos.shape[0]
    cdef int d_max = direction.shape[0]
    cdef int i_max = bin_edges.shape[0] - 1
    cdef int j_max = pos.shape[1] - 1
    cdef int k_max = pos.shape[1]
    cdef int f_max = f.shape[0]

    cdef double[:, :] variogram = np.zeros((d_max, len(bin_edges)-1))
    cdef np.int64_t[:, :] counts = np.zeros((d_max, len(bin_edges)-1), dtype=np.int64)
    cdef int i, j, k, m, d
    cdef double dist

    cdef int num_threads_c = set_num_threads(num_threads)

    for i in prange(i_max, nogil=True, num_threads=num_threads_c):
        for j in range(j_max):
            for k in range(j+1, k_max):
                dist = dist_euclid(dim, pos, j, k)
                if dist < bin_edges[i] or dist >= bin_edges[i+1]:
                    continue  # skip if not in current bin
                for d in range(d_max):
                    if not dir_test(
                      dim, pos, dist, direction, angles_tol, bandwidth, k, j, d
                    ):
                        continue  # skip if not in current direction
                    for m in range(f_max):
                        # skip no data values
                        if not (isnan(f[m, k]) or isnan(f[m, j])):
                            counts[d, i] += 1
                            variogram[d, i] += estimator_func(f[m, k] - f[m, j])
                    # once we found a fitting direction
                    # break the search if directions are separated
                    if separate_dirs:
                        break

    normalization_func_vec(variogram, counts)
    return np.asarray(variogram), np.asarray(counts)


def unstructured(
    const double[:, :] f,
    const double[:] bin_edges,
    const double[:, :] pos,
    str estimator_type='m',
    str distance_type='e',
    num_threads=None,
):
    cdef int dim = pos.shape[0]
    cdef _dist_func distance

    if distance_type == 'e':
        distance = dist_euclid
    else:
        distance = dist_haversine
        if dim != 2:
            raise ValueError(f'Haversine: dim = {dim} != 2')

    if pos.shape[1] != f.shape[1]:
        raise ValueError(f'len(pos) = {pos.shape[1]} != len(f) = {f.shape[1])}')

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

    cdef double[:] variogram = np.zeros(len(bin_edges)-1)
    cdef np.int64_t[:] counts = np.zeros(len(bin_edges)-1, dtype=np.int64)
    cdef int i, j, k, m
    cdef double dist

    cdef int num_threads_c = set_num_threads(num_threads)

    for i in prange(i_max, nogil=True, num_threads=num_threads_c):
        for j in range(j_max):
            for k in range(j+1, k_max):
                dist = distance(dim, pos, j, k)
                if dist < bin_edges[i] or dist >= bin_edges[i+1]:
                    continue  # skip if not in current bin
                for m in range(f_max):
                    # skip no data values
                    if not (isnan(f[m, k]) or isnan(f[m, j])):
                        counts[i] += 1
                        variogram[i] += estimator_func(f[m, k] - f[m, j])

    normalization_func(variogram, counts)
    return np.asarray(variogram), np.asarray(counts)


def structured(
    const double[:, :] f,
    str estimator_type='m',
    num_threads=None,
):
    cdef _estimator_func estimator_func = choose_estimator_func(estimator_type)
    cdef _normalization_func normalization_func = (
        choose_estimator_normalization(estimator_type)
    )

    cdef int i_max = f.shape[0] - 1
    cdef int j_max = f.shape[1]
    cdef int k_max = i_max + 1

    cdef double[:] variogram = np.zeros(k_max)
    cdef np.int64_t[:] counts = np.zeros(k_max, dtype=np.int64)
    cdef int i, j, k

    cdef int num_threads_c = set_num_threads(num_threads)

    with nogil, parallel(num_threads=num_threads_c):
        for i in range(i_max):
            for j in range(j_max):
                for k in prange(1, k_max-i):
                    counts[k] += 1
                    variogram[k] += estimator_func(f[i, j] - f[i+k, j])

    normalization_func(variogram, counts)
    return np.asarray(variogram)


def ma_structured(
    const double[:, :] f,
    const bint[:, :] mask,
    str estimator_type='m',
    num_threads=None,
):
    cdef _estimator_func estimator_func = choose_estimator_func(estimator_type)
    cdef _normalization_func normalization_func = (
        choose_estimator_normalization(estimator_type)
    )

    cdef int i_max = f.shape[0] - 1
    cdef int j_max = f.shape[1]
    cdef int k_max = i_max + 1

    cdef double[:] variogram = np.zeros(k_max)
    cdef np.int64_t[:] counts = np.zeros(k_max, dtype=np.int64)
    cdef int i, j, k

    cdef int num_threads_c = set_num_threads(num_threads)

    with nogil, parallel(num_threads=num_threads_c):
        for i in range(i_max):
            for j in range(j_max):
                for k in prange(1, k_max-i):
                    if not mask[i, j] and not mask[i+k, j]:
                        counts[k] += 1
                        variogram[k] += estimator_func(f[i, j] - f[i+k, j])

    normalization_func(variogram, counts)
    return np.asarray(variogram)
