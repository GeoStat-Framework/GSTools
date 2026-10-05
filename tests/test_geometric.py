"""
This is the unittest of the geometric tools module.
"""

import unittest

import numpy as np

from gstools.tools.geometric import (
    matrix_anisometrize,
    matrix_anisotropify,
    matrix_derotate,
    matrix_detransform,
    matrix_isometrize,
    matrix_isotropify,
    matrix_rotate,
    matrix_scale,
    matrix_transform,
    no_of_angles,
    set_anis,
    set_scale,
)


# Frozen copies of the PRE-refactor implementations: the bit-identical
# regression reference (oracle a of the non-stationarity redesign spec).
def _ref_isotropify(dim, anis):
    anis = set_anis(dim, anis)
    return np.diag(np.concatenate(([1.0], 1.0 / anis)))


def _ref_anisotropify(dim, anis):
    anis = set_anis(dim, anis)
    return np.diag(np.concatenate(([1.0], anis)))


def _ref_isometrize(dim, angles, anis):
    return np.matmul(_ref_isotropify(dim, anis), matrix_derotate(dim, angles))


def _ref_anisometrize(dim, angles, anis):
    return np.matmul(matrix_rotate(dim, angles), _ref_anisotropify(dim, anis))


class TestSetScale(unittest.TestCase):
    def test_scalar_broadcasts_to_dim_vector(self):
        np.testing.assert_array_equal(set_scale(3, 2.0), [2.0, 2.0, 2.0])
        np.testing.assert_array_equal(set_scale(1, 0.5), [0.5])

    def test_exact_dim_vector_passes(self):
        np.testing.assert_array_equal(set_scale(2, [1.0, 0.5]), [1.0, 0.5])

    def test_short_vector_raises(self):
        # exact-match discipline: NO padding (unlike set_anis) — B5 history
        with self.assertRaises(ValueError):
            set_scale(3, [1.0, 2.0])

    def test_long_vector_raises(self):
        with self.assertRaises(ValueError):
            set_scale(2, [1.0, 2.0, 3.0])

    def test_2d_array_raises(self):
        with self.assertRaises(ValueError):
            set_scale(2, [[1.0, 2.0]])

    def test_nonpositive_raises(self):
        with self.assertRaises(ValueError):
            set_scale(2, 0.0)
        with self.assertRaises(ValueError):
            set_scale(2, [1.0, -0.5])


class TestTransformInverse(unittest.TestCase):
    def test_detransform_is_exact_inverse(self):
        rng = np.random.default_rng(42)
        for dim in (1, 2, 3, 4):
            for _ in range(20):
                angles = rng.uniform(-np.pi, np.pi, no_of_angles(dim))
                scale = rng.uniform(0.2, 5.0, dim)
                M = matrix_transform(dim, angles, scale)
                Minv = matrix_detransform(dim, angles, scale)
                np.testing.assert_allclose(Minv @ M, np.eye(dim), atol=1e-12)

    def test_uniform_dilation(self):
        # matrix_scale with a scalar is uniform dilation (M10 affinity r):
        # inexpressible in the axis-0-pinned anis parameterization.
        np.testing.assert_array_equal(matrix_scale(3, 2.0), 2.0 * np.eye(3))


class TestWrapperBitIdentical(unittest.TestCase):
    """Oracle (a): randomized angles/anis — new wrappers bit-identical to old."""

    def test_all_four_wrappers_bit_identical(self):
        rng = np.random.default_rng(0)
        for dim in (2, 3, 4):
            for _ in range(50):
                angles = rng.uniform(-np.pi, np.pi, no_of_angles(dim))
                anis = rng.uniform(0.1, 4.0, dim - 1)
                np.testing.assert_array_equal(
                    matrix_isotropify(dim, anis), _ref_isotropify(dim, anis)
                )
                np.testing.assert_array_equal(
                    matrix_anisotropify(dim, anis),
                    _ref_anisotropify(dim, anis),
                )
                np.testing.assert_array_equal(
                    matrix_isometrize(dim, angles, anis),
                    _ref_isometrize(dim, angles, anis),
                )
                np.testing.assert_array_equal(
                    matrix_anisometrize(dim, angles, anis),
                    _ref_anisometrize(dim, angles, anis),
                )

    def test_wrappers_still_pad_short_anis(self):
        # set_anis leniency is preserved for the CovModel-facing wrappers.
        np.testing.assert_array_equal(
            matrix_isotropify(3, 0.5), _ref_isotropify(3, 0.5)
        )


if __name__ == "__main__":
    unittest.main()
