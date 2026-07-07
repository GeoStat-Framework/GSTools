"""
This is the unittest of the correlogram module.
"""

import unittest

import numpy as np

import gstools as gs


class TestCorrelogram(unittest.TestCase):
    def setUp(self):
        self.model = gs.Gaussian(dim=1, var=0.5, len_scale=2.0)
        self.cross_corr = 0.8
        self.secondary_var = 1.5
        self.primary_mean = 1.0
        self.secondary_mean = 0.5

    def test_markov_model1_covariances(self):
        """Test MM1 covariance computation at zero lag."""
        mm1 = gs.MarkovModel1(
            primary_model=self.model,
            cross_corr=self.cross_corr,
            secondary_var=self.secondary_var,
            primary_mean=self.primary_mean,
            secondary_mean=self.secondary_mean,
        )

        C_Z0, C_Y0, C_YZ0 = mm1.compute_covariances()

        # Check primary variance
        self.assertAlmostEqual(C_Z0, self.model.sill)
        # Check secondary variance
        self.assertAlmostEqual(C_Y0, self.secondary_var)
        # Check cross-covariance formula
        expected_C_YZ0 = self.cross_corr * np.sqrt(C_Z0 * C_Y0)
        self.assertAlmostEqual(C_YZ0, expected_C_YZ0)

    def test_markov_model1_cross_covariance(self):
        """Test MM1 cross-covariance formula at distance h."""
        mm1 = gs.MarkovModel1(
            primary_model=self.model,
            cross_corr=self.cross_corr,
            secondary_var=self.secondary_var,
        )

        # Test at zero lag
        C_YZ_0 = mm1.cross_covariance(0.0)
        _, _, C_YZ0_expected = mm1.compute_covariances()
        self.assertAlmostEqual(C_YZ_0, C_YZ0_expected)

        # Test MM1 formula: C_YZ(h) = (C_YZ(0) / C_Z(0)) * C_Z(h)
        h = 1.0
        C_YZ_h = mm1.cross_covariance(h)
        C_Z0, _, C_YZ0 = mm1.compute_covariances()
        C_Z_h = self.model.covariance(h)
        expected = (C_YZ0 / C_Z0) * C_Z_h
        self.assertAlmostEqual(C_YZ_h, expected)

    def test_markov_model1_cross_covariance_array(self):
        """Test MM1 cross-covariance with array input."""
        mm1 = gs.MarkovModel1(
            primary_model=self.model,
            cross_corr=self.cross_corr,
            secondary_var=self.secondary_var,
        )

        h_array = np.array([0.0, 0.5, 1.0, 2.0])
        C_YZ_array = mm1.cross_covariance(h_array)

        # Check array shape
        self.assertEqual(C_YZ_array.shape, h_array.shape)

        # Verify each element matches scalar computation
        for i, h in enumerate(h_array):
            C_YZ_single = mm1.cross_covariance(h)
            self.assertAlmostEqual(C_YZ_array[i], C_YZ_single)

    def test_validation_cross_corr(self):
        """Test parameter validation for cross_corr."""
        # cross_corr too large
        with self.assertRaises(ValueError):
            gs.MarkovModel1(
                primary_model=self.model,
                cross_corr=1.5,
                secondary_var=self.secondary_var,
            )
        # cross_corr too small
        with self.assertRaises(ValueError):
            gs.MarkovModel1(
                primary_model=self.model,
                cross_corr=-1.5,
                secondary_var=self.secondary_var,
            )

    def test_validation_secondary_var(self):
        """Test parameter validation for secondary_var."""
        # negative variance
        with self.assertRaises(ValueError):
            gs.MarkovModel1(
                primary_model=self.model,
                cross_corr=self.cross_corr,
                secondary_var=-1.0,
            )
        # zero variance
        with self.assertRaises(ValueError):
            gs.MarkovModel1(
                primary_model=self.model,
                cross_corr=self.cross_corr,
                secondary_var=0.0,
            )


if __name__ == "__main__":
    unittest.main()
