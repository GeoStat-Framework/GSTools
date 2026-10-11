"""
This is the unittest of the cokriging module.
"""

import unittest

import numpy as np

import gstools as gs


class TestCokriging(unittest.TestCase):
    def setUp(self):
        # Simple 1D test case
        self.model = gs.Gaussian(dim=1, var=2, len_scale=2)
        self.cond_pos = ([0.3, 1.9, 1.1, 3.3, 4.7],)
        self.cond_val = np.array([0.47, 0.56, 0.74, 1.47, 1.74])
        self.sec_cond_val = np.array([1.8, 1.2, 2.1, 2.9, 2.4])
        self.pos = np.linspace(0, 5, 51)
        # Dummy secondary data
        self.sec_data = np.random.RandomState(42).rand(len(self.pos))

    def test_secondary_data_required(self):
        """Test that secondary_data is required on call."""
        correlogram = gs.MarkovModel1(
            self.model, cross_corr=0.5, secondary_var=1.0
        )
        scck = gs.cokriging.SimpleCollocated(
            correlogram, self.cond_pos, self.cond_val
        )
        with self.assertRaises(ValueError):
            scck(self.pos)

    def test_correlogram_type_required(self):
        """Test that first argument must be a Correlogram."""
        with self.assertRaises(TypeError):
            gs.cokriging.SimpleCollocated(
                self.model, self.cond_pos, self.cond_val
            )

    def test_icck_secondary_cond_required(self):
        """Test ICCK requires secondary conditioning data."""
        correlogram = gs.MarkovModel1(
            self.model, cross_corr=0.5, secondary_var=1.0
        )
        with self.assertRaises(ValueError):
            gs.cokriging.IntrinsicCollocated(
                correlogram,
                self.cond_pos,
                self.cond_val,
                secondary_cond_pos=None,
                secondary_cond_val=None,
            )

    def test_icck_secondary_cond_length(self):
        """Test ICCK secondary conditioning data length validation."""
        correlogram = gs.MarkovModel1(
            self.model, cross_corr=0.5, secondary_var=1.0
        )
        with self.assertRaises(ValueError):
            gs.cokriging.IntrinsicCollocated(
                correlogram,
                self.cond_pos,
                self.cond_val,
                self.cond_pos,
                self.sec_cond_val[:3],  # Wrong length
            )

    def test_zero_correlation_equals_sk(self):
        """Test that ρ=0 gives Simple Kriging results."""
        # Reference: Simple Kriging
        sk = gs.krige.Simple(
            self.model, self.cond_pos, self.cond_val, mean=0.0
        )
        sk_field, sk_var = sk(self.pos, return_var=True)

        # SCCK with ρ=0
        correlogram_scck = gs.MarkovModel1(
            self.model, cross_corr=0.0, secondary_var=1.5
        )
        scck = gs.cokriging.SimpleCollocated(
            correlogram_scck, self.cond_pos, self.cond_val
        )
        scck_field, scck_var = scck(
            self.pos, secondary_data=self.sec_data, return_var=True
        )
        np.testing.assert_allclose(scck_field, sk_field, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(scck_var, sk_var, rtol=1e-6, atol=1e-9)

        # ICCK with ρ=0
        correlogram_icck = gs.MarkovModel1(
            self.model, cross_corr=0.0, secondary_var=1.5
        )
        icck = gs.cokriging.IntrinsicCollocated(
            correlogram_icck,
            self.cond_pos,
            self.cond_val,
            self.cond_pos,
            self.sec_cond_val,
        )
        icck_field, icck_var = icck(
            self.pos, secondary_data=self.sec_data, return_var=True
        )
        np.testing.assert_allclose(icck_field, sk_field, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(icck_var, sk_var, rtol=1e-6, atol=1e-9)

    def test_scck_variance_formula(self):
        """Test SCCK variance formula."""
        cross_corr = 0.7
        secondary_var = 1.5

        # Get SK variance
        sk = gs.krige.Simple(
            self.model, self.cond_pos, self.cond_val, mean=0.0
        )
        _, sk_var = sk(self.pos, return_var=True)

        # Calculate expected SCCK variance
        C_Z0 = self.model.sill
        C_Y0 = secondary_var
        C_YZ0 = cross_corr * np.sqrt(C_Z0 * C_Y0)
        k = C_YZ0 / C_Z0

        numerator = k * sk_var
        denominator = C_Y0 - (k**2) * (C_Z0 - sk_var)
        lambda_Y0 = np.where(
            np.abs(denominator) < 1e-15, 0.0, numerator / denominator
        )
        expected_var = sk_var * (1.0 - lambda_Y0 * k)
        expected_var = np.maximum(0.0, expected_var)

        # Actual SCCK variance
        correlogram = gs.MarkovModel1(
            self.model, cross_corr=cross_corr, secondary_var=secondary_var
        )
        scck = gs.cokriging.SimpleCollocated(
            correlogram, self.cond_pos, self.cond_val
        )
        _, actual_var = scck(
            self.pos, secondary_data=self.sec_data, return_var=True
        )
        np.testing.assert_allclose(
            actual_var, expected_var, rtol=1e-6, atol=1e-9
        )

    def test_icck_variance_formula(self):
        """Test ICCK variance formula."""
        cross_corr = 0.7
        secondary_var = 1.5

        # Get SK variance
        sk = gs.krige.Simple(
            self.model, self.cond_pos, self.cond_val, mean=0.0
        )
        _, sk_var = sk(self.pos, return_var=True)

        # Expected ICCK variance
        C_Z0 = self.model.sill
        C_Y0 = secondary_var
        C_YZ0 = cross_corr * np.sqrt(C_Z0 * C_Y0)
        rho_squared = (C_YZ0**2) / (C_Y0 * C_Z0)
        expected_var = (1.0 - rho_squared) * sk_var

        # Actual ICCK variance
        correlogram = gs.MarkovModel1(
            self.model, cross_corr=cross_corr, secondary_var=secondary_var
        )
        icck = gs.cokriging.IntrinsicCollocated(
            correlogram,
            self.cond_pos,
            self.cond_val,
            self.cond_pos,
            self.sec_cond_val,
        )
        _, actual_var = icck(
            self.pos, secondary_data=self.sec_data, return_var=True
        )
        np.testing.assert_allclose(
            actual_var, expected_var, rtol=1e-6, atol=1e-9
        )

    def test_perfect_correlation_variance(self):
        """Test that ρ=±1 gives near-zero variance for ICCK."""
        for rho in [-1.0, 1.0]:
            correlogram = gs.MarkovModel1(
                self.model, cross_corr=rho, secondary_var=1.5
            )
            icck = gs.cokriging.IntrinsicCollocated(
                correlogram,
                self.cond_pos,
                self.cond_val,
                self.cond_pos,
                self.sec_cond_val,
            )
            _, icck_var = icck(
                self.pos, secondary_data=self.sec_data, return_var=True
            )
            self.assertTrue(np.allclose(icck_var, 0.0, atol=1e-12))

    def test_variance_reduction(self):
        """Test that cokriging reduces variance compared to simple kriging."""
        cross_corr = 0.8
        secondary_var = 1.5

        # Get SK variance
        sk = gs.krige.Simple(
            self.model, self.cond_pos, self.cond_val, mean=0.0
        )
        _, sk_var = sk(self.pos, return_var=True)

        # Get ICCK variance
        correlogram = gs.MarkovModel1(
            self.model, cross_corr=cross_corr, secondary_var=secondary_var
        )
        icck = gs.cokriging.IntrinsicCollocated(
            correlogram,
            self.cond_pos,
            self.cond_val,
            self.cond_pos,
            self.sec_cond_val,
        )
        _, icck_var = icck(
            self.pos, secondary_data=self.sec_data, return_var=True
        )

        # ICCK variance ≤ SK variance
        self.assertTrue(np.all(icck_var <= sk_var + 1e-8))
        self.assertTrue(np.mean(icck_var) < np.mean(sk_var))

    def test_scck_field_matches_full_system(self):
        """SCCK reuse-field must equal a from-scratch (n+1) cokriging solve."""
        rho, sec_var, mZ, mY = 0.7, 1.5, 0.3, -0.2
        corr = gs.MarkovModel1(
            self.model, cross_corr=rho, secondary_var=sec_var,
            primary_mean=mZ, secondary_mean=mY,
        )
        scck = gs.cokriging.SimpleCollocated(corr, self.cond_pos, self.cond_val)
        field = scck(self.pos, secondary_data=self.sec_data, return_var=False)

        C_Z0 = self.model.sill
        C_YZ0 = rho * np.sqrt(C_Z0 * sec_var)
        k = C_YZ0 / C_Z0
        cov = self.model.covariance
        P = np.asarray(self.cond_pos[0])
        n = len(P)
        Czz = cov(np.abs(P[:, None] - P[None, :]))
        ref = np.empty_like(self.pos, dtype=float)
        for j, u0 in enumerate(np.atleast_1d(self.pos)):
            cz0 = cov(np.abs(P - u0))
            A = np.zeros((n + 1, n + 1))
            b = np.zeros(n + 1)
            A[:n, :n] = Czz
            A[:n, n] = k * cz0
            A[n, :n] = k * cz0
            A[n, n] = sec_var
            b[:n] = cz0
            b[n] = C_YZ0
            w = np.linalg.solve(A, b)
            ref[j] = (
                mZ + w[:n] @ (self.cond_val - mZ)
                + w[n] * (self.sec_data[j] - mY)
            )
        np.testing.assert_allclose(field, ref, rtol=1e-6, atol=1e-9)

    def test_icck_field_matches_full_system(self):
        """ICCK reuse-field must equal a from-scratch (2n+1) cokriging solve.

        Uses eval points that don't coincide with conditioning locations to
        keep the (2n+1) system full-rank (it is singular at exact conditioning
        locations under Markov Model 1).
        """
        rho, sec_var, mZ, mY = 0.7, 1.5, 0.3, -0.2
        corr = gs.MarkovModel1(
            self.model, cross_corr=rho, secondary_var=sec_var,
            primary_mean=mZ, secondary_mean=mY,
        )
        icck = gs.cokriging.IntrinsicCollocated(
            corr, self.cond_pos, self.cond_val,
            self.cond_pos, self.sec_cond_val,
        )
        # Use points that don't coincide with cond_pos (0.3,1.1,1.9,3.3,4.7)
        eval_pos = ([np.array([0.75, 1.5, 2.5, 4.0])],)
        sec_eval = np.array([0.4, 0.6, 0.3, 0.8])
        field = icck(eval_pos[0], secondary_data=sec_eval, return_var=False)

        C_Z0 = self.model.sill
        C_YZ0 = rho * np.sqrt(C_Z0 * sec_var)
        C_Y0 = sec_var
        k = C_YZ0 / C_Z0
        cov = self.model.covariance
        P = np.asarray(self.cond_pos[0])
        n = len(P)
        Dzz = cov(np.abs(P[:, None] - P[None, :]))
        Dyz = k * Dzz
        Dyy = (C_Y0 / C_Z0) * Dzz
        ref = np.empty(len(eval_pos[0][0]), dtype=float)
        for j, u0 in enumerate(eval_pos[0][0]):
            cz0 = cov(np.abs(P - u0))
            cyz0 = k * cz0
            cy0 = (C_Y0 / C_Z0) * cz0
            A = np.zeros((2 * n + 1, 2 * n + 1))
            b = np.zeros(2 * n + 1)
            A[:n, :n] = Dzz
            A[:n, n:2 * n] = Dyz
            A[:n, 2 * n] = cyz0
            A[n:2 * n, :n] = Dyz
            A[n:2 * n, n:2 * n] = Dyy
            A[n:2 * n, 2 * n] = cy0
            A[2 * n, :n] = cyz0
            A[2 * n, n:2 * n] = cy0
            A[2 * n, 2 * n] = C_Y0
            b[:n] = cz0
            b[n:2 * n] = cyz0
            b[2 * n] = C_YZ0
            w = np.linalg.solve(A, b)
            ref[j] = (
                mZ + w[:n] @ (self.cond_val - mZ)
                + w[n:2 * n] @ (self.sec_cond_val - mY)
                + w[2 * n] * (sec_eval[j] - mY)
            )
        np.testing.assert_allclose(field, ref, rtol=1e-6, atol=1e-9)


    def test_icck_nan_in_cond_val(self):
        """A NaN primary value must be ignored, not crash ICCK."""
        cond_val = np.array([0.47, np.nan, 0.74, 1.47, 1.74])
        corr = gs.MarkovModel1(self.model, cross_corr=0.7, secondary_var=1.5)
        icck = gs.cokriging.IntrinsicCollocated(
            corr, self.cond_pos, cond_val,
            self.cond_pos, self.sec_cond_val,
        )
        field = icck(self.pos, secondary_data=self.sec_data, return_var=False)
        self.assertEqual(field.shape, self.pos.shape)
        self.assertTrue(np.all(np.isfinite(field)))

    def test_icck_non_collocated_secondary_rejected(self):
        """Secondary positions that differ from cond_pos must raise."""
        corr = gs.MarkovModel1(self.model, cross_corr=0.7, secondary_var=1.5)
        bogus = ([99.0, 98.0, 97.0, 96.0, 95.0],)
        with self.assertRaises(ValueError):
            gs.cokriging.IntrinsicCollocated(
                corr, self.cond_pos, self.cond_val,
                bogus, self.sec_cond_val,
            )

    def test_field_stored_is_cokriging_result(self):
        """self.field must hold the cokriging result, not the SK field."""
        corr = gs.MarkovModel1(self.model, cross_corr=0.7, secondary_var=1.5)
        scck = gs.cokriging.SimpleCollocated(corr, self.cond_pos, self.cond_val)
        field = scck(self.pos, secondary_data=self.sec_data, return_var=False)
        np.testing.assert_allclose(scck.field, field, rtol=1e-12, atol=1e-12)

    def test_only_mean_not_supported(self):
        """only_mean=True must raise a clear error, not corrupt or crash."""
        corr = gs.MarkovModel1(self.model, cross_corr=0.7, secondary_var=1.5)
        scck = gs.cokriging.SimpleCollocated(corr, self.cond_pos, self.cond_val)
        with self.assertRaises(NotImplementedError):
            scck(self.pos, secondary_data=self.sec_data, only_mean=True)

    def test_secondary_data_wrong_length(self):
        """A wrong-length secondary array must give a clear ValueError."""
        corr = gs.MarkovModel1(self.model, cross_corr=0.7, secondary_var=1.5)
        scck = gs.cokriging.SimpleCollocated(corr, self.cond_pos, self.cond_val)
        with self.assertRaisesRegex(ValueError, "secondary_data"):
            scck(self.pos, secondary_data=np.ones(3))

    def test_secondary_data_structured_mesh(self):
        """Flat secondary data must work on a structured 2D mesh."""
        m2 = gs.Gaussian(dim=2, var=2.0, len_scale=2.0)
        c2 = gs.MarkovModel1(m2, cross_corr=0.7, secondary_var=1.5)
        cond_pos = ([0.5, 2.0, 4.0, 6.0, 8.0], [1.0, 3.0, 5.0, 2.0, 4.0])
        scck = gs.cokriging.SimpleCollocated(c2, cond_pos, self.cond_val)
        gx, gy = np.linspace(0, 10, 6), np.linspace(0, 10, 5)
        sec = np.ones(30)  # 6 * 5, flat
        field = scck(
            (gx, gy), secondary_data=sec,
            mesh_type="structured", return_var=False,
        )
        self.assertEqual(field.shape, (6, 5))

    def test_condsrf_gives_clear_error(self):
        """Wrapping cokriging in CondSRF must raise a clear, documented error."""
        corr = gs.MarkovModel1(self.model, cross_corr=0.7, secondary_var=1.5)
        scck = gs.cokriging.SimpleCollocated(corr, self.cond_pos, self.cond_val)
        cond_srf = gs.CondSRF(scck)
        with self.assertRaisesRegex(ValueError, "CondSRF"):
            cond_srf(self.pos)

    def test_zero_corr_with_normalizer_equals_sk(self):
        """rho=0 + nonlinear normalizer must reduce to normalizer-aware SK."""
        normalizer = gs.normalizer.LogNormal()
        corr = gs.MarkovModel1(self.model, cross_corr=0.0, secondary_var=1.5)
        scck = gs.cokriging.SimpleCollocated(
            corr, self.cond_pos, self.cond_val, normalizer=normalizer
        )
        ck_field = scck(self.pos, secondary_data=self.sec_data, return_var=False)

        sk = gs.krige.Simple(
            self.model, self.cond_pos, self.cond_val,
            mean=0.0, normalizer=gs.normalizer.LogNormal(),
        )
        sk_field = sk(self.pos, return_var=False)
        np.testing.assert_allclose(ck_field, sk_field, rtol=1e-6, atol=1e-9)


    def test_icck_summate_matches_oracle_with_chunks(self):
        """Rewritten _summate must match the full-system oracle, incl. chunking."""
        rho, sec_var, mZ, mY = 0.7, 1.5, 0.3, -0.2
        corr = gs.MarkovModel1(
            self.model, cross_corr=rho, secondary_var=sec_var,
            primary_mean=mZ, secondary_mean=mY,
        )
        icck = gs.cokriging.IntrinsicCollocated(
            corr, self.cond_pos, self.cond_val,
            self.cond_pos, self.sec_cond_val,
        )
        # force multiple chunks to exercise the per-chunk path
        field, var = icck(
            self.pos, secondary_data=self.sec_data,
            return_var=True, chunk_size=7,
        )
        self.assertEqual(field.shape, self.pos.shape)
        self.assertTrue(np.all(var >= 0.0))
        # independently re-derive reference using full (2n+1) system on non-coinciding points
        C_Z0 = self.model.sill
        C_YZ0 = rho * np.sqrt(C_Z0 * sec_var)
        k = C_YZ0 / C_Z0
        cov = self.model.covariance
        P = np.asarray(self.cond_pos[0])
        n = len(P)
        Dzz = cov(np.abs(P[:, None] - P[None, :]))
        # use only non-coinciding eval points for oracle comparison
        eval_pos = np.array([0.75, 1.5, 2.5, 4.0])
        sec_eval = np.array([0.4, 0.6, 0.3, 0.8])
        field_sub, _ = icck(
            ([eval_pos],), secondary_data=sec_eval,
            return_var=True, chunk_size=2,
        )
        ref = np.empty(len(eval_pos), dtype=float)
        for j, u0 in enumerate(eval_pos):
            cz0 = cov(np.abs(P - u0))
            A = np.zeros((2 * n + 1, 2 * n + 1))
            b = np.zeros(2 * n + 1)
            A[:n, :n] = Dzz
            A[:n, n:2 * n] = k * Dzz
            A[:n, 2 * n] = k * cz0
            A[n:2 * n, :n] = k * Dzz
            A[n:2 * n, n:2 * n] = (sec_var / C_Z0) * Dzz
            A[n:2 * n, 2 * n] = (sec_var / C_Z0) * cz0
            A[2 * n, :n] = k * cz0
            A[2 * n, n:2 * n] = (sec_var / C_Z0) * cz0
            A[2 * n, 2 * n] = sec_var
            b[:n] = cz0
            b[n:2 * n] = k * cz0
            b[2 * n] = C_YZ0
            w = np.linalg.solve(A, b)
            ref[j] = (
                mZ + w[:n] @ (self.cond_val - mZ)
                + w[n:2 * n] @ (self.sec_cond_val - mY)
                + w[2 * n] * (sec_eval[j] - mY)
            )
        np.testing.assert_allclose(field_sub, ref, rtol=1e-6, atol=1e-9)

    def test_top_level_exports(self):
        """Correlogram and CollocatedCokriging must be top-level exported."""
        self.assertTrue(hasattr(gs, "Correlogram"))
        self.assertTrue(hasattr(gs, "CollocatedCokriging"))
        self.assertIs(gs.Correlogram, gs.cokriging.Correlogram)
        self.assertIs(gs.CollocatedCokriging, gs.cokriging.CollocatedCokriging)


if __name__ == "__main__":
    unittest.main()
