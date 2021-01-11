# -*- coding: utf-8 -*-
"""
This is the unittest of the Normalizer class.
"""

import unittest
import numpy as np
import gstools as gs


def _rel_err(a, b):
    return np.abs(a / ((a + b) / 2) - 1)


class TestNormalizer(unittest.TestCase):
    def setUp(self):
        self.seed = 20210111
        self.rng = gs.random.RNG(self.seed)
        self.samp = self.rng.random.normal(11.1, 2.25, 1000)
        self.lmb = 2.5

    def test_fitting(self):
        # boxcox with given data to init
        bc_samples = gs.normalize.BoxCox(lmbda=self.lmb).denormalize(self.samp)
        bc_norm = gs.normalize.BoxCox(data=bc_samples)
        self.assertLess(_rel_err(self.lmb, bc_norm.lmbda), 1e-2)
        self.assertAlmostEqual(
            bc_norm.likelihood(bc_samples),
            np.exp(bc_norm.loglikelihood(bc_samples)),
        )
        # yeo-johnson with calling fit
        yj_norm = gs.normalize.YeoJohnson(lmbda=self.lmb)
        yj_samples = yj_norm.denormalize(self.samp)
        yj_norm.fit(yj_samples)
        self.assertLess(_rel_err(self.lmb, yj_norm.lmbda), 1e-2)
        self.assertAlmostEqual(
            yj_norm.likelihood(yj_samples),
            np.exp(yj_norm.loglikelihood(yj_samples)),
        )
        # modulus with calling fit
        mo_norm = gs.normalize.Modulus(lmbda=self.lmb)
        mo_samples = mo_norm.denormalize(self.samp)
        mo_norm.fit(mo_samples)
        self.assertLess(_rel_err(self.lmb, mo_norm.lmbda), 1e-2)
        self.assertAlmostEqual(
            mo_norm.likelihood(mo_samples),
            np.exp(mo_norm.loglikelihood(mo_samples)),
        )
        # manly with calling fit
        ma_norm = gs.normalize.Manly(lmbda=self.lmb)
        ma_samples = ma_norm.denormalize(self.samp)
        ma_norm.fit(ma_samples)
        self.assertLess(_rel_err(self.lmb, ma_norm.lmbda), 1e-2)
        # self.assertAlmostEqual(
        #     ma_norm.likelihood(ma_samples),
        #     np.exp(ma_norm.loglikelihood(ma_samples)),
        # )  # this is comparing infs


if __name__ == "__main__":
    unittest.main()
