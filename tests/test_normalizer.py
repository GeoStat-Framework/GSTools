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
        self.mean = 11.1
        self.std = 2.25
        self.smp = self.rng.random.normal(self.mean, self.std, 1000)
        self.lmb = 1.5

    def test_fitting(self):
        # boxcox with given data to init
        bc_samples = gs.normalizer.BoxCox(lmbda=self.lmb).denormalize(self.smp)
        bc_norm = gs.normalizer.BoxCox(data=bc_samples)
        self.assertLess(_rel_err(self.lmb, bc_norm.lmbda), 1e-2)
        self.assertAlmostEqual(
            bc_norm.likelihood(bc_samples),
            np.exp(bc_norm.loglikelihood(bc_samples)),
        )
        # yeo-johnson with calling fit
        yj_norm = gs.normalizer.YeoJohnson(lmbda=self.lmb)
        yj_samples = yj_norm.denormalize(self.smp)
        yj_norm.fit(yj_samples)
        self.assertLess(_rel_err(self.lmb, yj_norm.lmbda), 1e-2)
        self.assertAlmostEqual(
            yj_norm.likelihood(yj_samples),
            np.exp(yj_norm.loglikelihood(yj_samples)),
        )
        # modulus with calling fit
        mo_norm = gs.normalizer.Modulus(lmbda=self.lmb)
        mo_samples = mo_norm.denormalize(self.smp)
        mo_norm.fit(mo_samples)
        self.assertLess(_rel_err(self.lmb, mo_norm.lmbda), 1e-2)
        self.assertAlmostEqual(
            mo_norm.likelihood(mo_samples),
            np.exp(mo_norm.loglikelihood(mo_samples)),
        )
        # manly with calling fit
        ma_norm = gs.normalizer.Manly(lmbda=self.lmb)
        ma_samples = ma_norm.denormalize(self.smp)
        ma_norm.fit(ma_samples)
        self.assertLess(_rel_err(self.lmb, ma_norm.lmbda), 1e-2)
        # self.assertAlmostEqual(
        #     ma_norm.likelihood(ma_samples),
        #     np.exp(ma_norm.loglikelihood(ma_samples)),
        # )  # this is comparing infs

    def test_boxcox(self):
        # without shift
        bc = gs.normalizer.BoxCox(lmbda=0)
        self.assertTrue(
            np.all(
                np.isclose(self.smp, bc.normalize(bc.denormalize(self.smp)))
            )
        )
        bc.lmbda = self.lmb
        self.assertTrue(
            np.all(
                np.isclose(self.smp, bc.normalize(bc.denormalize(self.smp)))
            )
        )
        # with shift
        bc = gs.normalizer.BoxCoxShift(lmbda=0, shift=1.1)
        self.assertTrue(
            np.all(
                np.isclose(self.smp, bc.normalize(bc.denormalize(self.smp)))
            )
        )
        bc.lmbda = self.lmb
        self.assertTrue(
            np.all(
                np.isclose(self.smp, bc.normalize(bc.denormalize(self.smp)))
            )
        )

    def test_yeojohnson(self):
        yj = gs.normalizer.YeoJohnson(lmbda=0)
        self.assertTrue(
            np.all(
                np.isclose(
                    self.smp - self.mean,
                    yj.normalize(yj.denormalize(self.smp - self.mean)),
                )
            )
        )
        yj.lmbda = 2
        self.assertTrue(
            np.all(
                np.isclose(
                    self.smp - self.mean,
                    yj.normalize(yj.denormalize(self.smp - self.mean)),
                )
            )
        )
        # with shift
        yj.lmbda = self.lmb
        self.assertTrue(
            np.all(
                np.isclose(
                    self.smp - self.mean,
                    yj.normalize(yj.denormalize(self.smp - self.mean)),
                )
            )
        )

    def test_modulus(self):
        mo = gs.normalizer.Modulus(lmbda=0)
        self.assertTrue(
            np.all(
                np.isclose(self.smp, mo.normalize(mo.denormalize(self.smp)))
            )
        )
        mo.lmbda = self.lmb
        self.assertTrue(
            np.all(
                np.isclose(self.smp, mo.normalize(mo.denormalize(self.smp)))
            )
        )

    def test_manly(self):
        ma = gs.normalizer.Manly(lmbda=0)
        self.assertTrue(
            np.all(
                np.isclose(self.smp, ma.normalize(ma.denormalize(self.smp)))
            )
        )
        ma.lmbda = self.lmb
        self.assertTrue(
            np.all(
                np.isclose(self.smp, ma.normalize(ma.denormalize(self.smp)))
            )
        )

    def test_parameterless(self):
        no = gs.normalizer.LogNormal()
        self.assertTrue(
            np.all(
                np.isclose(self.smp, no.normalize(no.denormalize(self.smp)))
            )
        )
        no = gs.normalizer.Normalizer()
        self.assertTrue(
            np.all(
                np.isclose(self.smp, no.normalize(no.denormalize(self.smp)))
            )
        )

    def test_compare(self):
        norm1 = gs.normalizer.BoxCox()
        norm2 = gs.normalizer.BoxCox(lmbda=0.5)
        norm3 = gs.normalizer.YeoJohnson()
        norm4 = "this is not a normalizer"
        # check campare
        self.assertTrue(norm1 == norm1)
        self.assertTrue(norm1 != norm2)
        self.assertTrue(norm1 != norm3)
        self.assertTrue(norm1 != norm4)

    def test_check(self):
        self.assertRaises(ValueError, gs.field.Field, gs.Cubic(), normalizer=5)

    def test_auto_fit(self):
        x = y = range(60)
        pos = gs.generate_grid([x, y])
        model = gs.Gaussian(dim=2, var=1, len_scale=10)
        srf = gs.SRF(
            model, seed=20170519, normalizer=gs.normalizer.LogNormal()
        )
        srf(pos)
        ids = np.arange(srf.field.size)
        samples = np.random.RandomState(20210201).choice(
            ids, size=60, replace=False
        )
        # sample conditioning points from generated field
        cond_pos = pos[:, samples]
        cond_val = srf.field[samples]
        krige = gs.krige.Ordinary(
            model=gs.Stable(dim=2),
            cond_pos=cond_pos,
            cond_val=cond_val,
            normalizer=gs.normalizer.BoxCox(),
            fit_normalizer=True,
            fit_variogram=True,
        )
        # test fitting during kriging
        self.assertTrue(np.abs(krige.normalizer.lmbda - 0.0) < 1e-1)
        self.assertAlmostEqual(krige.model.len_scale, 10.2677, places=4)
        self.assertAlmostEqual(
            krige.model.sill,
            krige.normalizer.normalize(cond_val).var(),
            places=4,
        )
        # test fitting during vario estimate
        bin_center, gamma, normalizer = gs.vario_estimate(
            cond_pos,
            cond_val,
            normalizer=gs.normalizer.BoxCox,
            fit_normalizer=True,
        )
        model = gs.Stable(dim=2)
        model.fit_variogram(bin_center, gamma)
        self.assertAlmostEqual(model.var, 0.6426670183, places=4)
        self.assertAlmostEqual(model.len_scale, 9.635193952, places=4)
        self.assertAlmostEqual(model.nugget, 0.001617908408, places=4)
        self.assertAlmostEqual(model.alpha, 2.0, places=4)


if __name__ == "__main__":
    unittest.main()
