"""This is the unittest of the transform submodule."""

import unittest

import numpy as np

import gstools as gs


class TestTransform(unittest.TestCase):
    def setUp(self):
        self.cov_model = gs.Gaussian(dim=2, var=1.5, len_scale=4.0)
        self.mean = 0.3
        self.mode_no = 100

        self.seed = 825718662
        self.x_grid = np.linspace(0.0, 12.0, 48)
        self.y_grid = np.linspace(0.0, 10.0, 46)

        self.methods = [
            "binary",
            "boxcox",
            "zinnharvey",
            "normal_force_moments",
            "normal_to_lognormal",
            "normal_to_uniform",
            "normal_to_arcsin",
            "normal_to_uquad",
        ]

    def test_transform_normal(self):
        srf = gs.SRF(self.cov_model, mean=self.mean, mode_no=self.mode_no)
        srf((self.x_grid, self.y_grid), seed=self.seed, mesh_type="structured")
        for method in self.methods:
            srf.transform(method, store=method)
        std = np.sqrt(srf.model.var)
        self.assertTrue(set(self.methods) == set(srf.field_names[1:]))
        # force moments
        self.assertAlmostEqual(srf["normal_force_moments"].mean(), srf.mean)
        self.assertAlmostEqual(srf["normal_force_moments"].var(), std**2)
        # binary
        np.testing.assert_allclose(
            np.unique(srf.binary), srf.mean + np.array([-std, std])
        )
        # boxcox
        np.testing.assert_allclose(
            srf.field, gs.normalizer.BoxCox().normalize(srf.boxcox)
        )
        with self.assertWarns(Warning):
            srf.transform("boxcox", store="boxcox_warn", lmbda=2)
        # lognormal
        np.testing.assert_allclose(srf.field, np.log(srf.normal_to_lognormal))
        srf.transform("boxcox", store="boxcox2", lmbda=0)
        np.testing.assert_allclose(srf.boxcox2, srf.normal_to_lognormal)
        # unifrom
        self.assertTrue(np.all(srf.normal_to_uniform < 1))
        self.assertTrue(np.all(srf.normal_to_uniform > 0))
        # how to test arcsin and uquad?!

        # discrete
        values = [-1, 0, 1]
        thresholds = [-0.9, 0.1]
        srf.transform(
            "discrete", values=values, thresholds=thresholds, store="f1"
        )
        np.testing.assert_allclose(np.unique(srf.f1), [-1, 0, 1])

        values = [-1, 0, 1]
        srf.transform(
            "discrete", values=values, thresholds="arithmetic", store="f2"
        )
        np.testing.assert_allclose(np.unique(srf.f2), [-1.0, 0.0, 1.0])

        values = [-1, 0, 0.5, 1]
        srf.transform(
            "discrete", values=values, thresholds="equal", store="f3"
        )
        np.testing.assert_allclose(np.unique(srf.f3), values)
        # checks
        with self.assertRaises(ValueError):
            srf.transform("discrete", values=values, thresholds=[1])
        with self.assertRaises(ValueError):
            srf.transform("discrete", values=values, thresholds=[1, 3, 2])

        # function
        srf.transform("function", function=lambda x: 2 * x, store="f4")
        np.testing.assert_allclose(2 * srf.field, srf.f4)
        with self.assertRaises(ValueError):
            srf.transform("function", function=None)

        # unknown method
        with self.assertRaises(ValueError):
            srf.transform("foobar")

    def test_transform_denormal(self):
        srf_fail = gs.SRF(
            model=self.cov_model,
            mean=self.mean,
            mode_no=self.mode_no,
            trend=lambda x, y: x,
        )
        srf_fail((self.x_grid, self.y_grid), mesh_type="structured")
        with self.assertRaises(ValueError):
            srf_fail.transform("zinnharvey")

        srf_fail = gs.SRF(
            model=self.cov_model,
            mean=lambda x, y: x,
            mode_no=self.mode_no,
        )
        srf_fail((self.x_grid, self.y_grid), mesh_type="structured")
        with self.assertRaises(ValueError):
            srf_fail.transform("zinnharvey")

        srf = gs.SRF(
            model=self.cov_model,
            mean=self.mean,
            mode_no=self.mode_no,
            normalizer=gs.normalizer.LogNormal,
        )
        srf((self.x_grid, self.y_grid), seed=self.seed, mesh_type="structured")

        for method in self.methods:
            if method in ("normal_to_lognormal", "boxcox"):
                continue
            with self.assertRaises(ValueError):
                srf.transform(method, store=method)

        for method in self.methods:
            srf.transform(method, store=method, process=True)
        std = np.sqrt(srf.model.var)
        self.assertTrue(set(self.methods) == set(srf.field_names[1:]))
        # force moments
        self.assertAlmostEqual(
            np.log(srf["normal_force_moments"]).mean(), srf.mean
        )
        self.assertAlmostEqual(
            np.log(srf["normal_force_moments"]).var(), std**2
        )
        # binary
        np.testing.assert_allclose(
            np.unique(np.log(srf.binary)), srf.mean + np.array([-std, std])
        )
        # boxcox
        np.testing.assert_allclose(
            np.log(srf.field),
            gs.normalizer.BoxCox().normalize(np.log(srf.boxcox)),
        )
        # lognormal
        np.testing.assert_allclose(srf.field, np.log(srf.normal_to_lognormal))
        # unifrom
        self.assertTrue(np.all(np.log(srf.normal_to_uniform) < 1))
        self.assertTrue(np.all(np.log(srf.normal_to_uniform) > 0))

        # discrete
        values = [-1, 0, 1]
        thresholds = [-0.9, 0.1]
        srf.transform(
            "discrete",
            values=values,
            thresholds=thresholds,
            store="f1",
            process=True,
        )
        np.testing.assert_allclose(np.unique(np.log(srf.f1)), [-1, 0, 1])

        values = [-1, 0, 1]
        srf.transform(
            "discrete",
            values=values,
            thresholds="arithmetic",
            store="f2",
            process=True,
        )
        np.testing.assert_allclose(np.unique(np.log(srf.f2)), [-1.0, 0.0, 1.0])

        values = [-1, 0, 0.5, 1]
        srf.transform(
            "discrete",
            values=values,
            thresholds="equal",
            store="f3",
            process=True,
        )
        np.testing.assert_allclose(np.unique(np.log(srf.f3)), values)


if __name__ == "__main__":
    unittest.main()
