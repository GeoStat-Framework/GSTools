"""Tests for gstools.mps.model."""

import unittest
import warnings
from pathlib import Path

import numpy as np

import gstools as gs
from gstools.mps.direct_sampling import DirectSampling
from gstools.mps.model import MPSModel
from gstools.mps.post_processing import PostProcessing
from gstools.mps.training_image import (
    TrainingImage,
    Variable,
)


class TestMPSModelNonstationarySpec(unittest.TestCase):
    def _ti(self, shape=(10, 10)):
        return gs.mps.TrainingImage(
            np.random.default_rng(0).integers(0, 2, shape)
        )

    def test_defaults(self):
        m = MPSModel(self._ti())
        self.assertIsNone(m.rotation)
        self.assertIsNone(m.scale)
        self.assertEqual(m.zones, [])

    def test_specs_stored(self):
        rot = np.pi / 4
        sc = [1.0, 0.5]
        m = MPSModel(self._ti(), rotation=rot, scale=sc)
        self.assertEqual(m.rotation, rot)
        np.testing.assert_array_equal(m.scale, sc)

    def test_rotation_on_1d_ti_raises(self):
        # no_of_angles(1) == 0 — every rotation form is degenerate on a 1-D TI.
        ti1d = gs.mps.TrainingImage(
            np.arange(30, dtype=float), categorical=False
        )
        with self.assertRaisesRegex(ValueError, "1-D|rotation"):
            MPSModel(ti1d, rotation=0.5)

    def test_scale_on_1d_ti_allowed(self):
        ti1d = gs.mps.TrainingImage(
            np.arange(30, dtype=float), categorical=False
        )
        m = MPSModel(ti1d, scale=2.0)  # dilation along the single axis
        self.assertEqual(m.scale, 2.0)

    def test_nonpositive_scale_raises_at_construction(self):
        with self.assertRaisesRegex(ValueError, "positive|> 0"):
            MPSModel(self._ti(), scale=0.0)
        with self.assertRaisesRegex(ValueError, "positive|> 0"):
            MPSModel(self._ti(), scale=np.full((10, 10), -1.0))

    def test_callable_specs_accepted(self):
        m = MPSModel(
            self._ti(),
            rotation=lambda x, y: x * 0.01,
            scale=lambda x, y: 1.0 + 0 * x,
        )
        self.assertTrue(callable(m.rotation))
        self.assertTrue(callable(m.scale))

    def test_zone_variable_set_must_match(self):
        rng = np.random.default_rng(0)
        primary = gs.mps.TrainingImage(
            [Variable("a", rng.integers(0, 2, (8, 8)))]
        )
        wrong_name = gs.mps.TrainingImage(
            [Variable("b", rng.integers(0, 2, (8, 8)))]
        )
        z = gs.mps.Zone(wrong_name, where=np.ones((5, 5), dtype=bool))
        with self.assertRaisesRegex(ValueError, "variable"):
            MPSModel(primary, zones=[z])

    def test_zone_variable_kind_must_match(self):
        rng = np.random.default_rng(0)
        primary = gs.mps.TrainingImage(
            [Variable("a", rng.integers(0, 2, (8, 8)), categorical=True)]
        )
        wrong_kind = gs.mps.TrainingImage(
            [Variable("a", rng.random((8, 8)), categorical=False)]
        )
        z = gs.mps.Zone(wrong_kind, where=np.ones((5, 5), dtype=bool))
        with self.assertRaisesRegex(ValueError, "categorical|kind"):
            MPSModel(primary, zones=[z])

    def test_zone_ndim_must_match(self):
        z = gs.mps.Zone(
            gs.mps.TrainingImage(np.zeros((5, 5, 5))),
            where=np.ones((5, 5), dtype=bool),
        )
        with self.assertRaisesRegex(ValueError, "dim"):
            MPSModel(self._ti(), zones=[z])

    def test_non_zone_in_zones_raises(self):
        with self.assertRaises(TypeError):
            MPSModel(self._ti(), zones=[self._ti()])

    def test_zone_codes_checked_against_primary_penalty_matrix(self):
        # The scan indexes the PRIMARY TI's penalty_matrix with codes read from
        # the selected zone, so an out-of-range zone code must be rejected at
        # construction rather than raising IndexError mid-simulation.
        rng = np.random.default_rng(0)
        primary = gs.mps.TrainingImage(
            rng.integers(0, 2, (8, 8)).astype(float),
            categorical=True,
            penalty_matrix=np.array([[0.0, 1.0], [1.0, 0.0]]),
        )
        out_of_range = gs.mps.TrainingImage(
            rng.integers(0, 4, (8, 8)).astype(float), categorical=True
        )
        z = gs.mps.Zone(out_of_range, where=np.ones((5, 5), dtype=bool))
        with self.assertRaisesRegex(ValueError, "penalty_matrix"):
            MPSModel(primary, zones=[z])
        in_range = gs.mps.TrainingImage(
            rng.integers(0, 2, (8, 8)).astype(float), categorical=True
        )
        MPSModel(
            primary, zones=[gs.mps.Zone(in_range, where=np.ones((5, 5), bool))]
        )

    def test_repr_array_by_shape_callable_by_name(self):
        def my_rot(x, y):
            return x * 0.0

        m1 = MPSModel(self._ti(), rotation=my_rot, scale=np.ones((10, 10, 2)))
        r = repr(m1)
        self.assertIn("my_rot", r)
        self.assertIn("shape=(10, 10, 2)", r)
        m2 = MPSModel(self._ti())
        self.assertNotIn("rotation", repr(m2))


class TestThresholdValidation(unittest.TestCase):
    """threshold=None is the DSBC spelling; any other value must be > 0."""

    def _ti(self):
        return gs.mps.TrainingImage(np.zeros((8, 8)), categorical=True)

    def test_none_is_the_default_and_warns_nothing(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertIsNone(gs.mps.MPSModel(self._ti()).threshold)
            self.assertIsNone(
                gs.mps.MPSModel(self._ti(), threshold=None).threshold
            )

    def test_non_positive_rejected(self):
        # 0.0 is not a DSBC selector: DS acceptance is ``d < threshold``, so
        # 0.0 would accept nothing. It is an ordinary invalid value.
        for bad in (0.0, -0.1):
            with self.subTest(threshold=bad):
                with self.assertRaises(ValueError):
                    gs.mps.MPSModel(self._ti(), threshold=bad)
                model = gs.mps.MPSModel(self._ti())
                with self.assertRaises(ValueError):
                    model.threshold = bad

    def test_positive_threshold_unaffected(self):
        model = gs.mps.MPSModel(self._ti(), threshold=0.05)
        self.assertAlmostEqual(model.threshold, 0.05)


class TestModelPostProcessingSetting(unittest.TestCase):
    """The MPSModel post_processing parameter (Me13 §4)."""

    def _ti(self):
        return gs.mps.TrainingImage(np.zeros((8, 8)), categorical=True)

    def test_default_is_off(self):
        self.assertIsNone(gs.mps.MPSModel(self._ti()).post_processing)

    def test_accepts_object_and_none(self):
        post = PostProcessing(2, factor=2.0)
        model = gs.mps.MPSModel(self._ti(), post_processing=post)
        self.assertIs(model.post_processing, post)
        model.post_processing = None
        self.assertIsNone(model.post_processing)

    def test_rejects_other_types(self):
        for bad in (1, 0, {"passes": 1}):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    gs.mps.MPSModel(self._ti(), post_processing=bad)

    def test_factor_below_one_accepted(self):
        model = gs.mps.MPSModel(
            self._ti(), post_processing=PostProcessing(1, factor=0.5)
        )
        self.assertEqual(model.post_processing.factor, 0.5)


class TestMPSModelValidation(unittest.TestCase):
    """MPSModel setters validate and warnings point at user code."""

    def setUp(self):
        # 1-D categorical TI: alternating 0/1, length 20
        arr1d = np.tile([0, 1], 10).astype(float)
        self.ti1d = TrainingImage(arr1d, categorical=True, n_neighbors=8)

        # 2-D categorical TI: 8×8 checkerboard
        self.ti2d = TrainingImage(
            (np.indices((8, 8)).sum(axis=0) % 2).astype(float),
            categorical=True,
            n_neighbors=8,
        )

        rng = np.random.default_rng(0)
        self.ti2d_rand = TrainingImage(
            rng.integers(0, 2, size=(20, 20)).astype(float),
            categorical=True,
            n_neighbors=8,
        )

        # 1-D continuous TI
        self.ti1d_cont = TrainingImage(
            np.linspace(0.0, 1.0, 20),
            categorical=False,
            distance="l1",
            n_neighbors=8,
        )

        self.x1d = np.arange(10, dtype=float)
        self.x2d = np.arange(6, dtype=float)
        self.y2d = np.arange(6, dtype=float)

    def test_mpsmodel_setters_validate(self):
        m = MPSModel(self.ti1d)
        # setters normalize/validate in one place
        m.scan_fraction = 0.5
        self.assertAlmostEqual(m.scan_fraction, 0.5)
        with self.assertRaises(ValueError):
            m.scan_fraction = 0.0  # out of (0, 1]
        with self.assertRaises(ValueError):
            m.threshold = -1.0

    def test_threshold_warning_points_at_user_code(self):
        import warnings as _w

        # constructor path: the warning must originate from THIS test file,
        # not from inside gstools/mps/model.py
        with _w.catch_warnings(record=True) as rec:
            _w.simplefilter("always")
            MPSModel(self.ti1d, threshold=5.0)
        self.assertTrue(any(Path(x.filename).name != "model.py" for x in rec))
        # direct-setter path: same expectation
        m = MPSModel(self.ti1d)
        with _w.catch_warnings(record=True) as rec:
            _w.simplefilter("always")
            m.threshold = 5.0
        self.assertTrue(any(Path(x.filename).name != "model.py" for x in rec))


class TestModelMutability(unittest.TestCase):
    """Every MPSModel parameter is settable, validated and atomic (as CovModel)."""

    @staticmethod
    def _mv_ti(shape=(6, 6), names=("a", "b")):
        return TrainingImage([Variable(n, np.zeros(shape)) for n in names])

    def _values(self, model):
        return {
            name: getattr(model, name)
            for name in (
                "ti",
                "scan_fraction",
                "threshold",
                "rotation",
                "scale",
                "zones",
                "post_processing",
                "scan_path",
            )
        }

    def test_every_parameter_settable(self):
        model = MPSModel(self._mv_ti())
        zone = gs.mps.Zone(self._mv_ti(), where=lambda x, y: x >= 3)
        new = {
            "ti": self._mv_ti((8, 8)),
            "scan_fraction": 0.5,
            "threshold": 0.1,
            "rotation": 0.3,
            "scale": 2.0,
            "zones": [zone],
            "post_processing": PostProcessing(
                2, overrides={"a": {"n_neighbors": 4}}
            ),
            "scan_path": "random",
        }
        for name, value in new.items():
            with self.subTest(name):
                setattr(model, name, value)
        for name, value in new.items():
            with self.subTest(name, check="read back"):
                self.assertEqual(getattr(model, name), value)

    def test_failed_set_leaves_model_unchanged(self):
        ti = self._mv_ti()
        post = PostProcessing(1, overrides={"a": {"n_neighbors": 4}})
        model = MPSModel(ti, post_processing=post)
        before = self._values(model)
        resolved = model._resolved
        other_vars = gs.mps.Zone(
            self._mv_ti(names=("x", "y")), where=lambda x, y: x >= 3
        )
        bad = {
            "zones": [other_vars],
            "post_processing": PostProcessing(1, overrides={"missing": {}}),
            "ti": self._mv_ti(names=("c", "d")),  # lacks override "a"
        }
        for name, value in bad.items():
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    setattr(model, name, value)
                self.assertEqual(self._values(model), before)
                self.assertIs(model._resolved, resolved)

    def test_ti_setter_rechecks_overrides(self):
        model = MPSModel(
            self._mv_ti(),
            post_processing=PostProcessing(1, overrides={"a": {}}),
        )
        with self.assertRaisesRegex(ValueError, "not found"):
            model.ti = self._mv_ti(names=("b", "c"))
        model.ti = self._mv_ti(names=("a", "c"))  # keeps "a": accepted

    def test_ti_setter_rechecks_zones_and_rotation(self):
        zone = gs.mps.Zone(self._mv_ti(), where=lambda x, y: x >= 3)
        model = MPSModel(self._mv_ti(), zones=[zone], rotation=0.2)
        with self.assertRaises(ValueError):  # zone variables differ
            model.ti = self._mv_ti(names=("c", "d"))
        with self.assertRaises(ValueError):  # zone is 2-D
            model.ti = TrainingImage(
                [Variable("a", np.zeros(6)), Variable("b", np.zeros(6))]
            )
        rotated = MPSModel(TrainingImage(np.zeros((6, 6))), rotation=0.2)
        with self.assertRaises(ValueError):  # no rotation in 1-D
            rotated.ti = TrainingImage(np.zeros(6))

    def test_setters_validate(self):
        model = MPSModel(self._mv_ti())
        for name, bad in (
            ("scan_fraction", 0.0),
            ("threshold", -1.0),
            ("scan_path", "bad"),
            ("scale", -1.0),
            ("rotation", np.nan),
        ):
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    setattr(model, name, bad)


class TestNonFiniteModelParameters(unittest.TestCase):
    """Non-finite MPSModel parameters are rejected."""

    def setUp(self):
        rng = np.random.default_rng(416)
        self.codes = rng.integers(0, 2, (12, 12)).astype(float)  # {0, 1}
        self.cont = rng.random((12, 12))

    def test_non_finite_parameters_rejected(self):
        nan, inf = np.nan, np.inf
        ti = TrainingImage(self.cont, categorical=False)

        cases = {  # parameter: (builder taking the bad value, bad values)
            "threshold": (lambda v: MPSModel(ti, threshold=v), [nan]),
            "post_processing factor": (
                lambda v: MPSModel(
                    ti, post_processing=PostProcessing(1, factor=v)
                ),
                [nan, inf, 1e-320],
            ),
            "rotation": (
                lambda v: MPSModel(ti, rotation=v),
                [np.array([nan])],
            ),
            "scale": (lambda v: MPSModel(ti, scale=v), [inf]),
        }
        for name, (build, values) in cases.items():
            for value in values:
                with self.subTest(parameter=name, value=value):
                    with self.assertRaises(ValueError):
                        build(value)


class TestPostProcessingOverrides(unittest.TestCase):
    """Post-processing overrides are validated against the TI when the model is set."""

    def setUp(self):
        rng = np.random.default_rng(416)
        self.codes = rng.integers(0, 2, (12, 12)).astype(float)  # {0, 1}
        self.cont = rng.random((12, 12))
        self.pos = [np.arange(8.0)] * 2

    def _two_variable_ti(self, **kw):
        return TrainingImage(
            [
                Variable(
                    name, self.cont, categorical=False, n_neighbors=4, **kw
                )
                for name in ("a", "b")
            ]
        )

    def _post_ti(self, model):
        return model._resolved.post_ti

    def test_override_inherits_unset_fields(self):
        penalty = [[0, 0.5], [0.5, 0]]
        ti = TrainingImage(
            self.codes,
            n_neighbors=7,
            penalty_matrix=penalty,
            distance_power=0.5,
            cond_weight=2.0,
        )
        model = MPSModel(
            ti,
            post_processing=PostProcessing(
                1, overrides={"field": {"n_neighbors": 8}}
            ),
        )
        post = self._post_ti(model).variable("field")
        self.assertEqual(post.n_neighbors, 8)
        self.assertEqual(post.distance, ti.variable().distance)
        np.testing.assert_array_equal(post.penalty_matrix, penalty)
        self.assertEqual(post.cond_weight, 2.0)
        self.assertEqual(post.distance_power, 0.5)
        self.assertIs(post.data, ti.variable().data)
        self.assertEqual(ti.variable().n_neighbors, 7)  # primary untouched

    def test_no_override_reuses_primary_variables(self):
        ti = self._two_variable_ti()
        post_ti = self._post_ti(
            MPSModel(ti, post_processing=PostProcessing(1))
        )
        for name in ("a", "b"):
            self.assertIs(post_ti.variable(name), ti.variable(name))

    def test_override_unknown_variable_raises(self):
        ti = self._two_variable_ti()
        for name in ("z", "field"):
            with self.subTest(name):
                with self.assertRaisesRegex(ValueError, "not found"):
                    MPSModel(
                        ti,
                        post_processing=PostProcessing(
                            1, overrides={name: {}}
                        ),
                    )

    def test_override_for_array_and_named_variable(self):
        # An array-built TI names its variable "field"; a Variable keeps its name.
        array_ti = TrainingImage(self.codes)
        named_ti = TrainingImage(Variable("a", self.codes))
        for ti, name in ((array_ti, "field"), (named_ti, "a")):
            with self.subTest(name):
                model = MPSModel(
                    ti,
                    post_processing=PostProcessing(
                        1, overrides={name: {"n_neighbors": 5}}
                    ),
                )
                self.assertEqual(
                    self._post_ti(model).variable(name).n_neighbors, 5
                )

    def test_override_weight_and_radius_take_effect(self):
        z = np.zeros((4, 4))
        ti = TrainingImage(
            [Variable("a", z, weight=1.0), Variable("b", z, weight=1.0)]
        )
        model = MPSModel(
            ti,
            post_processing=PostProcessing(
                1, overrides={"a": {"weight": 3.0, "max_radius": 2.0}}
            ),
        )
        post_ti = self._post_ti(model)
        self.assertEqual(post_ti.weights, {"a": 0.75, "b": 0.25})
        self.assertEqual(post_ti.variable("a").max_radius, 2.0)
        self.assertIsNone(post_ti.variable("b").max_radius)

    def test_override_explicit_none_and_default_are_applied(self):
        # max_radius=None lifts the primary's limit; n_neighbors equal to the
        # Variable default (32) is applied, not inherited.
        ti = TrainingImage(
            Variable("a", self.codes, n_neighbors=7, max_radius=3.0)
        )
        post = PostProcessing(
            1, overrides={"a": {"max_radius": None, "n_neighbors": 32}}
        )
        var = self._post_ti(MPSModel(ti, post_processing=post)).variable()
        self.assertIsNone(var.max_radius)
        self.assertEqual(var.n_neighbors, 32)

    def test_post_weight_rule_mixed_weights_rejected(self):
        # Post-pass weights must be set on every variable or on none; a mixed
        # set fails at construction, not mid-simulation.
        weighted = self._two_variable_ti(weight=1.0)
        cases = {
            "unset on override": (weighted, {"a": {"weight": None}}),
            "set on override": (
                self._two_variable_ti(),
                {"a": {"weight": 2.0}},
            ),
        }
        for name, (ti, overrides) in cases.items():
            with self.subTest(name):
                with self.assertRaisesRegex(ValueError, "weight"):
                    MPSModel(
                        ti,
                        post_processing=PostProcessing(1, overrides=overrides),
                    )

    def test_override_invalid_value_rejected(self):
        ti = self._two_variable_ti()
        for fields in (
            {"n_neighbors": 0},
            {"max_radius": 0.5},
            {"distance": "bogus"},
            {"penalty_matrix": [[0, 1], [1, 0]]},  # continuous variable
        ):
            with self.subTest(fields):
                with self.assertRaises(ValueError):
                    MPSModel(
                        ti,
                        post_processing=PostProcessing(
                            1, overrides={"a": fields}
                        ),
                    )

    def test_override_penalty_matrix_checks_zone_codes(self):
        ti = TrainingImage(self.codes)
        zone = gs.mps.Zone(
            TrainingImage(np.full((12, 12), 2.0)), where=lambda x, y: x >= 3
        )
        post = PostProcessing(
            1, overrides={"field": {"penalty_matrix": [[0, 1], [1, 0]]}}
        )
        with self.assertRaisesRegex(ValueError, "penalty_matrix"):
            MPSModel(ti, zones=[zone], post_processing=post)

    def test_categorical_primary_rejects_variation_override(self):
        # A continuous override of a categorical variable would invent
        # categories through the mean shift.
        ti = TrainingImage(
            [Variable("c", self.codes), Variable("d", self.codes)]
        )
        post = PostProcessing(1, overrides={"c": {"distance": "variation"}})
        with self.assertRaises(ValueError):
            MPSModel(ti, post_processing=post)


class TestVariationPostPassValidation(unittest.TestCase):
    """A variation post-pass needs at least two neighbours."""

    def setUp(self):
        rng = np.random.default_rng(416)
        self.codes = rng.integers(0, 2, (12, 12)).astype(float)  # {0, 1}
        self.cont = rng.random((12, 12))
        self.pos = [np.arange(8.0)] * 2

    def test_variation_post_pass_below_two_neighbours_raises(self):
        # round(2 / 2) = 1 neighbour has no local mean, so variation would see
        # only zero distances: raised at construction. Post-processing off, or
        # a non-variation metric at one neighbour, stays valid.
        ti = TrainingImage(
            self.cont, categorical=False, distance="variation", n_neighbors=2
        )
        with self.assertRaisesRegex(ValueError, "variation"):
            MPSModel(ti, post_processing=PostProcessing(1, factor=2))
        MPSModel(ti)
        l1 = TrainingImage(self.codes, n_neighbors=1)
        model = MPSModel(l1, post_processing=PostProcessing(1, factor=2))
        self.assertEqual(
            DirectSampling(model, seed=1)(self.pos, store=False).shape, (8, 8)
        )

    def test_variation_override_below_two_neighbours_raises(self):
        ti = TrainingImage(
            self.cont, categorical=False, distance="variation", n_neighbors=4
        )
        post = PostProcessing(1, overrides={"field": {"n_neighbors": 1}})
        with self.assertRaisesRegex(ValueError, "variation"):
            MPSModel(ti, post_processing=post)

    def test_variation_post_pass_setter_leaves_state_unchanged(self):
        ti = TrainingImage(
            self.cont, categorical=False, distance="variation", n_neighbors=2
        )
        post = PostProcessing(1, factor=1)
        model = MPSModel(ti, post_processing=post)
        with self.assertRaisesRegex(ValueError, "variation"):
            model.post_processing = PostProcessing(1, factor=2)
        self.assertIs(model.post_processing, post)

        model = MPSModel(ti)
        with self.assertRaisesRegex(ValueError, "variation"):
            model.post_processing = PostProcessing(1, factor=2)
        self.assertIsNone(model.post_processing)


class TestTinyScanWarning(unittest.TestCase):
    """Fewer than two scan candidates per node warns."""

    def setUp(self):
        rng = np.random.default_rng(416)
        self.codes = rng.integers(0, 2, (12, 12)).astype(float)  # {0, 1}

    def test_tiny_scan_warns_main_and_post_pass(self):
        # Fewer than 2 candidates per node, from scan_fraction or from
        # scan_fraction / PostProcessing.factor, warns.
        ti = TrainingImage(self.codes)
        with self.assertWarnsRegex(UserWarning, "fewer than 2"):
            MPSModel(ti, scan_fraction=0.005)
        with self.assertWarnsRegex(UserWarning, "fewer than 2"):
            MPSModel(
                ti,
                scan_fraction=0.5,
                post_processing=PostProcessing(1, factor=100),
            )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            MPSModel(
                ti,
                scan_fraction=0.5,
                post_processing=PostProcessing(1, factor=2),
            )


if __name__ == "__main__":
    unittest.main()
