"""Tests for gstools.mps.training_image."""

import unittest
import warnings
from unittest import mock

import numpy as np
from mps_fixtures import ignoring, mv_dist, uni_dist

import gstools as gs
from gstools.mps.direct_sampling import DirectSampling
from gstools.mps.model import MPSModel
from gstools.mps.training_image import TrainingImage, Variable, _data_range


class TestVariable(unittest.TestCase):
    def test_basic_categorical(self):
        data = np.array([[0, 1], [1, 0]], dtype=int)
        v = Variable("cat", data, categorical=True, n_neighbors=8)
        self.assertEqual(v.name, "cat")
        self.assertTrue(v.categorical)
        self.assertEqual(v.n_neighbors, 8)
        self.assertIsNone(v.weight)
        self.assertIsNone(v.p_norm)
        self.assertIsNone(v.variation_p_norm)
        self.assertIsNone(v.d_max)

    def test_data_defensive_copy(self):
        data = np.array([0, 1, 2, 3])
        v = Variable("x", data)
        data[0] = 99
        self.assertEqual(v.data[0], 0)

    def test_continuous_attributes(self):
        v = Variable(
            "p", np.linspace(0, 1, 20), categorical=False, distance="l2"
        )
        self.assertEqual(v.p_norm, 2.0)
        self.assertIsNone(v.variation_p_norm)
        self.assertGreater(v.d_max, 0)

    def test_variation_distance(self):
        v = Variable(
            "p",
            np.linspace(0, 1, 20),
            categorical=False,
            distance="variation",
            n_neighbors=2,
        )
        self.assertIsNone(v.p_norm)
        self.assertEqual(v.variation_p_norm, 2.0)

    def test_invalid_name_raises(self):
        with self.assertRaisesRegex(ValueError, "identifier"):
            Variable("123bad", np.zeros(4))
        with self.assertRaisesRegex(ValueError, "identifier"):
            Variable("", np.zeros(4))
        with self.assertRaisesRegex(ValueError, "identifier"):
            Variable("a b", np.zeros(4))

    def test_name_none_rejected(self):
        with self.assertRaisesRegex(ValueError, "identifier"):
            Variable(None, np.zeros(4))

    def test_n_neighbors_is_read_only(self):
        v = Variable("x", np.zeros(4))
        with self.assertRaises(AttributeError):
            v.n_neighbors = 16

    def test_replace_shares_data_buffer(self):
        v = Variable("x", np.zeros((10, 10)))
        v2 = v.replace(n_neighbors=16)
        self.assertTrue(np.shares_memory(v.data, v2.data))
        self.assertEqual(v2.n_neighbors, 16)
        self.assertEqual(v2.name, "x")
        self.assertIs(v2.categorical, v.categorical)

    def test_replace_rejects_invalid_overrides(self):
        v = Variable("x", np.zeros((4, 4)))
        for overrides in (
            {"data": np.ones((4, 4))},  # defines what the variable *is*
            {"categorical": False},  # ditto
            {"not_a_real_field": 1},
        ):
            with self.subTest(field=next(iter(overrides))):
                with self.assertRaises(ValueError):
                    v.replace(**overrides)

    def test_replace_leaves_caller_array_writeable(self):
        arr = np.zeros((4, 4))
        v = Variable("x", arr)
        v.replace(n_neighbors=5)
        self.assertTrue(arr.flags.writeable)
        self.assertFalse(v.data.flags.writeable)

    def test_replace_runs_no_data_checks(self):
        v = Variable("x", np.array([0, 1, 0, 1]), penalty_matrix=1 - np.eye(2))
        with mock.patch(
            "gstools.mps.training_image._check_category_codes"
        ) as check:
            v2 = v.replace(n_neighbors=5)
            self.assertEqual(check.call_count, 0)
            v.replace(penalty_matrix=[[0, 1], [1, 0]])
            self.assertEqual(check.call_count, 1)  # a new matrix is checked
        self.assertEqual(v2.n_neighbors, 5)

    def test_replace_copies_derived_facts_without_warning(self):
        data = np.array([0.0, 1.0, np.nan, 3.0])
        with self.assertWarns(UserWarning):
            v = Variable("x", data, categorical=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            v2 = v.replace(n_neighbors=3)
        self.assertTrue(v2.has_nan)
        self.assertEqual(v2.d_max, v.d_max)

    def test_replace_distance_revalidates_n_neighbors(self):
        v = Variable(
            "x", np.linspace(0, 1, 10), categorical=False, n_neighbors=1
        )
        with self.assertRaisesRegex(ValueError, "n_neighbors >= 2"):
            v.replace(distance="variation")

    def test_replace_applies_override_and_carries_the_rest(self):
        # replace() uses the constructor's field validators, so per field it
        # is enough to confirm the override arrives and nothing else moves.
        v = Variable(
            "x",
            np.zeros((4, 4)),
            n_neighbors=10,
            max_radius=5.0,
            weight=0.3,
            distance_power=0.5,
            cond_weight=2.0,
        )
        carried = (
            "distance",
            "n_neighbors",
            "max_radius",
            "weight",
            "distance_power",
            "cond_weight",
        )
        for field, value in [
            ("n_neighbors", 20),
            ("max_radius", 7.0),
            ("max_radius", None),  # None is an override, not "leave as is"
            ("weight", 0.9),
            ("distance_power", 1.0),
            ("cond_weight", 3.0),
        ]:
            with self.subTest(field=field, value=value):
                v2 = v.replace(**{field: value})
                self.assertEqual(getattr(v2, field), value)
                for other in carried:
                    if other != field:
                        self.assertEqual(getattr(v2, other), getattr(v, other))

    def test_replace_distance_change_reparses_p_norm(self):
        v = Variable("x", np.linspace(0.0, 1.0, 10), categorical=False)
        self.assertEqual(v.p_norm, 1.0)  # default "l1"
        v2 = v.replace(distance="l2")
        self.assertEqual(v2.p_norm, 2.0)
        self.assertIsNone(v2.variation_p_norm)

    def test_replace_validates_changed_fields(self):
        cat = Variable("x", np.array([0, 1, 0, 1]))
        variation = Variable(
            "x",
            np.linspace(0.0, 1.0, 10),
            categorical=False,
            distance="variation",
            n_neighbors=4,
        )
        for var, overrides in [
            (cat, {"max_radius": 0.0}),
            (cat, {"distance_power": -1.0}),
            (cat, {"penalty_matrix": np.array([[0.0, 2.0], [1.0, 0.0]])}),
            (variation, {"n_neighbors": 1}),  # variation needs >= 2
        ]:
            with self.subTest(field=next(iter(overrides))):
                with self.assertRaises(ValueError):
                    var.replace(**overrides)

    def test_distance_power_and_cond_weight_on_variable(self):
        v = Variable(
            "x", np.zeros((4, 4)), distance_power=0.5, cond_weight=2.0
        )
        self.assertAlmostEqual(v.distance_power, 0.5)
        self.assertAlmostEqual(v.cond_weight, 2.0)
        self.assertAlmostEqual(Variable("x", np.zeros(4)).distance_power, 0.0)
        with self.assertRaises(ValueError):
            Variable("x", np.zeros((4, 4)), distance_power=-1.0)

    def test_univariate_sugar_forwards_distance_power_and_cond_weight(self):
        # distance_power/cond_weight live on the Variable only; the TI just
        # forwards the univariate sugar to its anonymous Variable.
        ti = TrainingImage(
            np.zeros((4, 4)), distance_power=0.5, cond_weight=2.0
        )
        self.assertFalse(hasattr(ti, "distance_power"))
        self.assertAlmostEqual(ti.variable().distance_power, 0.5)
        self.assertAlmostEqual(ti.variable().cond_weight, 2.0)

    def test_read_only_name(self):
        v = Variable("x", np.zeros(4))
        with self.assertRaises(AttributeError):
            v.name = "y"

    def test_read_only_categorical(self):
        v = Variable("x", np.zeros(4))
        with self.assertRaises(AttributeError):
            v.categorical = False

    def test_weight_stored(self):
        v = Variable("x", np.zeros(4), weight=0.7)
        self.assertAlmostEqual(v.weight, 0.7)

    def test_repr(self):
        v = Variable("x", np.zeros((4, 4)))
        self.assertIn("Variable", repr(v))
        self.assertIn("x", repr(v))

    def test_max_radius_none_default(self):
        v = Variable("x", np.zeros(4))
        self.assertIsNone(v.max_radius)

    def test_max_radius_stored(self):
        v = Variable("x", np.zeros(4), max_radius=50.0)
        self.assertAlmostEqual(v.max_radius, 50.0)

    def test_max_radius_zero_raises(self):
        with self.assertRaisesRegex(ValueError, "max_radius"):
            Variable("x", np.zeros(4), max_radius=0.0)

    def test_max_radius_negative_raises(self):
        with self.assertRaisesRegex(ValueError, "max_radius"):
            Variable("x", np.zeros(4), max_radius=-5.0)

    def test_max_radius_below_one_raises(self):
        with self.assertRaisesRegex(ValueError, "max_radius"):
            Variable("x", np.zeros(4), max_radius=0.5)

    def test_max_radius_read_only(self):
        v = Variable("x", np.zeros(4), max_radius=10.0)
        with self.assertRaises(AttributeError):
            v.max_radius = 20.0

    def test_read_only_view_base_mutation_does_not_leak(self):
        # The Variable holds its own copy of a read-only view's data.
        base = np.zeros((4, 4))
        view = base.view()
        view.flags.writeable = False
        v = Variable("x", view)
        base[0, 0] = 7.0
        self.assertEqual(v.data[0, 0], 0.0)


class TestTrainingImage(unittest.TestCase):
    def setUp(self):
        arr_cat = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        self.ti_cat = TrainingImage(arr_cat, categorical=True)

        arr_cont = np.linspace(0.0, 1.0, 20)
        self.ti_cont = TrainingImage(
            arr_cont, categorical=False, distance="l1"
        )

    def test_properties(self):
        np.testing.assert_array_equal(
            self.ti_cat.variable().data,
            np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float),
        )
        self.assertEqual(self.ti_cat.ndim, 2)
        self.assertEqual(self.ti_cat.shape, (3, 3))
        self.assertTrue(self.ti_cat.variable().categorical)
        self.assertEqual(
            self.ti_cat.variable().distance, "l1"
        )  # default ignored for cat
        self.assertIsInstance(repr(self.ti_cat), str)
        self.assertIn("TrainingImage", repr(self.ti_cat))

        self.assertEqual(self.ti_cont.ndim, 1)
        self.assertEqual(self.ti_cont.shape, (20,))
        self.assertFalse(self.ti_cont.variable().categorical)
        self.assertEqual(self.ti_cont.variable().distance, "l1")

    def test_raise(self):
        with self.assertRaises(ValueError):
            TrainingImage(np.ones(10), categorical=False, distance="l0")
        with self.assertRaises(ValueError):
            TrainingImage(np.ones(10), categorical=False, distance="labc")
        with self.assertRaises(ValueError):
            TrainingImage(np.ones(10), categorical=False, distance="invalid")

    def test_distance_categorical(self):
        # Identical events → 0.0
        a = np.array([0.0, 1.0, 0.0])
        dist = uni_dist(self.ti_cat, a, a)
        self.assertAlmostEqual(dist, 0.0)

        # Completely mismatched, uniform weights → 1.0
        b = np.array([1.0, 0.0, 1.0])
        dist = uni_dist(self.ti_cat, a, b)
        self.assertAlmostEqual(dist, 1.0)

        # One of three mismatched → 1/3
        c = np.array([1.0, 1.0, 0.0])
        dist = uni_dist(self.ti_cat, a, c)
        self.assertAlmostEqual(dist, 1.0 / 3.0)

        # Two of four mismatched → 0.5 (spec-required half-mismatch case)
        a4 = np.array([0.0, 1.0, 0.0, 1.0])
        c4 = np.array([1.0, 0.0, 0.0, 1.0])
        dist = uni_dist(self.ti_cat, a4, c4)
        self.assertAlmostEqual(dist, 0.5)

    def test_distance_continuous(self):
        x = np.array([0.0, 0.5, 1.0])
        y = np.array([0.2, 0.3, 0.8])

        # l1
        ti_l1 = TrainingImage(
            np.linspace(0.0, 1.0, 10), categorical=False, distance="l1"
        )
        self.assertAlmostEqual(uni_dist(ti_l1, x, x), 0.0)
        self.assertAlmostEqual(uni_dist(ti_l1, x, y), 0.2, places=6)

        # l2
        ti_l2 = TrainingImage(
            np.linspace(0.0, 1.0, 10), categorical=False, distance="l2"
        )
        self.assertAlmostEqual(uni_dist(ti_l2, x, x), 0.0)
        self.assertAlmostEqual(uni_dist(ti_l2, x, y), 0.2, places=6)

        # lp (p=3.5) — non-uniform diffs [0.3, 0.1, 0.3] distinguish lp from l1/l2
        y_lp = np.array([0.3, 0.4, 0.7])
        ti_lp = TrainingImage(
            np.linspace(0.0, 1.0, 10), categorical=False, distance="l3.5"
        )
        self.assertAlmostEqual(uni_dist(ti_lp, x, x), 0.0)
        self.assertAlmostEqual(uni_dist(ti_lp, x, y_lp), 0.2680, places=3)
        self.assertGreater(uni_dist(ti_lp, x, y_lp), uni_dist(ti_l1, x, y_lp))

        # variation (default p=2)
        ti_var = TrainingImage(
            np.linspace(0.0, 1.0, 10), categorical=False, distance="variation"
        )
        self.assertAlmostEqual(uni_dist(ti_var, x, x), 0.0)
        self.assertAlmostEqual(uni_dist(ti_var, x, y), 0.094281, places=5)
        # constant shift → distance = 0 (key behavioral property of variation distance)
        self.assertAlmostEqual(uni_dist(ti_var, x, x + 0.15), 0.0, places=10)

        # variation1 (L^1 aggregation)
        ti_var1 = TrainingImage(
            np.linspace(0.0, 1.0, 10), categorical=False, distance="variation1"
        )
        self.assertAlmostEqual(uni_dist(ti_var1, x, x), 0.0)
        self.assertAlmostEqual(uni_dist(ti_var1, x, y), 0.08889, places=4)
        self.assertAlmostEqual(uni_dist(ti_var1, x, x + 0.15), 0.0, places=10)
        # L^1 < L^2 for non-uniform diffs
        self.assertLess(uni_dist(ti_var1, x, y), uni_dist(ti_var, x, y))

        # variation2 explicit matches variation (default p)
        ti_var2 = TrainingImage(
            np.linspace(0.0, 1.0, 10), categorical=False, distance="variation2"
        )
        self.assertAlmostEqual(
            uni_dist(ti_var2, x, y), uni_dist(ti_var, x, y), places=10
        )

    def test_distance_weights(self):
        a = np.array([0.0, 1.0, 0.0])
        b = np.array([1.0, 1.0, 0.0])  # first element differs

        # cond_weight=2 on first node → it gets weight 0.5 (double)
        d1 = uni_dist(
            self.ti_cat, a, b, cond_mask=[True, False, False], cond_weight=1.0
        )
        d2 = uni_dist(
            self.ti_cat, a, b, cond_mask=[True, False, False], cond_weight=2.0
        )
        self.assertGreater(d2, d1)

        # distance_power shifts weight toward closer neighbours — use non-uniform
        # differences so the weighted sums actually differ: diffs = [0, 0, 0.5]
        ti_p = TrainingImage(
            np.linspace(0.0, 1.0, 10),
            categorical=False,
            distance="l1",
            distance_power=1.0,
        )
        ti_flat = TrainingImage(
            np.linspace(0.0, 1.0, 10),
            categorical=False,
            distance="l1",
            distance_power=0.0,
        )
        x = np.array([0.0, 0.5, 1.0])
        z = np.array([0.0, 0.5, 0.5])  # only third element differs
        lags = np.array([1.0, 2.0, 3.0])
        d_power = uni_dist(ti_p, x, z, lag_norms=lags)
        d_flat = uni_dist(ti_flat, x, z, lag_norms=lags)
        # power=1 weights far neighbours less → smaller distance for far mismatch
        self.assertLess(d_power, d_flat)

    def test_distance_empty_event(self):
        dist = uni_dist(self.ti_cat, np.array([]), np.array([]))
        self.assertAlmostEqual(dist, 0.0)

    def test_variation_lp_parsing(self):
        """variation<p> string is parsed correctly and rejects bad inputs."""
        data = np.linspace(0.0, 1.0, 10)
        for spec in ("variation", "variation1", "variation1.5", "variation2"):
            ti = TrainingImage(data, categorical=False, distance=spec)
            self.assertEqual(ti.variable().distance, spec)
        # invalid suffix
        with self.assertRaises(ValueError):
            TrainingImage(data, categorical=False, distance="variationX")
        # non-positive exponent
        with self.assertRaises(ValueError):
            TrainingImage(data, categorical=False, distance="variation0")
        with self.assertRaises(ValueError):
            TrainingImage(data, categorical=False, distance="variation-1")

    def test_warning_prefixes(self):
        # threshold>1 warning carries an "MPSModel:" prefix
        ti = TrainingImage(np.ones(4), categorical=True)
        with self.assertWarnsRegex(UserWarning, r"MPSModel:"):
            MPSModel(ti, threshold=5.0)
        # NaN warning comes from Variable construction
        nan_data = np.array([np.nan, 1.0, 2.0])
        with self.assertWarns(UserWarning) as cm:
            Variable("x", nan_data, categorical=False)
        self.assertTrue(
            any("Variable" in str(w.message) for w in cm.warnings),
            "Expected 'Variable' in NaN warning message",
        )


class TestUnivariateSugarKwargConflicts(unittest.TestCase):
    def test_sugar_kwargs_rejected_for_variable_input(self):
        # Every univariate-sugar kwarg is rejected, for both a bare Variable
        # and a list of them, and the message names the offending kwarg.
        data = np.array([0, 1, 0, 1])
        for kwarg, value in [
            ("categorical", False),
            ("distance", "l2"),
            ("n_neighbors", 5),
            ("max_radius", 3.0),
            ("penalty_matrix", np.array([[0.0, 1.0], [1.0, 0.0]])),
            ("distance_power", 1.0),
            ("cond_weight", 2.0),
        ]:
            for listed in (False, True):
                var = Variable("a", data)
                container = [var] if listed else var
                with self.subTest(kwarg=kwarg, listed=listed):
                    with self.assertRaisesRegex(ValueError, kwarg):
                        TrainingImage(container, **{kwarg: value})

    def test_sugar_kwargs_absent_accepted(self):
        ti = TrainingImage([Variable("a", np.zeros((4, 4)))])
        self.assertEqual(ti.variable("a").name, "a")


class TestMultivariateTrainingImage(unittest.TestCase):
    def test_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "same shape"):
            TrainingImage(
                [
                    Variable("a", np.zeros((4, 4))),
                    Variable("b", np.zeros((3, 3))),
                ]
            )

    def test_equal_weights(self):
        ti = TrainingImage(
            [
                Variable("a", np.zeros((4, 4), dtype=int)),
                Variable("b", np.ones((4, 4), dtype=int)),
            ]
        )
        self.assertTrue(ti.multivariate)
        self.assertEqual([v.name for v in ti.variables], ["a", "b"])
        self.assertAlmostEqual(ti.weights["a"], 0.5)
        self.assertAlmostEqual(ti.weights["b"], 0.5)
        self.assertEqual(ti.shape, (4, 4))
        self.assertEqual(ti.ndim, 2)

    def test_weights_must_sum_positive(self):
        # all-zero weights are invalid
        with self.assertRaisesRegex(ValueError, "positive"):
            TrainingImage(
                [
                    Variable("a", np.zeros((4, 4), dtype=int), weight=0.0),
                    Variable("b", np.zeros((4, 4), dtype=int), weight=0.0),
                ]
            )

    def test_distance_weighted_sum(self):
        ti = TrainingImage(
            [
                Variable("a", np.zeros((4, 4), dtype=int), weight=0.5),
                Variable("b", np.zeros((4, 4), dtype=int), weight=0.5),
            ]
        )
        de_sg = {"a": np.array([1, 1]), "b": np.array([0, 0])}
        de_ti = {"a": np.array([0, 0]), "b": np.array([0, 0])}
        self.assertAlmostEqual(mv_dist(ti, de_sg, de_ti), 0.5)

    def test_per_variable_categorical_and_distance(self):
        ti = TrainingImage(
            [
                Variable("cat", np.zeros((4, 4), dtype=int), categorical=True),
                Variable(
                    "cont",
                    np.linspace(0, 100, 16).reshape(4, 4),
                    categorical=False,
                    distance="l2",
                ),
            ]
        )
        self.assertTrue(ti.variable("cat").categorical)
        self.assertFalse(ti.variable("cont").categorical)
        self.assertAlmostEqual(ti.variable("cont").d_max, 100.0)

    def test_univariate_variation_p_preserved(self):
        ti = TrainingImage(
            np.linspace(0, 100, 16).reshape(4, 4),
            categorical=False,
            distance="variation1.5",
        )
        self.assertFalse(ti.multivariate)
        self.assertEqual(ti.variable().variation_p_norm, 1.5)
        self.assertIsNone(ti.variable().p_norm)
        self.assertEqual(ti.weights, {"field": 1.0})

    def test_variable_accessor(self):
        v_a = Variable("a", np.arange(16).reshape(4, 4))
        v_b = Variable("b", np.zeros((4, 4), dtype=int))
        ti = TrainingImage([v_a, v_b])
        self.assertIs(ti.variable("a"), v_a)  # same object
        # an array-built TI has one variable, named "field"
        uni = TrainingImage(np.zeros((4, 4), dtype=int))
        v_uni = uni.variable()
        self.assertEqual(v_uni.name, "field")
        with self.assertRaises(KeyError):
            uni.variable("a")

    def test_mixing_none_and_explicit_weights_raises(self):
        with self.assertRaisesRegex(ValueError, "mixing"):
            TrainingImage(
                [
                    Variable("a", np.zeros((4, 4)), weight=0.5),
                    Variable("b", np.zeros((4, 4)), weight=None),
                ]
            )

    def test_duplicate_names_raises(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            TrainingImage(
                [
                    Variable("x", np.zeros((4, 4))),
                    Variable("x", np.zeros((4, 4))),
                ]
            )


class TestVariationNNeighbors(unittest.TestCase):
    def test_variation_n1_raises_at_variable_construction(self):
        with self.assertRaisesRegex(ValueError, "variation"):
            Variable(
                "x",
                np.linspace(0.0, 1.0, 20),
                categorical=False,
                distance="variation",
                n_neighbors=1,
            )

    def test_variation_n2_ok(self):
        v = Variable(
            "x",
            np.linspace(0.0, 1.0, 20),
            categorical=False,
            distance="variation",
            n_neighbors=2,
        )
        ti = TrainingImage([v])
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.5))
        self.assertEqual(ds.ti.variable().n_neighbors, 2)

    def test_non_variation_n1_ok(self):
        v = Variable(
            "x",
            np.linspace(0.0, 1.0, 20),
            categorical=False,
            distance="l2",
            n_neighbors=1,
        )
        ti = TrainingImage([v])
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.5))
        self.assertEqual(ds.ti.variable().n_neighbors, 1)

    def test_variation_rejects_one_neighbour_atomically(self):
        # Variation guard raises on replace(); the original Variable/DS
        # are unaffected (replace() never mutates, so this is trivially
        # atomic -- verified anyway).
        v = Variable(
            "x",
            np.linspace(0.0, 1.0, 20),
            categorical=False,
            distance="variation",
            n_neighbors=3,
        )
        ti = TrainingImage([v])
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.5))
        with self.assertRaises(ValueError):
            v.replace(n_neighbors=1)
        self.assertEqual(v.n_neighbors, 3)  # unchanged
        self.assertEqual(ds.ti.variable().n_neighbors, 3)  # unchanged


class TestNaNTrainingImage(unittest.TestCase):
    """Masked / incomplete TIs: NaN cells are undefined (warn + handle).

    Contract: NaN cells are treated as undefined. They are (a) excluded from
    the continuous data range ``d_max``, (b) excluded per-position from every
    candidate distance, and (c) never pasted into the field. Construction warns
    once. A TI with no fully-defined cell cannot be simulated and raises.
    """

    def test_nan_excluded_from_dmax(self):
        # A NaN must not collapse d_max to the 1.0 fallback.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ti = TrainingImage(
                np.array([0.0, np.nan, 100.0]),
                categorical=False,
                distance="l1",
            )
        self.assertAlmostEqual(ti.variable()._d_max, 100.0)

    def test_construction_warns_on_nan(self):
        with self.assertWarns(UserWarning):
            TrainingImage(
                np.array([0.0, np.nan, 1.0]), categorical=False, distance="l1"
            )

    def test_partial_nan_continuous_runs_finite(self):
        # A NaN patch must not crash and must produce finite output within
        # the defined data range.
        data = np.random.RandomState(0).rand(20, 20)
        data[5:8, 5:8] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ti = TrainingImage(data, categorical=False, distance="l1")
        ds = DirectSampling(MPSModel(ti, scan_fraction=1.0))
        field = ds([np.arange(15, dtype=float)] * 2, seed=1)
        self.assertEqual(field.shape, (15, 15))
        self.assertTrue(np.all(np.isfinite(field)))
        self.assertGreaterEqual(field.min(), np.nanmin(data))
        self.assertLessEqual(field.max(), np.nanmax(data))

    def test_partial_nan_categorical_runs_subset(self):
        # A categorical NaN patch must not crash, and output must be a
        # subset of the *defined* categories.
        data = np.random.RandomState(0).randint(0, 2, (20, 20)).astype(float)
        data[5:8, 5:8] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ti = TrainingImage(data, categorical=True)
        ds = DirectSampling(MPSModel(ti, scan_fraction=1.0))
        field = ds([np.arange(15, dtype=float)] * 2, seed=1)
        self.assertTrue(np.all(np.isfinite(field)))
        self.assertTrue(set(np.unique(field)).issubset({0.0, 1.0}))

    def test_all_nan_ti_raises(self):
        # No fully-defined cell → cannot paste any value → clear error.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ti = TrainingImage(
                np.full((5, 5), np.nan), categorical=False, distance="l2"
            )
        ds = DirectSampling(MPSModel(ti, scan_fraction=1.0))
        with self.assertRaises(ValueError):
            ds([np.arange(3, dtype=float)] * 2, seed=0)


class TestTrainingImageWindow(unittest.TestCase):
    """TrainingImage.window(slices) sub-region views (M10 para [40])."""

    def _cont_ti(self):
        rng = np.random.default_rng(0)
        data = rng.random((20, 30)) * 10.0
        return gs.mps.TrainingImage(data, categorical=False), data

    def test_window_is_view_no_copy(self):
        ti, data = self._cont_ti()
        win = ti.window(np.s_[5:15, 10:25])
        self.assertEqual(win.shape, (10, 15))
        self.assertTrue(
            np.shares_memory(win.variable().data, ti.variable().data)
        )
        np.testing.assert_array_equal(win.variable().data, data[5:15, 10:25])

    def test_window_d_max_from_slice(self):
        # d_max of the view is the range of the SLICED data, not the parent's.
        data = np.zeros((10, 10))
        data[0, 0] = 100.0  # parent range 100, outside the window
        data[5:, 5:] = np.arange(25).reshape(5, 5)  # window range 24
        ti = gs.mps.TrainingImage(data, categorical=False)
        win = ti.window(np.s_[5:10, 5:10])
        self.assertEqual(win.variable().d_max, 24.0)
        self.assertEqual(ti.variable().d_max, 100.0)

    def test_window_preserves_config(self):
        rng = np.random.default_rng(1)
        ti = gs.mps.TrainingImage(
            [
                Variable(
                    "a",
                    rng.integers(0, 3, (12, 12)),
                    weight=2.0,
                    n_neighbors=7,
                ),
                Variable(
                    "b",
                    rng.random((12, 12)),
                    categorical=False,
                    distance="l2",
                    weight=1.0,
                    max_radius=4.0,
                ),
            ]
        )
        win = ti.window(np.s_[2:10, 0:6])
        self.assertEqual(win.shape, (8, 6))
        self.assertTrue(win.multivariate)
        self.assertEqual(win.weights, ti.weights)
        self.assertEqual(win.variable("a").n_neighbors, 7)
        self.assertEqual(win.variable("b").distance, "l2")
        self.assertEqual(win.variable("b").max_radius, 4.0)

    def test_open_ended_slices(self):
        ti, data = self._cont_ti()
        win = ti.window(np.s_[:10, 15:])
        np.testing.assert_array_equal(win.variable().data, data[:10, 15:])

    def test_strided_slice_raises(self):
        ti, _ = self._cont_ti()
        with self.assertRaisesRegex(ValueError, "step"):
            ti.window(np.s_[0:10:2, 0:10])

    def test_out_of_bounds_raises(self):
        ti, _ = self._cont_ti()
        with self.assertRaisesRegex(ValueError, "bounds"):
            ti.window(np.s_[0:25, 0:10])  # axis 0 has length 20
        with self.assertRaisesRegex(ValueError, "bounds"):
            ti.window(np.s_[-3:10, 0:10])  # negative indices rejected

    def test_degenerate_slice_raises(self):
        ti, _ = self._cont_ti()
        with self.assertRaisesRegex(ValueError, "degenerate|empty"):
            ti.window(np.s_[5:5, 0:10])

    def test_wrong_slice_count_raises(self):
        ti, _ = self._cont_ti()
        with self.assertRaisesRegex(ValueError, "slice"):
            ti.window(np.s_[0:10])

    def test_non_iterable_raises_valueerror(self):
        # Non-iterable arguments (int, None) must raise ValueError, not TypeError.
        ti, _ = self._cont_ti()
        with self.assertRaisesRegex(ValueError, "slice"):
            ti.window(5)
        with self.assertRaisesRegex(ValueError, "slice"):
            ti.window(None)

    def test_1d_single_slice(self):
        ti = gs.mps.TrainingImage(
            np.arange(50, dtype=float), categorical=False
        )
        win = ti.window(np.s_[10:30])
        self.assertEqual(win.shape, (20,))


class TestPenaltyMatrixVariable(unittest.TestCase):
    """Variable construction validation for penalty_matrix."""

    def test_default_none(self):
        v = Variable("x", np.array([0, 1, 0, 1]))
        self.assertIsNone(v.penalty_matrix)

    def test_accept_valid_asymmetric(self):
        t = np.array([[0.0, 0.5, 0.8], [0.3, 0.0, 0.6], [0.9, 0.4, 0.0]])
        v = Variable("x", np.array([0, 1, 2, 0]), penalty_matrix=t)
        np.testing.assert_array_equal(v.penalty_matrix, t)
        # not symmetric, and that's fine
        self.assertFalse(np.allclose(v.penalty_matrix, v.penalty_matrix.T))

    def test_penalty_matrix_read_only(self):
        t = 1.0 - np.eye(2)
        v = Variable("x", np.array([0, 1, 0, 1]), penalty_matrix=t)
        with self.assertRaises(ValueError):
            v.penalty_matrix[0, 1] = 0.1

    def test_reject_bad_shape(self):
        for t in (np.zeros((2, 3)), np.zeros((2, 2, 2))):
            with self.subTest(shape=t.shape):
                with self.assertRaisesRegex(ValueError, "square"):
                    Variable("x", np.array([0, 1]), penalty_matrix=t)

    def test_reject_nonzero_diagonal(self):
        t = np.array([[0.0, 0.5], [0.3, 0.1]])
        with self.assertRaisesRegex(ValueError, "diagonal"):
            Variable("x", np.array([0, 1, 0, 1]), penalty_matrix=t)

    def test_reject_bad_off_diagonal(self):
        # zero, negative, and >1 are all invalid off-diagonal penalties.
        for val in (0.0, -0.2, 1.5):
            t = np.array([[0.0, val], [0.3, 0.0]])
            with self.subTest(val=val):
                with self.assertRaisesRegex(ValueError, "off-diagonal"):
                    Variable("x", np.array([0, 1, 0, 1]), penalty_matrix=t)

    def test_reject_with_categorical_false(self):
        t = 1.0 - np.eye(2)
        with self.assertRaisesRegex(ValueError, "categorical"):
            Variable(
                "x",
                np.linspace(0.0, 1.0, 4),
                categorical=False,
                penalty_matrix=t,
            )

    def test_reject_bad_data_codes(self):
        # C = 3, valid codes {0, 1, 2}: non-integer, out-of-range, negative.
        t = 1.0 - np.eye(3)
        for data in ([0.0, 1.5, 2.0], [0.0, 1.0, 3.0], [0.0, -1.0, 2.0]):
            with self.subTest(data=data):
                with self.assertRaisesRegex(ValueError, "integer"):
                    Variable("x", np.array(data), penalty_matrix=t)

    def test_nan_data_ignored_by_range_check(self):
        # NaN cells are undefined and must not trip the integer/range check.
        t = 1.0 - np.eye(3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            v = Variable("x", np.array([0.0, np.nan, 2.0]), penalty_matrix=t)
        self.assertIsNotNone(v.penalty_matrix)

    def test_repr_includes_penalty_matrix_shape(self):
        t = 1.0 - np.eye(3)
        v = Variable("x", np.array([0, 1, 2, 0]), penalty_matrix=t)
        self.assertIn("penalty_matrix=(3, 3)", repr(v))
        v_none = Variable("y", np.array([0, 1, 2, 0]))
        self.assertNotIn("penalty_matrix", repr(v_none))

    def test_window_view_retains_penalty_matrix(self):
        t = np.array([[0.0, 0.5, 0.8], [0.3, 0.0, 0.6], [0.9, 0.4, 0.0]])
        rng = np.random.default_rng(2)
        ti = gs.mps.TrainingImage(
            [Variable("a", rng.integers(0, 3, (12, 12)), penalty_matrix=t)]
        )
        win = ti.window(np.s_[2:10, 0:6])
        np.testing.assert_array_equal(win.variable("a").penalty_matrix, t)

    def test_simulation_stays_subset_of_ti_codes(self):
        """End-to-end: a penalty_matrix must not invent values."""
        t = np.array([[0.0, 0.5, 0.8], [0.3, 0.0, 0.6], [0.9, 0.4, 0.0]])
        rng = np.random.default_rng(5)
        ti_data = rng.integers(0, 3, (25, 25)).astype(float)
        ti = TrainingImage(ti_data, n_neighbors=6, penalty_matrix=t)
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.5, threshold=0.1))
        out = ds([np.arange(15, dtype=float)] * 2, seed=42)
        self.assertTrue(
            set(np.unique(out).tolist()) <= set(np.unique(ti_data).tolist())
        )


class TestTrainingImageSemantics(unittest.TestCase):
    """Named-variable equivalence and common-support value adjustment."""

    def setUp(self):
        rng = np.random.default_rng(416)
        self.codes = rng.integers(0, 2, (12, 12)).astype(float)  # {0, 1}
        self.cont = rng.random((12, 12))
        self.pos = [np.arange(8.0)] * 2

    def test_named_variable_univariate_ti(self):
        # TrainingImage(Variable("f", ...)) must act like TrainingImage(array).
        named = TrainingImage(Variable("f", self.codes, n_neighbors=6))
        anon = TrainingImage(self.codes, n_neighbors=6)
        np.testing.assert_array_equal(
            named.variable().data, anon.variable().data
        )
        self.assertEqual(
            named.variable().categorical, anon.variable().categorical
        )
        self.assertEqual(named.variable().distance, anon.variable().distance)
        cond = ([[1.0, 4.0], [2.0, 5.0]], [5.0, 6.0])  # not TI values
        for conditioned in (False, True):
            fields = []
            for ti in (named, anon):
                ds = DirectSampling(MPSModel(ti, scan_fraction=0.5), seed=3)
                if conditioned:
                    with ignoring(UserWarning):  # values absent from the TI
                        ds.set_condition(*cond)
                fields.append(ds(self.pos, store=False))
            with self.subTest(conditioned=conditioned):
                np.testing.assert_array_equal(*fields)
        self.assertEqual((fields[0][1, 2], fields[0][4, 5]), (5.0, 6.0))


class TestTrainingImageInputValidation(unittest.TestCase):
    """Non-finite and inapplicable TI/Variable inputs are rejected."""

    def setUp(self):
        rng = np.random.default_rng(416)
        self.codes = rng.integers(0, 2, (12, 12)).astype(float)  # {0, 1}
        self.cont = rng.random((12, 12))

    def test_infinite_ti_values_rejected(self):
        for bad in (np.inf, -np.inf):
            data = self.cont.copy()
            data[0, 0] = bad
            builders = {
                "Variable": lambda: Variable("v", data, categorical=False),
                "TrainingImage": lambda: TrainingImage(
                    data, categorical=False
                ),
            }
            for name, build in builders.items():
                with self.subTest(name, value=bad):
                    with self.assertRaises(ValueError):
                        build()

    def test_distance_on_categorical_data_rejected(self):
        # `distance` only applies to continuous data; the categorical default
        # would silently ignore it.
        with self.assertRaises(ValueError):
            TrainingImage(self.cont, distance="l2")
        with self.assertRaises(ValueError):
            Variable("v", self.cont, distance="l2")
        Variable("v", self.cont, categorical=True, distance="l1")  # accepted

    def test_non_finite_parameters_rejected(self):
        nan, inf = np.nan, np.inf
        bad_matrix = np.array([[0, nan, 1], [1, 0, 1], [1, 1, 0.0]])

        def variable(field):
            return lambda v: Variable(
                "v", self.cont, categorical=False, **{field: v}
            )

        cases = {  # parameter: (builder taking the bad value, bad values)
            "distance": (
                lambda v: TrainingImage(
                    self.cont, categorical=False, distance=v
                ),
                ["lnan", "linf", "l-1", "variationinf", "variationnan"],
            ),
            "weight": (variable("weight"), [nan, -1, inf]),
            "max_radius": (variable("max_radius"), [nan, inf]),
            "distance_power": (variable("distance_power"), [nan, -1, inf]),
            "cond_weight": (variable("cond_weight"), [nan, -1, inf]),
            "penalty_matrix": (
                lambda v: Variable("v", self.codes, penalty_matrix=v),
                [bad_matrix],
            ),
        }
        for name, (build, values) in cases.items():
            for value in values:
                with self.subTest(parameter=name, value=value):
                    with self.assertRaises(ValueError):
                        build(value)


class TestExactIntegerInputs(unittest.TestCase):
    def test_fractional_n_neighbors_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            Variable("a", np.zeros((4, 4)), n_neighbors=2.9)

    def test_lossy_integer_data_rejected(self):
        data = np.array([[2**60 + 3, 0]], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "not exactly representable"):
            Variable("a", data)


class TestDataRange(unittest.TestCase):
    def test_float64_spread_overflow_rejected(self):
        with self.assertRaisesRegex(ValueError, "not finite"):
            _data_range(np.array([-1e308, 1e308]))

    def test_int8_full_range(self):
        self.assertEqual(
            _data_range(np.array([-128, 127], dtype=np.int8)), 255.0
        )

    def test_float32_wide_range_is_finite(self):
        result = _data_range(np.array([-3e38, 3e38], dtype=np.float32))
        self.assertTrue(np.isfinite(result))
        self.assertAlmostEqual(result / 6e38, 1.0, places=6)

    def test_64_bit_integer_extremes_use_exact_spread(self):
        full = float(2**64 - 1)
        self.assertEqual(
            _data_range(np.array([0, 2**64 - 1], dtype=np.uint64)), full
        )
        self.assertEqual(
            _data_range(np.array([-(2**63), 2**63 - 1], dtype=np.int64)), full
        )

    def test_constant_and_all_nan_return_one(self):
        self.assertEqual(_data_range(np.full(4, 2.5)), 1.0)
        self.assertEqual(_data_range(np.full(4, np.nan)), 1.0)


class TestOneVariableModel(unittest.TestCase):
    """A TrainingImage is always a list of named variables internally."""

    def setUp(self):
        self.d = (np.indices((12, 12)).sum(axis=0) % 3).astype(float)

    def test_array_input_named_field(self):
        ti = TrainingImage(self.d)
        self.assertEqual(ti.variables[0].name, "field")
        self.assertEqual(ti.weights, {"field": 1.0})
        self.assertFalse(ti.multivariate)

    def test_single_variable_weight_consistent(self):
        d = self.d
        self.assertEqual(
            TrainingImage(Variable("a", d, weight=5.0)).weights, {"a": 1.0}
        )
        for data in (
            Variable("a", d, weight=0.0),
            [Variable("a", d, weight=0.0)],
        ):
            with self.assertRaisesRegex(ValueError, "positive"):
                TrainingImage(data)

    def test_non_variable_sequence_rejected(self):
        with self.assertRaises(TypeError):
            TrainingImage([self.d, self.d])

    def test_tuple_of_variables_accepted(self):
        ti = TrainingImage((Variable("a", self.d), Variable("b", self.d)))
        self.assertEqual([v.name for v in ti.variables], ["a", "b"])

    def test_removed_single_variable_accessors(self):
        ti = TrainingImage(self.d)
        for attr in ("data", "categorical", "distance_type"):
            self.assertFalse(hasattr(ti, attr), attr)

    def test_sugar_forwards_only_given_kwargs(self):
        v = TrainingImage(self.d, n_neighbors=7).variable()
        self.assertEqual(v.n_neighbors, 7)
        self.assertIsNone(v.weight)

    def test_variable_name_must_be_identifier(self):
        with self.assertRaises(ValueError):
            Variable(None, self.d)


if __name__ == "__main__":
    unittest.main()
