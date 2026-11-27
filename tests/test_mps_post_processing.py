"""Tests for gstools.mps.post_processing."""

import unittest
import warnings

import numpy as np
from mps_fixtures import as_dict

import gstools as gs
from gstools.mps.direct_sampling import DirectSampling
from gstools.mps.model import MPSModel
from gstools.mps.post_processing import PostProcessing
from gstools.mps.training_image import SEARCH_FIELDS, TrainingImage, Variable


class TestPostProcessingValidation(unittest.TestCase):
    def test_defaults_and_read_back(self):
        post = PostProcessing(2)
        self.assertEqual(post.passes, 2)
        self.assertEqual(post.factor, 1.0)
        self.assertIsNone(post.path)
        self.assertEqual(post.overrides, {})
        self.assertIsNone(post.mask)

    def test_all_fields_read_back(self):
        mask = np.ones((4, 4), dtype=bool)
        post = PostProcessing(
            3,
            factor=2,
            path="same",
            overrides={"a": {"n_neighbors": 4}},
            mask=mask,
        )
        self.assertEqual(
            (post.passes, post.factor, post.path), (3, 2.0, "same")
        )
        self.assertEqual(post.overrides, {"a": {"n_neighbors": 4}})
        np.testing.assert_array_equal(post.mask, mask)

    def test_mask_and_path_are_read_only_copies(self):
        mask = np.ones((4, 4), dtype=bool)
        path = np.arange(8).reshape(4, 2)
        post = PostProcessing(1, path=path, mask=mask)
        mask[0, 0] = False
        path[0, 0] = 99
        self.assertTrue(post.mask[0, 0])
        self.assertEqual(post.path[0, 0], 0)
        self.assertFalse(post.mask.flags.writeable)
        self.assertFalse(post.path.flags.writeable)

    def test_invalid_arguments_raise_value_error(self):
        cases = {
            "passes 0.9": dict(passes=0.9),
            "passes 0": dict(passes=0),
            "passes -1": dict(passes=-1),
            "passes nan": dict(passes=np.nan),
            "factor 0": dict(passes=1, factor=0),
            "factor -1": dict(passes=1, factor=-1),
            "factor nan": dict(passes=1, factor=np.nan),
            "factor inf": dict(passes=1, factor=np.inf),
            "factor subnormal": dict(passes=1, factor=1e-320),
            "unknown field": dict(passes=1, overrides={"a": {"colour": 1}}),
            "data override": dict(passes=1, overrides={"a": {"data": 1}}),
            "non-bool mask": dict(passes=1, mask=np.ones((3, 3))),
            "bad path name": dict(passes=1, path="bogus"),
            "1-D path": dict(passes=1, path=np.array([1, 2, 3])),
        }
        for name, kwargs in cases.items():
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    PostProcessing(**kwargs)

    def test_fractional_passes_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            PostProcessing(1.5)

    def test_overrides_must_be_nested_dict(self):
        for bad in ([("a", {})], {"a": [("n_neighbors", 4)]}, {1: {}}):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    PostProcessing(1, overrides=bad)

    def test_every_search_field_is_accepted(self):
        PostProcessing(1, overrides={"a": {f: None for f in SEARCH_FIELDS}})

    def test_overrides_are_copied(self):
        given = {"a": {"n_neighbors": 4}}
        post = PostProcessing(1, overrides=given)
        given["a"]["n_neighbors"] = 99
        post.overrides["a"]["n_neighbors"] = 77
        self.assertEqual(post.overrides, {"a": {"n_neighbors": 4}})

    def test_properties_are_read_only(self):
        post = PostProcessing(1)
        for name in ("passes", "factor", "path", "overrides", "mask"):
            with self.subTest(name):
                with self.assertRaises(AttributeError):
                    setattr(post, name, 1)

    def test_repr_shows_non_defaults(self):
        self.assertEqual(repr(PostProcessing(2)), "PostProcessing(2)")
        text = repr(PostProcessing(2, factor=2, path="same"))
        self.assertEqual(text, "PostProcessing(2, factor=2.0, path='same')")


class TestPostPassesKeepConditioning(unittest.TestCase):
    def test_post_passes_never_change_conditioned_values(self):
        # Multivariate: one node conditioned in "a" only. Joint post-passes
        # may re-simulate "b" there but must leave the datum of "a" alone.
        rng = np.random.default_rng(7)
        ti = TrainingImage(
            [
                Variable("a", rng.integers(0, 3, (20, 20)).astype(float)),
                Variable("b", rng.integers(0, 2, (20, 20)).astype(float)),
            ]
        )
        mask = np.ones((10, 10), dtype=bool)
        model = MPSModel(
            ti,
            scan_fraction=0.5,
            threshold=0.1,
            post_processing=PostProcessing(2, mask=mask),
        )
        ds = DirectSampling(model, seed=3)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # partial NaN drops nothing
            ds.set_condition([[4.0], [5.0]], {"a": [2.0], "b": [np.nan]})
        out = ds([np.arange(10.0)] * 2, store=False)
        out = as_dict(ds, out)
        self.assertEqual(out["a"][4, 5], 2.0)


class TestPostProcessingExports(unittest.TestCase):
    def test_exported_from_mps(self):
        self.assertIs(gs.mps.PostProcessing, PostProcessing)
        self.assertIn("PostProcessing", gs.mps.__all__)


if __name__ == "__main__":
    unittest.main()
