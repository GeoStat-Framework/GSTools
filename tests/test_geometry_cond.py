"""This is the unittest of borehole data and conditioning of a LayerStack."""

import unittest
import warnings

import numpy as np

import gstools as gs
from gstools.geometry import BoreholeData, from_well_logs


class TestGeometryCond(unittest.TestCase):
    def setUp(self):
        self.model = gs.Gaussian(dim=2, var=0.2, len_scale=20)
        # contacts (MD below collar) of interfaces 0..3, bottom to top
        self.tops = {
            "A": [5.0, 3.0, 1.0, 0.0],  # z = 0, 2, 4, 5
            "B": [6.0, 4.5, 4.5, 2.0],  # layer 1 absent
            "C": [np.nan, np.nan, 3.0, 1.0],  # ends inside layer 1
        }
        self.collar = {
            "A": (10, 10, 5.0),
            "B": (30, 20, 6.0),
            "C": (40, 40, 7.0),
        }
        self.td = {"C": 4.5}

    def stack(self, base=None, seed=7):
        layers = [
            gs.Layer.from_model(self.model, mean=np.log(2.0)) for _ in range(3)
        ]
        if base is None:
            base = gs.SRF(gs.Gaussian(dim=2, var=0.5, len_scale=30))
        return gs.LayerStack(layers, base=base, dim=3, seed=seed)

    def data(self, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return from_well_logs(
                self.tops, layers=3, collar=self.collar, td=self.td, **kwargs
            )

    def test_borehole_data(self):
        pos = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
        data = BoreholeData(
            pos, [0, 1, 1], [2.0, np.nan, 0.0], lower=[0, 1, 0]
        )
        self.assertEqual(len(data), 3)
        self.assertEqual(data.kind, "thickness")
        # zero thickness is "absent": bound t <= 0
        self.assertTrue(np.isnan(data.value[2]))
        self.assertEqual(data.upper[2], 0.0)
        self.assertEqual(len(data.for_layer(1)), 2)
        self.assertEqual(len(data[data.layer == 0]), 1)
        self.assertIn("BoreholeData", repr(data))
        with self.assertRaises(ValueError):  # neither equality nor bounded
            BoreholeData(pos, [0, 1, 1], [2.0, np.nan, 1.0])
        with self.assertRaises(ValueError):  # value above upper
            BoreholeData(pos, 0, [2.0, 1.0, 1.0], upper=1.5)
        with self.assertRaises(ValueError):  # negative thickness bound
            BoreholeData(pos, 0, np.nan, lower=-1.0, upper=1.0)
        with self.assertRaises(ValueError):  # non-finite position
            BoreholeData([[np.nan], [0.0]], 0, 1.0)
        with self.assertRaises(ValueError):  # length mismatch
            BoreholeData(pos, [0, 1], 1.0)
        with self.assertRaises(ValueError):
            BoreholeData(pos, 0, 1.0, kind="foo")
        elev = BoreholeData(
            pos, 0, [np.nan, 1.0, 1.0], kind="elevation", upper=[0, 9, 9]
        )
        self.assertEqual(elev.lower[0], -np.inf)

    def test_merge_duplicates(self):
        pos = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
        data = BoreholeData(
            pos,
            0,
            [2.0, 2.0, np.nan, 1.0],
            lower=[0, 0, 1.0, 0],
            well=["w1", "w2", "w3", "w4"],
        )
        merged = data.merge_duplicates()
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged.value[0], 2.0)
        self.assertEqual(merged.lower[0], 1.0)
        self.assertEqual(merged.well[0], "w1+w2+w3")
        conflict = BoreholeData(pos[:, :2], 0, [2.0, 3.0], well=["P", "Q"])
        with self.assertRaisesRegex(ValueError, "#P\\+#Q"):
            conflict.merge_duplicates()
        empty = BoreholeData(
            pos[:, :2], 0, np.nan, lower=[2.0, 0.0], upper=[3.0, 1.0]
        )
        self.assertRaises(ValueError, empty.merge_duplicates)

    def test_from_well_logs(self):
        data, rejected = self.data()
        # C: contact 2 has unobserved lower contacts -> sum of thicknesses
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["well"], "C")
        eq = data.is_equality
        rows = {
            (w, lay): (v, lo, up)
            for w, lay, v, lo, up in zip(
                data.well, data.layer, data.value, data.lower, data.upper
            )
        }
        self.assertEqual(rows[("A", -1)][0], 0.0)  # base datum
        self.assertEqual(rows[("A", 0)][0], 2.0)
        self.assertEqual(rows[("A", 2)][0], 1.0)
        self.assertTrue(np.isnan(rows[("B", 1)][0]))  # absent
        self.assertEqual(rows[("B", 1)][2], 0.0)
        self.assertEqual(rows[("C", 2)][0], 2.0)
        self.assertEqual(rows[("C", 1)][1], 1.5)  # t >= partial thickness
        self.assertEqual(np.sum(~eq), 2)
        with self.assertRaises(ValueError):
            self.data(strict=True)
        # out of order contacts are reported, never dropped silently
        tops = dict(self.tops, D=[1.0, 2.0, np.nan, np.nan])
        collar = dict(self.collar, D=(0, 0, 0.0))
        with self.assertWarns(UserWarning):
            _, rej = from_well_logs(tops, layers=3, collar=collar)
        self.assertIn("D", [r["well"] for r in rej])

    def test_erosion_flags(self):
        # absent-and-eroded must be explicit once erosion is enabled
        with self.assertRaises(ValueError):
            self.data(eroded={"A": []})
        data, _ = self.data(eroded={"A": [], "B": [1], "C": []})
        self.assertFalse(np.any((data.well == "B") & (data.layer == 1)))
        data, _ = self.data(eroded={"A": [2], "B": [], "C": []})
        sel = (data.well == "A") & (data.layer == 2)
        self.assertTrue(data.truncated[sel][0])
        self.assertEqual(data.lower[sel][0], 1.0)

    def test_condition_exact(self):
        data, _ = self.data()
        stack = self.stack()
        stack.condition(data, seed=42)
        res = stack.residuals(data)
        self.assertLess(res.max(), 1e-8)
        # logged labels at the wells (away from the contacts)
        fac = gs.FaciesField(stack)
        pnt = np.array(
            [[10, 10, 0.5], [10, 10, 3.0], [10, 10, 4.5], [30, 20, 3.0]]
        ).T
        np.testing.assert_array_equal(fac.unstructured(pnt), [0, 1, 2, 2])
        # exact contact reproduction via interfaces
        ifaces = stack.interfaces(np.array([[10.0], [10.0]]))[:, 0]
        np.testing.assert_allclose(ifaces, [0, 2, 4, 5], atol=1e-8)
        # layer 0 is only reached by wells A and B
        self.assertEqual(stack.layers[0].field.krige.cond_no, 2)
        self.assertEqual(stack.layers[2].field.krige.cond_no, 3)

    def test_condition_inequality(self):
        data, _ = self.data()
        stack = self.stack()
        stack.condition(data, seed=1)
        thick = stack.thicknesses(np.array([[40.0], [40.0]]))
        self.assertGreaterEqual(thick[1, 0], 1.5)
        first = stack.interfaces(np.array([[40.0], [40.0]]))
        stack.resample_inequalities(seed=1)  # same seed, same draw
        np.testing.assert_allclose(
            first, stack.interfaces(np.array([[40.0], [40.0]]))
        )
        stack.resample_inequalities(seed=2)
        self.assertLess(stack.residuals(data).max(), 1e-8)
        with self.assertRaises(ValueError):  # inequality with unknown mean
            self.stack().condition(data, unbiased=True)

    def test_stale_cache(self):
        data, _ = self.data()
        stack = self.stack()
        pnt = np.array([[20.0, 35.0], [15.0, 30.0]])
        stack.condition(data, seed=3)
        first = stack.interfaces(pnt)
        tops = {w: np.asarray(t) + 0.5 for w, t in self.tops.items()}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            other, _ = from_well_logs(
                tops, layers=3, collar=self.collar, td=self.td
            )
        stack.condition(other, seed=3)
        self.assertFalse(np.allclose(first, stack.interfaces(pnt)))
        self.assertLess(stack.residuals(other).max(), 1e-8)

    def test_condition_errors(self):
        data, _ = self.data()
        det = self.stack(base=0.0)
        with self.assertWarns(UserWarning):  # base data ignored
            det.condition(data)
        const = gs.LayerStack([gs.Layer(0.0)] * 3, dim=3)
        with self.assertWarns(UserWarning):
            const.condition(data)
        elev = BoreholeData([[0.0], [0.0]], 0, 1.0, kind="elevation")
        self.assertRaises(ValueError, self.stack().condition, elev)
        bad = BoreholeData([[0.0], [0.0]], 5, 1.0)
        self.assertRaises(ValueError, self.stack().condition, bad)
        noisy = BoreholeData([[0.0], [0.0]], 0, 1.0, err=0.1)
        self.assertRaises(NotImplementedError, self.stack().condition, noisy)
        # a user-built CondSRF layer is re-conditioned in place
        krige = gs.Krige(self.model, [[0.0], [0.0]], [0.5], mean=0.0)
        cond = gs.CondSRF(krige, cond_method="error", seed=2)
        stack = gs.LayerStack(
            [gs.Layer(cond)] + self.stack().layers[1:], dim=3, base=0.0
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stack.condition(data)
        self.assertIs(stack.layers[0].field, cond)
        self.assertEqual(cond.krige.cond_no, 2)


if __name__ == "__main__":
    unittest.main()
