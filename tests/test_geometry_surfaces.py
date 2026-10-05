"""This is the unittest of the SurfaceStack (elevation parameterisation)."""

import unittest
import warnings

import numpy as np

import gstools as gs
from gstools.geometry import (
    NO_LABEL,
    BoreholeData,
    Composite,
    HalfSpace,
    from_surface_points,
    from_well_logs,
)
from gstools.tools import generate_grid


def _h0(x):
    return np.sin(x / 7.0)


def _h1(x):
    return 0.5 + 0 * x


def _h2(x):
    return np.cos(x / 5.0) + 0.2


class TestGeometrySurfaces(unittest.TestCase):
    def setUp(self):
        self.model = gs.Gaussian(dim=1, var=1.0, len_scale=10)
        self.x = np.linspace(0, 100, 101)
        self.z = np.linspace(-3, 8, 45)

    def stack(self, relation="onlap", seed=11):
        surfs = [
            gs.Surface.from_model(self.model, mean=mu) for mu in (0, 2, 4)
        ]
        return gs.SurfaceStack(surfs, relation=relation, dim=2, seed=seed)

    def test_labels(self):
        # crossing constant/callable surfaces
        surfs = [gs.Surface(_h0), gs.Surface(_h1), gs.Surface(_h2)]
        pos = np.array([[0.0] * 5, [-0.5, 0.2, 0.7, 1.1, 1.3]])
        # x=0: h = (0, 0.5, 1.2)
        onlap = gs.SurfaceStack(surfs, relation="onlap", dim=2)
        np.testing.assert_array_equal(onlap.labels(pos), [-1, 0, 1, 1, 2])
        # x=35: h = (sin 5 = -0.96, 0.5, cos 7 + 0.2 = 0.95)
        pos = np.array([[35.0] * 3, [0.0, 0.7, 1.0]])
        np.testing.assert_array_equal(onlap.labels(pos), [0, 1, 2])
        # erode: f_i = min(f_{i+1}, h_i); x = 80: h = (-0.98, 0.5, -0.15)
        erode = gs.SurfaceStack(surfs, relation="erode", dim=2)
        ifc = erode.interfaces(np.array([[80.0]]))[:, 0]
        np.testing.assert_allclose(ifc, [np.sin(80 / 7), _h2(80), _h2(80)])
        onl = onlap.interfaces(np.array([[80.0]]))[:, 0]
        np.testing.assert_allclose(onl, [np.sin(80 / 7), 0.5, 0.5])
        for stack in (onlap, erode):
            ifaces = stack.interfaces(self.x[None, :])
            self.assertTrue(np.all(np.diff(ifaces, axis=0) >= 0))
            fac = gs.FaciesField(stack)
            np.testing.assert_array_equal(
                fac.structured((self.x, self.z)).ravel(),
                fac.unstructured(generate_grid((self.x, self.z))),
            )
        self.assertRaises(ValueError, gs.SurfaceStack, surfs[:1], dim=2)
        self.assertRaises(ValueError, gs.SurfaceStack, surfs, relation="x")

    def test_erode_equals_halfspaces(self):
        erode = gs.SurfaceStack(
            [gs.Surface(_h0), gs.Surface(_h1), gs.Surface(_h2)],
            relation="erode",
            dim=2,
            below_label=NO_LABEL,
            above_label=2,
        )
        comp = Composite(
            [
                HalfSpace(func, dim=2, label=i, priority=i)
                for i, func in enumerate((_h0, _h1, _h2))
            ]
        )
        pos = generate_grid((self.x, np.linspace(-2, 2, 81)))
        np.testing.assert_array_equal(erode.labels(pos), comp.labels(pos))

    def test_consistency(self):
        stack = self.stack()
        pts = np.random.RandomState(0).uniform(0, 100, (2, 300))
        full = stack.labels(pts)
        np.testing.assert_array_equal(full[:50], stack.labels(pts[:, :50]))
        ifaces = stack.interfaces(self.x[None, :])
        # true pinch-outs exist and are exact zeros
        self.assertTrue(np.any(ifaces[1] == ifaces[0]))
        idx, w_val = stack.strat_coords((self.x, self.z), "structured")
        self.assertTrue(np.all((w_val[np.isfinite(w_val)] >= 0)))
        self.assertTrue(np.all((w_val[np.isfinite(w_val)] < 1)))

    def test_condition(self):
        data, rej = from_surface_points(
            x=[10, 10, 10, 50, 50, 80, 80, 80],
            y=None,
            z=[0.5, 2.0, 4.0, 1.8, 3.9, -0.2, 3.0, 3.0],
            surface=[0, 1, 2, 1, 2, 0, 1, 2],
            surfaces=3,
            well=["A"] * 3 + ["B"] * 2 + ["C"] * 3,
            td={"B": 1.0},
        )
        self.assertEqual(rej, [])
        # B misses surface 0 (h_0 <= td); C: coincident contacts 1 and 2
        self.assertEqual(len(data), 9)
        self.assertEqual(np.sum(~data.is_equality), 2)
        for relation in ("onlap", "erode"):
            if relation == "erode":
                data, _ = from_surface_points(
                    x=[10, 10, 10, 80, 80, 80],
                    y=None,
                    z=[0.5, 2.0, 4.0, -0.2, 3.0, 3.0],
                    surface=[0, 1, 2, 0, 1, 2],
                    surfaces=3,
                    relation="erode",
                )
            stack = self.stack(relation=relation)
            stack.condition(data, seed=1)
            self.assertLess(stack.residuals(data).max(), 1e-8)
            ifaces = stack.interfaces(data.pos)
            # layer 1 absent at C: thickness exactly zero
            self.assertEqual((ifaces[2] - ifaces[1])[-1], 0.0)
        with self.assertRaises(ValueError):
            self.stack().condition(BoreholeData([[0.0]], 0, 1.0))

    def test_erode_preserves_contact_without_upper_picks(self):
        data, rejected = from_surface_points(
            [0.0], None, [0.0], [0], surfaces=3, relation="erode"
        )
        self.assertEqual(rejected, [])
        for j in (1, 2):
            np.testing.assert_array_equal(data.for_layer(j).lower, [0.0])
        stack = gs.SurfaceStack(
            [
                gs.Surface.from_model(self.model, mean=mean)
                for mean in (0.0, -2.0, -3.0)
            ],
            relation="erode",
            dim=2,
            seed=1,
        )
        stack.condition(data, seed=1)
        self.assertLess(stack.residuals(data).max(), 1e-8)
        self.assertAlmostEqual(stack.interfaces([[0.0]])[0, 0], 0.0)

        # Well logs retain the stronger collar bound, including a
        # contact picked exactly at the collar elevation.
        for depth in (0.0, 2.0):
            logs, rejected = from_well_logs(
                {"A": [depth, np.nan, np.nan]},
                layers=2,
                collar={"A": (0.0, 5.0)},
                kind="elevation",
                relation="erode",
            )
            self.assertEqual(rejected, [])
            for j in (1, 2):
                np.testing.assert_array_equal(logs.for_layer(j).lower, [5.0])

    def test_well_logs_elevation(self):
        tops = {"A": [5.0, np.nan, 1.0], "B": [np.nan, 2.0, 1.0]}
        collar = {"A": (10.0, 5.0), "B": (60.0, 6.0)}
        data, rej = from_well_logs(
            tops,
            layers=2,
            collar=collar,
            td={"B": 3.0},
            kind="elevation",
        )
        self.assertEqual(rej, [])
        rows = {
            (w, lay): (v, lo, up)
            for w, lay, v, lo, up in zip(
                data.well, data.layer, data.value, data.lower, data.upper
            )
        }
        self.assertEqual(rows[("A", 0)][0], 0.0)
        self.assertEqual(rows[("A", 1)][2], 4.0)  # missing pick: h_1 <= c_2
        self.assertEqual(rows[("B", 0)][2], 3.0)  # below td: h_0 <= z_td
        stack = gs.SurfaceStack(
            [gs.Surface.from_model(self.model, mean=m) for m in (0, 2, 4)],
            dim=2,
            seed=5,
        )
        stack.condition(data, seed=2)
        self.assertLess(stack.residuals(data).max(), 1e-8)
        with self.assertRaises(ValueError):
            from_well_logs(
                tops,
                layers=2,
                collar=collar,
                kind="elevation",
                stacking="down",
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, rej = from_surface_points(
                [0, 0], None, [2.0, 1.0], [0, 1], surfaces=2
            )
        self.assertEqual(len(rej), 1)  # out of order


if __name__ == "__main__":
    unittest.main()
