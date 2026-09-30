"""This is the unittest of the stochastic geometry base, links and stacks."""

import os
import tempfile
import unittest
from statistics import NormalDist

import numpy as np

import gstools as gs
from gstools.geometry import (
    NO_LABEL,
    Composite,
    HalfSpace,
    LogLink,
    Region,
    StratColumn,
    set_link,
)
from gstools.tools import generate_grid

HAS_PYVISTA = False
try:
    import pyvista  # noqa: F401

    HAS_PYVISTA = True
except ImportError:
    pass


class _Box(Region):
    def __init__(self, low, upp, **kwargs):
        super().__init__(len(low), **kwargs)
        self.low, self.upp = np.array(low), np.array(upp)

    def contains(self, pos):
        pos = np.asarray(pos, dtype=np.double).reshape(self.dim, -1)
        low, upp = self.low[:, None], self.upp[:, None]
        return np.all((pos >= low) & (pos < upp), axis=0)


class _Count:
    """Lateral callable recording the number of points per evaluation."""

    def __init__(self, func):
        self.func = func
        self.sizes = []

    def __call__(self, *lat):
        self.sizes.append(np.size(lat[0]))
        return self.func(*lat)


class TestGeometry(unittest.TestCase):
    def setUp(self):
        self.model = gs.Gaussian(dim=2, var=0.2, len_scale=20)
        self.x = np.linspace(0, 50, 23)
        self.y = np.linspace(0, 40, 17)
        self.z = np.linspace(-1, 9, 31)
        self.pts = np.random.RandomState(0).uniform(0, 50, (3, 500))
        self.pts[2] *= 0.2

    def stack(self, seed=7, **kwargs):
        layers = [
            gs.Layer.from_model(self.model, mean=np.log(2.0)) for _ in range(3)
        ]
        return gs.LayerStack(layers, dim=3, seed=seed, **kwargs)

    # --- base.py --------------------------------------------------------

    def test_region(self):
        box = _Box([0, 0], [1, 1], label=3)
        pos = np.array([[0.5, 1.5, 0.0], [0.5, 0.5, 0.99]])
        np.testing.assert_array_equal(box.contains(pos), [True, False, True])
        np.testing.assert_array_equal(box.labels(pos), [3, np.nan, 3])
        axes = (np.linspace(-1, 2, 7), np.linspace(-1, 2, 5))
        np.testing.assert_array_equal(
            box.labels(axes, mesh_type="structured"),
            box.labels(generate_grid(axes)),
        )
        self.assertIn("_Box", repr(box))

    def test_composite(self):
        old = _Box([0, 0], [2, 2], label=1, priority=0)
        young = _Box([1, 1], [3, 3], label=2, priority=1)
        pos = np.array([[0.5, 1.5, 2.5, 5.0], [0.5, 1.5, 2.5, 5.0]])
        comp = Composite([young, old])
        np.testing.assert_array_equal(comp.labels(pos), [1, 2, 2, np.nan])
        # ties keep insertion order
        tie_a = _Box([0, 0], [2, 2], label=1)
        tie_b = _Box([0, 0], [2, 2], label=5)
        self.assertEqual(Composite([tie_a, tie_b]).labels(pos)[0], 5)
        self.assertEqual(Composite([tie_b, tie_a]).labels(pos)[0], 1)
        # nesting
        nested = Composite([Composite([old]), _Box([4, 4], [6, 6], label=7)])
        np.testing.assert_array_equal(nested.labels(pos), [1, 1, np.nan, 7])
        axes = (np.linspace(-1, 4, 9), np.linspace(-1, 4, 8))
        np.testing.assert_array_equal(
            comp.labels(axes, mesh_type="structured"),
            comp.labels(generate_grid(axes)),
        )
        self.assertRaises(ValueError, Composite, [])
        self.assertRaises(ValueError, Composite, [old, _Box([0], [1])])

    def test_halfspace(self):
        above = HalfSpace(lambda x: 0.1 * x, dim=2, label=4)
        below = HalfSpace(lambda x: 0.1 * x, dim=2, label=4, below=True)
        pos = np.array([[0.0, 10.0, 10.0], [0.0, 0.5, 1.0]])
        np.testing.assert_array_equal(above.contains(pos), [True, False, True])
        np.testing.assert_array_equal(
            below.contains(pos), [False, True, False]
        )
        axes = (np.linspace(0, 10, 11), np.linspace(-1, 2, 13))
        for reg in (above, below):
            np.testing.assert_array_equal(
                reg.labels(axes, mesh_type="structured"),
                reg.labels(generate_grid(axes)),
            )
        srf = gs.SRF(gs.Gaussian(dim=1, nugget=0.1), seed=1)
        self.assertRaises(ValueError, HalfSpace, srf, dim=2)

    def test_strat_column(self):
        old = gs.LayerStack(
            [gs.Layer(np.log(2.0), name="a0"), gs.Layer(np.log(2.0))],
            base=0.0,
            dim=2,
        )  # interfaces 0, 2, 4
        mid = gs.SurfaceStack(
            [gs.Surface(lambda x: 1.0 + x / 50.0), gs.Surface(3.0)], dim=2
        )  # base 1 .. 3, top 3
        young = gs.LayerStack(
            [gs.Layer(np.log(1.5))], base=lambda x: 2.5 + 0 * x, dim=2
        )  # 2.5 .. 4
        col = StratColumn(
            [old, mid, young], ["erode", "onlap"], topography=5.0
        )
        pos = np.array(
            [
                [0, 0, 0, 0, 0, 0, 100, 100],
                [0.5, 1.5, 2.8, 3.5, 4.2, 6.0, 2.9, 3.2],
            ]
        )
        # 1.5: eroded by mid; 3.5: onlap fills; 4.2: old's top unit must
        # NOT reappear above the erosive group; 2.9 at x=100: below the
        # erosion surface, old unit 1 survives
        np.testing.assert_array_equal(
            col.labels(pos), [0, 2, 2, 3, 4, 4, 1, 3]
        )
        axes = (np.linspace(0, 100, 11), np.linspace(-1, 6, 29))
        np.testing.assert_array_equal(
            col.labels(axes, mesh_type="structured"),
            col.labels(generate_grid(axes)),
        )
        self.assertEqual(col.names, {0.0: "a0"})
        onlap = StratColumn([old, mid], ["onlap"])
        # onlap: old units kept, mid only above old's top (4)
        np.testing.assert_array_equal(
            onlap.labels(np.array([[0, 0], [1.5, 4.5]])), [0, 3]
        )
        self.assertRaises(ValueError, StratColumn, [old, mid], [])
        self.assertRaises(ValueError, StratColumn, [old, mid], ["foo"])

    # --- thickness.py ---------------------------------------------------

    def test_link(self):
        link = LogLink()
        thick = np.array([0.5, 1.0, 7.0])
        np.testing.assert_allclose(
            link.thickness(link.gauss_value(thick)), thick
        )
        self.assertTrue(np.isnan(link.gauss_value(0.0)))
        mu_g, var_g = LogLink.from_moments(5.0, 0.3)
        self.assertAlmostEqual(np.exp(mu_g + var_g / 2), 5.0)
        self.assertAlmostEqual(np.exp(var_g) - 1, 0.09)
        mu_g, var_g = LogLink.from_median(2.0, 0.5)
        self.assertAlmostEqual(np.exp(mu_g), 2.0)
        self.assertAlmostEqual(var_g, 0.25)
        # std instead of cv, exactly one of both
        np.testing.assert_allclose(
            LogLink.from_moments(5.0, std=1.5), LogLink.from_moments(5.0, 0.3)
        )
        self.assertRaises(ValueError, LogLink.from_moments, 5.0)
        self.assertRaises(ValueError, LogLink.from_moments, 5.0, 0.3, 1.5)
        # central interval: reproduces the given percentiles
        for prob, q_lo in ((0.8, 0.1), (0.9, 0.05)):
            mu_g, var_g = LogLink.from_percentiles(2.0, 8.0, prob=prob)
            z = NormalDist().inv_cdf(q_lo)
            sig = np.sqrt(var_g)
            self.assertAlmostEqual(np.exp(mu_g + z * sig), 2.0)
            self.assertAlmostEqual(np.exp(mu_g - z * sig), 8.0)
        self.assertAlmostEqual(np.exp(mu_g), 4.0)  # geometric midpoint
        self.assertRaises(ValueError, LogLink.from_percentiles, 8.0, 2.0)
        self.assertRaises(ValueError, LogLink.from_percentiles, 0.0, 2.0)
        self.assertRaises(ValueError, LogLink.from_percentiles, 2.0, 8.0, 1.0)
        low, upp = link.gauss_bounds([0.0, 1.0, 0.0], [np.inf, 2.0, 0.0])
        np.testing.assert_allclose(low, [-np.inf, 0.0, -np.inf])
        np.testing.assert_allclose(upp, [np.inf, np.log(2.0), np.log(1e-6)])
        self.assertIsInstance(set_link("log"), LogLink)
        self.assertIs(set_link(link), link)
        self.assertEqual(set_link(LogLink, absent=0.1).absent, 0.1)
        self.assertRaises(ValueError, set_link, "foo")
        self.assertRaises(ValueError, set_link, link, absent=1.0)
        self.assertIn("LogLink", repr(link))

    # --- layers.py, deterministic --------------------------------------

    def test_layers_deterministic(self):
        # thicknesses 1, 2, 0 (absent via -inf), 3 over base 0.5
        layers = [
            gs.Layer(0.0),
            gs.Layer(lambda x: np.log(2.0) + 0 * x),
            gs.Layer(-np.inf),
            gs.Layer(np.log(3.0), label=10),
        ]
        stack = gs.LayerStack(layers, base=0.5, dim=2, above_label=99)
        pos = np.array(
            [[0] * 7, [0.0, 0.5, 1.2, 1.5, 3.4, 3.5, 6.5]], dtype=float
        )
        # half-open [f_i, f_i+1): zero-thickness layer 2 is skipped
        np.testing.assert_array_equal(
            stack.labels(pos), [-1, 0, 0, 1, 1, 10, 99]
        )
        ifaces = stack.interfaces(np.zeros((1, 1)))
        np.testing.assert_allclose(ifaces[:, 0], [0.5, 1.5, 3.5, 3.5, 6.5])
        down = gs.LayerStack(layers, base=0.5, dim=2, stacking="down")
        np.testing.assert_allclose(
            down.interfaces(np.zeros((1, 1)))[:, 0],
            [0.5, -0.5, -2.5, -2.5, -5.5],
        )
        np.testing.assert_array_equal(
            down.labels(np.array([[0, 0, 0], [1.0, 0.0, -1.0]])), [-1, 0, 1]
        )
        # 3d
        stack3 = gs.LayerStack([gs.Layer(0.0)], base=lambda x, y: x, dim=3)
        pos3 = np.array([[1.0, 1.0, 2.0], [0.0, 0.0, 5.0], [0.5, 1.5, 2.5]])
        np.testing.assert_array_equal(stack3.labels(pos3), [-1, 0, 0])
        # transparent outside labels and mask
        clear = gs.LayerStack(
            layers[:1],
            base=0.0,
            dim=2,
            below_label=NO_LABEL,
            above_label=NO_LABEL,
            mask=lambda x: x < 5,
            outside_label=42,
        )
        pos = np.array([[0, 0, 0, 9], [-1.0, 0.5, 2.0, 0.5]])
        np.testing.assert_array_equal(
            clear.labels(pos), [np.nan, 0, np.nan, 42]
        )
        fac = gs.FaciesField(clear, background=-5)
        np.testing.assert_array_equal(fac(pos), [-5, 0, -5, 42])
        self.assertRaises(ValueError, gs.LayerStack, layers, stacking="x")
        self.assertRaises(ValueError, gs.LayerStack, [])

    def test_strat_coords(self):
        stack = gs.LayerStack(
            [gs.Layer(0.0), gs.Layer(np.log(2.0))], base=0.0, dim=2
        )
        pos = np.array([[0.0] * 5, [-1.0, 0.0, 0.5, 2.0, 3.5]])
        idx, w_val = stack.strat_coords(pos)
        np.testing.assert_array_equal(idx, [-1, 0, 0, 1, 2])
        np.testing.assert_allclose(w_val, [np.nan, 0.0, 0.5, 0.5, np.nan])
        axes = (np.linspace(0, 3, 4), np.linspace(-1, 4, 11))
        idx_s, w_s = stack.strat_coords(axes, mesh_type="structured")
        idx_u, w_u = stack.strat_coords(generate_grid(axes))
        np.testing.assert_array_equal(idx_s, idx_u)
        np.testing.assert_array_equal(w_s, w_u)
        region = stack.region(1)
        np.testing.assert_array_equal(region.contains(pos), [0, 0, 0, 1, 0])
        self.assertRaises(ValueError, stack.region, 2)

    # --- field.py -------------------------------------------------------

    def test_facies_field(self):
        stack = self.stack()
        fac = gs.FaciesField(stack)
        grid = fac.structured((self.x, self.y, self.z))
        self.assertEqual(grid.shape, (23, 17, 31))
        self.assertEqual(grid.dtype, np.int64)
        np.testing.assert_array_equal(
            grid.ravel(),
            fac.unstructured(generate_grid((self.x, self.y, self.z))),
        )
        self.assertEqual(fac.field_names, ["field"])
        fac(self.pts, store="foo")
        self.assertIn("foo", fac.field_names)
        fac.delete_fields()
        self.assertEqual(fac.field_names, [])
        self.assertIn("FaciesField", repr(fac))
        self.assertRaises(ValueError, fac, self.pts, store="__init__")
        frac = gs.FaciesField(_Box([0, 0], [1, 1], label=0.5), background=0)
        self.assertEqual(frac(np.array([[0.5], [0.5]])).dtype, np.double)
        named = gs.FaciesField(
            gs.LayerStack([gs.Layer(0.0, name="sand")], dim=2)
        )
        self.assertEqual(named.names, {0: "sand"})
        self.assertEqual(
            gs.FaciesField(_Box([0], [1]), names={1: "a"}).names, {1: "a"}
        )
        self.assertIsInstance(
            gs.FaciesField([_Box([0], [1])]).geometry, Composite
        )

    @unittest.skipIf(not HAS_PYVISTA, "pyvista not installed")
    def test_export(self):
        fac = gs.FaciesField(self.stack())
        fac.structured((self.x, self.y, self.z))
        with tempfile.TemporaryDirectory() as tmp:
            fac.vtk_export(os.path.join(tmp, "facies"))
            self.assertTrue(os.path.isfile(os.path.join(tmp, "facies.vtr")))
        surf = gs.geometry.interfaces_to_pyvista(
            fac.geometry, (self.x, self.y)
        )
        self.assertEqual(len(surf), 4)

    def test_topology(self):
        lab = np.array([[0, 0, 1], [0, 2, 1]])
        topo = gs.geometry.topology(lab)
        self.assertEqual(topo, {(0, 1): 1, (0, 2): 2, (1, 2): 1})
        self.assertEqual(gs.geometry.topology(lab, background=2), {(0, 1): 1})

    # --- Field-backed layers --------------------------------------------

    def test_consistency(self):
        stack = self.stack()
        fac = gs.FaciesField(stack)
        full = fac.unstructured(self.pts)
        np.testing.assert_array_equal(
            full[:37], fac.unstructured(self.pts[:, :37])
        )
        perm = np.random.RandomState(1).permutation(500)
        np.testing.assert_array_equal(
            full[perm], fac.unstructured(self.pts[:, perm])
        )
        self.assertTrue(
            np.all(np.diff(stack.interfaces(self.pts[:2]), axis=0) >= 0)
        )
        # the lateral collapse evaluates surfaces only at n_lat points
        lat = generate_grid((self.x, self.y))
        np.testing.assert_array_equal(
            stack.interfaces(lat), stack.interfaces(lat.copy())
        )

    def test_cache_and_reset(self):
        stack = self.stack()
        lat = self.pts[:2]
        first = stack.interfaces(lat)
        self.assertIs(stack._ifaces(lat), stack._ifaces(lat.copy()))
        stack.clear_cache()
        np.testing.assert_array_equal(first, stack.interfaces(lat))
        stack.reset(seed=8)
        self.assertFalse(np.array_equal(first, stack.interfaces(lat)))
        stack.reset(seed=7)
        np.testing.assert_array_equal(first, stack.interfaces(lat))
        stack.reset()  # keep master seed
        np.testing.assert_array_equal(first, stack.interfaces(lat))
        stack.reset(seed=None)  # random
        self.assertFalse(np.array_equal(first, stack.interfaces(lat)))

    def test_lateral_dedupe(self):
        # 20 vertical columns with 25 points each
        lat = np.random.RandomState(2).uniform(0, 50, (2, 20))
        pts = np.vstack(
            [np.repeat(lat, 25, axis=1), np.tile(np.linspace(-1, 9, 25), 20)]
        )
        base = _Count(lambda x, y: 0.01 * x)
        stack = self.stack(base=base)
        full = stack.labels(pts)
        self.assertEqual(base.sizes, [20])
        # same labels as evaluating every point on its own
        for i in np.random.RandomState(3).permutation(500)[:40]:
            self.assertEqual(full[i], stack.labels(pts[:, i : i + 1])[0])
        # per-position arrays are tied to the given positions: no dedupe
        arr = np.arange(500, dtype=float) * 1e-3
        stack = self.stack(base=arr)
        np.testing.assert_array_equal(stack.interfaces(pts[:2])[0], arr)
        self.assertEqual(stack.labels(pts).shape, (500,))
        # structured grids are never deduplicated
        base = _Count(lambda x, y: 0.0 * x)
        self.stack(base=base).labels((self.x, self.y, self.z), "structured")
        self.assertEqual(base.sizes, [len(self.x) * len(self.y)])

    def test_surface_cache(self):
        lat = np.random.RandomState(2).uniform(0, 50, (2, 20))
        pts = np.vstack([np.repeat(lat, 5, axis=1), np.zeros(100)])
        surf = _Count(lambda x, y: 0.01 * x - 0.2)
        half = HalfSpace(surf, dim=3, label=4)
        first = half.labels(pts)
        np.testing.assert_array_equal(first, half.labels(pts.copy()))
        self.assertEqual(surf.sizes, [20])
        half.clear_cache()
        np.testing.assert_array_equal(first, half.labels(pts))
        self.assertEqual(surf.sizes, [20, 20])
        # topography of a column: cached and deduplicated as well
        topo = _Count(lambda x, y: 4.0 + 0.0 * x)
        col = StratColumn([self.stack()], topography=topo)
        first = col.labels(pts)
        np.testing.assert_array_equal(first, col.labels(pts))
        self.assertEqual(topo.sizes, [20])
        col.clear_cache()
        col.labels(pts)
        self.assertEqual(topo.sizes, [20, 20])
        col.labels((self.x, self.y, self.z), "structured")
        self.assertEqual(topo.sizes[-1], len(self.x) * len(self.y))

    def test_derived_seeds(self):
        stack = self.stack(seed=3)
        thick = stack.thicknesses(self.pts[:2])
        corr = np.corrcoef(thick[0], thick[1])[0, 1]
        self.assertLess(abs(corr), 0.99)
        seeds = [lay.seed for lay in stack.layers]
        self.assertEqual(len(set(seeds)), len(seeds))
        same = [gs.Layer.from_model(self.model, seed=5) for _ in range(2)]
        with self.assertWarns(UserWarning):
            gs.LayerStack(same, dim=3)

    def test_guards(self):
        stack = self.stack()
        fac = gs.FaciesField(stack)
        fac(self.pts)
        self.model.nugget = 0.1  # mutated after construction
        self.assertRaises(ValueError, fac, self.pts)
        self.model.nugget = 0.0
        fac(self.pts)
        nug = gs.Gaussian(dim=2, nugget=0.1)
        self.assertRaises(
            ValueError, self.stack().__class__, [gs.Layer.from_model(nug)]
        )
        wrong_dim = gs.Layer.from_model(gs.Gaussian(dim=3))
        self.assertRaises(ValueError, gs.LayerStack, [wrong_dim], dim=3)
        temporal = gs.Gaussian(dim=1, temporal=True)
        self.assertRaises(
            ValueError, gs.LayerStack, [gs.Layer.from_model(temporal)], dim=3
        )
        latlon = gs.Gaussian(latlon=True)
        self.assertRaises(
            ValueError, gs.LayerStack, [gs.Layer.from_model(latlon)], dim=3
        )
        normed = gs.SRF(self.model, normalizer=gs.normalizer.LogNormal)
        with self.assertWarns(UserWarning):
            gs.LayerStack([gs.Layer(normed)], dim=3)
        self.assertIn("mean_thickness=", repr(stack.layers[0]))
        self.assertIn("thickness=", repr(gs.Layer(0.0)))


if __name__ == "__main__":
    unittest.main()
