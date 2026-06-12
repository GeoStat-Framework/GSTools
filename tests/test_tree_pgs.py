"""
This is the unittest for tree-based PGS functionality.
"""

import unittest

import numpy as np

import gstools as gs


class TestTreePGS(unittest.TestCase):
    """Test tree-based plurigaussian simulation."""

    def test_tree_2d_simple(self):
        """Test 2D tree-based PGS with simple ellipse decision."""
        n1 = 100
        n2 = 100
        pos = [np.arange(n1), np.arange(n2)]

        model1 = gs.Gaussian(dim=2, var=1, len_scale=10)
        srf1 = gs.SRF(model1, seed=20170519)
        field1 = srf1.structured(pos)

        model2 = gs.Gaussian(dim=2, var=1, len_scale=10)
        srf2 = gs.SRF(model2, seed=20160519)
        field2 = srf2.structured(pos)

        # Define a simple ellipse decision function
        def ellipse(data, key1, key2, c1, c2, s1, s2):
            x, y = data[key1] - c1, data[key2] - c2
            return (x / s1) ** 2 + (y / s2) ** 2 <= 1

        # Simple tree with one decision and two leaf nodes
        config = {
            "root": {
                "type": "decision",
                "func": ellipse,
                "args": {
                    "key1": "Z1",
                    "key2": "Z2",
                    "c1": 0,
                    "c2": 0,
                    "s1": 2.0,
                    "s2": 2.0,
                },
                "yes_branch": "phase1",
                "no_branch": "phase0",
            },
            "phase1": {"type": "leaf", "action": 1},
            "phase0": {"type": "leaf", "action": 0},
        }

        pgs = gs.PGS(2, [field1, field2])
        P = pgs(tree=config)

        # Check that we get valid phase values
        self.assertTrue(np.all((P == 0) | (P == 1)))
        # Check that both phases are present
        self.assertTrue(0 in P)
        self.assertTrue(1 in P)
        # Check shape matches
        self.assertEqual(P.shape, (n1, n2))

    def test_tree_2d_multi_phase(self):
        """Test 2D tree-based PGS with multiple phases."""
        n1 = 80
        n2 = 80
        pos = [np.arange(n1), np.arange(n2)]

        model = gs.Gaussian(dim=2, var=1, len_scale=10)
        srf = gs.SRF(model)
        field1 = srf.structured(pos, seed=12345)
        field2 = srf.structured(pos, seed=54321)

        def threshold(data, key, value):
            return data[key] > value

        # Tree with multiple decisions creating 3 phases
        config = {
            "root": {
                "type": "decision",
                "func": threshold,
                "args": {"key": "Z1", "value": 0.5},
                "yes_branch": "phase2",
                "no_branch": "node1",
            },
            "node1": {
                "type": "decision",
                "func": threshold,
                "args": {"key": "Z2", "value": -0.5},
                "yes_branch": "phase1",
                "no_branch": "phase0",
            },
            "phase2": {"type": "leaf", "action": 2},
            "phase1": {"type": "leaf", "action": 1},
            "phase0": {"type": "leaf", "action": 0},
        }

        pgs = gs.PGS(2, [field1, field2])
        P = pgs(tree=config)

        # Check valid phases
        self.assertTrue(np.all((P >= 0) & (P <= 2)))
        # Check shape
        self.assertEqual(P.shape, (n1, n2))
        # Check that all three phases exist
        self.assertTrue(0 in P)
        self.assertTrue(1 in P)
        self.assertTrue(2 in P)

    def test_tree_3d(self):
        """Test 3D tree-based PGS."""
        n1 = 30
        n2 = 30
        n3 = 30
        pos = [np.arange(n1), np.arange(n2), np.arange(n3)]

        model1 = gs.Gaussian(dim=3, var=1, len_scale=10)
        srf1 = gs.SRF(model1, seed=20170519)
        field1 = srf1.structured(pos)

        model2 = gs.Gaussian(dim=3, var=1, len_scale=10)
        srf2 = gs.SRF(model2, seed=20160519)
        field2 = srf2.structured(pos)

        model3 = gs.Gaussian(dim=3, var=1, len_scale=10)
        srf3 = gs.SRF(model3, seed=20150519)
        field3 = srf3.structured(pos)

        def sphere(data, key1, key2, key3, c1, c2, c3, r):
            x = data[key1] - c1
            y = data[key2] - c2
            z = data[key3] - c3
            return x**2 + y**2 + z**2 <= r**2

        config = {
            "root": {
                "type": "decision",
                "func": sphere,
                "args": {
                    "key1": "Z1",
                    "key2": "Z2",
                    "key3": "Z3",
                    "c1": 0,
                    "c2": 0,
                    "c3": 0,
                    "r": 1.5,
                },
                "yes_branch": "phase1",
                "no_branch": "phase0",
            },
            "phase1": {"type": "leaf", "action": 1},
            "phase0": {"type": "leaf", "action": 0},
        }

        pgs = gs.PGS(3, [field1, field2, field3])
        P = pgs(tree=config)

        # Check valid phases
        self.assertTrue(np.all((P == 0) | (P == 1)))
        # Check shape
        self.assertEqual(P.shape, (n1, n2, n3))
        # Check both phases present
        self.assertTrue(0 in P)
        self.assertTrue(1 in P)

    def test_tree_multi_field(self):
        """Test tree-based PGS with more fields than dimensions."""
        n1 = 60
        n2 = 60
        pos = [np.arange(n1), np.arange(n2)]

        model = gs.Gaussian(dim=2, var=1, len_scale=10)
        srf = gs.SRF(model)
        field1 = srf.structured(pos, seed=111)
        field2 = srf.structured(pos, seed=222)
        field3 = srf.structured(pos, seed=333)
        field4 = srf.structured(pos, seed=444)

        def ellipse(data, key1, key2, c1, c2, s1, s2):
            x, y = data[key1] - c1, data[key2] - c2
            return (x / s1) ** 2 + (y / s2) ** 2 <= 1

        # Use all 4 fields in decision tree
        config = {
            "root": {
                "type": "decision",
                "func": ellipse,
                "args": {
                    "key1": "Z1",
                    "key2": "Z2",
                    "c1": 0.5,
                    "c2": 0.5,
                    "s1": 1.5,
                    "s2": 1.5,
                },
                "yes_branch": "phase1",
                "no_branch": "node1",
            },
            "node1": {
                "type": "decision",
                "func": ellipse,
                "args": {
                    "key1": "Z3",
                    "key2": "Z4",
                    "c1": -0.5,
                    "c2": -0.5,
                    "s1": 1.5,
                    "s2": 1.5,
                },
                "yes_branch": "phase2",
                "no_branch": "phase0",
            },
            "phase2": {"type": "leaf", "action": 2},
            "phase1": {"type": "leaf", "action": 1},
            "phase0": {"type": "leaf", "action": 0},
        }

        pgs = gs.PGS(2, [field1, field2, field3, field4])
        P = pgs(tree=config)

        # Check valid phases
        self.assertTrue(np.all((P >= 0) & (P <= 2)))
        # Check shape
        self.assertEqual(P.shape, (n1, n2))

    def test_compute_lithotype(self):
        """Test compute_lithotype method."""
        n1 = 50
        n2 = 50
        pos = [np.arange(n1), np.arange(n2)]

        model = gs.Gaussian(dim=2, var=1, len_scale=10)
        srf = gs.SRF(model)
        field1 = srf.structured(pos, seed=98765)
        field2 = srf.structured(pos, seed=56789)

        def threshold(data, key, value):
            return data[key] > value

        config = {
            "root": {
                "type": "decision",
                "func": threshold,
                "args": {"key": "Z1", "value": 0},
                "yes_branch": "phase1",
                "no_branch": "phase0",
            },
            "phase1": {"type": "leaf", "action": 1},
            "phase0": {"type": "leaf", "action": 0},
        }

        pgs = gs.PGS(2, [field1, field2])

        # Test computing lithotype with tree config
        L = pgs.compute_lithotype(tree=config)
        self.assertEqual(L.shape, (n1, n2))
        self.assertTrue(np.all((L == 0) | (L == 1)))

        # Test computing P and then lithotype without passing tree again
        P = pgs(tree=config)
        self.assertEqual(P.shape, (n1, n2))
        L2 = pgs.compute_lithotype()
        self.assertEqual(L2.shape, (n1, n2))
        self.assertTrue(np.all((L2 == 0) | (L2 == 1)))

    def test_assertions(self):
        """Test error handling for tree-based PGS."""
        n = 30
        pos = [np.arange(n), np.arange(n)]

        model = gs.Gaussian(dim=2, var=1, len_scale=10)
        srf = gs.SRF(model)
        field1 = srf.structured(pos, seed=111)
        field2 = srf.structured(pos, seed=222)

        pgs = gs.PGS(2, [field1, field2])

        # Test error when neither lithotypes nor tree provided
        self.assertRaises(ValueError, pgs)

        # Test error when trying to compute lithotype without tree
        self.assertRaises(ValueError, pgs.compute_lithotype)

    def test_tree_config_missing_root(self):
        """Test that tree config without root raises error."""
        n = 30
        pos = [np.arange(n), np.arange(n)]

        model = gs.Gaussian(dim=2, var=1, len_scale=10)
        srf = gs.SRF(model, seed=123)
        field1 = srf.structured(pos)
        field2 = srf.structured(pos)

        # Config without 'root' key
        bad_config = {
            "node1": {"type": "leaf", "action": 0},
        }

        pgs = gs.PGS(2, [field1, field2])
        self.assertRaises(KeyError, pgs, tree=bad_config)

    def test_tree_1d(self):
        """Test 1D tree-based PGS."""
        n = 100
        x = np.arange(n)

        model = gs.Gaussian(dim=1, var=1, len_scale=10)
        srf = gs.SRF(model, seed=42)
        field = srf.structured((x,))

        def threshold(data, key, value):
            return data[key] > value

        config = {
            "root": {
                "type": "decision",
                "func": threshold,
                "args": {"key": "Z1", "value": 0},
                "yes_branch": "phase1",
                "no_branch": "phase0",
            },
            "phase1": {"type": "leaf", "action": 1},
            "phase0": {"type": "leaf", "action": 0},
        }

        pgs = gs.PGS(1, field)
        P = pgs(tree=config)

        # Check valid phases
        self.assertTrue(np.all((P == 0) | (P == 1)))
        # Check shape
        self.assertEqual(P.shape, (n,))
        # Check both phases present
        self.assertTrue(0 in P)
        self.assertTrue(1 in P)

    def test_tree_rotated_ellipse(self):
        """Test tree with rotated ellipse decision boundary."""
        n1 = 80
        n2 = 80
        pos = [np.arange(n1), np.arange(n2)]

        model = gs.Gaussian(dim=2, var=1, len_scale=10)
        srf = gs.SRF(model)
        field1 = srf.structured(pos, seed=777)
        field2 = srf.structured(pos, seed=888)

        def ellipse_rotated(data, key1, key2, c1, c2, s1, s2, angle):
            x, y = data[key1] - c1, data[key2] - c2
            theta = np.deg2rad(angle)
            c, s = np.cos(theta), np.sin(theta)
            x_rot, y_rot = x * c + y * s, -x * s + y * c
            return (x_rot / s1) ** 2 + (y_rot / s2) ** 2 <= 1

        config = {
            "root": {
                "type": "decision",
                "func": ellipse_rotated,
                "args": {
                    "key1": "Z1",
                    "key2": "Z2",
                    "c1": 0,
                    "c2": 0,
                    "s1": 2.5,
                    "s2": 0.8,
                    "angle": -45,
                },
                "yes_branch": "phase1",
                "no_branch": "phase0",
            },
            "phase1": {"type": "leaf", "action": 1},
            "phase0": {"type": "leaf", "action": 0},
        }

        pgs = gs.PGS(2, [field1, field2])
        P = pgs(tree=config)

        # Check valid phases
        self.assertTrue(np.all((P == 0) | (P == 1)))
        # Check shape
        self.assertEqual(P.shape, (n1, n2))
        # Check both phases present
        self.assertTrue(0 in P)
        self.assertTrue(1 in P)
