"""Tests for gstools.mps.runner."""

import unittest

import numpy as np

from gstools.mps.neighbors import (
    _precompute_offsets,
    _select_neighbors,
)
from gstools.mps.runner import _build_dag_base
from gstools.mps.simulate import _order_positions


class TestPostPassDag(unittest.TestCase):
    """_build_dag_base edge orientation for the two vmap/pos_map regimes."""

    def setUp(self):
        self.shape = (5, 5)
        self.offsets = _precompute_offsets(self.shape)
        # Sequential order so visit position == raster index (readable asserts).
        self.order = np.stack(
            np.unravel_index(np.arange(25), self.shape), axis=-1
        )
        self.pos = _order_positions(self.order, self.shape)
        self.main_vmap = {"v": self.pos.copy()}
        self.post_vmap = {"v": np.full(25, -1, dtype=np.intp)}
        self.n_k = {"v": 4}

    @staticmethod
    def _pairs(out_edges):
        """(source, target) pairs of a CSR successor structure."""
        indptr, targets, _ = out_edges
        return [
            (a, int(b))
            for a in range(len(indptr) - 1)
            for b in targets[indptr[a] : indptr[a + 1]]
        ]

    def _edges(self, vmap, pos_map):
        _, out_edges, _ = _build_dag_base(
            self.order,
            self.shape,
            self.offsets,
            vmap,
            self.n_k,
            None,
            pos_map,
        )
        return set(self._pairs(out_edges))

    def _reads(self, i, vmap):
        """Visit positions node ``i`` selects as neighbours under ``vmap``."""
        coords, _ = _select_neighbors(
            self.order[i],
            self.offsets,
            np.array(self.shape),
            self.shape,
            vmap["v"],
            i,
            None,
            None,
            self.n_k["v"],
        )
        if not len(coords):
            return []
        return [
            int(p)
            for p in self.pos[np.ravel_multi_index(coords.T, self.shape)]
        ]

    def test_pos_map_none_matches_main_path(self):
        # Default (pos_map=None) must reproduce the main-path
        # edge set, which the order-gated vmap already orients for us.
        self.assertEqual(
            self._edges(self.main_vmap, None),
            self._edges(self.main_vmap, self.pos),
        )

    def test_all_edges_run_forward_in_visit_order(self):
        # Acyclicity: the visit order itself is a topological order.
        for a, b in self._edges(self.post_vmap, self.pos):
            self.assertLess(a, b)

    def test_indegree_counts_each_pair_once(self):
        # The read relation is not exactly symmetric (n_neighbors truncation
        # lets a boundary node reach farther than its neighbours reach back),
        # so an edge can be discovered from either endpoint. Double-counting
        # indegree would deadlock the dispatcher, as only one decrement ever
        # arrives per pair.
        indegree, out_edges, _ = _build_dag_base(
            self.order,
            self.shape,
            self.offsets,
            self.post_vmap,
            self.n_k,
            None,
            self.pos,
        )
        pairs = self._pairs(out_edges)
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertEqual(int(indegree["v"].sum()), len(pairs))

    def test_every_read_pair_is_ordered(self):
        """No unsynchronized pair: each node's own selected neighbours all
        carry an edge with it, which is what makes erasing a node's value
        inside its worker safe under concurrency.

        Node 0 also pins the write-after-read direction the main path never
        has — it reads only *later*-positioned nodes (their pre-pass values),
        so the DAG must order it before them. Under the order-gated main-path
        vmap node 0 would read nothing at all.
        """
        edges = self._edges(self.post_vmap, self.pos)
        for i in range(25):
            for j in self._reads(i, self.post_vmap):
                pair = (j, i) if j < i else (i, j)
                self.assertIn(
                    pair, edges, f"node {i} reads {j} unsynchronized"
                )
        node0 = self._reads(0, self.post_vmap)
        self.assertTrue(node0, "node 0 read nothing — not fully informed")
        self.assertTrue(all(p > 0 for p in node0))
        self.assertEqual(self._reads(0, self.main_vmap), [])


if __name__ == "__main__":
    unittest.main()
