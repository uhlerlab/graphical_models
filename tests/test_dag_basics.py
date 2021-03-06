from unittest import TestCase
import unittest
import numpy as np
from graphical_models import DAG, CycleError


class TestDAG(TestCase):
    def setUp(self):
        self.d = DAG(arcs={(1, 2), (1, 3), (3, 4), (2, 4), (3, 5)})

    def test_neighbors(self):
        self.assertEqual(self.d.neighbors_of(1), {2, 3})
        self.assertEqual(self.d.neighbors_of(2), {1, 4})
        self.assertEqual(self.d.neighbors_of(3), {1, 4, 5})
        self.assertEqual(self.d.neighbors_of(4), {2, 3})
        self.assertEqual(self.d.neighbors_of(5), {3})

        self.assertEqual(self.d.neighbors_of({1, 2}), {1, 2, 3, 4})

    def test_children(self):
        self.assertEqual(self.d.children_of(1), {2, 3})
        self.assertEqual(self.d.children_of(2), {4})
        self.assertEqual(self.d.children_of(3), {4, 5})
        self.assertEqual(self.d.children_of(4), set())
        self.assertEqual(self.d.children_of(5), set())

        self.assertEqual(self.d.children_of({1, 2}), {2, 3, 4})

    def test_parents(self):
        self.assertEqual(self.d.parents_of(1), set())
        self.assertEqual(self.d.parents_of(2), {1})
        self.assertEqual(self.d.parents_of(3), {1})
        self.assertEqual(self.d.parents_of(4), {2, 3})
        self.assertEqual(self.d.parents_of(5), {3})

        self.assertEqual(self.d.parents_of({3, 4}), {1, 2, 3})

    def test_descendants_of(self):
        self.assertEqual(self.d.descendants_of(1), {2, 3, 4, 5})
        self.assertEqual(self.d.descendants_of(2), {4})
        self.assertEqual(self.d.descendants_of(3), {4, 5})
        self.assertEqual(self.d.descendants_of(4), set())
        self.assertEqual(self.d.descendants_of(5), set())

        self.assertTrue(self.d.is_ancestor_of(1, 2))
        self.assertTrue(self.d.is_ancestor_of(1, 3))
        self.assertTrue(self.d.is_ancestor_of(1, 4))
        self.assertTrue(self.d.is_ancestor_of(1, 5))

    def test_descendants_of_multiple(self):
        d = DAG(arcs={(0, 1), (2, 3)})
        self.assertEqual(d.descendants_of({0, 2}), {1, 3})

    def test_ancestors_of(self):
        self.assertEqual(self.d.ancestors_of(1), set())
        self.assertEqual(self.d.ancestors_of(2), {1})
        self.assertEqual(self.d.ancestors_of(3), {1})
        self.assertEqual(self.d.ancestors_of(4), {1, 2, 3})
        self.assertEqual(self.d.ancestors_of(5), {1, 3})

    def test_ancestors_of_multiple(self):
        d = DAG(arcs={(0, 1), (2, 3)})
        self.assertEqual(d.ancestors_of({1, 3}), {0, 2})

    def test_add_node(self):
        self.d.add_node(6)
        self.assertEqual(self.d.nodes, set(range(1, 7)))

    def test_add_arc(self):
        self.d.add_arc(2, 3)
        self.assertEqual(self.d._children[2], {3, 4})
        self.assertEqual(self.d._neighbors[2], {1, 3, 4})
        self.assertEqual(self.d._parents[3], {1, 2})
        self.assertEqual(self.d._neighbors[3], {1, 2, 4, 5})
        self.assertEqual(self.d.descendants_of(2), {3, 4, 5})
        self.assertEqual(self.d.ancestors_of(3), {1, 2})

    def test_topological_sort(self):
        t = self.d.topological_sort()
        ixs = {node: t.index(node) for node in self.d.nodes}
        for i, j in self.d.arcs:
            self.assertTrue(ixs[i] < ixs[j])

    def test_add_arc_cycle(self):
        with self.assertRaises(CycleError) as cm:
            self.d.add_arc(2, 1)
        self.assertEqual(cm.exception.cycle, [1, 2, 1])
        with self.assertRaises(CycleError):
            self.d.add_arc(4, 1)
        with self.assertRaises(CycleError) as cm:
            self.d.add_arc(5, 1)
        self.assertEqual(cm.exception.cycle, [1, 3, 5, 1])

    # def test_reversible_arcs(self):
    #     pass
    #
    # def test_shd(self):
    #     pass

    def test_amat(self):
        amat, nodes = self.d.to_amat()
        for (i, j), val in np.ndenumerate(amat):
            if val == 1:
                self.assertTrue((nodes[i], nodes[j]) in self.d.arcs)
            elif val == 0:
                self.assertTrue((nodes[i], nodes[j]) not in self.d.arcs)

    def test_incident_arcs(self):
        self.assertEqual(self.d.incident_arcs(1), {(1, 2), (1, 3)})
        self.assertEqual(self.d.incident_arcs(2), {(1, 2), (2, 4)})
        self.assertEqual(self.d.incident_arcs(3), {(1, 3), (3, 4), (3, 5)})
        self.assertEqual(self.d.incident_arcs(4), {(2, 4), (3, 4)})
        self.assertEqual(self.d.incident_arcs(5), {(3, 5)})

    def test_properties(self):
        d = DAG(arcs={(0, 1), (1, 2)})
        self.assertEqual(d.nnodes, 3)
        self.assertEqual(d.num_arcs, 2)
        self.assertEqual(d.skeleton, {frozenset({0, 1}), frozenset({1, 2})})
        self.assertEqual(d.sparsity, 2/3)

    def test_degrees(self):
        d = DAG(arcs={(0, 1), (0, 2), (1, 3), (2, 3), (0, 4)})
        indegrees = {0: 0, 1: 1, 2: 1, 3: 2, 4: 1}
        outdegrees = {0: 3, 1: 1, 2: 1, 3: 0, 4: 0}
        self.assertEqual(d.in_degrees, indegrees)
        self.assertEqual(d.out_degrees, outdegrees)
        self.assertEqual(d.max_in_degree, 2)
        self.assertEqual(d.max_out_degree, 3)

        for k, v in indegrees.items():
            self.assertEqual(d.indegree_of(k), v)
        for k, v in outdegrees.items():
            self.assertEqual(d.outdegree_of(k), v)

    def test_outgoing_incoming(self):
        self.assertEqual(self.d.incoming_arcs(4), {(2, 4), (3, 4)})
        self.assertEqual(self.d.outgoing_arcs(1), {(1, 2), (1, 3)})

    # def test_vstructs(self):
    #     pass
    #
    # def test_to_cpdag(self):
    #     pass


if __name__ == '__main__':
    unittest.main()
