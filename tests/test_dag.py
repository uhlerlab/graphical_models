from unittest import TestCase
import unittest
import numpy as np
import networkx as nx
from graphical_models import DAG, AncestralGraph, CycleError


class TestDAG(TestCase):
    def setUp(self):
        self.d = DAG(arcs={(1, 2), (1, 3), (3, 4), (2, 4), (3, 5)})

    def test_neighbors(self):
        self.assertEqual(self.d._neighbors[1], {2, 3})
        self.assertEqual(self.d._neighbors[2], {1, 4})
        self.assertEqual(self.d._neighbors[3], {1, 4, 5})
        self.assertEqual(self.d._neighbors[4], {2, 3})
        self.assertEqual(self.d._neighbors[5], {3})

    def test_children(self):
        self.assertEqual(self.d._children[1], {2, 3})
        self.assertEqual(self.d._children[2], {4})
        self.assertEqual(self.d._children[3], {4, 5})
        self.assertEqual(self.d._children[4], set())
        self.assertEqual(self.d._children[5], set())

    def test_parents(self):
        self.assertEqual(self.d._parents[1], set())
        self.assertEqual(self.d._parents[2], {1})
        self.assertEqual(self.d._parents[3], {1})
        self.assertEqual(self.d._parents[4], {2, 3})
        self.assertEqual(self.d._parents[5], {3})

    def test_downstream(self):
        self.assertEqual(self.d.descendants_of(1), {2, 3, 4, 5})
        self.assertEqual(self.d.descendants_of(2), {4})
        self.assertEqual(self.d.descendants_of(3), {4, 5})
        self.assertEqual(self.d.descendants_of(4), set())
        self.assertEqual(self.d.descendants_of(5), set())

    def test_upstream(self):
        self.assertEqual(self.d.ancestors_of(1), set())
        self.assertEqual(self.d.ancestors_of(2), {1})
        self.assertEqual(self.d.ancestors_of(3), {1})
        self.assertEqual(self.d.ancestors_of(4), {1, 2, 3})
        self.assertEqual(self.d.ancestors_of(5), {1, 3})

    def test_sources(self):
        self.assertEqual(self.d.sources(), {1})

    def test_sinks(self):
        self.assertEqual(self.d.sinks(), {4, 5})

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

    def test_interventional_cpdag_2node(self):
        d = DAG(arcs={(0, 1)})
        c = d.interventional_cpdag([{1}], cpdag=d.cpdag())
        self.assertEqual(c.arcs, {(0, 1)})
        self.assertEqual(c.edges, set())
        c = d.interventional_cpdag([{0}], cpdag=d.cpdag())
        self.assertEqual(c.arcs, {(0, 1)})
        self.assertEqual(c.edges, set())

    def test_interventional_cpdag_3node(self):
        d = DAG(arcs={(0, 1), (0, 2), (1, 2)})
        c = d.interventional_cpdag([{0}], cpdag=d.cpdag())
        self.assertEqual(c.arcs, {(0, 1), (0, 2)})
        self.assertEqual(c.edges, {frozenset((1, 2))})

    # def test_reversible_arcs(self):
    #     pass
    #
    # def test_shd(self):
    #     pass

    # def test_amat(self):
    #     amat, nodes = self.d.to_amat()
    #     for (i, j), val in np.ndenumerate(amat):
    #         if val == 1:
    #             self.assertTrue((nodes[i], nodes[j]) in self.d.arcs)
    #         elif val == 0:
    #             self.assertTrue((nodes[i], nodes[j]) not in self.d.arcs)

    def test_incident_arcs(self):
        self.assertEqual(self.d.incident_arcs(1), {(1, 2), (1, 3)})
        self.assertEqual(self.d.incident_arcs(2), {(1, 2), (2, 4)})
        self.assertEqual(self.d.incident_arcs(3), {(1, 3), (3, 4), (3, 5)})
        self.assertEqual(self.d.incident_arcs(4), {(2, 4), (3, 4)})
        self.assertEqual(self.d.incident_arcs(5), {(3, 5)})

    def test_shd(self):
        d1 = DAG(arcs={(0, 1), (0, 2)})
        d2 = DAG(arcs={(1, 0), (1, 2)})
        self.assertEqual(d1.shd(d2), 3)
        self.assertEqual(d2.shd(d1), 3)

        d1 = DAG()
        d2 = DAG(arcs={(0, 1), (1, 2)})
        self.assertEqual(d1.shd(d2), 2)
        self.assertEqual(d2.shd(d1), 2)

        d1 = DAG(arcs={(0, 1), (1, 2)})
        d2 = DAG(arcs={(0, 1), (2, 1)})
        self.assertEqual(d1.shd(d2), 1)
        self.assertEqual(d2.shd(d1), 1)

    def test_dsep(self):
        d = DAG(arcs={(1, 2), (2, 3)})  # chain
        self.assertTrue(d.dsep(1, 3, {2}))
        self.assertFalse(d.dsep(1, 3))

        d = DAG(arcs={(2, 1), (2, 3)})  # confounder
        self.assertTrue(d.dsep(1, 3, {2}))
        self.assertFalse(d.dsep(1, 3))

        d = DAG(arcs={(1, 3), (2, 3)})  # v-structure
        self.assertTrue(d.dsep(1, 2))
        self.assertFalse(d.dsep(1, 2, {3}))

        d = DAG(arcs={(1, 3), (2, 3), (3, 4), (4, 5)})  # v-structure with chain
        self.assertTrue(d.dsep(1, 2))
        self.assertFalse(d.dsep(1, 2, {5}))

    def test_is_invariant(self):
        d = DAG(arcs={(1, 2), (2, 3)})
        self.assertTrue(d.is_invariant(1, 3))
        self.assertTrue(d.is_invariant(2, 3))
        self.assertFalse(d.is_invariant(3, 3))
        self.assertFalse(d.is_invariant(1, 3, cond_set=3))
        self.assertFalse(d.is_invariant(2, 3, cond_set=3))
        self.assertTrue(d.is_invariant(2, 3, cond_set=1))
        self.assertTrue(d.is_invariant(1, 3, cond_set=2))

    def test_marginal_mag(self):
        d = DAG(arcs={(1, 2), (1, 3)})
        self.assertEqual(d.marginal_mag(1), AncestralGraph(bidirected={(2, 3)}))

        d = DAG(arcs={(1, 2), (1, 3), (2, 3)})
        self.assertEqual(d.marginal_mag(1), AncestralGraph(directed={(2, 3)}))

    def test_markov_blanket(self):
        d = DAG(arcs={(1, 2), (2, 3), (2, 4), (3, 5), (6, 3), (7, 4), (8, 4)})
        self.assertEqual(d.markov_blanket_of(2), {1, 3, 4, 6, 7, 8})

    def test_init_dag(self):
        d = DAG(arcs={(0, 1)})
        d2 = DAG(dag=d)
        self.assertEqual(d2.arcs, {(0, 1)})

    def test_str(self):
        d = DAG(arcs={(0, 1)})
        self.assertEqual(str(d), "[0][1|0]")

    def test_copy(self):
        d = DAG(arcs={(0, 1)})
        d2 = d.copy()
        self.assertEqual(d2.arcs, {(0, 1)})

    def test_induced_subgraph(self):
        d = DAG(arcs={(0, 1), (1, 2)})
        d2 = d.induced_subgraph({0, 1})
        self.assertEqual(d2.arcs, {(0, 1)})

    # def test_vstructs(self):
    #     pass
    #
    # def test_to_cpdag(self):
    #     pass

    def test_atomic_verification(self):
        # Windmill graph G^* from Appendix C
        # a,b,c,d,e,f,g,h are mapped to 0,1,2,3,4,5,6,7
        windmill = DAG(arcs={(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(1,2),(3,4),(5,6),(7,0)})
        I = windmill.atomic_verification()
        self.assertEqual(len(I), 4)
        self.assertEqual(0 in I or 7 in I, True)
        self.assertEqual(1 in I or 2 in I, True)
        self.assertEqual(3 in I or 4 in I, True)
        self.assertEqual(5 in I or 6 in I, True)
        self.assertEqual(windmill.interventional_cpdag([{node} for node in I], cpdag=windmill.cpdag()).num_edges, 0)
    
        # Small clique on odd number of nodes.
        small_odd_clique = nx.complete_graph(49)
        directed_small_odd_clique = DAG.from_nx(nx.DiGraph([(u,v) for (u,v) in small_odd_clique.edges() if u < v]))
        I = directed_small_odd_clique.atomic_verification()
        self.assertEqual(len(I), 24)
        self.assertEqual(directed_small_odd_clique.interventional_cpdag([{node} for node in I], cpdag=directed_small_odd_clique.cpdag()).num_edges, 0)
   
        # Small clique on even number of nodes.
        small_even_clique = nx.complete_graph(50)
        directed_small_even_clique = DAG.from_nx(nx.DiGraph([(u,v) for (u,v) in small_even_clique.edges() if u < v]))
        I = directed_small_even_clique.atomic_verification()
        self.assertEqual(len(I), 25)
        self.assertEqual(directed_small_even_clique.interventional_cpdag([{node} for node in I], cpdag=directed_small_even_clique.cpdag()).num_edges, 0)

        # Large clique on odd number of nodes. Can take some time.
        large_odd_clique = nx.complete_graph(149)
        directed_large_odd_clique = DAG.from_nx(nx.DiGraph([(u,v) for (u,v) in large_odd_clique.edges() if u < v]))
        I = directed_large_odd_clique.atomic_verification()
        self.assertEqual(len(I), 74)
        self.assertEqual(directed_large_odd_clique.interventional_cpdag([{node} for node in I], cpdag=directed_large_odd_clique.cpdag()).num_edges, 0)
   
        # Large clique on even number of nodes. Can take some time.
        large_even_clique = nx.complete_graph(150)
        directed_large_even_clique = DAG.from_nx(nx.DiGraph([(u,v) for (u,v) in large_even_clique.edges() if u < v]))
        I = directed_large_even_clique.atomic_verification()
        self.assertEqual(len(I), 75)
        self.assertEqual(directed_large_even_clique.interventional_cpdag([{node} for node in I], cpdag=directed_large_even_clique.cpdag()).num_edges, 0)
   
        # Random tree on 100 nodes: Generate random tree skeleton then BFS from vertex 0
        tree = nx.random_tree(100)
        directed_tree = DAG.from_nx(nx.bfs_tree(tree, 0))
        I = directed_tree.atomic_verification()
        self.assertEqual(I, {0})
        self.assertEqual(directed_tree.interventional_cpdag([{node} for node in I], cpdag=directed_tree.cpdag()).num_edges, 0)
    
        # Random graph with tree skeleton on 100 nodes
        tree = nx.random_tree(100)
        tree_skel = DAG.from_nx(nx.DiGraph([(u,v) for (u,v) in tree.edges() if u < v]))
        I = tree_skel.atomic_verification()
        self.assertEqual(tree_skel.interventional_cpdag([{node} for node in I], cpdag=tree_skel.cpdag()).num_edges, 0)

if __name__ == '__main__':
    unittest.main()
