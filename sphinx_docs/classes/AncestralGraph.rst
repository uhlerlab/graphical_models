.. _ancestral_graph:

**************
AncestralGraph
**************

Overview
********
.. currentmodule:: graphical_models.classes.mags.ancestral_graph
.. autoclass:: AncestralGraph

Copying
-------
.. autosummary::
   :toctree: generated

   AncestralGraph.copy
   AncestralGraph.induced_subgraph

Information about nodes
-----------------------
.. autosummary::
   :toctree: generated

   AncestralGraph.parents_of
   AncestralGraph.children_of
   AncestralGraph.spouses_of
   AncestralGraph.neighbors_of
   AncestralGraph.descendants_of
   AncestralGraph.ancestors_of
   AncestralGraph.district_of
   AncestralGraph.markov_blanket_of

Graph modification
------------------
.. autosummary::
   :toctree: generated

   AncestralGraph.add_node
   AncestralGraph.remove_node
   AncestralGraph.add_directed
   AncestralGraph.remove_directed
   AncestralGraph.add_bidirected
   AncestralGraph.remove_bidirected
   AncestralGraph.add_undirected
   AncestralGraph.remove_undirected
   AncestralGraph.add_nodes_from
   AncestralGraph.remove_edge
   AncestralGraph.remove_edges

Graph properties
----------------
.. autosummary::
   :toctree: generated

   AncestralGraph.legitimate_mark_changes
   AncestralGraph.discriminating_triples
   AncestralGraph.discriminating_paths
   AncestralGraph.is_maximal
   AncestralGraph.c_components
   AncestralGraph.colliders
   AncestralGraph.vstructures
   AncestralGraph.has_directed
   AncestralGraph.has_bidirected
   AncestralGraph.has_undirected
   AncestralGraph.has_any_edge

Ordering
--------
.. autosummary::
   :toctree: generated

   AncestralGraph.topological_sort

Comparison to other AncestralGraphs
-----------------------------------
.. autosummary::
   :toctree: generated

   AncestralGraph.shd_skeleton
   AncestralGraph.markov_equivalent
   AncestralGraph.is_imap
   AncestralGraph.is_minimal_imap

Separation Statements
---------------------
.. autosummary::
   :toctree: generated

   AncestralGraph.msep
   AncestralGraph.msep_from_given

Conversion to/from other formats
--------------------------------
.. autosummary::
   :toctree: generated

   AncestralGraph.to_amat
   AncestralGraph.from_amat