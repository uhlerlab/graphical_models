

# === BUILT-IN
import itertools as itr
from collections import defaultdict
from copy import deepcopy
from functools import reduce
from math import prod
from typing import Dict, Hashable, List

# === THIRD-PARTY
import numpy as np
import torch
from torch.nn.functional import log_softmax
from einops import repeat, einsum
from tqdm import tqdm
from scipy.special import logsumexp

from graphical_models.classes.dags.dag import DAG
from graphical_models.classes.dags.functional_dag import FunctionalDAG
# === LOCAL
from graphical_models.utils import core_utils


def marginalize(table, ixs):
    return torch.logsumexp(table, dim=tuple(ixs))


def no_warn_log(x, eps=1e-10):
    # ixs = x > eps
    res = torch.log(x)
    # res[~ixs] = -torch.inf
    return res


def repeat_dimensions(
    tensor, 
    curr_dims, 
    new_dims, 
    dim_sizes, 
    add_new=True
):
    start_dims = " ".join([f"d{ix}" for ix in curr_dims])
    end_dims = " ".join([f"d{ix}" for ix in new_dims])
    if add_new:
        start_dims += " d_new"
        end_dims += " d_new"
    repeat_pattern = start_dims + " -> " + end_dims
    repeats = {f"d{ix}": dim_sizes[ix] for ix in new_dims if ix not in curr_dims}
    new_tensor = repeat(tensor, repeat_pattern, **repeats)
    return new_tensor


def add_variable(
    table, 
    current_variables, 
    conditional, 
    node2dims, 
    parents
):
    log_conditional = no_warn_log(conditional)

    log_conditional = repeat_dimensions(
        log_conditional,
        parents,
        current_variables,
        node2dims,
    )
    
    new_table = table.reshape(table.shape + (1, )) + log_conditional
    return new_table


class DiscreteDAGTorch(FunctionalDAG):
    def __init__(
        self, 
        nodes, 
        arcs, 
        conditionals: Dict[Hashable, torch.Tensor], 
        node2parents: Dict[Hashable, List],
        node_alphabets: Dict[Hashable, List],
        device=None
    ):
        super().__init__(set(nodes), arcs)
        self.conditionals = conditionals
        self.node2parents = node2parents
        self.node_alphabets = node_alphabets
        self.node2dims = {
            node: len(alphabet) 
            for node, alphabet in self.node_alphabets.items()
        }
        self._node_list = list(nodes)
        self._node2ix = core_utils.ix_map_from_list(self._node_list)
        
        for node, parents in node2parents.items():
            expected_shape = tuple([len(node_alphabets[p]) for p in parents + [node]])
            if conditionals is not None:
                assert conditionals[node].shape == expected_shape
                
        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
    def copy(self):
        return deepcopy(self)
    
    def get_marginals(self, marginal_nodes: List[Hashable], log=False):
        ancestor_subgraph = self.ancestral_subgraph(set(marginal_nodes))
        t = ancestor_subgraph.topological_sort()

        current_nodes = [t[0]]
        added_nodes = {t[0]}
        log_table = torch.log(self.conditionals[t[0]])
        
        for new_node in t[1:]:
            node2ix = {node: ix for ix, node in enumerate(current_nodes)}

            log_table = add_variable(
                log_table, 
                current_nodes,
                self.conditionals[new_node], 
                self.node2dims, 
                self.node2parents[new_node]
            )
            current_nodes.append(new_node)
            added_nodes.add(new_node)

            # === MARGINALIZE ANY NODE WHERE ALL CHILDREN HAVE BEEN ADDED
            marginalizable_nodes = {
                node for node in current_nodes 
                if (ancestor_subgraph.children_of(node) <= added_nodes)
                and (node not in marginal_nodes)
            }
            if len(marginalizable_nodes) > 0:
                log_table = marginalize(log_table, [node2ix[node] for node in marginalizable_nodes])
                current_nodes = [
                    node for node in current_nodes
                    if node not in marginalizable_nodes
                ]
        
        if not log:
            table = torch.exp(log_table)
        else:
            table = log_table
        return repeat_dimensions(table, current_nodes, marginal_nodes, None, add_new=False)
    
    
    def get_marginals_torch(self, marginal_nodes):
        ancestor_subgraph = self.ancestral_subgraph(set(marginal_nodes))
        topsort = ancestor_subgraph.topological_sort()
        
        shapes = {}
        for node in topsort:
            shape = " ".join([f"d{parent}" for parent in self.node2parents[node] + [node]])
            shapes[node] = shape
        
        start_pattern = ", ".join([shapes[node] for node in topsort])
        end_pattern = " ".join([f"d{node}" for node in marginal_nodes])
        pattern = f"{start_pattern} -> {end_pattern}"
        
        tensors = [self.conditionals[node] for node in topsort]
        
        return einsum(*tensors, pattern)
                