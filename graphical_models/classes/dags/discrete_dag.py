# === BUILT-IN
import itertools as itr
from typing import Dict, List, Hashable
from copy import deepcopy
from functools import reduce

# === THIRD-PARTY
import numpy as np
from tqdm import tqdm
from einops import repeat
from scipy.special import logsumexp

# === LOCAL
from graphical_models.utils import core_utils
from graphical_models.classes.dags.dag import DAG


def repeat_dimensions(tensor, curr_dims, new_dims, dim_sizes, add_new=True):
    start_dims = " ".join([f"d{ix}" for ix in curr_dims])
    end_dims = " ".join([f"d{ix}" for ix in new_dims])
    if add_new:
        start_dims += " d_new"
        end_dims += " d_new"
    repeat_pattern = start_dims + " -> " + end_dims
    repeats = {f"d{ix}": dim_sizes[ix] for ix in new_dims if ix not in curr_dims}
    new_tensor = repeat(tensor, repeat_pattern, **repeats)
    return new_tensor


def get_conditional(data, node, vals, parent_ixs, parent_alphabets):
    if len(parent_ixs) == 0:
        return np.array([(data[:, node] == val).mean() for val in vals])
    else:
        nvals = len(vals)
        conditional = np.ones(list(map(len, parent_alphabets)) + [nvals]) * 1/nvals
        for parent_vals in itr.product(*parent_alphabets):
            ixs = (data[:, parent_ixs] == parent_vals).all(axis=1)
            subdata = data[ixs, :]
            if subdata.shape[0] > 0:
                conditional[tuple(parent_vals)] = get_conditional(subdata, node, vals, [], [])
        return conditional


def add_variable(table, current_variables, conditional, node2dims, parents):
    log_conditional = np.log(conditional)

    log_conditional = repeat_dimensions(
        log_conditional,
        parents,
        current_variables,
        node2dims
    )
    
    table = table.reshape(table.shape + (1, )) + log_conditional
    return table


def marginalize(table, ixs):
    return logsumexp(table, axis=tuple(ixs))


class DiscreteDAG(DAG):
    def __init__(
        self, 
        nodes, 
        arcs, 
        conditionals: Dict[Hashable, np.ndarray], 
        node2parents: Dict[Hashable, List],
        node_alphabets: Dict[Hashable, List]
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

    def set_conditional(self, node, cpt):
        self.conditionals[node] = cpt

    def sample(self, nsamples: int = 1, progress=False) -> np.array:
        samples = np.zeros((nsamples, len(self._nodes)), dtype=int)
        t = self.topological_sort()
        t = t if not progress else tqdm(t)

        for node in t:
            parents = self.node2parents[node]
            if len(parents) == 0:
                vals = np.random.choice(
                    self.node_alphabets[node], 
                    p=self.conditionals[node], 
                    size=nsamples
                )
            else:
                parent_ixs = [self._node2ix[p] for p in parents]
                parent_vals = samples[:, parent_ixs]
                dists = [self.conditionals[node][tuple(v)] for v in parent_vals]
                vals = [np.random.choice(self.node_alphabets[node], p=d) for d in dists]
            samples[:, self._node2ix[node]] = vals

        return samples

    def sample_single_node_interventional(self, iv_node, value, nsamples):
        samples = np.zeros((nsamples, len(self._nodes)), dtype=int)
        t = self.topological_sort()

        for node in t:
            parents = self.node2parents[node]
            if node == iv_node:
                vals = [value] * nsamples
            else:
                if len(parents) == 0:
                    vals = np.random.choice(
                        self.node_alphabets[node], 
                        p=self.conditionals[node], 
                        size=nsamples
                    )
                else:
                    parent_ixs = [self._node2ix[p] for p in parents]
                    parent_vals = samples[:, parent_ixs]
                    dists = [self.conditionals[node][tuple(v)] for v in parent_vals]
                    vals = [np.random.choice(self.node_alphabets[node], p=d) for d in dists]
            samples[:, self._node2ix[node]] = vals

        return samples

    def get_hard_interventional_dag(self, target_node, value):
        assert len(self.parents_of(target_node)) == 0
        node_alphabet = self.node_alphabets[target_node]
        target_conditional = np.array([0 if v == value else 1 for v in node_alphabet])
        new_conditionals = {
            node: self.conditionals[node] if node != target_node else target_conditional
            for node in self.nodes  
        }
        return DiscreteDAG(
            nodes=self.nodes,
            arcs=self.arcs,
            conditionals=new_conditionals,
            node2parents=deepcopy(self.node2parents),
            node_alphabets=self.node_alphabets
        )

    def get_marginals(self, marginal_nodes: List[Hashable], log=False):
        ancestor_subgraph = self.ancestral_subgraph(set(marginal_nodes))
        t = ancestor_subgraph.topological_sort()

        current_nodes = [t[0]]
        added_nodes = {t[0]}
        log_table = np.log(self.conditionals[t[0]])
        
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
            table = np.exp(log_table)
        else:
            table = log_table
        return repeat_dimensions(table, current_nodes, marginal_nodes, None, add_new=False)

    def get_marginal(self, node, verbose=False, log=False):
        ancestor_subgraph = self.ancestral_subgraph(node)
        t = ancestor_subgraph.topological_sort()
        if verbose: print(f"Ancestor subgraph: {ancestor_subgraph}")

        current_nodes = [t[0]]
        added_nodes = {t[0]}
        table = np.log(self.conditionals[t[0]])
        
        for new_node in t[1:]:
            node2ix = {node: ix for ix, node in enumerate(current_nodes)}

            if verbose: print(f"====== Adding {new_node} to {current_nodes} ======")
            table = add_variable(
                table, 
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
                and (node != new_node)
            }
            if verbose: print(f"Marginalizing {marginalizable_nodes}")
            if len(marginalizable_nodes) > 0:
                table = marginalize(table, [node2ix[node] for node in marginalizable_nodes])
                current_nodes = [
                    node for node in current_nodes
                    if node not in marginalizable_nodes
                ]
            
            if verbose: print(f"Shape: {table.shape}")
                
        return np.exp(table)

    def get_conditional(self, marginal_nodes, cond_nodes):
        all_nodes = marginal_nodes + cond_nodes
        full_marginal = self.get_marginals(all_nodes, log=True)
        cond_marginal = marginalize(full_marginal, list(range(len(marginal_nodes))))
        cond_marginal = cond_marginal.reshape((1, ) * len(marginal_nodes) + cond_marginal.shape)
        conditional = full_marginal - cond_marginal
        return np.exp(conditional)

    def get_conditional2(self, marginal_nodes, cond_nodes):
        marginal_nodes_no_repeats = [node for node in marginal_nodes if node not in cond_nodes]
        all_nodes = marginal_nodes_no_repeats + cond_nodes
        full_marginal = self.get_marginals(all_nodes, log=True)
        cond_marginal = marginalize(full_marginal, list(range(len(marginal_nodes_no_repeats))))
        cond_marginal = cond_marginal.reshape((1, ) * len(marginal_nodes_no_repeats) + cond_marginal.shape)
        # TODO: account for division by zero
        conditional = full_marginal - cond_marginal
        if len(marginal_nodes) != len(marginal_nodes_no_repeats):
            marginal_nodes_repeat = [node for node in marginal_nodes if node in cond_nodes]
            start_dims = " ".join([f"d{ix}" for ix in all_nodes])
            end_dims = " ".join([
                f"d{node}" if node in marginal_nodes_no_repeats else f"r{node}" 
                for node in marginal_nodes
            ])
            end_dims += " " + " ".join([f"d{node}" for node in cond_nodes])
            pattern = start_dims + " -> " + end_dims
            repeats = {f"r{node}": self.node2dims[node] for node in marginal_nodes_repeat}
            conditional = repeat(conditional, pattern, **repeats)
            ones = [np.eye(self.node2dims[node]) for node in marginal_nodes_repeat]
            if len(ones) > 1:
                raise NotImplementedError
            else:
                ones = ones[0]
                rep_node = marginal_nodes_repeat[0]
                repeats = {f"d{node}": self.node2dims[node] for node in all_nodes if node != rep_node}
                ones = repeat(ones, f"d{rep_node} r{rep_node} -> {end_dims}", **repeats)
            conditional = conditional * ones
        return np.exp(conditional)

    def get_mean_and_variance(self, node):
        alphabet = self.node_alphabets[node]
        marginal = self.get_marginal(node)
        terms = [val * marg for val, marg in zip(alphabet, marginal)]
        mean = sum(terms)
        terms = [(val - mean)**2 * marg for val, marg in zip(alphabet, marginal)]
        variance = sum(terms)
        return mean, variance

    @classmethod
    def fit_mle(cls, dag: DAG, data):
        conditionals = dict()
        node_alphabets = dict()
        nodes = dag.topological_sort()
        node2parents = dict()
        for node in nodes:
            parents = list(dag.parents_of(node))
            node2parents[node] = parents
            alphabet = list(range(max(data[:, node]) + 1))
            node_alphabets[node] = alphabet
            if len(parents) == 0:
                conditionals[node] = get_conditional(data, node, alphabet, [], [])
            else:
                parent_alphabets = [node_alphabets[p] for p in parents]
                conditionals[node] = get_conditional(data, node, alphabet, parents, parent_alphabets)
        
        return DiscreteDAG(
            nodes,
            dag.arcs,
            conditionals=conditionals,
            node2parents=node2parents,
            node_alphabets=node_alphabets
        )


if __name__ == "__main__":
    conditional0 = 0.5 * np.ones(2)
    conditional1 = np.array([[0.1, 0.9], [0.9, 0.1]])
    conditional2 = np.array([[[0.1, 0.9], [0.9, 0.1]], [[0.8, 0.2], [0.2, 0.8]]])
    ddag = DiscreteDAG(
        [0, 1, 2],
        arcs={(0, 1), (0, 2), (1, 2)},
        conditionals={
            0: conditional0, 
            1: conditional1, 
            2: conditional2
        },
        node_alphabets={0: [0, 1], 1: [0, 1], 2: [0, 1]}
    )
    table = ddag.get_marginal(2)