# === BUILT-IN
import itertools as itr
from typing import Dict, List, Hashable

# === THIRD-PARTY
import numpy as np
from tqdm import tqdm
from einops import repeat
from scipy.special import logsumexp

# === LOCAL
from graphical_models.utils import core_utils
from graphical_models.classes.dags.dag import DAG


def get_conditional(data, node, vals, parent_ixs, parent_alphabets):
    if len(parent_ixs) == 0:
        return [(data[:, node] == val).mean() for val in vals]
    else:
        nvals = len(vals)
        conditional = np.ones(list(map(len, parent_alphabets)) + [nvals]) * 1/nvals
        for parent_vals in itr.product(*parent_alphabets):
            ixs = (data[:, parent_ixs] == parent_vals).all(axis=1)
            subdata = data[ixs, :]
            if subdata.shape[0] > 0:
                conditional[tuple(parent_vals)] = get_conditional(subdata, node, vals, [], [])
        return conditional


def add_variable(table, conditional, parent_ixs):
    log_conditional = np.log(conditional)
    K = conditional.shape[-1]
    previous_nodes = list(range(len(table.shape)))
    nonparent_ixs = [ix for ix in previous_nodes if ix not in parent_ixs]
    if len(nonparent_ixs) > 0:
        start_dims = " ".join([f"d{ix}" for ix in parent_ixs]) + f" d_new"
        new_dims = " ".join([(f"d{ix}" if ix in parent_ixs else f"n{ix}") for ix in previous_nodes]) + f" d_new"
        pattern = start_dims + " -> " + new_dims
        repeats = {f"n{ix}": table.shape[ix] for ix in nonparent_ixs}
        log_conditional = repeat(log_conditional, pattern, **repeats)
    
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
        node_alphabets: Dict[Hashable, List]
    ):
        super().__init__(set(nodes), arcs)
        self.conditionals = conditionals
        self.node_alphabets = node_alphabets
        self._node_list = list(nodes)
        self._node2ix = core_utils.ix_map_from_list(self._node_list)

    def set_conditional(self, node, cpt):
        self.conditionals[node] = cpt

    def sample(self, nsamples: int = 1, progress=False) -> np.array:
        samples = np.zeros((nsamples, len(self._nodes)), dtype=int)
        t = self.topological_sort()
        t = t if not progress else tqdm(t)

        for node in t:
            parents = list(self._parents[node])
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
            parents = list(self._parents[node])
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

    def get_joint_probability_table(self):
        table = np.ones([len(self.node_alphabets[node]) for node in self._node_list])
        for node in self._node_list:
            node_ix = self._node2ix[node]
            parents = self.parents_of(node)
            if len(parents) == 0:
                cond = self.conditionals[node]
                def mul(arr):
                    return arr * cond
                table = np.apply_along_axis(mul, node_ix, table)
            else:
                
                parent_ixs = [self._node2ix[p] for p in parents]
        return table


    def get_marginal(self, node, verbose=False):
        ancestor_subgraph = self.ancestral_subgraph(node)
        t = ancestor_subgraph.topological_sort()
        if verbose: print(ancestor_subgraph)

        unmarginalized_nodes = t[:-1]
        added_nodes = {t[0]}
        table = np.log(self.conditionals[t[0]])
        
        for node in t[1:]:
            node2ix = {node: ix for ix, node in enumerate(unmarginalized_nodes)}
            parent_ixs = [node2ix[p] for p in self.parents_of(node)]

            if verbose: print(f"====== Adding {node} to {added_nodes} ======")
            table = add_variable(table, self.conditionals[node], parent_ixs)
            added_nodes.add(node)

            # === MARGINALIZE ANY NODE WHERE ALL CHILDREN HAVE BEEN ADDED
            marginalizable_nodes = {
                node for node in unmarginalized_nodes 
                if ancestor_subgraph.children_of(node) <= added_nodes
            }
            if verbose: print(f"Marginalizing {marginalizable_nodes}")
            if len(marginalizable_nodes) > 0:
                table = marginalize(table, [node2ix[node] for node in marginalizable_nodes])
                unmarginalized_nodes = [
                    node for node in unmarginalized_nodes
                    if node not in marginalizable_nodes
                ]
            if verbose: print(f"Shape: {table.shape}")
            if verbose: print(f"Unmarginalized: {unmarginalized_nodes}")
                
        return np.exp(table)

    @classmethod
    def fit_mle(cls, dag: DAG, data):
        conditionals = dict()
        node_alphabets = dict()
        nodes = dag.topological_sort()
        for node in nodes:
            parents = list(dag.parents_of(node))
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