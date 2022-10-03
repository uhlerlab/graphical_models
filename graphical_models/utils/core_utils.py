# === IMPORTS: BUILT-IN ===
import random
import itertools as itr
from typing import Iterable, Callable, Any

# === IMPORTS: THIRD-PARTY ===
import numpy as np
import networkx as nx
from numpy import abs
from numpy.linalg import inv, lstsq
from networkx.algorithms import bipartite


def get_extended_cov(
    samples: np.ndarray
):
    n = samples.shape[0]
    extended_samples = np.hstack((samples, np.ones((n, 1))))
    return extended_samples.T @ extended_samples


def get_extended_cov2(
    cov,
    mean,
    n: int
):
    p = len(mean)
    extended_cov = np.zeros((p+1, p+1))
    extended_cov[-1, -1] = n
    extended_cov[-1, :-1] = mean * n
    extended_cov[:-1, -1] = mean * n
    extended_cov[:-1, :-1] = cov * (n - 1) + np.outer(mean, mean) * n
    return extended_cov


def linear_regression_mle(
    extended_cov: np.ndarray,
    input_ixs: list, 
    output_ixs: int
):
    C = extended_cov
    p = extended_cov.shape[0] - 1
    ii = input_ixs + [p]
    coefs, _, _, _ = lstsq(C[np.ix_(ii, ii)], C[ii, output_ixs], rcond=None)
    bias = coefs[-1]
    variance = C[output_ixs, output_ixs] - coefs.T @ C[np.ix_(ii, ii)] @ coefs
    
    return coefs[:-1], bias, variance


def ix_map_from_list(l):
    return {e: i for i, e in enumerate(l)}


def defdict2dict(defdict, keys):
    factory = defdict.default_factory
    d = {k: factory(v) for k, v in defdict.items()}
    for k in keys:
        if k not in d:
            d[k] = factory()
    return d


def powerset(s: Iterable, r_min=0, r_max=None) -> Iterable:
    if r_max is None: r_max = len(s)
    return map(set, itr.chain(*(itr.combinations(s, r) for r in range(r_min, r_max+1))))


def powerset_predicate(s: Iterable, predicate: Callable[[Any], bool]) -> Iterable:
    for set_size in range(len(s)+1):
        any_satisfy = False
        for subset in itr.combinations(s, set_size):
            if predicate(subset):
                any_satisfy = True
                yield set(subset)
        if not any_satisfy:
            break


def to_set(o) -> set:
    if not isinstance(o, set):
        try:
            return set(o)
        except TypeError:
            if o is None:
                return set()
            return {o}
    return o


def to_list(o):
    if not isinstance(o, list):
        try:
            return list(o)
        except TypeError:
            if o is None:
                return []
            return [o]
    return o


def is_symmetric(matrix, tol=1e-8):
    return (abs(matrix - matrix.T) <= tol).all()


def random_max(d, minimize=False):
    max_val = max(d.values()) if not minimize else min(d.values())
    max_keys = [k for k, v in d.items() if v == max_val]
    if len(max_keys) == 1:
        return max_keys[0]
    else:
        return random.choice(max_keys)


def iszero(a, atol=1e-8):
    return abs(a) < atol

'''
Given an undirected graph H (networkx graph object), output the minimum vertex cover.
Since H is a forest (and hence is bipartite), we can use Konig's theorem to compute the minimum vertex cover.
However, networkx requires us to process connected components one at a time.
Konig's theorem: In bipartite graph, size maximum matching = size of minimum vertex cover.
'''
def compute_minimum_vertex_cover(H):
    assert bipartite.is_bipartite(H)
    mvc = set()
    for V in nx.connected_components(H):
        cc = H.subgraph(V)
        assert bipartite.is_bipartite(cc)
        matching_for_cc = nx.bipartite.eppstein_matching(cc)
        mvc_for_cc = nx.bipartite.to_vertex_cover(cc, matching_for_cc)
        mvc.update(mvc_for_cc)
    return mvc

if __name__ == '__main__':
    res = list(powerset_predicate(set(range(10)), lambda ss: len(ss) < 4))
