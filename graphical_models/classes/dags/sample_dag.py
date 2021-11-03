from .dag import DAG
import numpy as np
from graphical_models.utils import core_utils
from typing import Callable
from graphical_models.classes.interventions import Intervention, SoftInterventionalDistribution, PerfectInterventionalDistribution
from tqdm import trange


class SampleDAG(DAG):
    def __init__(self, nodes, arcs):
        super().__init__(set(nodes), arcs)
        self.conditionals = dict()
        self._node_list = list(nodes)
        self._node2ix = core_utils.ix_map_from_list(self._node_list)

    def set_conditional(self, node, conditional_distribution: Callable[[np.ndarray], np.ndarray]):
        self.conditionals[node] = conditional_distribution

    def sample(self, nsamples: int = 1) -> np.array:  # TODO: parallelize?
        samples = np.zeros((nsamples, len(self._nodes)))
        t = self.topological_sort()
        for node in t:
            node_ix = self._node2ix[node]
            parent_ixs = [self._node2ix[p] for p in self._parents[node]]
            samples[:, node_ix] = self.conditionals[node](samples[:, parent_ixs])
        return samples

    def sample_interventional(self, intervention, nsamples: int = 1) -> np.ndarray:
        samples = np.zeros((nsamples, len(self._nodes)))

        t = self.topological_sort()
        for node in t:
            ix = self._node2ix[node]
            parent_ixs = [self._node2ix[p] for p in self._parents[node]]
            parent_vals = samples[:, parent_ixs]

            interventional_dist = intervention.get(node)
            if interventional_dist is not None:
                if isinstance(interventional_dist, SoftInterventionalDistribution):
                    samples[:, ix] = interventional_dist.sample(parent_vals, self, node)
                elif isinstance(interventional_dist, PerfectInterventionalDistribution):
                    samples[:, ix] = interventional_dist.sample(nsamples)
            else:
                samples[:, ix] = self.conditionals[node](samples[:, parent_ixs])

        return samples
