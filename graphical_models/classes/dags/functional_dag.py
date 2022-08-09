# === IMPORTS: BUILT-IN ===
from typing import Callable, Dict, Hashable, Optional

# === IMPORTS: THIRD-PARTY ===
import numpy as np

# === IMPORTS: LOCAL ===
from .dag import DAG


class FunctionalDAG(DAG):
    def __init__(self, nodes, arcs):
        super().__init__(set(nodes), arcs)
    
    def sample(self, nsamples: int) -> np.ndarray:
        pass
    
    def log_probability(self, samples: np.ndarray):
        pass
    
    def predict_from_parents(self, node, parent_vals) -> np.ndarray:
        pass