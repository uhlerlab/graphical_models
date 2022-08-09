from unittest import TestCase
import unittest
import numpy as np
from graphical_models.classes.dags.gaussdag import GaussDAG


class TestDAG(TestCase):
    def test_sample(self):
        dag = GaussDAG(nodes=[0, 1], arcs={(0, 1)})
        samples = dag.sample(100)
        self.assertEqual(samples.shape, (100, 2))
        
    def test_log_probability(self):
        dag = GaussDAG(nodes=[0, 1], arcs={(0, 1)})
        samples = dag.sample(100)
        log_prob = dag.log_probability(samples)
        self.assertEqual(log_prob.shape, (100, ))
        
    def test_predict_from_parents(self):
        dag = GaussDAG(nodes=[0, 1], arcs={(0, 1)})
        parent_vals = np.arange(100).reshape(-1, 1)
        predictions = dag.predict_from_parents(1, parent_vals)
        self.assertEqual(predictions.shape, (100, ))
        self.assertTrue((predictions == np.arange(100)).all())
        
        
if __name__ == '__main__':
    unittest.main()