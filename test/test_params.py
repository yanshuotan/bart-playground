import unittest
import numpy as np

import sys
from os.path import abspath, dirname
# Add the parent directory (module) to the search path
sys.path.append(abspath(dirname(dirname(__file__))))

from src.params import Tree
from src.util import Dataset

class TestTree(unittest.TestCase):

    def setUp(self):
        X, y = np.random.rand(100, 5), np.random.rand(100)
        dataset = Dataset(X, y, None)
        self.tree = Tree(data=dataset)

    def test_traverse_tree(self):
        self.tree.vars = np.array([0, -1, -1], dtype=int)
        self.tree.thresholds = np.array([0.5, np.nan, np.nan])
        self.tree.leaf_vals = np.array([np.nan, 1.0, -1.0])

        X_test = np.array([[0.4], [0.6]])
        node_ids = self.tree.traverse_tree(X_test)

        self.assertEqual(node_ids[0], 1, "Sample 1 should be assigned to the left leaf node!")
        self.assertEqual(node_ids[1], 2, "Sample 2 should be assigned to the right leaf node!")

    def test_evaluate(self):
        self.tree.vars = np.array([0, -1, -1], dtype=int)
        self.tree.thresholds = np.array([0.5, np.nan, np.nan])
        self.tree.leaf_vals = np.array([np.nan, 1.0, -1.0])

        X_test = np.array([[0.4], [0.6]])
        outputs = self.tree.evaluate(X_test)

        self.assertEqual(outputs[0], 1.0)
        self.assertEqual(outputs[1], -1.0)

    def test_split_leaf(self):
        self.tree.split_leaf(0, var=0, threshold=0.5, left_val=1.0, right_val=-1.0)

        self.assertEqual(self.tree.vars[0], 0)
        self.assertEqual(self.tree.thresholds[0], 0.5)
        self.assertEqual(self.tree.leaf_vals[1], 1.0)
        self.assertEqual(self.tree.leaf_vals[2], -1.0)

    def test_update_n(self):
        self.tree.vars = np.array([0, -1, -1], dtype=int)
        self.tree.thresholds = np.array([0.5, np.nan, np.nan])
        self.tree.leaf_vals = np.array([np.nan, 1.0, -1.0])
        self.tree.n = np.zeros(3, dtype=int)
        self.tree.node_indicators = np.zeros((100, 3), dtype=bool)
        self.tree.node_indicators[:, 0] = True

        success = self.tree.update_n(0)

        self.assertTrue(success, "All nodes should have counts greater than 0")
        self.assertEqual(
            self.tree.n[1], 
            np.sum(self.tree.data.X[:, 0] <= 0.5), 
            "Left child count should match number of samples <= 0.5"
        )
        self.assertEqual(
            self.tree.n[2], 
            np.sum(self.tree.data.X[:, 0] > 0.5), 
            "Right child count should match number of samples > 0.5"
        )

    def test_prune_split(self):
        self.tree.vars = np.array([0, -1, -1, -2, -2, -2, -2, -2], dtype=int)
        self.tree.thresholds = np.array([0.5, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        self.tree.leaf_vals = np.array([np.nan, 1.0, -1.0, np.nan, np.nan, np.nan, np.nan, np.nan])
        self.tree.n = np.array([-2, 100, 50, -2, -2, -2, -2, -2], dtype=int)

        self.tree.prune_split(0)
        self.assertEqual(self.tree.vars[0], -1)
        self.assertTrue(np.isnan(self.tree.thresholds[0]))

    def test_set_leaf_value(self):
        self.tree.split_leaf(0, var=0, threshold=0.5, left_val=1.0, right_val=-1.0)
        self.tree.set_leaf_value(1, 2.0)
        self.assertEqual(self.tree.leaf_vals[1], 2.0)

    def test_is_leaf(self):
        self.tree.split_leaf(0, var=0, threshold=0.5, left_val=1.0, right_val=np.nan)
        self.assertTrue(self.tree.is_leaf(1))
        self.assertFalse(self.tree.is_leaf(0))

    def test_is_split_node(self):
        self.tree.split_leaf(0, var=0, threshold=0.5, left_val=1.0, right_val=np.nan)
        self.assertTrue(self.tree.is_split_node(0))
        self.assertFalse(self.tree.is_split_node(1))

    def test_is_terminal_split_node(self):
        self.tree.split_leaf(0, var=0, threshold=0.5, left_val=1.0, right_val=np.nan)
        self.tree.split_leaf(2, var=1, threshold=0.7, left_val=2.0, right_val=-2.0)
        self.assertTrue(self.tree.is_terminal_split_node(2))
        self.assertFalse(self.tree.is_terminal_split_node(0))

    # def test_get_n_leaves(self):
    #     self.tree.split_leaf(0, var=0, threshold=0.5, left_val=1.0, right_val=np.nan)
    #     self.tree.split_leaf(2, var=1, threshold=0.7, left_val=2.0, right_val=-2.0)
    #     self.assertEqual(self.tree.get_n_leaves(), 3)

if __name__ == "__main__":
    unittest.main()
