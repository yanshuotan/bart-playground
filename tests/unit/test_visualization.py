import unittest
import os
import numpy as np
from graphviz import Digraph
from bart_playground import visualize_tree
from bart_playground import Tree
from bart_playground import Dataset


class TestVisualizeTree(unittest.TestCase):
    def setUp(self):
        X, y = np.random.rand(100, 5), np.random.rand(100)
        dataset = Dataset(X, y)
        self.tree = Tree.new(dataX=dataset.X)
        self.tree.vars = np.array([0, 1, -1, -1, -1, -2, -2, -2], dtype=int)
        self.tree.thresholds = np.array([0.5, 0.7, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        self.tree.leaf_vals = np.array([np.nan, np.nan, -1.0, 1.0, 2.0, np.nan, np.nan, np.nan])
        self.filename = "test/test_tree"

    def test_visualize_tree(self):
        dot = visualize_tree(self.tree, filename=self.filename, format="png")
        self.assertIsInstance(dot, Digraph, "The returned object should be a Digraph instance.")
        expected_filepath = f"{self.filename}.png"
        dot.render(self.filename, format="png")
        self.assertTrue(os.path.exists(expected_filepath), "The output image file was not created correctly.")

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)
        expected_filepath = f"{self.filename}.png"
        if os.path.exists(expected_filepath):
            os.remove(expected_filepath)

if __name__ == "__main__":
    unittest.main()
