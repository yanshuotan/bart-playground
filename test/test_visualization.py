import unittest
import os
import numpy as np
from graphviz import Digraph

import sys
from os.path import abspath, dirname
# Add the parent directory (module) to the search path
sys.path.append(abspath(dirname(dirname(__file__))))

from src.visualization import visualize_tree  # Replace with the correct filename

class TreeStructure:
    def __init__(self):
        self.vars = np.array([0, -1, 1, -2, -2, -1, -1], dtype=int)
        self.thresholds = np.array([0.5, np.nan, 0.7, np.nan, np.nan, np.nan, np.nan])

class TreeParams:
    def __init__(self):
        self.leaf_vals = np.array([np.nan, 1.0, np.nan, np.nan, np.nan, 2.0, -2.0])

class TestVisualizeTree(unittest.TestCase):
    def setUp(self):
        self.tree_structure = TreeStructure()
        self.tree_params = TreeParams()
        self.filename = "test/test_tree"

    def test_visualize_tree(self):
        dot = visualize_tree(self.tree_structure, self.tree_params, filename=self.filename, format="png")
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
