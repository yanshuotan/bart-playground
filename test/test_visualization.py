import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary libraries
from graphviz import Digraph
from src.visualization import visualize_tree  # Replace with the correct filename
import numpy as np

# Define the tree structure and parameters manually
class TreeStructure:
    """
    Mock class to mimic tree structure for visualization.
    """
    def __init__(self):
        self.var = np.array([0, -1, 1, -2, -2, -1, -1], dtype=int)  # -1: Leaf, 0/1: Split variables
        self.split = np.array([0.5, np.nan, 0.7, np.nan, np.nan, np.nan, np.nan])  # Split thresholds

class TreeParams:
    """
    Mock class to mimic tree parameters for visualization.
    """
    def __init__(self):
        self.leaf_vals = np.array([np.nan, 1.0, np.nan, np.nan, np.nan, 2.0, -2.0])  # Leaf node values

# Instantiate tree structure and parameters
tree_structure = TreeStructure()
tree_params = TreeParams()

# Test the visualize_tree function
dot = visualize_tree(tree_structure, tree_params, filename="test/test_tree", format="png")