import unittest
import numpy as np
import sys
from os.path import abspath, dirname
# Add the parent directory (module) to the search path
sys.path.append(abspath(dirname(dirname(__file__))))
from src.util import Dataset
from src.moves import Grow, Prune, Change, Swap
from src.params import Parameters, Tree
from DataGenerator import *  # Import the updated DataGenerator class

class TestMoves(unittest.TestCase):

    def setUp(self):
        #Set up the test with a specific dataset and trees.
 
        self.generator = DataGenerator(n_samples=100, n_features=3, noise=0.1, random_seed=42)
        self.X, self.y = self.generator.generate(scenario="linear")
        self.thresholds = self.X # Use all covariate values as potential thresholds

        # Create dataset and parameters
        self.dataset = Dataset(self.X, self.y, self.thresholds)
        self.trees = [Tree(data=self.dataset) for _ in range(5)]
        self.params = Parameters(self.trees, None, self.dataset)
        self.rng = np.random.default_rng()

    def test_grow_move(self):
        move = Grow(self.params, trees_changed=[0])
        move.propose(self.rng)
        self.assertTrue(move.proposed.trees[0].is_split_node(0), "Grow move should create a split node.")

    def test_prune_move(self):
        # Simulate an initial split
        self.params.trees[0].split_leaf(0, var=0, threshold=0.5, left_val=1.0, right_val=-1.0)

        move = Prune(self.params, trees_changed=[0])
        move.propose(self.rng)
        self.assertTrue(move.proposed.trees[0].is_leaf(0), "Prune move should revert the node to a leaf.")

    def test_change_move(self):
        # Simulate an initial split
        self.params.trees[0].split_leaf(0, var=0, threshold=0.5, left_val=1.0, right_val=-1.0)

        move = Change(self.params, trees_changed=[0])
        move.propose(self.rng)
        self.assertNotEqual(move.proposed.trees[0].thresholds[0], 0.5, "Change move should modify the split threshold.")

    def test_swap_move(self):
        # Simulate a deeper tree
        self.params.trees[0].split_leaf(0, var=0, threshold=0.5, left_val=1.0, right_val=-1.0)
        self.params.trees[0].split_leaf(1, var=1, threshold=0.7, left_val=2.0, right_val=-2.0)

        move = Swap(self.params, trees_changed=[0])
        move.propose(self.rng)
        self.assertNotEqual(
            move.proposed.trees[0].vars[0],
            move.proposed.trees[0].vars[1],
            "Swap move should exchange parent and child spliconditions."
        )


if __name__ == "__main__":
    unittest.main()

