import cProfile
import pstats
import os
import unittest
import bartz
import numpy as np
from sklearn.model_selection import train_test_split
from bart_playground.bart import DefaultBART
from bart_playground.DataGenerator import DataGenerator


class TestDefaultBART(unittest.TestCase):
    def setUp(self):
        # Set up the test with a specific dataset and trees.

        proposal_probs = {"grow": 0.5, "swap": 0.1, "prune": 0.4}
        self.generator = DataGenerator(n_samples=50, n_features=5, noise=0.1, random_seed=42)
        self.X, self.y = self.generator.generate(scenario="piecewise_flat")
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, random_state=42)

        # Initialize the DefaultBART with a preprocessor
        self.bart = DefaultBART(ndpost=100, nskip=10, n_trees=20, proposal_probs=proposal_probs)
        self.bart.fit(X_train, y_train)

    def test_initialization(self):
        self.assertIsNotNone(self.bart.preprocessor, "DefaultBART should have a preprocessor.")
        self.assertIsNotNone(self.bart.sampler, "DefaultBART should have a sampler.")


if __name__ == "__main__":
    profile_filename = "output.prof"
    cProfile.run("unittest.main()", profile_filename)

    stats = pstats.Stats(profile_filename)
    stats.strip_dirs()

    stats.sort_stats("tottime").print_stats(20)

    os.remove(profile_filename)
