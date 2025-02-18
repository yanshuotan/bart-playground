import cProfile
import pstats
import os
import unittest
import numpy as np
from bart_playground import DefaultPreprocessor
from bart_playground import DataGenerator  # Import the updated DataGenerator class
from bart_playground import DefaultBART

class TestDefaultBART(unittest.TestCase):

    def setUp(self):
        #Set up the test with a specific dataset and trees.
        self.generator = DataGenerator(n_samples=50, n_features=2, noise=0.1, random_seed=42)
        self.X, self.y = self.generator.generate(scenario="piecewise_linear")
  

        # Initialize the DefaultBART with a preprocessor
        self.preprocessor = DefaultPreprocessor(max_bins=10)
        self.preprocessor.fit(self.X, self.y)
        self.bart = DefaultBART(ndpost=100, n_trees=20)

    def test_initialization(self):
        self.assertIsNotNone(self.bart.preprocessor, "DefaultBART should have a preprocessor.")
        self.assertIsNotNone(self.bart.sampler, "DefaultBART should have a sampler.")

    def test_fit(self):
        self.bart.fit(self.X, self.y)
        self.assertIsNotNone(self.bart.ndpost, "BART should have the number of samples")
        self.assertIsNotNone(self.bart.preprocessor, "BART should have the preprocessor")
        self.assertIsNotNone(self.bart.trace, "BART should have the trace")
        self.assertIsNotNone(self.bart.nskip, "BART should have the number of samples to skip")
        self.assertIsNotNone(self.bart.sampler, "BART should have a sampler")
        pass
      
    def test_predict(self):
        self.bart.fit(self.X, self.y)
        predictions = self.bart.predict(self.X)
        self.assertEqual(predictions.shape[0], self.X.shape[0], "Predictions should have the same number of rows as X.")
        pass
    
    def test_posterior_f(self):
        self.bart.fit(self.X, self.y)
        posterior_samples = self.bart.posterior_f(self.X)
        predictions = self.bart.predict(self.X)
        self.assertEqual(posterior_samples.shape, (self.X.shape[0], self.bart.ndpost), "Posterior samples shape mismatch.")
        self.assertAlmostEqual(posterior_samples.mean(axis = 1).mean(), predictions.mean(), "Posterior samples should match the predictions")
        pass

if __name__ == "__main__":
    profile_filename = "output.prof"
    cProfile.run("unittest.main()", profile_filename)

    stats = pstats.Stats(profile_filename)
    stats.strip_dirs()

    stats.sort_stats("tottime").print_stats(20)

    os.remove(profile_filename)

