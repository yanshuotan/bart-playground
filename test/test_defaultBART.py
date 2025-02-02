import unittest
import numpy as np
from bart_playground import DefaultPreprocessor
from bart_playground import DataGenerator  # Import the updated DataGenerator class
from bart_playground import DefaultBART
from bart_playground import Dataset

class TestDefaultBART(unittest.TestCase):

    def setUp(self):
        #Set up the test with a specific dataset and trees.
        self.generator = DataGenerator(n_samples=200, n_features=2, noise=0.1, random_seed=42)
        self.X, self.y = self.generator.generate(scenario="piecewise_linear")
  

        # Initialize the DefaultBART with a preprocessor
        self.preprocessor = DefaultPreprocessor(max_bins=10)
        self.preprocessor.fit(self.X, self.y)
        self.bart = DefaultBART()

    def test_initialization(self):
        self.assertIsNotNone(self.bart.preprocessor, "DefaultBART should have a preprocessor.")
        self.assertIsNotNone(self.bart.sampler, "DefaultBART should have a sampler.")

    def test_fit(self):
        print(self.bart.preprocessor)
        #print(self.preprocessor.thresholds)
        data_use = self.preprocessor.transform(self.X, self.y)
        #print(data_use)
        self.bart.fit(data_use)
        print("Running the fit test")
        #self.assertIsNotNone(self.bart.posterior_samples, "DefaultBART should store posterior samples after fitting.")
        pass
      
    def test_predict(self):
        #self.bart.fit(self.X, self.y)
        #predictions = self.bart.predict(self.X)
        #self.assertEqual(predictions.shape[0], self.X.shape[0], "Predictions should have the same number of rows as X.")
        pass
    
    def test_posterior_f(self):
        #self.bart.fit(self.X, self.y)
        #posterior_samples = self.bart.posterior_f(self.X, n_samples=10)
        #self.assertEqual(posterior_samples.shape, (10, self.X.shape[0]), "Posterior samples shape mismatch.")
        pass

if __name__ == "__main__":
    unittest.main()


