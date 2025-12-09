import unittest
import numpy as np
import sys
from os.path import abspath, dirname
# Add the parent directory (module) to the search path
sys.path.append(abspath(dirname(dirname(__file__))))
from bart_playground import DefaultPreprocessor
from bart_playground import DataGenerator  # Import the updated DataGenerator class


class TestDefaultPreprocessor(unittest.TestCase):

    def setUp(self):
        #Set up the test case with a specific scenario using the DataGenerator.
        # Generate tied X scenario data with 50% ties
        self.generator = DataGenerator(n_samples=100, n_features=2, noise=0.1, random_seed=42)
        self.X, self.y = self.generator.generate(scenario="tied_x", tie_percentage=0.5)

        # Initialize the preprocessor with a max_bins parameter
        self.max_bins = 5
        self.preprocessor = DefaultPreprocessor(max_bins=self.max_bins)

    def test_initialization(self):
        self.assertEqual(self.preprocessor.max_bins, self.max_bins)
        self.assertIsNone(self.preprocessor.splits)

    def test_fit(self):
        self.preprocessor.fit(self.X, self.y)
        self.assertIsNotNone(self.preprocessor.thresholds)
        self.assertEqual(len(self.preprocessor.thresholds), self.X.shape[1])
        for var, thresholds in self.preprocessor.thresholds.items():
            self.assertTrue(len(thresholds) <= self.max_bins)
            self.assertTrue(np.all(thresholds >= np.min(self.X[:, var])))
            self.assertTrue(np.all(thresholds <= np.max(self.X[:, var])))

    def test_transform(self):
        self.preprocessor.fit(self.X, self.y)
        dataset = self.preprocessor.transform(self.X, self.y)
        self.assertEqual(dataset.X.shape, self.X.shape)
        self.assertEqual(dataset.y.shape, self.y.shape)
        self.assertIsInstance(self.preprocessor.thresholds, dict)

    def test_fit_transform(self):
        dataset = self.preprocessor.fit_transform(self.X, self.y)
        self.assertEqual(dataset.X.shape, self.X.shape)
        self.assertEqual(dataset.y.shape, self.y.shape)
        self.assertIsInstance(self.preprocessor.thresholds, dict)

    def test_transform_y(self):
        self.preprocessor.fit(self.X, self.y)
        y_transformed = self.preprocessor.transform_y(self.y)
        self.assertTrue(np.all(y_transformed >= -0.5) and np.all(y_transformed <= 0.5))

    def test_backtransform_y(self):
        self.preprocessor.fit(self.X, self.y)
        y_transformed = self.preprocessor.transform_y(self.y)
        y_backtransformed = self.preprocessor.backtransform_y(y_transformed)
        np.testing.assert_almost_equal(self.y, y_backtransformed, decimal=6)

    def test_pipeline(self):
        self.preprocessor.fit(self.X, self.y)
        y_transformed = self.preprocessor.transform_y(self.y)
        y_backtransformed = self.preprocessor.backtransform_y(y_transformed)
        dataset = self.preprocessor.transform(self.X, self.y)
        self.assertEqual(dataset.X.shape, self.X.shape)
        self.assertEqual(dataset.y.shape, y_transformed.shape)
        np.testing.assert_almost_equal(self.y, y_backtransformed, decimal=6)


if __name__ == "__main__":
    unittest.main()

