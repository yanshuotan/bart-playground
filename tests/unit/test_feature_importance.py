import unittest
import numpy as np

from bart_playground import DefaultBART, DefaultPreprocessor, DataGenerator


class TestFeatureImportanceDefaultBART(unittest.TestCase):

    def setUp(self):
        self.generator = DataGenerator(n_samples=100, n_features=4, noise=0.1, random_seed=123)
        self.X, self.y = self.generator.generate(scenario="sparse_linear")
        self.preprocessor = DefaultPreprocessor(max_bins=16)
        self.preprocessor.fit(self.X, self.y)
        # keep runtime modest
        self.model = DefaultBART(ndpost=200, nskip=200, n_trees=50, random_state=42)
        self.model.fit(self.X, self.y, quietly=True)

    def test_inclusion_probability_shape_and_bounds(self):
        prob = self.model.feature_inclusion_probability()
        print("inclusion_probability:", np.round(prob, 6))
        self.assertEqual(prob.shape, (self.X.shape[1],))
        self.assertTrue(np.all(prob >= 0.0))
        self.assertTrue(np.all(prob <= 1.0))

    def test_inclusion_frequency_split_and_per_draw(self):
        freq_split = self.model.feature_inclusion_frequency('split')
        freq_per_draw = self.model.feature_inclusion_frequency('per_draw')

        print("inclusion_frequency(split):", np.round(freq_split, 6))
        print("inclusion_frequency(per_draw):", np.round(freq_per_draw, 6))

        self.assertEqual(freq_split.shape, (self.X.shape[1],))
        self.assertEqual(freq_per_draw.shape, (self.X.shape[1],))

        sum_split = float(np.sum(freq_split))
        print("sum(split):", sum_split)
        # If there are any splits, 'split' normalization sums to ~1
        if sum_split > 0:
            self.assertAlmostEqual(sum_split, 1.0, places=6)

        sum_per_draw = float(np.sum(freq_per_draw))
        print("sum(per_draw):", sum_per_draw)
        if sum_per_draw > 0:
            self.assertAlmostEqual(sum_per_draw, 1.0, places=6)


if __name__ == '__main__':
    unittest.main()


