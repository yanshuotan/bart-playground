import cProfile
import pstats
import os
import unittest
import numpy as np
import xgboost as xgb

from bart_playground import DefaultPreprocessor
from bart_playground import DataGenerator
from bart_playground import DefaultBART


class TestInitFromXGBoost(unittest.TestCase):

    def setUp(self):
        # Generate a simple piecewise-linear dataset
        self.generator = DataGenerator(n_samples=50, n_features=2, noise=0.1, random_seed=42)
        self.X, self.y = self.generator.generate(scenario="piecewise_linear")

        # Preprocess the data
        self.preprocessor = DefaultPreprocessor(max_bins=10)
        self.data = self.preprocessor.fit_transform(self.X, self.y)
        self.X_t = self.data.X  # Transformed features
        self.y_t = self.data.y  # Transformed target

        # Initialize DefaultBART using the same preprocessor
        self.bart = DefaultBART(ndpost=100, n_trees=1)
        self.bart.preprocessor = self.preprocessor
        self.bart.data = self.data  # Prevent double preprocessing
        self.bart.is_fitted = True

    def test_init_from_xgboost_on_preprocessed_data(self):
        # Train XGBoost on transformed data with limited depth
        xgb_params = {
            "n_estimators": 1,
            "max_depth": 2,
            "learning_rate": 1.0,
            "random_state": 42,
            "tree_method": "exact",
            "grow_policy": "depthwise"
        }
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(self.X_t, self.y_t)

        # Initialize BART from this XGBoost model using transformed data
        self.bart.init_from_xgboost(xgb_model, self.X_t, self.y_t, debug=True)

        # Evaluate BART prediction before MCMC sampling
        params = self.bart.sampler.get_init_state()
        bart_pred = params.evaluate(self.X_t)  # On transformed data
        xgb_pred = xgb_model.predict(self.X_t, output_margin=True)

        diffs = bart_pred - xgb_pred
        print("Per-point diffs between BART and XGBoost:")
        print(diffs)

        # Check if all differences are (approximately) equal
        if np.allclose(diffs, diffs[0], atol=1e-8):
            print(f"✅ All diffs equal: offset = {diffs[0]}")
        else:
            print("❌ Diffs vary across points")

        # Optional: test offset-corrected predictions
        corrected = bart_pred - diffs[0]
        np.testing.assert_allclose(
            corrected,
            xgb_pred,
            atol=1e-6,
            err_msg="Even after offset correction, BART doesn't match XGBoost"
        )


if __name__ == "__main__":
    profile_filename = "output.prof"
    cProfile.run("unittest.main()", profile_filename)

    stats = pstats.Stats(profile_filename)
    stats.strip_dirs()
    stats.sort_stats("tottime").print_stats(20)

    os.remove(profile_filename)
