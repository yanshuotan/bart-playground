import cProfile
import pstats
import os
import unittest
import numpy as np
import xgboost as xgb
import itertools

from bart_playground import DefaultPreprocessor
from bart_playground import DataGenerator
from bart_playground import DefaultBART


def run_xgb_bart_test(n_estimators, max_depth, learning_rate, seed):
    print(f"\n=== Test with n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate}, seed={seed} ===")
    # Generate data
    generator = DataGenerator(n_samples=250, n_features=2, noise=0.1, random_seed=seed)
    X, y = generator.generate(scenario="piecewise_linear")

    # Preprocess
    preprocessor = DefaultPreprocessor(max_bins=50)
    data = preprocessor.fit_transform(X, y)
    X_t = data.X
    y_t = data.y

    # Fit XGBoost
    xgb_params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "random_state": seed,
        "tree_method": "exact",
        "grow_policy": "depthwise"
    }
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_t, y_t)

    # Init BART
    bart = DefaultBART(ndpost=100, n_trees=n_estimators)
    bart.preprocessor = preprocessor
    bart.data = data
    bart.is_fitted = True
    bart.init_from_xgboost(xgb_model, X_t, y_t, debug=False)

    # Compare predictions
    params = bart.sampler.get_init_state()
    bart_pred = params.evaluate(X_t)
    xgb_pred = xgb_model.predict(X_t, output_margin=True)

    diffs = bart_pred - xgb_pred
    print("Per-point diffs between BART and XGBoost:")
    print(diffs)

    if np.allclose(diffs, diffs[0], atol=1e-8):
        print(f"✅ All diffs equal: offset = {diffs[0]}")
    else:
        print("❌ Diffs vary across points")

    corrected = bart_pred - diffs[0]
    np.testing.assert_allclose(
        corrected,
        xgb_pred,
        atol=1e-6,
        err_msg="Even after offset correction, BART doesn't match XGBoost"
    )


class TestInitFromXGBoost(unittest.TestCase):
    def test_multiple_parameter_combinations(self):
        settings = list(itertools.product(
            [1, 5, 20],         # n_estimators
            [1, 3, 5],          # max_depth
            [0.1, 0.3, 1.0],    # learning_rate
            [42, 123]          # seeds
        ))

        for n_estimators, max_depth, lr, seed in settings:
            run_xgb_bart_test(n_estimators, max_depth, lr, seed)


if __name__ == "__main__":
    profile_filename = "output.prof"
    cProfile.run("unittest.main()", profile_filename)

    stats = pstats.Stats(profile_filename)
    stats.strip_dirs()
    stats.sort_stats("tottime").print_stats(20)

    os.remove(profile_filename)
