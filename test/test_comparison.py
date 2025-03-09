"""
Tests include:
  - Fitting and prediction shape checks.
  - Summary statistics of predictions (mean, std deviation).
  - RMSE accuracy comparisons.
  - Coverage of 95% posterior predictive intervals.
  - Runtime performance of model fitting.

For bart_playground, the DefaultBART model is used.
For bartz, we use a wrapper to expose a uniform interface.
"""

import time
import math
import numpy as np
import pytest

# Imports from bart_playground
from bart_playground import DefaultBART, DefaultPreprocessor, DataGenerator
# (The Tree and Parameters classes are part of bart_playground as provided.)

# Import bartz and its grove module (for tree info)
import bartz


# --- Bartz wrapper ---
class BartzWrapper:
    """
    A simple wrapper to make the bartz gbart output behave like a model
    with predict, posterior_predictive, and get_tree_info methods.
    """
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        """
        Predict returns the marginal posterior mean.
        X is provided with shape (n, p) (as produced by DataGenerator).
        Bartz expects predictors as (p, n), so we transpose.
        """
        X_t = X.T  # shape (p, n)
        yhat_samples = self.model.predict(X_t)  # Expected shape: (ndpost, n_test)
        return np.mean(yhat_samples, axis=0)

    def posterior_predictive(self, X):
        """
        Return the full posterior sample predictions.
        """
        X_t = X.T
        return self.model.predict(X_t)

    def get_tree_info(self):
        """
        Return tree information.
        This method is not used in the current tests.
        """
        return []


# --- Model creation helper functions ---
def create_bart_playground_model(X, y, ndpost=100, n_trees=20):
    """
    Initialize and fit a bart_playground model.
    """
    model = DefaultBART(ndpost=ndpost, nskip=100, n_trees=n_trees, max_bins=10, random_state=42)
    model.fit(X, y)
    return model


def create_bartz_model(X, y, ndpost=100, n_trees=20):
    """
    Initialize and fit a bartz model using the gbart interface.
    Note: bartz expects predictors as (p, n), so we transpose X.
    We set x_test equal to x_train for simplicity.
    """
    x_train = X.T
    model = bartz.BART.gbart(
        x_train, y,
        x_test=x_train,
        ndpost=ndpost,
        nskip=10,
        ntree=n_trees,
        seed=42
    )
    return BartzWrapper(model)


# --- Data generation ---
test_cases = [
    {"scenario": "linear", "n_samples": 100, "n_features": 3, "noise": 0.1},
    {"scenario": "piecewise_linear", "n_samples": 200, "n_features": 2, "noise": 0.2},
    {"scenario": "cyclic", "n_samples": 150, "n_features": 1, "noise": 0.05},
    {"scenario": "heteroscedastic", "n_samples": 120, "n_features": 4, "noise": 0.3},
    {"scenario": "multimodal", "n_samples": 80, "n_features": 2, "noise": 0.1},
    {"scenario": "tied_x", "n_samples": 100, "n_features": 3, "noise": 0.2},
]


@pytest.fixture(params=test_cases)
def data(request):
    """
    Generate data using DataGenerator.
    If the chosen scenario returns a dict (with multiple cases), pick the first one.
    """
    params = request.param
    generator = DataGenerator(
        n_samples=params["n_samples"],
        n_features=params["n_features"],
        noise=params["noise"],
        random_seed=42,
    )
    result = generator.generate(scenario=params["scenario"])
    if isinstance(result, dict):
        key = list(result.keys())[0]
        X, y = result[key]
    else:
        X, y = result
    return X, y, params


# --- Test functions ---
def test_fit_and_predict(data):
    """
    Test that both models fit without error and produce predictions of the correct shape.
    """
    X, y, params = data
    ndpost = 100
    n_trees = 20

    bp_model = create_bart_playground_model(X, y, ndpost=ndpost, n_trees=n_trees)
    bz_model = create_bartz_model(X, y, ndpost=ndpost, n_trees=n_trees)

    bp_pred = bp_model.predict(X)
    bz_pred = bz_model.predict(X)

    assert bp_pred.shape[0] == X.shape[0], "bart_playground: prediction row count mismatch."
    assert bz_pred.shape[0] == X.shape[0], "bartz: prediction row count mismatch."


def test_prediction_summary_statistics(data):
    """
    Compare summary statistics (mean and std deviation) of the predictions.
    Allow some tolerance since both algorithms are stochastic.
    """
    X, y, params = data
    ndpost = 100
    n_trees = 20

    bp_model = create_bart_playground_model(X, y, ndpost=ndpost, n_trees=n_trees)
    bz_model = create_bartz_model(X, y, ndpost=ndpost, n_trees=n_trees)

    bp_pred = bp_model.predict(X)
    bz_pred = bz_model.predict(X)

    bp_mean = np.mean(bp_pred)
    bz_mean = np.mean(bz_pred)
    assert np.isclose(bp_mean, bz_mean, rtol=0.2), (
        f"Prediction means differ: bart_playground {bp_mean:.3f} vs bartz {bz_mean:.3f}"
    )

    bp_std = np.std(bp_pred)
    bz_std = np.std(bz_pred)
    assert np.isclose(bp_std, bz_std, rtol=0.3), (
        f"Prediction std dev differ: bart_playground {bp_std:.3f} vs bartz {bz_std:.3f}"
    )


def test_accuracy(data):
    """
    Compare RMSE for both models on the training data.
    """
    X, y, params = data
    ndpost = 100
    n_trees = 20

    bp_model = create_bart_playground_model(X, y, ndpost=ndpost, n_trees=n_trees)
    bz_model = create_bartz_model(X, y, ndpost=ndpost, n_trees=n_trees)

    bp_pred = bp_model.predict(X)
    bz_pred = bz_model.predict(X)

    bp_rmse = np.sqrt(np.mean((bp_pred - y) ** 2))
    bz_rmse = np.sqrt(np.mean((bz_pred - y) ** 2))

    # Allow up to 30% relative difference in RMSE.
    assert np.isclose(bp_rmse, bz_rmse, rtol=0.3), (
        f"RMSE differs: bart_playground {bp_rmse:.3f} vs bartz {bz_rmse:.3f}"
    )


def test_posterior_coverage(data):
    """
    Test the coverage of the 95% posterior predictive intervals.
    """
    X, y, params = data
    ndpost = 100
    n_trees = 20

    bp_model = create_bart_playground_model(X, y, ndpost=ndpost, n_trees=n_trees)
    bz_model = create_bartz_model(X, y, ndpost=ndpost, n_trees=n_trees)

    bp_post = bp_model.posterior_f(X)  # Expected shape: (n, ndpost) or (ndpost, n)
    bz_post = bz_model.posterior_predictive(X)  # Expected shape: (ndpost, n)

    if bp_post.shape[0] == X.shape[0]:
        bp_post = bp_post.T

    bp_lower = np.percentile(bp_post, 2.5, axis=0)
    bp_upper = np.percentile(bp_post, 97.5, axis=0)
    bz_lower = np.percentile(bz_post, 2.5, axis=0)
    bz_upper = np.percentile(bz_post, 97.5, axis=0)

    bp_coverage = np.mean((y >= bp_lower) & (y <= bp_upper))
    bz_coverage = np.mean((y >= bz_lower) & (y <= bz_upper))

    assert 0.8 <= bp_coverage <= 0.98, f"bart_playground coverage {bp_coverage:.3f} out of expected range."
    assert 0.8 <= bz_coverage <= 0.98, f"bartz coverage {bz_coverage:.3f} out of expected range."
    assert np.isclose(bp_coverage, bz_coverage, rtol=0.15), (
        f"Coverage differs: bart_playground {bp_coverage:.3f} vs bartz {bz_coverage:.3f}"
    )


def test_runtime(data):
    """
    Measure runtime for fitting both models.
    Each fit should complete within a reasonable time (e.g., < 2 seconds).
    """
    X, y, params = data
    ndpost = 100
    n_trees = 20

    start_bp = time.time()
    _ = create_bart_playground_model(X, y, ndpost=ndpost, n_trees=n_trees)
    runtime_bp = time.time() - start_bp

    start_bz = time.time()
    _ = create_bartz_model(X, y, ndpost=ndpost, n_trees=n_trees)
    runtime_bz = time.time() - start_bz

    assert runtime_bp < 2, f"bart_playground model runtime too high: {runtime_bp:.2f} sec."
    assert runtime_bz < 2, f"bartz model runtime too high: {runtime_bz:.2f} sec."

    ratio = runtime_bp / (runtime_bz + 1e-6)
    assert ratio < 2, f"Runtime ratio too high (bart_playground/bartz = {ratio:.2f})."


# --- Main ---
if __name__ == "__main__":
    pytest.main()
