import numpy as np
import pytest
import xgboost as xgb

from bart_playground import DefaultBART, DefaultPreprocessor, DataGenerator

@pytest.mark.parametrize(
    "n_trees,max_depth,learning_rate,seed",
    [
        (1, 2, 1.0, 0),
        (3, 3, 0.5, 42),
        (5, 4, 1.0, 7),
    ]
)
def test_xgb_initialization_no_offset(n_trees, max_depth, learning_rate, seed):
    """
    Test that BART initialized from XGBoost (with base_score=0.0) reproduces raw XGBoost outputs exactly.
    """
    # Generate synthetic data
    generator = DataGenerator(n_samples=100, n_features=2, noise=0.1, random_seed=seed)
    X, y = generator.generate(scenario="piecewise_linear")

    # Preprocess data
    preprocessor = DefaultPreprocessor(max_bins=10)
    data = preprocessor.fit_transform(X, y)
    X_t, y_t = data.X, data.y

    # Train XGBoost with no mean-centering
    xgb_model = xgb.XGBRegressor(
        n_estimators=n_trees,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=seed,
        tree_method="exact",
        grow_policy="depthwise",
        base_score=0.0  # disable default mean-centering
    )
    xgb_model.fit(X_t, y_t)

    # Initialize BART from this XGBoost model
    bart = DefaultBART(ndpost=10, n_trees=n_trees)
    # Ensure consistent preprocessing inside BART
    bart.preprocessor = preprocessor
    bart.is_fitted = False  # allow init_from_xgboost to call fit_transform
    bart = bart.init_from_xgboost(xgb_model, X_t, y_t, debug=False)

    # Get initial BART parameters and predictions
    params = bart.sampler.get_init_state()
    bart_pred = params.evaluate(X_t)

    # Compare to XGBoost raw predictions
    xgb_pred = xgb_model.predict(X_t, output_margin=True)

    # Check for exact match within numerical tolerance
    assert bart_pred.shape == xgb_pred.shape
    assert np.allclose(bart_pred, xgb_pred, atol=1e-8), \
        f"BART init predictions differ from XGBoost by {np.max(np.abs(bart_pred - xgb_pred))}"