import numpy as np
import pytest
import xgboost as xgb

from bart_playground import DefaultBART, DefaultPreprocessor, DataGenerator

@pytest.fixture
def raw_data():
    # Generate a simple piecewise-linear dataset
    generator = DataGenerator(n_samples=40, n_features=2, noise=0.1, random_seed=1)
    X, y = generator.generate(scenario="piecewise_linear")
    return X, y

@pytest.fixture
def preprocessor(raw_data):
    X, y = raw_data
    pp = DefaultPreprocessor(max_bins=10)
    return pp

@pytest.fixture
def transformed_data(raw_data, preprocessor):
    X, y = raw_data
    data = preprocessor.fit_transform(X, y)
    return data.X, data.y, preprocessor

def test_sampler_runs_with_xgb_init(transformed_data):
    X_t, y_t, preprocessor = transformed_data

    # Train XGBoost on transformed data
    xgb_model = xgb.XGBRegressor(
        n_estimators=1,
        max_depth=2,
        learning_rate=1.0,
        random_state=42,
        tree_method="exact",
        grow_policy="depthwise"
    )
    xgb_model.fit(X_t, y_t)

    # Initialize BART without marking it fitted:
    model = DefaultBART(ndpost=10, nskip=2, n_trees=1)
    model.preprocessor = preprocessor

    # init_from_xgboost should call fit_transform internally
    model.init_from_xgboost(xgb_model, X_t, y_t, debug=False)

    # After init, sampler must have data and thresholds
    assert model.sampler.data.X.shape[0] == X_t.shape[0]

    # Run a few sampling iterations directly
    n_iter = 5
    trace = model.sampler.run(n_iter, quietly=True, n_skip=0)

    # Trace should include initial state + n_iter states
    assert isinstance(trace, list)
    assert len(trace) == n_iter + 1

    # Evaluate the final state on transformed features
    final_params = trace[-1]
    preds = final_params.evaluate(X_t)
    assert preds.shape == (X_t.shape[0],)
    assert np.all(np.isfinite(preds)), "Predictions contain non-finite values"

    # Optionally, compare raw init prediction to xgb_model output-margin
    bart_init_pred = trace[0].evaluate(X_t)
    xgb_pred = xgb_model.predict(X_t, output_margin=True)
    diffs = bart_init_pred - xgb_pred
    assert np.allclose(diffs, diffs[0], atol=1e-6), "Initial BART tree differs by non-constant offset"
