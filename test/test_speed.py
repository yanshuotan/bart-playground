import time

from bart_playground import DefaultBART
from bart_playground.util import DataGenerator


def _get_data(scenario: str = "piecewise_flat"):
    dgp_params = {"noise": 0.1, "n_samples": 10000, "n_features": 10}
    generator = DataGenerator(**dgp_params)
    X, y = generator.generate(scenario=scenario)
    return X, y


def test_speed():
    ndpost = 100
    nskip = 10
    n_trees = 20
    proposal_probs = {"grow": 0.25, "prune": 0.25, "change": 0.1, "swap": 0.1}
    random_state = 42
    temperature = 1.0
    bart = DefaultBART(
        ndpost=ndpost,
        nskip=nskip,
        n_trees=n_trees,
        proposal_probs=proposal_probs,
        random_state=random_state,
        temperature=temperature,
    )
    X, y = _get_data(scenario="piecewise_flat")
    s = time.time()
    bart.fit(X, y)
    print(f"Time taken: {time.time() - s} seconds")


if __name__ == "__main__":
    test_speed()
