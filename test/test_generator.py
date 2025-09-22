import numpy as np

from bart_playground.bandit.experiment_utils.bart_generator import (
    generate_data_from_defaultbart_prior,
)

def _default_params_call(n, p):
    return generate_data_from_defaultbart_prior(
        n=n,
        p=p,
        n_trees=1,
        tree_alpha=0.95,
        tree_beta=2.0,
        f_k=2.0,
        eps_nu=3.0,
        eps_lambda=1.0,
        random_state=42,
        max_depth=10,
        min_node_size=1,
        quick_decay=False,
        return_latent=True
    )
    
def test_all_close():
    n = 20000
    p = 2
    res = _default_params_call(n, p)
    X, y, f, sigma2, trees = res # type: ignore
    assert X.shape == (n, p)
    assert f.shape == (n,)
    assert np.allclose(np.sum([t.evaluate(X) for t in trees], axis=0), f)
