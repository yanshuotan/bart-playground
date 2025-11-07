import numpy as np
import pytest

from bart_playground.mcbart import MultiChainBART


def _toy_regression(n=40, d=3, rng=None):
    rng = np.random.default_rng(rng)
    X = rng.normal(size=(n, d))
    y = X[:, 0] * 2.0 - X[:, 1] * 1.5 + rng.normal(scale=0.1, size=n)
    return X.astype(float), y.astype(float)


@pytest.mark.filterwarnings("ignore::UserWarning")
class TestMultiChainBART:
    def test_fit_predict_runs(self):
        X, y = _toy_regression(n=50, d=3, rng=0)
        model = MultiChainBART(
            n_ensembles=3,
            random_state=123,
            ndpost=50,
            nskip=10,
            n_trees=20,
        )
        model.fit(X, y, quietly=True)
        preds = model.predict(X[:5])
        assert preds.shape == (5,)
        post = model.posterior_predict(X[:5])
        assert post.ndim == 2 and post.shape[0] == 5
        f_post = model.posterior_f(X[:5])
        assert f_post.ndim == 2 and f_post.shape[0] == 5
        # Clean up actors to avoid Ray leakage across tests
        model.clean_up()

    def test_deterministic_with_same_master_seed(self):
        X, y = _toy_regression(n=30, d=2, rng=1)
        args = dict(n_ensembles=4, ndpost=30, nskip=10, n_trees=10)

        m1 = MultiChainBART(random_state=999, **args)
        m1.fit(X, y, quietly=True)
        p1 = m1.predict(X[:7])
        post1 = m1.posterior_predict(X[:7])

        m2 = MultiChainBART(random_state=999, **args)
        m2.fit(X, y, quietly=True)
        p2 = m2.predict(X[:7])
        post2 = m2.posterior_predict(X[:7])

        # With SeedSequence spawning, same master seed should yield identical results
        np.testing.assert_allclose(p1, p2, rtol=0, atol=0)
        np.testing.assert_allclose(post1, post2, rtol=0, atol=0)

        m1.clean_up()
        m2.clean_up()

    def test_different_master_seeds_diverge(self):
        X, y = _toy_regression(n=30, d=2, rng=2)
        args = dict(n_ensembles=3, ndpost=25, nskip=10, n_trees=10)

        m1 = MultiChainBART(random_state=111, **args)
        m1.fit(X, y, quietly=True)
        p1 = m1.predict(X[:7])

        m2 = MultiChainBART(random_state=112, **args)
        m2.fit(X, y, quietly=True)
        p2 = m2.predict(X[:7])

        # Different seeds should very likely differ; allow rare equalities to pass by checking not allclose
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(p1, p2, rtol=0, atol=0)

        m1.clean_up()
        m2.clean_up()

