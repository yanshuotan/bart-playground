import numpy as np
import pytest

from bart_playground.bart import DefaultBART
from bart_playground.diagnostics import compute_diagnostics


def _make_data(n=60, p=4, seed=123):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float32)
    y = (0.7 * X[:, 0] - 0.3 * X[:, 1] + rng.normal(scale=0.3, size=n)).astype(np.float32)
    return X, y


def test_diagnostics_defaultbart_basic():
    X, y = _make_data(n=50, p=3, seed=10)
    model = DefaultBART(ndpost=30, nskip=10, n_trees=20, random_state=7)
    model.fit(X, y, quietly=True)

    diag = compute_diagnostics(model, key="eps_sigma2")

    assert set(["n_chains", "n_draws", "rhat", "ess_bulk", "mcse_mean", "acceptance"]).issubset(diag.keys())
    assert diag["n_chains"] == 1
    assert diag["n_draws"] == model.ndpost
    assert isinstance(diag["mcse_mean"], float)
    assert "overall" in diag["acceptance"]
    # ESS should be positive
    assert diag["ess_bulk"] > 0


def test_diagnostics_mcbart_two_chains():
    try:
        import ray  # noqa: F401
    except Exception:
        pytest.skip("Ray not available; skipping MultiChainBART diagnostics test.")

    from bart_playground.mcbart import MultiChainBART

    X, y = _make_data(n=40, p=3, seed=42)
    mcb = MultiChainBART(
        n_ensembles=2,
        bart_class=DefaultBART,
        random_state=1234,
        ndpost=30,
        nskip=10,
        n_trees=15,
    )
    try:
        mcb.fit(X, y, quietly=True)
        diag = compute_diagnostics(mcb, key="eps_sigma2")

        assert diag["n_chains"] == 2
        assert diag["n_draws"] > 0
        assert "overall" in diag["acceptance"]
        # With 2 chains, R-hat should be finite (not NaN or inf) if there are draws
        assert np.isfinite(diag["rhat"]) or np.isnan(diag["rhat"]) is False
        assert diag["ess_bulk"] > 0
    finally:
        # Ensure Ray actors are cleaned up to not leak resources across tests
        try:
            mcb.clean_up()
        except Exception:
            pass


