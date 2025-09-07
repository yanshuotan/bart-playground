import numpy as np

from bart_playground.bart import DefaultBART
from bart_playground.mcbart import MultiChainBART
from bart_playground.serializer import bart_from_json, bart_to_json
from bart_playground.serializer import multichain_from_json, multichain_to_json


def test_defaultbart_posterior_f_reproduced_after_json_roundtrip():
    rng = np.random.default_rng(0)
    n, p = 40, 5
    X = rng.standard_normal((n, p)).astype(np.float32)
    # simple linear-ish target with noise
    beta = np.array([1.0, -2.0, 0.5, 0.0, 0.0], dtype=np.float32)
    y = (X @ beta + rng.normal(0, 0.1, size=n)).astype(np.float32)

    # small model to keep runtime down
    model = DefaultBART(ndpost=20, nskip=10, n_trees=10, random_state=123, max_bins=32)
    model.fit(X, y, quietly=True)

    # baseline posterior_f
    f_ref = model.posterior_f(X)

    # serialize and restore
    s = bart_to_json(model, include_dataX=False, include_cache=True)
    restored = bart_from_json(s)

    # posterior_f from restored should match exactly because trace, rng, and preprocessor are preserved
    f_restored = restored.posterior_f(X)
    assert f_ref.shape == f_restored.shape
    np.testing.assert_allclose(f_ref, f_restored, rtol=0, atol=0)

def test_multichainbart_posterior_f_reproduced_against_restored_portable():
    rng = np.random.default_rng(7)
    n, p = 30, 4
    X = rng.standard_normal((n, p)).astype(np.float32)
    beta = np.array([0.7, -1.0, 0.3, 0.0], dtype=np.float32)
    y = (X @ beta + rng.normal(0, 0.05, size=n)).astype(np.float32)

    # Train a real MultiChainBART using Ray
    n_ensembles = 3
    master_seed = 2024
    mc = MultiChainBART(
        n_ensembles=n_ensembles,
        random_state=master_seed,
        ndpost=15,
        nskip=5,
        n_trees=8,
        max_bins=32,
    )
    mc.fit(X, y, quietly=True)
    ref = mc.posterior_f(X)

    # Serialize to portable MultiChain and compare posterior_f to the real MultiChainBART
    s = multichain_to_json(mc)
    restored = multichain_from_json(s)

    out = restored.posterior_f(X)
    assert out.shape == ref.shape
    np.testing.assert_allclose(ref, out, rtol=0, atol=0)
