import math

import numpy as np

from bart_playground.bart import DefaultBART, ParallelTemperingBART
from bart_playground.moves import (
    Change,
    Grow,
    Prune,
    _change_rule_probability,
    _normalized_variable_probs,
)
from bart_playground.samplers import DEFAULT_INFORMED_V1_CONFIG


def _initialized_model_state(seed=7):
    x0 = np.linspace(0.0, 1.0, 48)
    X = np.column_stack((x0, np.sin(3.0 * x0)))
    y = np.sin(8.0 * x0) + 0.2 * x0
    model = DefaultBART(
        ndpost=2,
        nskip=0,
        n_trees=1,
        max_bins=20,
        random_state=seed,
        proposal_kernel="informed_v1",
    )
    data = model.preprocessor.fit_transform(X, y)
    model.data = data
    model.sampler.add_data(data)
    model.sampler.add_thresholds(model.preprocessor.thresholds)
    return model, model.sampler.get_init_state()


def test_informed_grow_and_reverse_prune_transition_terms_cancel():
    model, state = _initialized_model_state()
    grow = Grow(
        state,
        np.array([0]),
        possible_thresholds=model.preprocessor.thresholds,
        tol=1,
        data_y=model.data.y,
        proposal_kernel="informed_v1",
        proposal_config=dict(DEFAULT_INFORMED_V1_CONFIG),
    )
    assert grow.propose(np.random.default_rng(21))

    # A grow from a stump produces exactly one terminal split, so this prune is
    # deterministically the reverse structural move.
    prune = Prune(
        grow.proposed,
        np.array([0]),
        possible_thresholds=model.preprocessor.thresholds,
        tol=1,
        data_y=model.data.y,
        proposal_kernel="informed_v1",
        proposal_config=dict(DEFAULT_INFORMED_V1_CONFIG),
    )
    assert prune.propose(np.random.default_rng(22))
    assert math.isclose(
        grow.log_tran_ratio + prune.log_tran_ratio,
        0.0,
        rel_tol=1e-10,
        abs_tol=1e-10,
    )


def test_local_global_change_mixture_is_normalized_and_reversible():
    model, _ = _initialized_model_state()
    thresholds = model.preprocessor.thresholds
    variable_probs = _normalized_variable_probs(len(thresholds), None)
    config = dict(DEFAULT_INFORMED_V1_CONFIG)
    old_var = 0
    old_threshold = thresholds[old_var][len(thresholds[old_var]) // 2]

    total = 0.0
    for new_var, values in thresholds.items():
        for new_threshold in values:
            total += _change_rule_probability(
                thresholds,
                variable_probs,
                old_var,
                old_threshold,
                new_var,
                new_threshold,
                config,
            )
    assert math.isclose(total, 1.0, rel_tol=1e-12, abs_tol=1e-12)

    new_threshold = thresholds[old_var][len(thresholds[old_var]) // 2 + 1]
    q_forward = _change_rule_probability(
        thresholds,
        variable_probs,
        old_var,
        old_threshold,
        old_var,
        new_threshold,
        config,
    )
    q_reverse = _change_rule_probability(
        thresholds,
        variable_probs,
        old_var,
        new_threshold,
        old_var,
        old_threshold,
        config,
    )
    assert q_forward > 0
    assert q_reverse > 0
    assert math.isfinite(math.log(q_reverse) - math.log(q_forward))


def test_informed_default_bart_smoke_and_instrumentation():
    x0 = np.linspace(0.0, 1.0, 64)
    X = np.column_stack((x0, x0**2, np.sin(6.0 * x0)))
    y = np.sin(9.0 * x0) + 0.1 * x0
    model = DefaultBART(
        ndpost=20,
        nskip=0,
        n_trees=8,
        max_bins=20,
        random_state=123,
        proposal_kernel="informed_v1",
        proposal_config=dict(DEFAULT_INFORMED_V1_CONFIG),
    )
    model.fit(X, y, quietly=True)
    assert np.isfinite(model.predict(X)).all()
    diagnostics = model.get_params()["proposal_diagnostics"]
    assert diagnostics["proposal_kernel"] == "informed_v1"
    assert sum(diagnostics["move_selected_counts"].values()) == 20 * 8
    assert diagnostics["event_counts"]


def test_pt_numerical_failure_is_rejected_and_counted():
    model = object.__new__(ParallelTemperingBART)
    model.swap_numerical_failure_counts = np.zeros(2, dtype=np.int64)
    model.store_swap_diagnostics = False

    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError("Internal algorithm failed to converge.")

    model._swap_collapsed_logliks = fail
    result = model._try_swap_collapsed_logliks([], 0, 1, 1.0, 2.0)
    assert result is None
    assert model.swap_numerical_failure_counts.tolist() == [1, 0]
