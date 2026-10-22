"""Checks of mixture identities and the estimator wall-time accounting."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'diagnosis'/'paper'))
from section7_costs import score
from section7_costs import scalar_summary
from section6_equal_time import budget_configurations, evaluate
from section6_methods import predictive_metrics
from core import prediction_variance_terms

def test_finite_mixture_decomposition_and_rmse_conventions():
    draws=np.random.default_rng(14).normal(size=(4,300,3))
    draws+=np.array([-3.,-1.,2.,5.])[:,None,None]
    y=np.array([1.,2.,-2.])
    result=score(draws,y,None,.9)
    means=draws.mean(axis=1)
    within=draws.var(axis=1,ddof=0).mean(axis=0)
    between=means.var(axis=0,ddof=0)
    np.testing.assert_allclose(within+between,draws.reshape(-1,3).var(axis=0))
    np.testing.assert_allclose(result['between_share_median'],np.median(between/(between+within)))
    np.testing.assert_allclose(result['mse_single'],result['mse_pooled']+result['mse_chain_specific'])
    np.testing.assert_allclose(result['rmse_single_mean_chainwise'],np.sqrt(((means-y)**2).mean(axis=1)).mean())
    assert result['agreement_screen_pass']==0

def test_pooled_default_is_charged_four_sequential_prefixes():
    ratios={'default':1.,'mtmh':7.5,'default_pt':4.,'mtmh_pt':12.}
    plans=budget_configurations(3.,ratios)
    for method,_,pooled,n in plans:
        charged=n/10000*ratios[method]*(4 if pooled else 1)
        assert charged <= 3.
        assert 3.-charged < 4*ratios[method]/10000
    assert plans[-1][3]==7500
    assert plans[0][3]==30000

def test_pooling_changes_estimator_not_constituent_chain_agreement():
    draws=np.random.default_rng(3).normal(size=(4,300,3))+np.arange(4)[:,None,None]
    y=np.zeros(3)
    single=evaluate(draws,y,y,False)
    pooled=evaluate(draws,y,y,True)
    assert pooled['mse'] < single['mse']
    assert pooled['rhat_median']==single['rhat_median']
    assert np.isfinite(scalar_summary(draws[:1].mean(axis=2))['rhat'])


def test_canonical_variance_and_predictive_rmse_across_sections():
    # Unequal chain errors distinguish sqrt(mean MSE) from mean(sqrt MSE).
    draws = np.random.default_rng(27).normal(size=(4, 100, 3))
    draws += np.array([0., 1., 3., 7.])[:, None, None]
    y = np.zeros(3)
    within, between = prediction_variance_terms(draws)
    np.testing.assert_allclose(within + between, draws.reshape(-1, 3).var(axis=0))
    rmse, _ = predictive_metrics(draws, np.ones((4, 100)), y, 1., 123)
    np.testing.assert_allclose(rmse, np.sqrt(((draws.mean(axis=1) - y) ** 2).mean()))
    np.testing.assert_allclose(rmse, evaluate(draws, y, None, False)['rmse_root_mean_mse'])
