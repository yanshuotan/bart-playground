"""Invariance checks use enumerable or analytic targets, not BART PCA appearance."""
from copy import deepcopy
import pickle

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from bart_playground import ConditionalSMCBART, SMCBART
from bart_playground.samplers import DefaultSampler
from bart_playground.priors import GlobalParamPrior
from bart_playground.smc import _restore_particle
from bart_playground.conditional_smc import _multinomial_ancestors


def test_original_default_transition_including_variance_is_identical():
    X=np.random.default_rng(91).normal(size=(30,2))
    y=X[:,0]+np.random.default_rng(3).normal(size=30)
    model=SMCBART(n_particles=4,n_trees=3,random_state=81)
    model.fit(X,y,quietly=True)
    assert type(model.sampler) is DefaultSampler
    assert type(model.sampler.global_prior) is GlobalParamPrior
    sampler=deepcopy(model.sampler)
    reference=DefaultSampler(deepcopy(sampler.prior),dict(sampler.proposals),
                             np.random.default_rng(),tol=1)
    reference.generator.bit_generator.state=deepcopy(sampler.generator.bit_generator.state)
    reference.tree_prior.generator=reference.global_prior.generator=reference.generator
    reference.moves_str_cache=deepcopy(sampler.moves_str_cache)
    reference.moves_cache_iterator=sampler.moves_cache_iterator
    reference.add_data(model.data)
    reference.add_thresholds(model.preprocessor.thresholds)
    state=_restore_particle(model.particles_[0],model.data.X)
    a=sampler.one_iter(deepcopy(state),temp=2.)
    b=reference.one_iter(deepcopy(state),temp=2.)
    np.testing.assert_array_equal(a.evaluate(),b.evaluate())
    np.testing.assert_array_equal(a.global_params['eps_sigma2'],b.global_params['eps_sigma2'])


def test_conditional_multinomial_fixes_only_the_reference_ancestor():
    rng=np.random.default_rng(1)
    w=np.log([.1,.2,.7])
    values=np.array([_multinomial_ancestors(w,rng,True) for _ in range(10000)])
    assert np.all(values[:,0]==0)
    np.testing.assert_allclose(np.bincount(values[:,1:].ravel(),minlength=3)/20000,[.1,.2,.7],atol=.012)


def test_single_particle_reference_path_cannot_be_overwritten():
    X=np.array([[0.],[1.]])
    m=ConditionalSMCBART(n_particles=1,n_trees=1,n_mcmc_steps=1,
                          betas=[0,.2,1.],fixed_eps_sigma2=.1).fit(X,np.array([-1.,1.]),quietly=True)
    before=[s.evaluate(X).copy() for s in m.reference_path_]
    m.step()
    for a,b in zip(before,m.reference_path_):
        np.testing.assert_array_equal(a,b.evaluate(X))
    assert m.last_diagnostics_['reference_retained']
    assert m.posterior_f(X).shape==(2,1)
    assert m.step().vars_histogram.shape==(1,)
    with pytest.raises(NotImplementedError,match='auxiliary'):
        m.sample_posterior(X)


@pytest.mark.parametrize('resampling',[True,False])
def test_particle_gibbs_gaussian_target_when_mutation_is_identity(resampling):
    # All features constant: Default cannot change any structure or leaf.
    # This tests conditional resampling and terminal selection, not rejuvenation.
    X=np.zeros((5,1)); y=np.array([-.2,-.1,0.,.1,.2])
    m=ConditionalSMCBART(n_particles=12,n_trees=1,n_mcmc_steps=1,
                          betas=[0,.15,.5,1.],resampling_schedule=[resampling]*3,
                          fixed_eps_sigma2=.2,random_state=182).fit(X,y,quietly=True)
    tau=m.sampler.tree_prior.f_sigma2
    variance=1/(1/tau+5/.2)
    mean=variance*m.data.y.sum()/.2
    samples=np.array([m.step().evaluate(X[:1])[0] for _ in range(1600)])[200:]
    assert samples.mean()==pytest.approx(mean,abs=.035)
    assert samples.var()==pytest.approx(variance,rel=.22)


def test_enumerated_structure_posterior_and_resume(tmp_path):
    X=np.array([[0.],[1.]])
    m=ConditionalSMCBART(n_particles=8,n_trees=1,n_mcmc_steps=2,tree_alpha=.6,
                          betas=[0,.1,.4,1.],max_bins=2,fixed_eps_sigma2=.1,
                          random_state=832).fit(X,np.array([-1.,1.]),quietly=True)
    tau=m.sampler.tree_prior.f_sigma2
    a=np.log(.4)+multivariate_normal.logpdf(m.data.y,cov=.1*np.eye(2)+tau)
    b=np.log(.6*(1-.6/4)**2)+multivariate_normal.logpdf(m.data.y,cov=(.1+tau)*np.eye(2))
    expected=np.exp(b-np.logaddexp(a,b))
    observed=np.mean([m.step().trees[0].n_leaves==2 for _ in range(900)])
    assert observed==pytest.approx(expected,abs=.06)
    path=tmp_path/'checkpoint.pkl'; m.save_checkpoint(path)
    with path.open('rb') as stream: resumed=pickle.load(stream)
    np.testing.assert_array_equal(m.step().evaluate(X),resumed.step().evaluate(X))


def test_pilot_schedule_is_frozen_across_particle_gibbs_updates():
    X=np.arange(8.)[:,None]; y=np.sin(X[:,0])
    m=ConditionalSMCBART(n_particles=4,n_trees=2,n_mcmc_steps=1,
                          random_state=67).fit(X,y,quietly=True)
    schedule=m.beta_history_.copy(); mask=m.fixed_resampling_.copy()
    for _ in range(3): m.step()
    assert schedule==m.beta_history_ and mask==m.fixed_resampling_
