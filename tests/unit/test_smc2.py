"""Supervisor SMC: PT target agreement, exact Default kernels and saved counts."""
from copy import deepcopy
import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from bart_playground import AdaptiveSMCBART
from bart_playground.bart import _collapsed_loglik_for_sampler_state, _refresh_tempered_state_for_sampler
from bart_playground.samplers import DefaultSampler
from bart_playground.smc2 import (_choose_beta, _importance, _pack, _restore,
                                  _pt_weight, _particle_task, PTWeight)


def data():
    X=np.random.default_rng(412).normal(size=(24,2))
    return X,X[:,0]+.3*np.random.default_rng(9).normal(size=24)


def prepared(trees=3):
    X,y=data()
    model=AdaptiveSMCBART(n_particles=5,n_trees=trees,n_mcmc_steps=4,random_state=73)
    model.data=model.preprocessor.fit_transform(X,y)
    sampler=model.sampler
    sampler.add_data(model.data)
    sampler.add_thresholds(model.preprocessor.thresholds)
    sampler.global_prior.fit_hyperparameters(model.data)
    state=sampler.get_init_state()
    for _ in range(3):
        state=sampler.one_iter(state,1.)
    return model,state


@pytest.mark.parametrize('trees',[1,3])
def test_temperature_weight_matches_actual_pt_implementation(trees):
    model,state=prepared(trees)
    summary=_pt_weight(model.sampler,state)
    assert summary.log_at(0)==0
    svd=model.sampler.likelihood.leaf_basis_svd(state.leaf_basis(np.arange(trees)))
    for beta in (.001,.1,.5,1.):
        actual=_collapsed_loglik_for_sampler_state(model.sampler,state,1/beta,
                                                  leaf_basis_svd=svd if trees>1 else None)
        assert summary.log_at(beta)==pytest.approx(actual,abs=2e-4,rel=1e-5)


def test_weight_is_integral_of_tempered_likelihood_not_tempered_marginal():
    model,state=prepared()
    summary=_pt_weight(model.sampler,state)
    basis=state.leaf_basis(np.arange(state.n_trees)).astype(float)
    sigma=summary.variance
    y=(model.data.y-state.evaluate(all_except=np.arange(state.n_trees))).astype(float)
    for beta in (.07,.6,1.):
        covariance=sigma/beta*np.eye(len(y))+summary.leaf_variance*(basis@basis.T)
        expected=multivariate_normal.logpdf(y,cov=covariance)+.5*len(y)*(
            (1-beta)*np.log(2*np.pi*sigma)-np.log(beta))
        assert summary.log_at(beta)==pytest.approx(expected,abs=3e-4)
    assert not np.isclose(summary.log_at(.2),.2*summary.log_at(1))


def test_ordinary_importance_ess_and_bisection_target():
    # Zero-leaf curves reduce to fixed log-likelihood slopes.
    weights=[PTWeight(np.array([]),np.array([]),float(x),.1,.01,24) for x in range(1,6)]
    next_beta=_choose_beta(weights,0,.5)
    logweights,ess,_=_importance(weights,0,next_beta)
    assert 0<next_beta<1
    assert ess/5==pytest.approx(.5,abs=1e-10)
    assert np.exp(logweights).sum()==pytest.approx(1)
    assert _choose_beta([weights[0]]*5,0,.5)==1


def test_mutation_is_exactly_k_original_default_sweeps_after_pt_leaf_lift():
    model,state=prepared()
    seed=np.random.SeedSequence(382)
    context={'prior':model.sampler.prior,'proposals':model.sampler.proposals,'tol':1,
             'preprocessor':model.preprocessor,'data':model.data,'test':model.data.X,
             'n_mcmc_steps':4,'max_prior_attempts':10000}
    _,got,_,saved,counts,_=_particle_task((0,_pack(state),seed,.3,2),context)
    rng=np.random.default_rng(seed)
    prior=deepcopy(model.sampler.prior)
    prior.tree_prior.generator=prior.global_prior.generator=rng
    sampler=DefaultSampler(prior,dict(model.sampler.proposals),rng,tol=1)
    sampler.add_data(model.data)
    sampler.add_thresholds(model.preprocessor.thresholds)
    assert type(sampler) is type(model.sampler) is DefaultSampler
    current=_restore(_pack(state),model.data.X)
    _refresh_tempered_state_for_sampler(sampler,current,1/.3)
    expected=[]
    for step in range(4):
        current=sampler.one_iter(current,1/.3)
        if step>=2:
            expected.append(model.preprocessor.backtransform_y(current.evaluate(model.data.X)).copy())
    np.testing.assert_array_equal(saved['preds'],expected)
    for ta,tb in zip(got.trees,current.trees):
        np.testing.assert_array_equal(ta.vars,tb.vars)
        np.testing.assert_array_equal(ta.leaf_vals,tb.leaf_vals)
    assert counts['move_selected_counts']==sampler.move_selected_counts
    assert sum(counts['move_selected_counts'].values())==4*3


def test_serial_parallel_and_checkpoint_resume_are_identical(tmp_path):
    X,y=data()
    common=dict(n_particles=5,n_trees=3,n_mcmc_steps=4,random_state=62)
    serial=AdaptiveSMCBART(**common).fit(X,y,retained_sweeps=2,X_test=X[:4])
    directory=tmp_path/'draws'
    checkpoint=tmp_path/'checkpoint.pkl'
    parallel=AdaptiveSMCBART(n_jobs=2,**common).fit(X,y,retained_sweeps=2,X_test=X[:4],
        output_dir=directory,checkpoint_path=checkpoint)
    assert serial.beta_history_==parallel.beta_history_
    assert serial.beta_history_[-1]==1
    for stage in serial.stage_diagnostics_[:-1]:
        assert stage['ess_ratio']==pytest.approx(.5,abs=1e-9)
    assert all(s['resampled'] for s in serial.stage_diagnostics_)
    assert serial.posterior_draws_['preds'].shape==(10,4)
    for key in serial.posterior_draws_:
        np.testing.assert_array_equal(serial.posterior_draws_[key],parallel.posterior_draws_[key])
    assert serial.mutation_counts_==parallel.mutation_counts_
    assert sum(serial.mutation_counts_['move_selected_counts'].values())==len(serial.stage_diagnostics_)*5*4*3
    resumed=AdaptiveSMCBART(n_jobs=1,**common).fit(X,y,retained_sweeps=2,X_test=X[:4],
        output_dir=directory,checkpoint_path=checkpoint,resume=True)
    np.testing.assert_array_equal(resumed.posterior_f(X[:4]),serial.posterior_f(X[:4]))
    with pytest.raises(ValueError,match='Checkpoint parameters'):
        AdaptiveSMCBART(**common).fit(X,y+1,retained_sweeps=2,X_test=X[:4],
            output_dir=directory,checkpoint_path=checkpoint,resume=True)


def test_interrupted_stage_can_resume_from_last_complete_checkpoint(tmp_path):
    X,y=data()
    settings=dict(n_particles=4,n_trees=2,n_mcmc_steps=3,random_state=33)
    path=tmp_path/'checkpoint.pkl'
    def stop(event):
        if event['phase']=='stage_complete':
            raise RuntimeError('stop after a committed stage')
    with pytest.raises(RuntimeError,match='stop after'):
        AdaptiveSMCBART(**settings).fit(X,y,checkpoint_path=path,progress_callback=stop)
    result=AdaptiveSMCBART(**settings).fit(X,y,checkpoint_path=path,resume=True)
    direct=AdaptiveSMCBART(**settings).fit(X,y)
    assert result.beta_history_==direct.beta_history_
    np.testing.assert_array_equal(result.posterior_f(X),direct.posterior_f(X))


def test_two_point_structure_probability_matches_integrated_posterior():
    X=np.array([[0.],[1.]])
    y=np.array([-1.,1.])
    model=AdaptiveSMCBART(n_particles=500,n_trees=1,n_mcmc_steps=10,
        tree_alpha=.6,tree_beta=2,max_bins=2,fixed_eps_sigma2=.1,
        random_state=172).fit(X,y)
    tau=model.sampler.tree_prior.f_sigma2
    a=np.log(.4)+multivariate_normal.logpdf(model.data.y,cov=.1*np.eye(2)+tau)
    b=np.log(.6*(1-.6/4)**2)+multivariate_normal.logpdf(model.data.y,cov=(.1+tau)*np.eye(2))
    expected=np.exp(b-np.logaddexp(a,b))
    actual=np.mean([state.trees[0].n_leaves==2 for state in model.particles_])
    assert actual==pytest.approx(expected,abs=.065)


@pytest.mark.parametrize('setting',[{'n_particles':0},{'n_mcmc_steps':0},{'n_jobs':0},
    {'ess_ratio':0},{'ess_ratio':1},{'tol':0},{'quick_decay':True}])
def test_invalid_configuration(setting):
    with pytest.raises(ValueError):
        AdaptiveSMCBART(**setting)
