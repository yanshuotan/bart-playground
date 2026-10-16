"""Run the supervisor's adaptive SMC on fixed-100 run0; checkpoint every beta."""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'diagnosis'))


def write_json(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray)
                                    else x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')
    for attempt in range(100):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt==99:
                raise
            time.sleep(.02)


def scores(preds, variances, truth, scale, seed=20261012):
    rng=np.random.default_rng(seed)
    n=len(preds)
    coefficient=2*np.arange(1,n+1)-n-1
    values=[]
    for j,y in enumerate(truth):
        order=np.lexsort((variances,preds[:,j]))
        draws=preds[order,j]+scale*np.sqrt(variances[order])*rng.standard_normal(n)
        values.append(np.mean(np.abs(draws-y))-np.sum(coefficient*np.sort(draws))/n**2)
    return float(np.sqrt(np.mean((preds.mean(axis=0)-truth)**2))),float(np.mean(values))


def geometry(preds):
    centers=preds.mean(axis=1)
    between=float(np.mean(np.sum((centers-centers.mean(axis=0))**2,axis=1)))
    within=float(np.mean(np.sum((preds-centers[:,None,:])**2,axis=2)))
    return {'R_BW':between/within,'between_centroid_variance':between,'within_population_variance':within}


def split_data(dataset):
    from fixed100_support import DATASET_CONFIGS,load_dataset
    from experiment_fixed100 import make_fixed100_splits
    sys.path.insert(0,str(ROOT/'diagnosis/paper'))
    from core import load_with_shape
    X,y=load_dataset(dataset)
    split=make_fixed100_splits(X,y,n_runs=1)[0]
    tag=DATASET_CONFIGS[dataset]['dataset_tag']
    source=ROOT/'diagnosis/store'/tag
    for name,key in [('train_idx','train_idx'),('fixed_test_idx','test_idx')]:
        saved=load_with_shape(source/'indices'/f'{tag}__run000__{name}.csv').astype(int).ravel()
        np.testing.assert_array_equal(saved,split[key])
    for name,key in [('subsample_X_test','X_test_fixed'),('subsample_y_test','y_test_fixed')]:
        saved=load_with_shape(source/name/f'{tag}__run000__{name}.csv')
        np.testing.assert_allclose(saved.reshape(split[key].shape),split[key])
    return split,tag,source


def repeat(dataset, index, root, settings):
    from bart_playground import AdaptiveSMCBART
    folder=Path(root)/dataset/f'replicate{index:03d}'
    folder.mkdir(parents=True,exist_ok=True)
    if settings['resume'] and (folder/'timing.json').exists():
        return json.loads((folder/'timing.json').read_text())
    split,tag,_=split_data(dataset)
    started=time.perf_counter()
    checkpoint=folder/'checkpoint.pkl'
    if (folder/'progress.json').exists():
        prior_elapsed=json.loads((folder/'progress.json').read_text()).get('elapsed_seconds',0.)
    else:
        prior_elapsed=0.
    def progress(event):
        write_json(folder/'progress.json',{'dataset':dataset,'replicate':index,
            'elapsed_seconds':prior_elapsed+time.perf_counter()-started,**event})
    with threadpool_limits(limits=1):
        model=AdaptiveSMCBART(n_particles=settings['n_particles'],n_trees=100,
            n_mcmc_steps=settings['mcmc_steps'],ess_ratio=.5,tol=1,
            n_jobs=settings['particle_jobs'],random_state=2024+index)
        model.fit(split['X_train'],split['y_train'],X_test=split['X_test_fixed'],
            retained_sweeps=settings['retained_sweeps'],output_dir=folder,
            checkpoint_path=checkpoint,resume=settings['resume'],progress_callback=progress,
            temp_dir=settings.get('temp_dir'),quietly=True)
        draws=model.posterior_draws_
        n=settings['n_particles']
        for key in ('preds','sigmas','vars_histogram'):
            array=draws[key]
            if not np.isfinite(array).all():
                raise ValueError('Nonfinite final output.')
            np.save(folder/f'terminal_{key}.npy',array[-n:])
        if not np.all(draws['sigmas']>0):
            raise ValueError('Nonpositive variance.')
        count=n*settings['retained_sweeps']
        rmses=np.lib.format.open_memmap(folder/'rmses.npy',mode='w+',dtype=float,shape=(count,))
        rmses[:]=np.sqrt(np.mean((draws['preds']-split['y_test_fixed'])**2,axis=1))
        rmses.flush()
        np.save(folder/'particle_id.npy',np.tile(np.arange(n),settings['retained_sweeps']))
        np.save(folder/'population_sweep.npy',np.repeat(np.arange(
            settings['mcmc_steps']-settings['retained_sweeps']+1,settings['mcmc_steps']+1),n))
        write_json(folder/'annealing.json',model.stage_diagnostics_)
        result={'dataset':dataset,'replicate':index,'seed':2024+index,'parameters':model.get_params(),
            'annealing_stages':len(model.stage_diagnostics_),'beta_history':model.beta_history_,
            'elapsed_seconds':prior_elapsed+time.perf_counter()-started,
            'worker_task_cpu_seconds':model.task_cpu_seconds_,
            'mutation_counts':model.mutation_counts_,
            'full_forest_sweeps':len(model.stage_diagnostics_)*n*settings['mcmc_steps'],
            'terminal_samples':n,'retained_samples':count,'effective_particle_workers':model.effective_workers_,
            'retained_final_temperature_sweeps':list(range(settings['mcmc_steps']-settings['retained_sweeps']+1,
                                                         settings['mcmc_steps']+1)),
            'initial_ancestors':int(len(np.unique(model.ancestry_))),
            'leaf_refreshes_before_rejuvenation':n*len(model.stage_diagnostics_)}
        assert sum(model.mutation_counts_['move_selected_counts'].values())==result['full_forest_sweeps']*100
        write_json(folder/'timing.json',result)
        progress({'phase':'complete','stage':len(model.stage_diagnostics_),'beta':1.})
        return result


def analyze(root,dataset,settings):
    sys.path.insert(0,str(ROOT/'diagnosis/paper'))
    from core import load_with_shape
    from section6_methods import predictive_baselines
    split,tag,source=split_data(dataset)
    truth=split['y_test_fixed']
    scale=float(np.ptp(split['y_train'])) or 1.
    ref=predictive_baselines(split['y_train'],truth)
    folder=Path(root)/dataset
    methods={}
    for label,prefix in [(f'smc2_terminal_{settings["n_particles"]}','terminal_'),
                         (f'smc2_final_{settings["retained_sweeps"]}_rounds','')]:
        preds=np.stack([np.load(folder/f'replicate{r:03d}'/f'{prefix}preds.npy') for r in range(4)])
        variances=np.stack([np.load(folder/f'replicate{r:03d}'/f'{prefix}sigmas.npy') for r in range(4)])
        methods[label]=(preds,variances)
    for method in ('default','default_pt','mtmh','mtmh_pt','default_long','default_long_tail'):
        path=source/'preds'/f'{tag}__run000__{method}__preds.csv'
        if not path.exists():
            continue
        p=load_with_shape(path).transpose(0,2,1)[:,3000:]
        v=load_with_shape(source/'sigmas'/f'{tag}__run000__{method}__sigmas.csv').reshape(4,-1)[:,3000:]
        methods[method]=(p,v)
        indices=np.linspace(0,p.shape[1]-1,settings['n_particles'],dtype=int)
        if settings['n_particles']<=p.shape[1]:
            methods[method+'_terminal_count']=(p[:,indices],v[:,indices])
    old=ROOT/'diagnosis/store_smc_population'/tag/'run000_p50_w0_s1000'
    if old.exists():
        for first,last in [(61,200),(861,1000)]:
            start,stop=(first-1)*50,last*50
            p=np.stack([np.load(old/'replicates'/f'replicate{r:03d}'/'preds.npy',mmap_mode='r')[start:stop] for r in range(4)])
            v=np.stack([np.load(old/'replicates'/f'replicate{r:03d}'/'sigmas.npy',mmap_mode='r')[start:stop] for r in range(4)])
            methods[f'smc1_{first}_{last}']=(p,v)
    rows=[]
    for name,(p,v) in methods.items():
        single=np.array([scores(a,b,truth,scale) for a,b in zip(p,v)])
        rows.append({'dataset':dataset,'method':name,'samples_per_repeat':p.shape[1],**geometry(p),
            'rmse_mean':single[:,0].mean(),'rmse_sd':single[:,0].std(ddof=1),
            'crps_mean':single[:,1].mean(),'crps_sd':single[:,1].std(ddof=1),
            'relative_rmse':single[:,0].mean()/ref['naive_rmse'],
            'relative_crps':single[:,1].mean()/ref['climatology_crps']})
    with (folder/'comparison.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    write_json(folder/'analysis.json',{'metrics':rows,'references':ref,
        'rank_rhat_reported':False,'flattened_mcmc_ess_reported':False,
        'terminal_count_selection':'Equally spaced indices from each baseline post-burn record',
        'terminal_particles_not_iid':True,'final_rounds_not_extra_updates':True,
        'comparison_is_equal_wall_time':False})
    print(json.dumps({'dataset':dataset,'comparison':rows}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',nargs='+',choices=['abalone','seoul_bike'],default=['abalone','seoul_bike'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--chain-jobs',type=int,default=4)
    parser.add_argument('--particle-jobs',type=int,default=1)
    parser.add_argument('--n-particles',type=int,default=1000)
    parser.add_argument('--mcmc-steps',type=int,default=100)
    parser.add_argument('--retained-sweeps',type=int,default=7)
    parser.add_argument('--temp-dir',type=Path)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--analyze-only',action='store_true')
    parser.add_argument('--skip-analysis',action='store_true')
    args=parser.parse_args()
    if not 1<=args.chain_jobs<=4 or not 1<=args.retained_sweeps<=args.mcmc_steps:
        parser.error('chain-jobs must be 1..4; retained-sweeps must be 1..K.')
    from bart_playground.smc2 import resolve_workers
    workers=resolve_workers(args.particle_jobs,args.n_particles)
    settings={**vars(args),'output':str(args.output.resolve()),
              'temp_dir':str(args.temp_dir.resolve()) if args.temp_dir else None}
    args.output.mkdir(parents=True,exist_ok=args.resume or args.analyze_only)
    if (args.output/'settings.json').exists():
        old=json.loads((args.output/'settings.json').read_text())
        for key in ('n_particles','mcmc_steps','retained_sweeps'):
            if settings[key]!=old[key]:
                raise ValueError(f'Cannot resume with a different {key}.')
    write_json(args.output/'settings.json',settings)
    print(f'{args.chain_jobs} populations at once, {workers} particle workers each; N={args.n_particles}, K={args.mcmc_steps}',flush=True)
    for dataset in args.datasets:
        if not args.analyze_only:
            with ProcessPoolExecutor(max_workers=args.chain_jobs,mp_context=mp.get_context('spawn')) as executor:
                futures=[executor.submit(repeat,dataset,r,args.output,settings) for r in range(4)]
                for future in as_completed(futures):
                    result=future.result()
                    print(json.dumps({'dataset':dataset,'replicate':result['replicate'],
                                     'stages':result['annealing_stages'],'seconds':result['elapsed_seconds']}),flush=True)
        if not args.skip_analysis:
            analyze(args.output,dataset,settings)


if __name__=='__main__':
    main()
