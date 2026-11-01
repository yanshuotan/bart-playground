"""Local fixed-100 run0 conditional-SMC tests; one selected endpoint per iteration.

No Default timing run. Four independent chains, one CPU each. Datasets run
sequentially. All output is separate from historical, modified-kernel SMC.
"""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'diagnosis'))

def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else str(a)),encoding='utf-8')
    temporary.replace(path)

def split_data(dataset):
    from fixed100_support import load_dataset,DATASET_CONFIGS
    from experiment_fixed100 import make_fixed100_splits
    from run_smc_population import validate_stored_split
    X,y=load_dataset(dataset)
    split=make_fixed100_splits(X,y,n_runs=1)[0]
    tag=DATASET_CONFIGS[dataset]['dataset_tag']
    validate_stored_split(split,ROOT/'diagnosis/store'/tag,tag)
    return split,tag

def worker(dataset,index,settings):
    from bart_playground import ConditionalSMCBART
    folder=Path(settings['output'])/dataset/f'chain{index:03d}'
    folder.mkdir(parents=True,exist_ok=True)
    if settings['resume'] and (folder/'timing.json').exists():
        return json.loads((folder/'timing.json').read_text())
    split,tag=split_data(dataset)
    started=time.perf_counter()
    checkpoint=folder/'checkpoint.pkl'
    fingerprint=hashlib.sha256()
    for key in ('X_train','y_train','X_test_fixed','y_test_fixed'):
        fingerprint.update(np.ascontiguousarray(split[key]).tobytes())
    digest=fingerprint.hexdigest()
    with threadpool_limits(limits=1):
        if settings['resume'] and checkpoint.exists():
            with checkpoint.open('rb') as stream: model=pickle.load(stream)
            if model.experiment_fingerprint_!=digest:
                raise ValueError('Checkpoint data differ from this split.')
            previous_elapsed=model.elapsed_seconds_
            mode='r+'
        else:
            model=ConditionalSMCBART(n_particles=settings['n_particles'],n_trees=100,
                n_mcmc_steps=2,ess_target=.8,resample_threshold=.5,
                pilot_seed=1729+index,random_state=2024+index)
            write_json(folder/'progress.json',{'phase':'pilot_and_initialization','elapsed_seconds':0})
            model.fit(split['X_train'],split['y_train'],quietly=True)
            model.experiment_fingerprint_=digest
            model.elapsed_seconds_=time.perf_counter()-started
            model.initialization_seconds_=model.elapsed_seconds_
            model.history_=[]
            model.save_checkpoint(checkpoint)
            previous_elapsed=0.
            mode='w+'
            write_json(folder/'schedule.json',{'betas':model.beta_history_,
                'fixed_resampling':model.fixed_resampling_,'pilot_seed':model.pilot_seed,
                'pilot':model.pilot_diagnostics_,'parameters':model.get_params()})
        count=settings['draws']
        arrays={name:np.lib.format.open_memmap(folder/f'{name}.npy',mode=mode,dtype=dtype,
                        shape=shape) for name,shape,dtype in [
            ('preds',(count,len(split['y_test_fixed'])),float),('sigmas',(count,),float),
            ('vars_histogram',(count,split['X_train'].shape[1]),np.int32)]}
        total=count+settings['burn']
        for iteration in range(model.iteration_,total):
            tick=time.perf_counter()
            state=model.step()
            if iteration>=settings['burn']:
                row=iteration-settings['burn']
                arrays['preds'][row]=model.preprocessor.backtransform_y(state.evaluate(split['X_test_fixed']))
                arrays['sigmas'][row]=state.global_params['eps_sigma2'][0]
                arrays['vars_histogram'][row]=state.vars_histogram
                for array in arrays.values(): array.flush()
            model.history_.append({'iteration':iteration+1,'seconds':time.perf_counter()-tick,
                'reference_retained':model.last_diagnostics_['reference_retained'],
                'selected_particle':model.last_diagnostics_['selected_particle']})
            model.elapsed_seconds_=previous_elapsed+time.perf_counter()-started
            model.save_checkpoint(checkpoint)
            write_json(folder/'progress.json',{'phase':'conditional_smc',
                'completed_iterations':iteration+1,'total_iterations':total,
                'retained_draws':max(0,iteration+1-settings['burn']),
                'stages':len(model.beta_history_)-1,'elapsed_seconds':model.elapsed_seconds_,
                'last_iteration_seconds':model.history_[-1]['seconds']})
        assert np.isfinite(arrays['preds']).all() and np.all(arrays['sigmas']>0)
        result={'dataset':dataset,'chain':index,'seed':2024+index,'pilot_seed':1729+index,
            'stages':len(model.beta_history_)-1,'initialization_seconds':model.initialization_seconds_,
            'elapsed_seconds':previous_elapsed+time.perf_counter()-started,
            'burn_iterations':settings['burn'],'retained_draws':count,
            'reference_retained_fraction':float(np.mean([h['reference_retained'] for h in model.history_])),
            'original_default_kernel':True,'extra_leaf_sweeps':0,'unchanged_variance_sampler':True,
            'variance_floor_hits':model.variance_floor_hits_,
            'minimum_mutation_variance':model.minimum_mutation_variance_,
            'history':model.history_,'fingerprint':digest,'parameters':model.get_params()}
        write_json(folder/'timing.json',result)
        write_json(folder/'progress.json',{'phase':'complete',**{k:v for k,v in result.items() if k not in ('history','parameters')}})
        return result

def analyze(dataset,settings):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    sys.path.insert(0,str(ROOT/'diagnosis/paper'))
    from core import load_with_shape
    from section6_methods import predictive_baselines
    from run_smc_population import unordered_geometry,unordered_scores
    split,tag=split_data(dataset)
    folder=Path(settings['output'])/dataset
    count=settings['draws']; truth=split['y_test_fixed']; scale=float(np.ptp(split['y_train'])) or 1.
    references=predictive_baselines(split['y_train'],truth)
    p=np.stack([np.load(folder/f'chain{r:03d}'/'preds.npy') for r in range(4)])
    v=np.stack([np.load(folder/f'chain{r:03d}'/'sigmas.npy') for r in range(4)])
    methods={'conditional_smc':(p,v)}
    source=ROOT/'diagnosis/store'/tag
    for method in ('default','default_pt','mtmh','mtmh_pt','default_long'):
        a=load_with_shape(source/'preds'/f'{tag}__run000__{method}__preds.csv').transpose(0,2,1)[:,3000:]
        b=load_with_shape(source/'sigmas'/f'{tag}__run000__{method}__sigmas.csv').reshape(4,-1)[:,3000:]
        if count>a.shape[1]: raise ValueError('Matched baseline count exceeds saved draws.')
        ix=np.linspace(0,a.shape[1]-1,count,dtype=int)
        methods[method]=(a[:,ix],b[:,ix])
    old=ROOT/'diagnosis/store_smc'/tag/'run000_p50_w0_s1000'/'replicates'
    if old.exists():
        ix=np.linspace(860*50,1000*50-1,count,dtype=int)
        a=np.stack([np.load(old/f'replicate{r:03d}'/'preds.npy',mmap_mode='r')[ix] for r in range(4)])
        b=np.stack([np.load(old/f'replicate{r:03d}'/'sigmas.npy',mmap_mode='r')[ix] for r in range(4)])
        methods['historical_smc_modified_kernel']=(a,b)
    rows=[]
    for name,(a,b) in methods.items():
        scores=np.array([unordered_scores(x,z,truth,scale) for x,z in zip(a,b)])
        rows.append({'method':name,'samples_per_chain':count,**unordered_geometry(a),
            'relative_rmse':float(np.sqrt(np.mean(scores[:,0]**2))/references['naive_rmse']),
            'relative_crps':float(scores[:,1].mean()/references['climatology_crps'])})
    # Same axes, same saved count for each method; the reference is not truth.
    pca=PCA(n_components=2).fit(np.concatenate([a.reshape(-1,len(truth)) for a,b in methods.values()]))
    fig,axes=plt.subplots(2,4,figsize=(13,6),layout='constrained',sharex=True,sharey=True)
    for ax,(name,(a,b)) in zip(axes.ravel(),methods.items()):
        for r in range(4):
            xy=pca.transform(a[r]); ax.scatter(xy[:,0],xy[:,1],s=10,alpha=.65,label=f'chain {r+1}')
            center=xy.mean(axis=0); ax.scatter(*center,s=60,marker='x',color=f'C{r}')
        ax.set_title(name.replace('_',' '),fontsize=9); ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    for ax in axes.ravel()[len(methods):]: ax.set_visible(False)
    fig.suptitle(f'{dataset} run0: {count} states per chain; exploratory short conditional-SMC test')
    fig.savefig(folder/'common_pca.png',dpi=150); plt.close(fig)
    times=[json.loads((folder/f'chain{r:03d}'/'timing.json').read_text()) for r in range(4)]
    write_json(folder/'analysis.json',{'dataset':dataset,'metrics':rows,'references':references,
        'timing':times,'pca_explained_variance_ratio':pca.explained_variance_ratio_,
        'pca_fit':'all displayed methods, equal saved counts, response scale, no standardization',
        'test_not_7000_draw_experiment':count!=7000,'pca_does_not_establish_bias':True,
        'four_chains_parallel_one_cpu_each':True})
    print(json.dumps({'dataset':dataset,'metrics':rows}),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',nargs='+',default=['abalone','seoul_bike'],choices=['abalone','seoul_bike'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--n-particles',type=int,default=50)
    parser.add_argument('--draws',type=int,default=40)
    parser.add_argument('--burn',type=int,default=10)
    parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--analyze-only',action='store_true')
    args=parser.parse_args()
    if min(args.n_particles,args.draws,args.jobs)<1 or args.burn<0 or args.jobs>4: parser.error('Invalid counts.')
    settings={**vars(args),'output':str(args.output.resolve()),
              'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    args.output.mkdir(parents=True,exist_ok=args.resume or args.analyze_only)
    old_path=args.output/'settings.json'
    if old_path.exists():
        old=json.loads(old_path.read_text())
        for k in ('n_particles','draws','burn'):
            if settings[k]!=old[k]: raise ValueError(f'Cannot resume with different {k}.')
    write_json(old_path,settings)
    for dataset in args.datasets:
        if not args.analyze_only:
            started=time.perf_counter()
            with ProcessPoolExecutor(max_workers=args.jobs,mp_context=mp.get_context('spawn')) as pool:
                futures=[pool.submit(worker,dataset,r,settings) for r in range(4)]
                for f in as_completed(futures):
                    result=f.result()
                    print(json.dumps({k:result[k] for k in ('dataset','chain','stages','elapsed_seconds')}),flush=True)
            write_json(args.output/dataset/'batch.json',{'wall_seconds_this_invocation':time.perf_counter()-started,
                'resume':args.resume,'workers':args.jobs,'particle_workers_per_chain':1})
        analyze(dataset,settings)

if __name__=='__main__': main()
