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
DATA_ROOT=Path(os.environ.get('CSMC_DATA_ROOT',str(ROOT))).resolve()
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'diagnosis'))

def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else str(a)),encoding='utf-8')
    for attempt in range(100):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt==99: raise
            time.sleep(.02)

def split_data(dataset):
    import fixed100_support as support
    support.UCI_CACHE_DIR=DATA_ROOT/'diagnosis/store/uci_cache'
    support.OPENML_CACHE_DIR=DATA_ROOT/'diagnosis/store/openml_cache'
    from fixed100_support import load_dataset,DATASET_CONFIGS
    from experiment_fixed100 import make_fixed100_splits
    from run_smc_population import validate_stored_split
    X,y=load_dataset(dataset)
    split=make_fixed100_splits(X,y,n_runs=1)[0]
    tag=DATASET_CONFIGS[dataset]['dataset_tag']
    validate_stored_split(split,DATA_ROOT/'diagnosis/store'/tag,tag)
    return split,tag

def worker(dataset,index,settings):
    from bart_playground import ConditionalSMCBART
    worker_commit=os.environ.get('CSMC_CODE_COMMIT') or subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
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
            if (model.n_particles!=settings['n_particles'] or model.n_mcmc_steps!=2
                    or model.sampler.tree_prior.n_trees!=100):
                raise ValueError('Checkpoint particle count or mutation settings differ.')
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
            'worker_code_commit':worker_commit,
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
    for r in range(4):
        chain_folder=folder/f'chain{r:03d}'
        np.save(chain_folder/'rmses.npy',np.sqrt(np.mean((p[r]-truth)**2,axis=1)))
        np.save(chain_folder/'conditional_iteration.npy',np.arange(settings['burn']+1,settings['burn']+count+1))
    methods={'conditional_smc':(p,v)}
    source=DATA_ROOT/'diagnosis/store'/tag
    full_long=load_with_shape(source/'preds'/f'{tag}__run000__default_long__preds.csv').transpose(0,2,1)[:,3000:]
    long_center=full_long.mean(axis=(0,1))
    for method in ('default','default_pt','mtmh','mtmh_pt','default_long'):
        a=load_with_shape(source/'preds'/f'{tag}__run000__{method}__preds.csv').transpose(0,2,1)[:,3000:]
        b=load_with_shape(source/'sigmas'/f'{tag}__run000__{method}__sigmas.csv').reshape(4,-1)[:,3000:]
        if count>a.shape[1]: raise ValueError('Matched baseline count exceeds saved draws.')
        ix=np.linspace(0,a.shape[1]-1,count,dtype=int)
        methods[method]=(a[:,ix],b[:,ix])
    old=DATA_ROOT/'diagnosis/store_smc'/tag/'run000_p50_w0_s1000'/'replicates'
    if old.exists():
        ix=np.linspace(860*50,1000*50-1,count,dtype=int)
        a=np.stack([np.load(old/f'replicate{r:03d}'/'preds.npy',mmap_mode='r')[ix] for r in range(4)])
        b=np.stack([np.load(old/f'replicate{r:03d}'/'sigmas.npy',mmap_mode='r')[ix] for r in range(4)])
        methods['historical_smc_modified_kernel']=(a,b)
    rows=[]
    for name,(a,b) in methods.items():
        scores=np.array([unordered_scores(x,z,truth,scale) for x,z in zip(a,b)])
        within=float(np.mean(np.sum((a-a.mean(axis=1)[:,None,:])**2,axis=2)))
        geometry=(unordered_geometry(a) if within>0 else {
            'R_BW':None,'within_population_variance':0.,
            'between_centroid_variance':float(np.mean(np.sum((a.mean(axis=1)-a.mean(axis=(0,1)))**2,axis=1)))})
        rows.append({'method':name,'samples_per_chain':count,**geometry,
            'mean_centroid_distance_to_full_long':float(np.linalg.norm(a.mean(axis=1)-long_center,axis=1).mean()),
            'distinct_predictions_per_chain':[int(len(np.unique(x,axis=0))) for x in a],
            'relative_rmse':float(np.sqrt(np.mean(scores[:,0]**2))/references['naive_rmse']),
            'relative_crps':float(scores[:,1].mean()/references['climatology_crps'])})
    # Same axes, same saved count for each method; the reference is not truth.
    pca=PCA(n_components=2,svd_solver='full').fit(np.concatenate([a.reshape(-1,len(truth)) for a,b in methods.values()]))
    from scipy.stats import gaussian_kde
    reference_xy=pca.transform(full_long.reshape(-1,len(truth)))
    all_xy=np.concatenate([reference_xy]+[pca.transform(a.reshape(-1,len(truth))) for a,b in methods.values()])
    low=all_xy.min(axis=0); high=all_xy.max(axis=0)
    pad=np.maximum((high-low)*.06,1e-6)
    gx=np.linspace(low[0]-pad[0],high[0]+pad[0],90)
    gy=np.linspace(low[1]-pad[1],high[1]+pad[1],90)
    xx,yy=np.meshgrid(gx,gy)
    density=gaussian_kde(reference_xy.T)(np.vstack([xx.ravel(),yy.ravel()])).reshape(xx.shape)
    levels=np.unique(np.quantile(density,[.5,.75,.9,.97]))
    np.savez(folder/'pca_axes.npz',components=pca.components_,mean=pca.mean_,
             explained_variance_ratio=pca.explained_variance_ratio_,long_reference_center=long_center)
    columns=3 if len(methods)<=6 else 4
    rows_count=(len(methods)+columns-1)//columns
    fig,axes=plt.subplots(rows_count,columns,figsize=(3*columns,3*rows_count),
                          layout='constrained',sharex=True,sharey=True,squeeze=False)
    names={'conditional_smc':f'Conditional SMC (N={settings["n_particles"]}, K=2)',
           'default':'Default','default_pt':'Default + PT','mtmh':'MTMH',
           'mtmh_pt':'MTMH + PT','default_long':'Default long',
           'historical_smc_modified_kernel':'Historical SMC (modified kernel)'}
    for ax,(name,(a,b)) in zip(axes.ravel(),methods.items()):
        ax.contour(xx,yy,density,levels=levels,colors='0.7',linewidths=.7)
        for r in range(4):
            xy=pca.transform(a[r]); ax.scatter(xy[:,0],xy[:,1],s=10,alpha=.65,label=f'chain {r+1}')
            center=xy.mean(axis=0); ax.scatter(*center,s=60,marker='x',color=f'C{r}')
        ax.set_title(names[name],fontsize=12)
        ax.set_xlabel(f'PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)')
        ax.set_ylabel(f'PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)')
    axes[0,0].legend(fontsize=7,loc='best',frameon=False)
    for ax in axes.ravel()[len(methods):]: ax.set_visible(False)
    label={'abalone':'Abalone','seoul_bike':'SeoulBike'}[dataset]
    fig.suptitle(f'{label}, run0: {count} selected states per chain\nShort local test; grey contours: stored long Default',fontsize=12)
    fig.savefig(folder/'common_pca.png',dpi=150); plt.close(fig)
    times=[json.loads((folder/f'chain{r:03d}'/'timing.json').read_text()) for r in range(4)]
    write_json(folder/'analysis.json',{'dataset':dataset,'metrics':rows,'references':references,
        'timing':times,'pca_explained_variance_ratio':pca.explained_variance_ratio_,
        'pca_fit':'all displayed methods, equal saved counts, response scale, no standardization',
        'grey_contours':'all saved post-burn long Default states; descriptive reference, not ground truth',
        'long_reference_samples_per_chain':full_long.shape[1],
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
    code_commit=os.environ.get('CSMC_CODE_COMMIT')
    if not code_commit:
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    settings={**vars(args),'output':str(args.output.resolve()),'data_root':str(DATA_ROOT),
              'code_commit':code_commit}
    args.output.mkdir(parents=True,exist_ok=args.resume or args.analyze_only)
    old_path=args.output/'settings.json'
    if old_path.exists():
        old=json.loads(old_path.read_text())
        for k in ('n_particles','draws','burn'):
            if settings[k]!=old[k]: raise ValueError(f'Cannot resume with different {k}.')
        settings['sampling_code_commit']=old.get('sampling_code_commit',old['code_commit'])
        if args.analyze_only:
            settings['analysis_code_commit']=settings['code_commit']
            settings['code_commit']=settings['sampling_code_commit']
    if old_path.exists():
        write_json(args.output/'latest_invocation.json',settings)
    else:
        write_json(old_path,settings)
    for dataset in args.datasets:
        already_complete=all((args.output/dataset/f'chain{r:03d}'/'timing.json').exists() for r in range(4))
        if not args.analyze_only and not (args.resume and already_complete):
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
