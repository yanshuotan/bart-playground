"""Compare equal-size posterior population windows without discarding at collection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

for _name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMBA_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_name]="1"

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"diagnosis"))
from run_smc_population import unordered_geometry, unordered_scores, write_json


def population_windows(n_sweeps, n_particles=50, width=140, stride=20):
    """Whole-population windows, including the final complete window."""
    if min(n_sweeps,n_particles,width,stride) <= 0 or width > n_sweeps:
        raise ValueError("Positive dimensions and a complete window are required.")
    starts = sorted(set(range(0,n_sweeps-width+1,stride)) | {n_sweeps-width})
    return [{"first_sweep":start+1,"last_sweep":start+width,
             "discarded_sweeps":start,"samples_per_replicate":width*n_particles,
             "slice_start":start*n_particles,"slice_stop":(start+width)*n_particles}
            for start in starts]


def score_set(preds, sigmas, truth, y_scale):
    """The same prediction-space geometry and predictive scores as earlier runs."""
    scores = [unordered_scores(p,v,truth,y_scale) for p,v in zip(preds,sigmas)]
    pooled = unordered_scores(preds.reshape(-1,len(truth)),sigmas.reshape(-1),truth,y_scale)
    noise = sigmas*y_scale**2
    panel = preds.mean(axis=2)
    result = {**unordered_geometry(preds),
        "rmse_single_mean":float(np.mean([x[0] for x in scores])),
        "rmse_single_sd":float(np.std([x[0] for x in scores],ddof=1)),
        "rmse_root_mean_mse":float(np.sqrt(np.mean([x[0]**2 for x in scores]))),
        "crps_single_mean":float(np.mean([x[1] for x in scores])),
        "crps_single_sd":float(np.std([x[1] for x in scores],ddof=1)),
        "rmse_pooled":pooled[0],"crps_pooled":pooled[1],
        "noise_variance_mean":float(noise.mean()),
        "noise_variance_repeat_mean_sd":float(np.std(noise.mean(axis=1),ddof=1)),
        "panel_mean_f":float(panel.mean()),
        "panel_mean_f_repeat_sd":float(np.std(panel.mean(axis=1),ddof=1))}
    for name,values in (("noise_variance",noise),("panel_mean_f",panel)):
        result[f"{name}_pairwise_wasserstein"] = float(np.mean([
            wasserstein_distance(values[i],values[j])
            for i in range(len(values)) for j in range(i+1,len(values))]))
    low,high=np.quantile(preds,[.05,.95],axis=1)
    result["latent_90pct_width_mean"] = float(np.mean(high-low))
    per_repeat = [{"replicate":r,"rmse":scores[r][0],"crps":scores[r][1],
        "noise_variance_mean":float(noise[r].mean()),"panel_mean_f":float(panel[r].mean()),
        "latent_90pct_width_mean":float((high[r]-low[r]).mean())}
        for r in range(len(preds))]
    return result,per_repeat


def validate_run(output, settings):
    n,s = settings["n_particles"],settings["retained_sweeps"]
    checks=[]
    for r in range(4):
        folder=output/"replicates"/f"replicate{r:03d}"
        p=np.load(folder/"preds.npy",mmap_mode="r")
        v=np.load(folder/"sigmas.npy",mmap_mode="r")
        h=np.load(folder/"vars_histogram.npy",mmap_mode="r")
        if p.shape != (n*s,settings["fixed_test_rows"]) or v.shape != (n*s,):
            raise ValueError(f"Unexpected posterior shape for repeat {r}.")
        if h.shape != (n*s,settings["n_features"]):
            raise ValueError("Unexpected feature-count shape.")
        if not (np.isfinite(p).all() and np.isfinite(v).all() and np.all(v>0)):
            raise ValueError("Non-finite output or nonpositive variance.")
        np.testing.assert_array_equal(np.load(folder/"particle_id.npy"),np.tile(np.arange(n),s))
        np.testing.assert_array_equal(np.load(folder/"population_sweep.npy"),np.repeat(np.arange(s),n))
        changed=np.any(np.diff(p.reshape(s,n,-1),axis=0)!=0,axis=2)
        if not changed.all():
            raise ValueError("An adjacent per-particle prediction vector did not change.")
        timing=json.loads((folder/"timing.json").read_text())
        if sum(timing["mutation_selected"].values()) != (settings["posterior_warmup_sweeps"]+s)*n*settings["n_trees"]:
            raise ValueError("Mutation count does not match the requested experiment.")
        stages=json.loads((folder/"annealing.json").read_text())
        if stages[-1]["beta"] != 1:
            raise ValueError("Annealing did not finish.")
        checks.append({"replicate":r,"predictions_shape":list(p.shape),
            "adjacent_particle_predictions_changed":int(changed.sum()),
            "annealing_stages":len(stages),"total_seconds":timing["total_seconds"]})
    return checks


def analyze_windows(output, source=None, width=140, stride=20):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from fixed100_support import load_dataset
    sys.path.insert(0,str(ROOT/"diagnosis/paper"))
    from core import load_with_shape

    output=Path(output)
    settings=json.loads((output/"settings.json").read_text())
    tag=settings["dataset_tag"]
    source=Path(source) if source is not None else ROOT/"diagnosis/store"/tag
    dest=output/"window_diagnostics"
    dest.mkdir(exist_ok=True)
    checks=validate_run(output,settings)
    _,y=load_dataset(settings["dataset"])
    idx=load_with_shape(output/"indices"/f"{tag}__run000__train_idx.csv").astype(int).ravel()
    truth=load_with_shape(output/"subsample_y_test"/f"{tag}__run000__subsample_y_test.csv").ravel()
    y_scale=float(np.ptp(y[idx])) or 1.
    p=np.stack([np.load(output/"replicates"/f"replicate{r:03d}"/"preds.npy") for r in range(4)])
    v=np.stack([np.load(output/"replicates"/f"replicate{r:03d}"/"sigmas.npy") for r in range(4)])
    windows=population_windows(settings["retained_sweeps"],settings["n_particles"],width,stride)
    metrics,replicates=[],[]
    for position,window in enumerate(windows):
        start,stop=window["slice_start"],window["slice_stop"]
        label={k:x for k,x in window.items() if not k.startswith("slice_")}
        score,repeats=score_set(p[:,start:stop],v[:,start:stop],truth,y_scale)
        metrics.append({**label,**score})
        replicates.extend({**label,**record} for record in repeats)
        if position%10==0 or position==len(windows)-1:
            pd.DataFrame(metrics).to_csv(dest/"population_window_metrics.csv",index=False)
            write_json(dest/"progress.json",{"phase":"window_scoring", "completed":position+1,"total":len(windows)})
            print(f"{tag}: window {position+1}/{len(windows)} ({window['first_sweep']}-{window['last_sweep']})",flush=True)
    frame=pd.DataFrame(metrics)
    frame.to_csv(dest/"population_window_metrics.csv",index=False)
    pd.DataFrame(replicates).to_csv(dest/"population_window_replicates.csv",index=False)

    baselines=[]
    baseline_replicates=[]
    for method in ("default","default_pt","mtmh","mtmh_pt","default_long","default_long_tail"):
        path=source/"preds"/f"{tag}__run000__{method}__preds.csv"
        if not path.is_file():
            continue
        a=load_with_shape(path).transpose(0,2,1)[:,3000:]
        b=load_with_shape(source/"sigmas"/f"{tag}__run000__{method}__sigmas.csv").reshape(4,-1)[:,3000:]
        if a.shape[1] != width*settings["n_particles"]:
            raise ValueError("Baseline and population windows must have the same sample count.")
        score,repeats=score_set(a,b,truth,y_scale)
        baselines.append({"method":method,"samples_per_replicate":a.shape[1],**score})
        baseline_replicates.extend({"method":method,**row} for row in repeats)
    baseline=pd.DataFrame(baselines)
    baseline.to_csv(dest/"baseline_comparison.csv",index=False)
    pd.DataFrame(baseline_replicates).to_csv(dest/"baseline_replicates.csv",index=False)
    # Explicitly separate consecutive disjoint windows from overlapping sensitivity windows.
    disjoint=frame.loc[(frame.discarded_sweeps%width)==0]
    disjoint.to_csv(dest/"nonoverlapping_windows.csv",index=False)
    anchors={0,60,140,200,280,400,420,560,700,840,settings["retained_sweeps"]-width}
    selected=frame.loc[frame.discarded_sweeps.isin(anchors)]
    selected.to_csv(dest/"selected_windows.csv",index=False)

    # With identical seeds, saving early states must not alter the sampler trajectory.
    old=output.parent/"run000_p50_w60_s140"
    replay=[]
    if old.is_dir() and settings["n_particles"]==50 and settings["posterior_warmup_sweeps"]==0 and settings["retained_sweeps"]>=200:
        for r in range(4):
            folder=old/"replicates"/f"replicate{r:03d}"
            previous_p=np.load(folder/"preds.npy")
            previous_v=np.load(folder/"sigmas.npy")
            np.testing.assert_array_equal(p[r,60*50:200*50],previous_p)
            np.testing.assert_array_equal(v[r,60*50:200*50],previous_v)
            replay.append({"replicate":r,"previous_61_to_200_window_reproduced_exactly":True})

    colors={"default":"#d17b0f","default_pt":"#287bb5","mtmh":"#9250ac",
            "mtmh_pt":"#bc4256","default_long":"#555555","default_long_tail":"#999999"}
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout="constrained")
    for ax,key,title in zip(axes,("R_BW","rmse_root_mean_mse","crps_single_mean"),
                            ("Prediction-space R_B/W","Posterior-mean RMSE","Predictive MC-CRPS")):
        ax.plot(frame.first_sweep,frame[key],marker=".",color="#147451",label="SMC windows",linewidth=2)
        for row in baselines:
            ax.axhline(row[key],color=colors[row["method"]],linestyle="--",alpha=.8,label=row["method"])
        if key=="R_BW":
            ax.set_yscale("log")
        ax.set(title=title,xlabel="first posterior sweep in window")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{tag.removeprefix('fixed100_')} run 0 | four repeats | each window: {width} sweeps, {width*settings['n_particles']:,} states\nOverlapping windows describe sensitivity, not independent replications; budgets differ between methods")
    fig.savefig(dest/"window_comparison.png",dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout="constrained")
    repeats_frame=pd.DataFrame(replicates)
    for ax,key,title in zip(axes,("noise_variance_mean","panel_mean_f","latent_90pct_width_mean"),
                            ("Noise variance","Fixed-panel mean f","Mean 90% latent interval width")):
        for r,values in repeats_frame.groupby("replicate"):
            ax.plot(values.first_sweep,values[key],label=f"repeat {r+1}")
        ax.set(title=title,xlabel="first posterior sweep in window"); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(dest/"window_scalar_sensitivity.png",dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout="constrained")
    for ax,key,title in zip(axes,("between_centroid_variance","within_population_variance"),
                            ("Between-repeat component B","Within-repeat component W")):
        ax.plot(frame.first_sweep,frame[key],color="#147451")
        ax.set(title=title,xlabel="first posterior sweep in window"); ax.grid(alpha=.2)
    fig.savefig(dest/"window_geometry_components.png",dpi=160); plt.close(fig)
    write_json(dest/"analysis_settings.json",{
        "collection_burn":settings["posterior_warmup_sweeps"],"recorded_sweeps":settings["retained_sweeps"],
        "particles":settings["n_particles"],"window_width_sweeps":width,"window_stride_sweeps":stride,
        "sweep_labels":"1-based inclusive; particle/sweep arrays remain zero-based",
        "baseline_excluded_prefix_states":3000,"score_seed":20261012,
        "samples_are_iid":False,"flattened_rank_rhat_reported":False,"flattened_mcmc_ess_reported":False,
        "notes":["Each window retains all particles in every selected sweep.",
                 "Overlapping windows are dependent; no best window is selected for the main conclusion.",
                 "No stationarity acceptance threshold is applied.",
                 "Baselines are comparison references, with different computational budgets."]})
    write_json(dest/"validation.json",{"replicates":checks,"windows":len(windows),
        "previous_run_replay":replay,"all_metrics_finite":bool(np.isfinite(frame.select_dtypes('number')).all().all())})
    write_json(dest/"progress.json",{"phase":"complete","completed":len(windows),"total":len(windows)})
    print(selected.to_string(index=False),flush=True)
    return frame


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--source",type=Path)
    parser.add_argument("--width",type=int,default=140)
    parser.add_argument("--stride",type=int,default=20)
    args=parser.parse_args()
    analyze_windows(args.output,args.source,args.width,args.stride)


if __name__=="__main__":
    main()
