import numpy as np
import pytest

from diagnosis.analyze_smc_windows import population_windows, score_set


def test_long_run_windows_keep_all_particles_and_include_legacy_and_final_ranges():
    windows=population_windows(1000,50,140,20)
    assert len(windows)==44
    assert (windows[0]["first_sweep"],windows[0]["last_sweep"])==(1,140)
    assert (windows[-1]["first_sweep"],windows[-1]["last_sweep"])==(861,1000)
    assert any((w["first_sweep"],w["last_sweep"])==(61,200) for w in windows)
    for w in windows:
        ids=np.arange(w["slice_start"],w["slice_stop"])
        assert len(ids)==w["samples_per_replicate"]==7000
        np.testing.assert_array_equal(np.bincount(ids%50),np.full(50,140))
        assert ids.min()//50+1==w["first_sweep"]
        assert ids.max()//50+1==w["last_sweep"]


def test_window_schedule_covers_final_window_when_stride_does_not_divide_range():
    windows=population_windows(11,2,4,3)
    assert [w["first_sweep"] for w in windows]==[1,4,7,8]
    assert windows[-1]["slice_stop"]==22
    assert len(population_windows(140,50))==1


@pytest.mark.parametrize("total,width,stride",[(100,140,20),(1000,0,20),(1000,140,0)])
def test_window_schedule_rejects_missing_or_invalid_windows(total,width,stride):
    with pytest.raises(ValueError):
        population_windows(total,50,width,stride)


def test_window_scores_use_mean_predictions_and_preserve_variance_units():
    p=np.tile(np.array([-1.,1.]),20).reshape(4,10,1)
    v=np.full((4,10),.25)
    metrics,repeats=score_set(p,v,np.zeros(1),4.)
    assert metrics["R_BW"]==0
    assert metrics["rmse_single_mean"]==metrics["rmse_pooled"]==0
    assert metrics["noise_variance_mean"]==4
    assert metrics["latent_90pct_width_mean"]==2
    assert metrics["crps_single_mean"]>0
    assert len(repeats)==4 and all(r["noise_variance_mean"]==4 for r in repeats)
