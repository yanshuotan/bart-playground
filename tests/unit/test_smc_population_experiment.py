import numpy as np
import pytest

from diagnosis.run_smc_population import (balanced_population_indices, unordered_geometry,
    unordered_scores, validate_stored_split, retained_block_diagnostics, save_csv)


def test_retained_blocks_use_posterior_mean_prediction_and_response_scale_variance():
    draws = np.zeros((2,2000,1))
    draws[:,:1000,0] = np.tile([-1.,1.],500)
    draws[:,1000:,0] = 2
    variances = np.ones((2,2000))
    variances[:,1000:] = 2
    rows = retained_block_diagnostics(draws,variances,np.zeros(1),3)
    assert len(rows) == 4
    for first,last in zip(rows[::2],rows[1::2]):
        assert first["rmse"] == 0  # Not the mean draw-wise RMSE of one.
        assert last["rmse"] == 2
        assert first["noise_variance"] == 9 and last["noise_variance"] == 18
        assert first["first_sample"] == 0 and last["last_sample_exclusive"] == 2000


@pytest.mark.parametrize("mismatch", [None,"train_idx","test_idx","X_test_fixed","y_test_fixed"])
def test_dataset_comparison_requires_identical_split_and_encoded_test_data(tmp_path, mismatch):
    tag = "fixed100_SeoulBike"
    split = {"train_idx":np.array([0,2,4]),"test_idx":np.array([1,3]),
             "X_test_fixed":np.arange(6,dtype=float).reshape(2,3),
             "y_test_fixed":np.array([10.,20.])}
    for name,key in (("train_idx","train_idx"),("fixed_test_idx","test_idx"),
                     ("subsample_X_test","X_test_fixed"),("subsample_y_test","y_test_fixed")):
        folder = "indices" if key.endswith("idx") else name
        save_csv(tmp_path/folder/f"{tag}__run000__{name}.csv",split[key])
    if mismatch is None:
        validate_stored_split(split,tmp_path,tag)
    else:
        split[mismatch] = split[mismatch].copy()
        split[mismatch].flat[0] += 1
        with pytest.raises(AssertionError):
            validate_stored_split(split,tmp_path,tag)


@pytest.mark.parametrize("particles,sweeps", [(50,140), (100,100)])
def test_matched_counts_preserve_complete_populations(particles, sweeps):
    indices = balanced_population_indices(particles, sweeps)
    assert len(indices) == len(np.unique(indices)) == 7000
    assert indices.min() == 0 and indices.max() == particles*sweeps-1
    np.testing.assert_array_equal(indices.reshape(-1, particles) % particles,
                                  np.tile(np.arange(particles), (7000//particles, 1)))
    if particles == 50:
        np.testing.assert_array_equal(indices, np.arange(7000))


def test_matched_counts_reject_incomplete_or_unavailable_populations():
    with pytest.raises(ValueError):
        balanced_population_indices(50, 139)
    with pytest.raises(ValueError):
        balanced_population_indices(60, 140)


def test_unordered_metrics_are_invariant_to_within_replicate_permutations():
    rng = np.random.default_rng(197)
    samples = rng.normal(size=(4, 31, 5))
    variances = rng.uniform(.1, .3, size=(4, 31))
    truth = rng.normal(size=5)
    shuffled = np.empty_like(samples)
    shuffled_variances = np.empty_like(variances)
    for replicate in range(4):
        order = rng.permutation(31)
        shuffled[replicate] = samples[replicate, order]
        shuffled_variances[replicate] = variances[replicate, order]
    for key, value in unordered_geometry(samples).items():
        np.testing.assert_allclose(unordered_geometry(shuffled)[key], value, atol=1e-14)
    for a, b, v, w in zip(samples, shuffled, variances, shuffled_variances):
        np.testing.assert_allclose(unordered_scores(a, v, truth, 3),
                                   unordered_scores(b, w, truth, 3), atol=1e-14)


def test_predictive_crps_includes_observation_variance():
    # The latent function is known at zero; CRPS of the predictive N(0,1)
    # against y=0 is (sqrt(2)-1)/sqrt(pi), rather than latent CRPS zero.
    n = 40000
    rmse, crps = unordered_scores(np.zeros((n, 1)), np.ones(n), np.zeros(1), 1)
    assert rmse == 0
    assert abs(crps - (np.sqrt(2)-1)/np.sqrt(np.pi)) < .008
