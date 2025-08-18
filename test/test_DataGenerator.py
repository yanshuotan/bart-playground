import pytest
import numpy as np

from bart_playground.DataGenerator import DataGenerator


# A list of scenarios to test against. We exclude those with special noise handling
# or dictionary-based returns for the standard SNR tests.
STANDARD_SCENARIOS = [
    "linear", "sparse_linear", "piecewise_flat", "cyclic", "imbalanced",
    "piecewise_linear", "tied_x", "tied_y", "friedman1", "friedman2",
    "friedman3", "linear_additive", "smooth_additive", "dgp_1", "dgp_2",
    "low_lei_candes", "high_lei_candes", "lss", "piecewise_linear_kunzel", "sum"
]


@pytest.mark.parametrize("scenario", STANDARD_SCENARIOS)
def test_snr_infinite_produces_noiseless_data(scenario):
    """
    Test that setting SNR to np.inf produces a noiseless dataset for a range of scenarios.
    """
    # Generate a noiseless version of the data first to get the ground truth
    generator_inf = DataGenerator(n_samples=1000, n_features=10, snr=np.inf, random_seed=42)
    _, y_inf = generator_inf.generate(scenario=scenario)

    # To get the true noiseless value, we can run the scenario method directly
    # This is a bit of a hack, but it isolates the test from the noise-adding logic
    generator_base = DataGenerator(n_samples=1000, n_features=10, random_seed=42)
    _, y_noiseless = getattr(generator_base, scenario)()

    assert np.allclose(y_inf, y_noiseless, atol=1e-6)


@pytest.mark.parametrize(
    "scenario", 
    ["linear", "friedman1", "smooth_additive", "low_lei_candes", "high_lei_candes", "piecewise_linear", "linear_additive"]
)
@pytest.mark.parametrize("snr_target", [0.3, 1.5, 3.0])
def test_snr_calculation(scenario, snr_target):
    """
    Test that the calculated signal-to-noise ratio is close to the requested value
    for multiple scenarios and SNR targets.
    """
    generator = DataGenerator(n_samples=2000, n_features=10, snr=snr_target, random_seed=42)
    
    # Generate the underlying noiseless signal to compare against
    generator_noiseless = DataGenerator(n_samples=2000, n_features=10, snr=np.inf, random_seed=42)
    _, y_noiseless = generator_noiseless.generate(scenario=scenario)

    # Now generate the noisy version
    _, y_noisy = generator.generate(scenario=scenario)
    
    signal_variance = np.var(y_noiseless)
    noise_variance = np.var(y_noisy - y_noiseless)

    # If signal variance is close to zero, SNR is not well-defined.
    if signal_variance < 1e-9:
        # In this case, noise variance should also be close to zero.
        assert np.isclose(noise_variance, 0)
        return

    # Avoid division by zero if noise is negligible
    if noise_variance > 1e-9:
        actual_snr = signal_variance / noise_variance
        # We use a loose tolerance because the randomness can cause variance
        assert np.isclose(actual_snr, snr_target, rtol=0.4)
    else:
        # If noise is very low, actual_snr will be huge, which is expected for high target SNR
        assert snr_target > 1000


def test_snr_raises_error_for_invalid_value():
    """
    Test that a ValueError is raised if a non-positive SNR is provided.
    """
    with pytest.raises(ValueError, match="Signal-to-noise ratio \\(SNR\\) must be positive."):
        generator = DataGenerator(snr=0)
        generator.generate(scenario="linear")
    
    with pytest.raises(ValueError, match="Signal-to-noise ratio \\(SNR\\) must be positive."):
        generator = DataGenerator(snr=-10)
        generator.generate(scenario="linear")


def test_snr_with_zero_signal_variance():
    """
    Test that SNR works correctly when the underlying signal has zero variance.
    The noise should also be zero.
    """
    
    class ZeroVarianceScenario(DataGenerator):
        def constant(self):
            X = self.rng.uniform(0, 1, (self.n_samples, self.n_features))
            y_noiseless = np.ones(self.n_samples)
            return X, y_noiseless

    generator = ZeroVarianceScenario(n_samples=100, snr=10)
    X, y = generator.generate(scenario="constant")
    
    # Since signal variance is 0, noise variance should also be 0
    assert np.var(y) == 0


def test_special_scenarios_ignore_snr():
    """
    Test that scenarios with special, hard-coded noise structures
    are not affected by the top-level SNR parameter.
    """
    # These scenarios add their own noise, so the variance should be consistent
    # regardless of the top-level `snr` parameter.
    special_scenarios = ["heteroscedastic", "multimodal"]

    for scenario in special_scenarios:
        gen_no_snr = DataGenerator(n_samples=1000, n_features=10, random_seed=42)
        _, y_no_snr = gen_no_snr.generate(scenario=scenario)

        gen_with_snr = DataGenerator(n_samples=1000, n_features=10, snr=1000, random_seed=42)
        _, y_with_snr = gen_with_snr.generate(scenario=scenario)

        # The output should be identical because the seed is the same and SNR should be ignored
        assert np.allclose(y_no_snr, y_with_snr) 