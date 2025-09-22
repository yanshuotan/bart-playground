import yaml
import numpy as np
from bart_playground.DataGenerator import DataGenerator

# --- Configuration ---
SCENARIOS = {
    "low_lei_candes": {"n_features": 10},
    "high_lei_candes": {"n_features": 100},
    "piecewise_linear_kunzel": {"n_features": 10},
    "linear_additive": {"n_features": 10}
}
N_SAMPLES = 10000000  # Large sample size for stable variance estimation
OUTPUT_FILE = "experiments/data_variances.yaml"

# --- Main Script ---
def calculate_variances():
    """
    Generates large, noiseless samples for specified DGPs, calculates the
    empirical variance of the outcomes, and saves them to a YAML file.
    """
    variances = {}
    
    print("Starting variance calculation for DGPs...")

    for scenario, params in SCENARIOS.items():
        print(f"  - Processing scenario: {scenario} with {params['n_features']} features...")
        
        # Initialize the data generator for a large, noiseless sample
        generator = DataGenerator(
            n_samples=N_SAMPLES,
            n_features=params['n_features'],
            snr=np.inf,  # Ensure no noise is added
            random_seed=42  # Use a fixed seed for reproducibility
        )
        
        # Generate the data
        _, y_noiseless = generator.generate(scenario=scenario)
        
        # Calculate the empirical variance
        variance = np.var(y_noiseless)
        variances[scenario] = {
            'signal_variance': float(variance),
            'required_noise_for_snr_1': float(variance)
        }
        
        print(f"    - Calculated signal variance: {variance:.4f}")

    # Save the results to a YAML file
    with open(OUTPUT_FILE, 'w') as f:
        yaml.dump(variances, f, default_flow_style=False)
        
    print(f"\nVariances successfully saved to {OUTPUT_FILE}")
    print("You can now use these 'required_noise_for_snr_1' values in your experiment configs.")

if __name__ == "__main__":
    calculate_variances() 