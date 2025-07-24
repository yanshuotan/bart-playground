For all three experiments, the data sets used, and settings of BART are kept identical except for the parameter being tested. All three experiments are run for 100 monte carlo replications. The seeding of both the data set generation and the BART fit is set based on a hash of the main seed and the parameters for the current run to ensure replicability, but unique seeding by data set / iteration / parameters. All the data sets are taken from the implementation in the DataGenerator class. The noise for each data set is set slightly differently, `low_lei_candes` is given a noise variance of 1 to match the original implementation, for `piecewise_linear`, the noise variance is selected for each monte carlo replication to give a signal to noise ratio of 1 (equal variance to the noiseless Y), and for `linear_additive` the noise variance is set to .1. This generates a range of difficulties across the data sets in terms of signal to noise.

## Datasets

### 1. Low Dimensional Smooth
- **DGP**: `low_lei_candes`
- **Features**: 10
- **Description**: 10-dimensional Multivariate normal covariates with mean zero and covariance matrix with 1 on diagonal and 0.01 off-diagonal. Response is generated as g(x₀) * g(x₁) + noise, where g(x) = 2/(1+exp(-12(x-0.5)))

### 2. Piecewise Linear
- **DGP**: `piecewise_linear`
- **Features**: 20
- **Description**: 20-dimensional Multivariate normal covariates with mean zero and covariance matrix with 1 on diagonal and 0.01 off-diagonal.Response is linear with three pieces based on x₁₉: x₁₉ > 4, x₁₉ < -4, and -4 ≤ x₁₉ ≤ 4. Coefficients sampled uniformly from [-15, 15]. Seeding is deterministic for the coefficients, so each monte carlo replication has the same values.

### 3. Linear Additive
- **DGP**: `linear_additive`
- **Features**: 10
- **Description**: 10-dimensional Multivariate normal covariates with mean zero and covariance matrix with 1 on diagonal. Response is a linear additive model with 10 features, and 5 fixed coefficient, active features. 

## BART Model Parameters

### Base Parameters (Common across all experiments)
- **Number of Trees**: 200
- **Posterior Samples**: 10,000
- **Burn-in Samples**: 1,000
- **Tree Prior Parameters**:
  - α (tree_alpha): 0.95
  - β (tree_beta): 2.0
  - f_k: 2.0
- **Proposal Probabilities**:
  - Grow: 0.25
  - Prune: 0.25
  - Change: 0.4
  - Swap: 0.1

## Experiment-Specific Parameters

### 1. Temperature Experiment
- **Temperatures Tested**: [1.0, 2.0, 5.0]
- **Number of Chains**: 10
- **Training Sample Sizes**: [100, 200, 500, 1,000, 10,000]
- **Test Sample Size**: 10,000
- **Main Seed**: 1-100 

### 2. Schedule Experiment
- **Temperature Schedules**:
  - Cosine: T(t) = t_min + 0.5 * (t_max - t_min) * (1 + cos(π * t/total_iters))
    - t_max = 5.0, t_min = 0.1
  - Linear: T(t) = t_max - (t_max - t_min) * (t/total_iters)
    - t_max = 5.0, t_min = 1.0
  - Exponential: T(t) = t_max * (γ^t)
    - t_max = 5.0, γ = 0.999
- **Total Iterations**: 11,000 (1,000 burn-in + 10,000 posterior)
- **Number of Chains**: 10
- **Training Sample Sizes**: [100, 200, 500, 1,000, 10,000]
- **Test Sample Size**: 10,000
- **Main Seed**: 1-100 

### 3. Initialization Experiment
- **Initialization Methods**:
  - From Scratch
  - XGBoost Initialization
- **XGBoost Parameters**:
  - max_depth: 4
  - learning_rate: 0.2
  - tree_method: "exact"
  - grow_policy: "depthwise"
  - base_score: 0.0
- **Number of Chains**: 10
- **Training Sample Sizes**: [100, 200, 500, 1,000, 10,000]
- **Test Sample Size**: 10,000
- **Main Seed**: 1-100 

## Evaluation Metrics
- Mean Squared Error (MSE)
- Coverage of 95% Prediction Intervals
- Gelman-Rubin Statistic


