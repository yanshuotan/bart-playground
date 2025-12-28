# bart-playground

A fast and modular implementation of Bayesian Additive Regression Trees (BART) for regression and classification tasks, with support for contextual bandits.

## Status

**⚠️ Experimental / Research Use Only**

This package is in early development and is **not production-ready**. The API is unstable and may change without notice. This software is provided for research purposes only.

## Installation


### Using pip (More Flexible)

Use this method to install with flexible version requirements:

```bash
pip install -e .
```

This installs the latest versions that satisfy the minimum requirements in [pyproject.toml](pyproject.toml), which should work but have not been extensively tested. 

###  Using Conda (For Reproducibility)

Use this method to install the exact dependency versions during development:

```bash
conda env create -f environment.yml
conda activate bartts
pip install -e .
```

## Quickstart

### Regression

```python
from bart_playground import DefaultBART, DataGenerator

# Generate synthetic data
X, y = DataGenerator(n_samples=200, n_features=5, noise=0.1, random_seed=1).generate("piecewise_linear")

# Fit model
model = DefaultBART(ndpost=200, nskip=50)
model.fit(X, y, quietly=True)

# Predictions
y_pred = model.predict(X)  # Posterior mean
y_post = model.posterior_predict(X)  # Full posterior samples (n_samples, n_posterior)
```

### Binary Classification

```python
from bart_playground import ProbitBART

# X: (n_samples, n_features), y: binary {0, 1}
model = ProbitBART(ndpost=200, nskip=50)
model.fit(X, y, quietly=True)

# Predictions
proba = model.predict_proba(X)  # (n_samples, 2) - probabilities for classes 0 and 1
y_hat = model.predict(X)  # Binary predictions
```

### Multiclass Classification

```python
from bart_playground import LogisticBART

# X: (n_samples, n_features), y: categorical {0, ..., K-1}
model = LogisticBART(ndpost=200, nskip=50)
model.fit(X, y, quietly=True)

# Predictions
proba = model.predict_proba(X)  # (n_samples, n_categories) - class probabilities
y_hat = model.predict(X)  # Class predictions
```

## Bandit Usage

The package includes BART-based agents for contextual bandit problems.

### Basic Usage

```python
from bart_playground.bandit.agents.bart_ts_agents import DefaultBARTTSAgent

# Initialize agent
agent = DefaultBARTTSAgent(
    n_arms=3, 
    n_features=4, 
    encoding='multi', 
    random_state=42,
    bart_kwargs={
        "ndpost": 100,
        "nskip": 50
    }
)

# Bandit loop
for t in range(100):
    x = ...  # Feature vector of length n_features
    arm = agent.choose_arm(x)  # Select arm
    y = ...  # Observe reward for the chosen arm
    agent.update_state(arm, x, y)  # Update agent state
```

### Simulation Framework

For running bandit simulations, use the `simulate` function:

```python
from bart_playground.bandit.experiment_utils.simulation import Scenario, simulate

class MyScenario(Scenario):
    def init_params(self):
        # Initialize scenario parameters
        pass
    
    def reward_function(self, x):
        # Compute expected rewards and noisy observations
        # Return {"outcome_mean": array, "reward": array}
        ...

# Run simulation
scenario = MyScenario(P=4, K=3, sigma2=1.0)
cum_regrets, time_agents = simulate(scenario, [agent], n_draws=100)
```

## System Architecture

Choose the section relevant to your use case:
- **Default (Regression/Classification)**: See [1. BART Core (Algorithm)](#1-bart-core-algorithm)
- **Contextual Bandits**: Expand [2. Bandit Loop (Optional)](#2-bandit-loop-optional)
- **Parallel Chains (Speedup)**: Expand [3. Parallelism Layer (Optional)](#3-parallelism-layer-optional)

```mermaid
flowchart TD
    User([User])
    
    subgraph Entries [Entry Points]
        API["Default Path: Regression / Classification"]
        Bandit["Bandit Path: Contextual Bandits"]
        Multi["Parallel Path: MultiChainBART (Ray)"]
    end

    User --> API
    User --> Bandit
    User --> Multi

    subgraph BARTLayer [Algorithm Layer: BART Core]
        BARTCore[BART Model]
        Gibbs[Gibbs Sampler]
        Trees[Tree Ensemble]
        BARTCore --- Gibbs
        BARTCore --- Trees
    end

    API --> BARTCore
    Bandit --> BARTCore
    Multi --> BARTCore
    
    %% Optional dependency link
    Bandit -.-> Multi
```

### 1. BART Core (Algorithm)

`bart-playground` uses a sampler-driven architecture where the core logic resides in the Gibbs sampler (Backfitting MCMC). The sampler iteratively proposes modifications to the tree ensemble and samples parameters from their posterior distributions.

#### 1.1 Sampler Iteration Skeleton

This diagram illustrates the `Sampler.run()` process, including the tree-by-tree Metropolis-Hastings (MH) steps and global parameter resampling.

```mermaid
sequenceDiagram
    participant B as BART
    participant S as Sampler
    participant M as Move
    participant P as Prior/Likelihood
    participant G as Global Prior

    B->>S: run(n_iter)
    
    loop For each iteration (1 to n_iter)
        loop For each Tree (1 to M)
            S->>S: sample_move()
            S->>M: propose(tol)
            M-->>S: proposed_Parameters
            S->>P: log_mh_ratio(move)
            Note right of P: Prior + Likelihood + Proposal Correction
            P-->>S: acceptance_prob
            
            alt Accepted
                S->>S: update current state
                S->>P: resample_leaf_vals()
                P-->>S: new μ
            end
        end
        
        S->>G: resample_global_params()
        Note right of G: Sample σ² (and s if Dirichlet)
        G-->>S: updated global_params
        
        S->>S: append Parameters to trace
    end
    S-->>B: trace (list of Parameters)
```

#### 1.2 State & Caching Overview

The model maintains a `trace` of `Parameters` objects. To ensure high performance, evaluation results are cached at both the tree and ensemble levels.

```mermaid
flowchart TD
    Data[Dataset X, y] --> Pre[Preprocessor]
    Pre --> BData[BART.data]
    
    subgraph Trace [Sampler.trace]
        P1[Parameters 1]
        P2[Parameters 2]
        PN[Parameters N]
    end
    
    BData --> Trace
    
    subgraph PState [Parameters State]
        Trees[Tree Ensemble]
        GParams[Global Params: σ², s]
        ECache[Ensemble Cache]
        FECache[Forest Eval Cache: Numba-optimized]
    end
    
    P1 --- PState
    
    subgraph TreeDetails [Single Tree]
        Structure[Structure: vars, thresholds]
        LeafVals[Leaf Values: μ]
        TCache[Tree Cache: leaf_ids, n, evals]
    end
    
    Trees --- TreeDetails

    %% Inference Path
    Trace --> Predict["predict(X) / posterior_f(X)"]
    Predict --> Evaluate{evaluate}
    Evaluate -- "Small X" --> FECache
    Evaluate -- "Large X" --> ECache
```

### 2. MultiChainBART (Parallelism)
<details>
<summary><b>Parallelism Layer</b></summary>

The `MultiChainBART` class coordinates parallel execution using Ray actors. It supports two modes:
1. **Standard Mode**: Single model per chain.
2. **Separate Models Mode** (`encoding='separate'`): Each chain maintains `K` independent models internally (one per arm).

The second mode is more complicated and suggested for most multi-arm bandit problems, so we present it here (with 2 chains) and omit the first mode for simplicity.

```mermaid
sequenceDiagram
    participant Client as Agent
    participant MC as MultiChainBART
    participant Worker1 as BARTActor 1
    participant Worker2 as BARTActor 2
    
    Note over Client, Worker2: Initialization (n_ensembles=2, n_models=K)
    Client->>MC: __init__()
    MC->>Worker1: remote(seed=s1, n_models=K)
    MC->>Worker2: remote(seed=s2, n_models=K)
    
    Note over Client, Worker2: Data Handling
    Client->>MC: fit(X, y)
    
    Note over Client, Worker2: Parallel Training (Separate Mode)
    loop For each Arm k
        Client->>MC: set_active_model(k)
        MC->>Worker1: set_active_model(k)
        MC->>Worker2: set_active_model(k)
        
        Client->>MC: fit(X_k, y_k)
        MC->>MC: Preprocessor.fit_transform(X_k, y_k)
        MC->>MC: ray.put(dataset) -> data_ref_k
        MC->>MC: ray.put(preprocessor) -> preproc_ref_k
        
        MC->>Worker1: fit.remote(data_ref_k, preproc_ref_k)
        MC->>Worker2: fit.remote(data_ref_k, preproc_ref_k)
        Note right of Worker2: Trains ONLY active model[k]
    end 
    
    Worker1-->>MC: Ready
    Worker2-->>MC: Ready
    MC-->>Client: self

    Note over Client, Worker2: Parallel Inference
    
    Client->>MC: posterior_f_batch(X)
    par Parallel Batch
        MC->>Worker1: posterior_f_batch.remote(X)
        MC->>Worker2: posterior_f_batch.remote(X)
        Note right of Worker2: Computes for ALL models
    end
    Worker1-->>MC: result_1 (K, n_samples, ndpost_per_chain)
    Worker2-->>MC: result_2 (K, n_samples, ndpost_per_chain)
    MC->>MC: concatenate -> (K, n_samples, total_ndpost) 
    
    MC-->>Client: Result
```

#### Key Concepts

- **Active Model**: For stateful operations like `fit()`, `MultiChainBART` sets an "active" model index on all actors. Subsequent calls operate only on that specific model (e.g., training arm $k$).
- **Batch APIs**: For inference, methods like `posterior_f_batch` trigger computation across **all** internal models simultaneously within each actor, reducing Ray overhead.
</details>

### 3. Bandit Usage

<details>
<summary><b>Bandit Usage</b></summary>

The bandit loop describes how the Agent interacts with the Environment (Scenario) and when model updates are triggered. It also highlights the "Feel-Good" mechanism, which optionally weights posterior samples towards those with higher historical utility.

To improve FGTS sampling, Sequential Monte Carlo and Reversible Jump MCMC could be used, but are left for future work.

```mermaid
sequenceDiagram
    participant Sim as Simulation
    participant Scen as Scenario
    participant Agent as BARTTSAgent
    participant Model as BART / MultiChainBART
    participant FG as Feel-Good Score

    loop for draw in n_draws
        Sim->>Scen: generate_covariates()
        Scen-->>Sim: x (Features)
        Sim->>Scen: reward_function(x)
        Scen-->>Sim: outcome_mean, reward[0..K-1]

        Sim->>Agent: choose_arm(x)

        alt initial random selections 
            Agent-->>Sim: arm ~ Uniform{0..K-1}
        else Thompson Sampling
            alt feel_good_lambda == 0
                Agent->>Agent: k = next posterior sample
            else feel_good_lambda != 0
                Agent->>FG: p(k) ∝ exp(lambda * S_k)
                FG-->>Agent: sample k ~ p (with replacement)
            end

            Agent->>Model: predict_trace(k, x)
            Model-->>Agent: f_k(x, arm=0..K-1)
            Agent-->>Sim: arm = argmax_a f_k(x,a)
        end

        Sim->>Agent: update_state(arm, x, reward[arm])
        Agent->>Agent: append (arm, x, y) to history

        alt should_refresh()
            Agent->>Model: fit(all history)
            Model-->>Agent: fitted
            Agent->>Agent: reset posterior queue
            opt feel_good_lambda != 0
                Agent->>FG: full recompute S_k using all_features
            end
        else no refresh
            opt feel_good_lambda != 0 AND fitted
                Agent->>FG: incremental update S_k
            end
        end
    end
```
</details>


## Caveats

- **No stability guarantees**: APIs may change without notice or documentation updates.
- **Research use only**: This package is not intended for production deployments.
- **Performance**: While optimized for speed, performance characteristics may vary with data size and model configuration.

