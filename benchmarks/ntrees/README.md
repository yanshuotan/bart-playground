# Number of Trees Benchmarks

This folder contains benchmark scripts and results for evaluating the effect of varying the number of trees in different models. We gradually reduce the number of trees and compare the results with other models where the initial number of trees is set to a small value.

The models compared include:
- **This package's default BART**
- [stochtree](https://github.com/StochasticTree/stochtree)
- [bartz](https://github.com/Gattocrucco/bartz)
- Random Forest (RF)
- XGBoost (XGB)

The benchmarks focus on predictive accuracy, uncertainty quantification (prediction intervals), and runtime across different datasets.