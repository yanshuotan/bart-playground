# Pointwise R-hat at two computational budgets

Modern rank-normalized split R-hat is calculated at each fixed test point and
summarized by its median and 90th percentile. Both budgets use the same number
of post-burn draws per chain, so they differ only in the iterations those draws
span: 10^4 for the short chains and 10^7 for the long ones.

| dataset | runs | draws/chain | short median | long median | short q90 | long q90 |
| --- | --- | --- | --- | --- | --- | --- |
| Abalone | 5 | 7000 | 1.061-1.080 | 1.000-1.000 | 1.165-1.194 | 1.000-1.001 |
| Friedman-S p20 | 5 | 7000 | 1.186-1.207 | 1.000-1.000 | 1.364-1.425 | 1.001-1.001 |
| Friedman-S p200 | 5 | 7000 | 1.166-1.200 | 1.000-1.001 | 1.335-1.375 | 1.001-1.001 |
| Friedman-S p100 | 5 | 7000 | 1.159-1.184 | 1.000-1.001 | 1.282-1.435 | 1.001-1.002 |
| Friedman | 5 | 7000 | 1.208-1.259 | 1.001-1.002 | 1.453-1.504 | 1.003-1.005 |
| Concrete | 5 | 7000 | 1.138-1.203 | 1.001-1.010 | 1.278-1.485 | 1.003-1.038 |
| CalHousing | 5 | 7000 | 1.228-1.289 | 1.012-1.025 | 1.458-1.641 | 1.054-1.075 |
| Airfoil | 5 | 7000 | 1.211-1.372 | 1.038-1.047 | 1.617-1.759 | 1.125-1.178 |
| CCPP | 5 | 7000 | 1.424-1.450 | 1.116-1.138 | 1.730-1.867 | 1.273-1.438 |
| CPUAct | 5 | 7000 | 1.465-1.600 | 1.205-1.280 | 1.737-2.046 | 1.475-1.671 |
| SeoulBike | 5 | 7000 | 1.263-1.428 | 1.189-1.242 | 1.711-1.823 | 1.605-1.676 |
