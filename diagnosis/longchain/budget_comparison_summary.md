# Chain separation at two computational budgets

The separation index divides the across-chain between/within ratio by the same
statistic on 4 consecutive blocks of one chain and multiplies by
4. Perfect mixing gives 1 at any autocorrelation. Both columns use the
same number of draws per chain, so they differ only in the iterations those draws span:
10^4 for the short chains and 10^7 for the long ones.

| dataset | runs | draws/chain | short index | long index |
| --- | --- | --- | --- | --- |
| Friedman-S p200 | 5 | 7000 | 1.67-2.54 | 0.56-1.14 |
| Friedman | 5 | 7000 | 3.01-4.23 | 0.67-1.48 |
| Friedman-S p100 | 5 | 7000 | 1.81-2.47 | 0.74-1.10 |
| Abalone | 5 | 7000 | 1.72-2.69 | 0.85-1.14 |
| Friedman-S p20 | 5 | 7000 | 2.43-3.09 | 0.90-1.10 |
| Concrete | 5 | 7000 | 2.42-5.61 | 1.08-1.60 |
| CalHousing | 5 | 7000 | 3.92-5.61 | 1.53-2.78 |
| Airfoil | 2 | 7000 | 4.63-7.12 | 2.49-3.68 |
| CCPP | 5 | 7000 | 4.93-7.64 | 4.39-5.61 |
| SeoulBike | 5 | 7000 | 5.21-6.50 | 6.53-13.38 |
