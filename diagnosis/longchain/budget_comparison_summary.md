# Chain separation at two computational budgets

The separation index divides the across-chain between/within ratio by the same
statistic on 4 consecutive blocks of one chain and multiplies by
4. Perfect mixing gives 1 at any autocorrelation. Both columns use the
same number of draws per chain, so they differ only in the iterations those draws span:
10^4 for the short chains and 10^7 for the long ones.

| dataset | runs | draws/chain | short index | long index |
| --- | --- | --- | --- | --- |
| Abalone | 5 | 7000 | 1.72-2.69 | 0.67-1.33 |
| Friedman-S p100 | 5 | 7000 | 1.81-2.47 | 0.67-1.18 |
| Friedman | 5 | 7000 | 3.01-4.23 | 0.79-1.26 |
| Friedman-S p200 | 5 | 7000 | 1.67-2.54 | 0.80-1.20 |
| Friedman-S p20 | 5 | 7000 | 2.43-3.09 | 0.84-1.14 |
| Concrete | 5 | 7000 | 2.42-5.61 | 0.98-2.41 |
| CalHousing | 5 | 7000 | 3.92-5.61 | 2.03-4.30 |
| Airfoil | 5 | 7000 | 4.63-7.12 | 3.94-5.57 |
| CCPP | 5 | 7000 | 4.93-7.64 | 7.62-9.43 |
| CPUAct | 5 | 7000 | 4.02-8.26 | 11.13-15.76 |
| SeoulBike | 5 | 7000 | 5.21-6.50 | 11.49-23.07 |
