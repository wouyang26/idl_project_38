# Wind comparison

## random_sub_wind, 107 training flights

20 distinct scored flights. The omitted warmup averages 2.85 seconds. Values average equally over flights within each run, then equally over runs.

| arch             |   mape |   regime_re |   signed_wh |   absolute_wh |   underprediction_wh |
|:-----------------|-------:|------------:|------------:|--------------:|---------------------:|
| adaptive16-leaky | 10.823 |       5.642 |      -0.668 |         0.743 |                0.705 |
| residual16-leaky | 10.740 |       5.375 |      -0.566 |         0.679 |                0.622 |
| tcn16-leaky      | 10.847 |       5.441 |      -0.475 |         0.624 |                0.550 |

Against residual16-leaky, adaptive16-leaky changed mean absolute energy error by +0.065 Wh and mean underprediction magnitude by +0.083 Wh. Absolute error fell in 1 of 3 runs and 23 of 60 matched flight/run observations.

Against tcn16-leaky, adaptive16-leaky changed mean absolute energy error by +0.119 Wh and mean underprediction magnitude by +0.156 Wh. Absolute error fell in 0 of 3 runs and 26 of 60 matched flight/run observations.

|                                 |   bias_w |   signed_wh |   induced_correction_wh |   profile_correction_wh |   parasitic_correction_wh |
|:--------------------------------|---------:|------------:|------------------------:|------------------------:|--------------------------:|
| ('adaptive16-leaky', 'climb')   |  -36.052 |      -0.211 |                  -0.040 |                   0.032 |                     0.006 |
| ('adaptive16-leaky', 'cruise')  |  -17.627 |      -0.434 |                  -0.092 |                  -0.085 |                     0.096 |
| ('adaptive16-leaky', 'descent') |   -2.826 |      -0.023 |                  -0.003 |                  -0.149 |                     0.003 |
| ('residual16-leaky', 'climb')   |  -34.149 |      -0.201 |                 nan     |                 nan     |                   nan     |
| ('residual16-leaky', 'cruise')  |  -14.594 |      -0.371 |                 nan     |                 nan     |                   nan     |
| ('residual16-leaky', 'descent') |    0.315 |       0.007 |                 nan     |                 nan     |                   nan     |
| ('tcn16-leaky', 'climb')        |  -33.319 |      -0.196 |                 nan     |                 nan     |                   nan     |
| ('tcn16-leaky', 'cruise')       |  -12.801 |      -0.308 |                 nan     |                 nan     |                   nan     |
| ('tcn16-leaky', 'descent')      |    2.449 |       0.030 |                 nan     |                 nan     |                   nan     |

Cruise contributes the largest absolute change in signed energy relative to residual16-leaky (-0.062 Wh per flight).

Cruise contributes the largest absolute change in signed energy relative to tcn16-leaky (-0.125 Wh per flight).

## wind, 107 training flights

20 distinct scored flights. The omitted warmup averages 2.85 seconds. Values average equally over flights within each run, then equally over runs.

| arch             |   mape |   regime_re |   signed_wh |   absolute_wh |   underprediction_wh |
|:-----------------|-------:|------------:|------------:|--------------:|---------------------:|
| adaptive16-leaky | 11.162 |       6.229 |      -0.912 |         0.933 |                0.923 |
| residual16-leaky | 11.253 |       6.432 |      -1.046 |         1.048 |                1.047 |
| tcn16-leaky      | 11.909 |       7.451 |      -1.234 |         1.234 |                1.234 |

Against residual16-leaky, adaptive16-leaky changed mean absolute energy error by -0.115 Wh and mean underprediction magnitude by -0.125 Wh. Absolute error fell in 2 of 3 runs and 41 of 60 matched flight/run observations.

Against tcn16-leaky, adaptive16-leaky changed mean absolute energy error by -0.301 Wh and mean underprediction magnitude by -0.311 Wh. Absolute error fell in 3 of 3 runs and 54 of 60 matched flight/run observations.

|                                 |   bias_w |   signed_wh |   induced_correction_wh |   profile_correction_wh |   parasitic_correction_wh |
|:--------------------------------|---------:|------------:|------------------------:|------------------------:|--------------------------:|
| ('adaptive16-leaky', 'climb')   |  -60.019 |      -0.332 |                  -0.080 |                  -0.026 |                     0.014 |
| ('adaptive16-leaky', 'cruise')  |  -21.730 |      -0.522 |                  -0.143 |                  -0.144 |                     0.199 |
| ('adaptive16-leaky', 'descent') |   -6.284 |      -0.059 |                  -0.007 |                  -0.204 |                     0.008 |
| ('residual16-leaky', 'climb')   |  -56.856 |      -0.314 |                 nan     |                 nan     |                   nan     |
| ('residual16-leaky', 'cruise')  |  -28.226 |      -0.669 |                 nan     |                 nan     |                   nan     |
| ('residual16-leaky', 'descent') |   -6.990 |      -0.063 |                 nan     |                 nan     |                   nan     |
| ('tcn16-leaky', 'climb')        |  -71.548 |      -0.397 |                 nan     |                 nan     |                   nan     |
| ('tcn16-leaky', 'cruise')       |  -32.556 |      -0.771 |                 nan     |                 nan     |                   nan     |
| ('tcn16-leaky', 'descent')      |   -7.514 |      -0.066 |                 nan     |                 nan     |                   nan     |

Cruise contributes the largest absolute change in signed energy relative to residual16-leaky (+0.148 Wh per flight).

Cruise contributes the largest absolute change in signed energy relative to tcn16-leaky (+0.249 Wh per flight).

Signed error is predicted minus measured. Underprediction magnitude clips each scored flight’s signed deficit at zero before averaging. Phase Wh decomposes signed error. Only total power is measured, so term corrections describe the fitted model.
