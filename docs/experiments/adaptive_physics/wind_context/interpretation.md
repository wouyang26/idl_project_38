# Wind context comparison

## random_sub_wind, 107 training flights

20 distinct scored flights. The omitted warmup averages 2.85 seconds. Values average equally over flights within each run, then equally over runs.

| arch             |   mape |   regime_re |   signed_wh |   absolute_wh |   underprediction_wh |
|:-----------------|-------:|------------:|------------:|--------------:|---------------------:|
| residual16-leaky | 10.740 |       5.375 |      -0.566 |         0.679 |                0.622 |
| tcn16-leaky      | 10.847 |       5.441 |      -0.475 |         0.624 |                0.550 |

Against tcn16-leaky, residual16-leaky changed mean absolute energy error by +0.054 Wh and mean underprediction magnitude by +0.072 Wh. Absolute error fell in 1 of 3 runs and 26 of 60 matched flight/run observations.

|                                 |   bias_w |   signed_wh |
|:--------------------------------|---------:|------------:|
| ('residual16-leaky', 'climb')   |  -34.149 |      -0.201 |
| ('residual16-leaky', 'cruise')  |  -14.594 |      -0.371 |
| ('residual16-leaky', 'descent') |    0.315 |       0.007 |
| ('tcn16-leaky', 'climb')        |  -33.319 |      -0.196 |
| ('tcn16-leaky', 'cruise')       |  -12.801 |      -0.308 |
| ('tcn16-leaky', 'descent')      |    2.449 |       0.030 |

Cruise contributes the largest absolute change in signed energy relative to tcn16-leaky (-0.063 Wh per flight).

## wind, 107 training flights

20 distinct scored flights. The omitted warmup averages 2.85 seconds. Values average equally over flights within each run, then equally over runs.

| arch             |   mape |   regime_re |   signed_wh |   absolute_wh |   underprediction_wh |
|:-----------------|-------:|------------:|------------:|--------------:|---------------------:|
| residual16-leaky | 11.253 |       6.432 |      -1.046 |         1.048 |                1.047 |
| tcn16-leaky      | 11.909 |       7.451 |      -1.234 |         1.234 |                1.234 |

Against tcn16-leaky, residual16-leaky changed mean absolute energy error by -0.186 Wh and mean underprediction magnitude by -0.187 Wh. Absolute error fell in 3 of 3 runs and 50 of 60 matched flight/run observations.

|                                 |   bias_w |   signed_wh |
|:--------------------------------|---------:|------------:|
| ('residual16-leaky', 'climb')   |  -56.856 |      -0.314 |
| ('residual16-leaky', 'cruise')  |  -28.226 |      -0.669 |
| ('residual16-leaky', 'descent') |   -6.990 |      -0.063 |
| ('tcn16-leaky', 'climb')        |  -71.548 |      -0.397 |
| ('tcn16-leaky', 'cruise')       |  -32.556 |      -0.771 |
| ('tcn16-leaky', 'descent')      |   -7.514 |      -0.066 |

Cruise contributes the largest absolute change in signed energy relative to tcn16-leaky (+0.102 Wh per flight).

Signed error is predicted minus measured. Underprediction magnitude clips each scored flight’s signed deficit at zero before averaging. Phase Wh decomposes signed error. Only total power is measured, so term corrections describe the fitted model.
