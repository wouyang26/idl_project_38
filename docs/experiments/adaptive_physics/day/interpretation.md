# Day comparison

## random_sub_windy_days, 108 training flights

11 distinct scored flights. The omitted warmup averages 2.83 seconds. Values average equally over flights within each run, then equally over runs.

| arch             |   mape |   regime_re |   signed_wh |   absolute_wh |   underprediction_wh |
|:-----------------|-------:|------------:|------------:|--------------:|---------------------:|
| residual16-leaky |  9.818 |       5.367 |      -0.455 |         0.570 |                0.512 |
| tcn16-leaky      | 10.010 |       5.619 |      -0.474 |         0.614 |                0.544 |

Against tcn16-leaky, residual16-leaky changed mean absolute energy error by -0.044 Wh and mean underprediction magnitude by -0.032 Wh. Absolute error fell in 2 of 3 runs and 18 of 33 matched flight/run observations.

|                                 |   bias_w |   signed_wh |
|:--------------------------------|---------:|------------:|
| ('residual16-leaky', 'climb')   |    3.315 |      -0.025 |
| ('residual16-leaky', 'cruise')  |  -13.056 |      -0.319 |
| ('residual16-leaky', 'descent') |  -12.123 |      -0.111 |
| ('tcn16-leaky', 'climb')        |    1.211 |      -0.039 |
| ('tcn16-leaky', 'cruise')       |  -11.739 |      -0.300 |
| ('tcn16-leaky', 'descent')      |  -15.578 |      -0.136 |

Descent contributes the largest absolute change in signed energy relative to tcn16-leaky (+0.025 Wh per flight).

## windy_days, 108 training flights

11 distinct scored flights. The omitted warmup averages 2.83 seconds. Values average equally over flights within each run, then equally over runs.

| arch             |   mape |   regime_re |   signed_wh |   absolute_wh |   underprediction_wh |
|:-----------------|-------:|------------:|------------:|--------------:|---------------------:|
| residual16-leaky | 10.077 |       5.020 |      -0.008 |         0.460 |                0.234 |
| tcn16-leaky      | 10.172 |       5.124 |       0.026 |         0.511 |                0.243 |

Against tcn16-leaky, residual16-leaky changed mean absolute energy error by -0.051 Wh and mean underprediction magnitude by -0.008 Wh. Absolute error fell in 3 of 3 runs and 20 of 33 matched flight/run observations.

|                                 |   bias_w |   signed_wh |
|:--------------------------------|---------:|------------:|
| ('residual16-leaky', 'climb')   |    8.354 |      -0.014 |
| ('residual16-leaky', 'cruise')  |    2.285 |       0.008 |
| ('residual16-leaky', 'descent') |    1.614 |      -0.003 |
| ('tcn16-leaky', 'climb')        |    7.156 |      -0.013 |
| ('tcn16-leaky', 'cruise')       |    3.848 |       0.037 |
| ('tcn16-leaky', 'descent')      |    2.186 |       0.001 |

Cruise contributes the largest absolute change in signed energy relative to tcn16-leaky (-0.029 Wh per flight).

Signed error is predicted minus measured. Underprediction magnitude clips each scored flight’s signed deficit at zero before averaging. Phase Wh decomposes signed error. Only total power is measured, so term corrections describe the fitted model.
