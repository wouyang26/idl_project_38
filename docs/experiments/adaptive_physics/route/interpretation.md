# Route comparison

## random_subsampled, 87 training flights

8 distinct scored flights. The omitted warmup averages 2.88 seconds. Values average equally over flights within each run, then equally over runs.

| arch             |   mape |   regime_re |   signed_wh |   absolute_wh |   underprediction_wh |
|:-----------------|-------:|------------:|------------:|--------------:|---------------------:|
| adaptive16-leaky |  9.926 |       5.652 |       0.971 |         0.998 |                0.013 |
| residual16-leaky |  9.689 |       5.355 |       0.865 |         0.898 |                0.017 |
| tcn16-leaky      | 10.037 |       5.898 |       1.069 |         1.072 |                0.002 |

Against residual16-leaky, adaptive16-leaky changed mean absolute energy error by +0.100 Wh and mean underprediction magnitude by -0.003 Wh. Absolute error fell in 1 of 3 runs and 7 of 24 matched flight/run observations.

Against tcn16-leaky, adaptive16-leaky changed mean absolute energy error by -0.075 Wh and mean underprediction magnitude by +0.011 Wh. Absolute error fell in 2 of 3 runs and 11 of 24 matched flight/run observations.

|                                 |   bias_w |   signed_wh |   induced_correction_wh |   profile_correction_wh |   parasitic_correction_wh |
|:--------------------------------|---------:|------------:|------------------------:|------------------------:|--------------------------:|
| ('adaptive16-leaky', 'climb')   |   16.329 |      -0.008 |                  -0.037 |                   0.010 |                     0.026 |
| ('adaptive16-leaky', 'cruise')  |   27.837 |       0.847 |                  -0.081 |                   0.129 |                     0.099 |
| ('adaptive16-leaky', 'descent') |   18.246 |       0.132 |                  -0.003 |                   0.014 |                     0.002 |
| ('residual16-leaky', 'climb')   |   17.506 |      -0.006 |                 nan     |                 nan     |                   nan     |
| ('residual16-leaky', 'cruise')  |   23.766 |       0.749 |                 nan     |                 nan     |                   nan     |
| ('residual16-leaky', 'descent') |   17.150 |       0.122 |                 nan     |                 nan     |                   nan     |
| ('tcn16-leaky', 'climb')        |   15.344 |      -0.003 |                 nan     |                 nan     |                   nan     |
| ('tcn16-leaky', 'cruise')       |   29.223 |       0.909 |                 nan     |                 nan     |                   nan     |
| ('tcn16-leaky', 'descent')      |   22.164 |       0.163 |                 nan     |                 nan     |                   nan     |

Cruise contributes the largest absolute change in signed energy relative to residual16-leaky (+0.099 Wh per flight).

Cruise contributes the largest absolute change in signed energy relative to tcn16-leaky (-0.061 Wh per flight).

## route, 134 training flights

8 distinct scored flights. The omitted warmup averages 2.88 seconds. Values average equally over flights within each run, then equally over runs.

| arch             |   mape |   regime_re |   signed_wh |   absolute_wh |   underprediction_wh |
|:-----------------|-------:|------------:|------------:|--------------:|---------------------:|
| adaptive16-leaky | 10.166 |       6.626 |       1.085 |         1.085 |                0.000 |
| residual16-leaky | 10.157 |       6.590 |       1.117 |         1.117 |                0.000 |
| tcn16-leaky      | 11.302 |       8.390 |       1.615 |         1.615 |                0.000 |

Against residual16-leaky, adaptive16-leaky changed mean absolute energy error by -0.032 Wh and mean underprediction magnitude by +0.000 Wh. Absolute error fell in 2 of 3 runs and 15 of 24 matched flight/run observations.

Against tcn16-leaky, adaptive16-leaky changed mean absolute energy error by -0.530 Wh and mean underprediction magnitude by +0.000 Wh. Absolute error fell in 3 of 3 runs and 24 of 24 matched flight/run observations.

|                                 |   bias_w |   signed_wh |   induced_correction_wh |   profile_correction_wh |   parasitic_correction_wh |
|:--------------------------------|---------:|------------:|------------------------:|------------------------:|--------------------------:|
| ('adaptive16-leaky', 'climb')   |   29.192 |       0.042 |                   0.063 |                  -0.027 |                     0.016 |
| ('adaptive16-leaky', 'cruise')  |   27.258 |       0.800 |                   0.096 |                   0.002 |                     0.057 |
| ('adaptive16-leaky', 'descent') |   29.628 |       0.243 |                   0.002 |                   0.136 |                    -0.001 |
| ('residual16-leaky', 'climb')   |   26.035 |       0.039 |                 nan     |                 nan     |                   nan     |
| ('residual16-leaky', 'cruise')  |   28.885 |       0.831 |                 nan     |                 nan     |                   nan     |
| ('residual16-leaky', 'descent') |   30.275 |       0.247 |                 nan     |                 nan     |                   nan     |
| ('tcn16-leaky', 'climb')        |   45.246 |       0.139 |                 nan     |                 nan     |                   nan     |
| ('tcn16-leaky', 'cruise')       |   39.717 |       1.164 |                 nan     |                 nan     |                   nan     |
| ('tcn16-leaky', 'descent')      |   37.743 |       0.312 |                 nan     |                 nan     |                   nan     |

Cruise contributes the largest absolute change in signed energy relative to residual16-leaky (-0.032 Wh per flight).

Cruise contributes the largest absolute change in signed energy relative to tcn16-leaky (-0.364 Wh per flight).

Signed error is predicted minus measured. Underprediction magnitude clips each scored flight’s signed deficit at zero before averaging. Phase Wh decomposes signed error. Only total power is measured, so term corrections describe the fitted model.
