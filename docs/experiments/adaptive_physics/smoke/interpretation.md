# Smoke comparison

## wind, 5 training flights

20 distinct scored flights. The omitted warmup averages 2.85 seconds. Values average equally over flights within each run, then equally over runs.

| arch             |   mape |   regime_re |   signed_wh |   absolute_wh |   underprediction_wh |
|:-----------------|-------:|------------:|------------:|--------------:|---------------------:|
| adaptive16-leaky | 12.586 |       6.381 |      -0.513 |         0.633 |                0.573 |
| residual16-leaky | 12.459 |       6.509 |      -0.627 |         0.698 |                0.662 |
| tcn16-leaky      | 21.948 |      14.858 |      -1.206 |         1.223 |                1.215 |

Against residual16-leaky, adaptive16-leaky changed mean absolute energy error by -0.065 Wh and mean underprediction magnitude by -0.089 Wh. Absolute error fell in 1 of 1 runs and 16 of 20 matched flight/run observations.

Against tcn16-leaky, adaptive16-leaky changed mean absolute energy error by -0.590 Wh and mean underprediction magnitude by -0.642 Wh. Absolute error fell in 1 of 1 runs and 17 of 20 matched flight/run observations.

|                                 |   bias_w |   signed_wh |   induced_correction_wh |   profile_correction_wh |   parasitic_correction_wh |
|:--------------------------------|---------:|------------:|------------------------:|------------------------:|--------------------------:|
| ('adaptive16-leaky', 'climb')   |  -47.652 |      -0.289 |                   0.010 |                   0.010 |                     0.000 |
| ('adaptive16-leaky', 'cruise')  |  -16.591 |      -0.408 |                   0.018 |                   0.105 |                    -0.004 |
| ('adaptive16-leaky', 'descent') |   19.011 |       0.184 |                   0.001 |                   0.099 |                    -0.004 |
| ('residual16-leaky', 'climb')   |  -51.109 |      -0.307 |                 nan     |                 nan     |                   nan     |
| ('residual16-leaky', 'cruise')  |  -19.201 |      -0.467 |                 nan     |                 nan     |                   nan     |
| ('residual16-leaky', 'descent') |   14.958 |       0.148 |                 nan     |                 nan     |                   nan     |
| ('tcn16-leaky', 'climb')        | -124.718 |      -0.688 |                 nan     |                 nan     |                   nan     |
| ('tcn16-leaky', 'cruise')       |  -28.882 |      -0.759 |                 nan     |                 nan     |                   nan     |
| ('tcn16-leaky', 'descent')      |   20.014 |       0.241 |                 nan     |                 nan     |                   nan     |

Cruise contributes the largest absolute change in signed energy relative to residual16-leaky (+0.058 Wh per flight).

Climb contributes the largest absolute change in signed energy relative to tcn16-leaky (+0.399 Wh per flight).

Signed error is predicted minus measured. Underprediction magnitude clips each scored flight’s signed deficit at zero before averaging. Phase Wh decomposes signed error. Only total power is measured, so term corrections describe the fitted model.
