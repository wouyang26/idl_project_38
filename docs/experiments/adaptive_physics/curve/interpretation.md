# Curve comparison

## random, 5 training flights

35 distinct scored flights. The omitted warmup averages 2.86 seconds. Values average equally over flights within each run, then equally over runs.

| arch             |   mape |   regime_re |   signed_wh |   absolute_wh |   underprediction_wh |
|:-----------------|-------:|------------:|------------:|--------------:|---------------------:|
| adaptive16-leaky | 10.303 |       6.616 |       0.242 |         0.695 |                0.226 |
| residual16-leaky | 10.269 |       6.552 |       0.271 |         0.643 |                0.186 |
| tcn16-leaky      | 11.754 |       8.600 |      -0.062 |         1.033 |                0.548 |

Against residual16-leaky, adaptive16-leaky changed mean absolute energy error by +0.052 Wh and mean underprediction magnitude by +0.040 Wh. Absolute error fell in 1 of 3 runs and 45 of 105 matched flight/run observations.

Against tcn16-leaky, adaptive16-leaky changed mean absolute energy error by -0.338 Wh and mean underprediction magnitude by -0.321 Wh. Absolute error fell in 3 of 3 runs and 73 of 105 matched flight/run observations.

|                                 |   bias_w |   signed_wh |   induced_correction_wh |   profile_correction_wh |   parasitic_correction_wh |
|:--------------------------------|---------:|------------:|------------------------:|------------------------:|--------------------------:|
| ('adaptive16-leaky', 'climb')   |   -6.209 |      -0.112 |                   0.022 |                  -0.121 |                    -0.003 |
| ('adaptive16-leaky', 'cruise')  |    5.488 |       0.192 |                   0.042 |                  -0.059 |                    -0.038 |
| ('adaptive16-leaky', 'descent') |   20.571 |       0.162 |                   0.009 |                   0.087 |                    -0.003 |
| ('residual16-leaky', 'climb')   |   -4.734 |      -0.099 |                 nan     |                 nan     |                   nan     |
| ('residual16-leaky', 'cruise')  |    5.726 |       0.186 |                 nan     |                 nan     |                   nan     |
| ('residual16-leaky', 'descent') |   22.864 |       0.184 |                 nan     |                 nan     |                   nan     |
| ('tcn16-leaky', 'climb')        |   -3.902 |      -0.115 |                 nan     |                 nan     |                   nan     |
| ('tcn16-leaky', 'cruise')       |   -5.670 |      -0.074 |                 nan     |                 nan     |                   nan     |
| ('tcn16-leaky', 'descent')      |   18.391 |       0.127 |                 nan     |                 nan     |                   nan     |

Descent contributes the largest absolute change in signed energy relative to residual16-leaky (-0.022 Wh per flight).

Cruise contributes the largest absolute change in signed energy relative to tcn16-leaky (+0.266 Wh per flight).

## random, 10 training flights

35 distinct scored flights. The omitted warmup averages 2.86 seconds. Values average equally over flights within each run, then equally over runs.

| arch             |   mape |   regime_re |   signed_wh |   absolute_wh |   underprediction_wh |
|:-----------------|-------:|------------:|------------:|--------------:|---------------------:|
| adaptive16-leaky |  9.506 |       5.514 |       0.006 |         0.574 |                0.284 |
| residual16-leaky |  9.507 |       5.798 |      -0.163 |         0.579 |                0.371 |
| tcn16-leaky      | 10.622 |       6.729 |      -0.197 |         0.743 |                0.470 |

Against residual16-leaky, adaptive16-leaky changed mean absolute energy error by -0.005 Wh and mean underprediction magnitude by -0.087 Wh. Absolute error fell in 1 of 3 runs and 42 of 105 matched flight/run observations.

Against tcn16-leaky, adaptive16-leaky changed mean absolute energy error by -0.169 Wh and mean underprediction magnitude by -0.186 Wh. Absolute error fell in 3 of 3 runs and 67 of 105 matched flight/run observations.

|                                 |   bias_w |   signed_wh |   induced_correction_wh |   profile_correction_wh |   parasitic_correction_wh |
|:--------------------------------|---------:|------------:|------------------------:|------------------------:|--------------------------:|
| ('adaptive16-leaky', 'climb')   |    0.550 |      -0.071 |                   0.071 |                  -0.153 |                     0.000 |
| ('adaptive16-leaky', 'cruise')  |    1.139 |       0.045 |                   0.141 |                  -0.400 |                     0.001 |
| ('adaptive16-leaky', 'descent') |    4.181 |       0.033 |                   0.014 |                  -0.057 |                    -0.001 |
| ('residual16-leaky', 'climb')   |   -0.630 |      -0.080 |                 nan     |                 nan     |                   nan     |
| ('residual16-leaky', 'cruise')  |   -2.356 |      -0.044 |                 nan     |                 nan     |                   nan     |
| ('residual16-leaky', 'descent') |   -4.063 |      -0.038 |                 nan     |                 nan     |                   nan     |
| ('tcn16-leaky', 'climb')        |   -7.129 |      -0.114 |                 nan     |                 nan     |                   nan     |
| ('tcn16-leaky', 'cruise')       |   -5.931 |      -0.172 |                 nan     |                 nan     |                   nan     |
| ('tcn16-leaky', 'descent')      |   11.013 |       0.088 |                 nan     |                 nan     |                   nan     |

Cruise contributes the largest absolute change in signed energy relative to residual16-leaky (+0.089 Wh per flight).

Cruise contributes the largest absolute change in signed energy relative to tcn16-leaky (+0.216 Wh per flight).

## random, 20 training flights

35 distinct scored flights. The omitted warmup averages 2.86 seconds. Values average equally over flights within each run, then equally over runs.

| arch             |   mape |   regime_re |   signed_wh |   absolute_wh |   underprediction_wh |
|:-----------------|-------:|------------:|------------:|--------------:|---------------------:|
| adaptive16-leaky |  9.708 |       5.904 |       0.557 |         0.701 |                0.072 |
| residual16-leaky |  9.877 |       6.288 |       0.616 |         0.754 |                0.069 |
| tcn16-leaky      | 11.284 |       7.894 |       0.743 |         1.128 |                0.192 |

Against residual16-leaky, adaptive16-leaky changed mean absolute energy error by -0.053 Wh and mean underprediction magnitude by +0.003 Wh. Absolute error fell in 2 of 3 runs and 65 of 105 matched flight/run observations.

Against tcn16-leaky, adaptive16-leaky changed mean absolute energy error by -0.426 Wh and mean underprediction magnitude by -0.120 Wh. Absolute error fell in 3 of 3 runs and 83 of 105 matched flight/run observations.

|                                 |   bias_w |   signed_wh |   induced_correction_wh |   profile_correction_wh |   parasitic_correction_wh |
|:--------------------------------|---------:|------------:|------------------------:|------------------------:|--------------------------:|
| ('adaptive16-leaky', 'climb')   |   15.717 |       0.004 |                   0.094 |                  -0.083 |                     0.002 |
| ('adaptive16-leaky', 'cruise')  |   15.244 |       0.403 |                   0.161 |                  -0.034 |                     0.018 |
| ('adaptive16-leaky', 'descent') |   18.994 |       0.150 |                   0.016 |                   0.067 |                     0.000 |
| ('residual16-leaky', 'climb')   |   25.396 |       0.042 |                 nan     |                 nan     |                   nan     |
| ('residual16-leaky', 'cruise')  |   17.926 |       0.450 |                 nan     |                 nan     |                   nan     |
| ('residual16-leaky', 'descent') |   16.272 |       0.124 |                 nan     |                 nan     |                   nan     |
| ('tcn16-leaky', 'climb')        |   13.078 |      -0.016 |                 nan     |                 nan     |                   nan     |
| ('tcn16-leaky', 'cruise')       |   21.255 |       0.533 |                 nan     |                 nan     |                   nan     |
| ('tcn16-leaky', 'descent')      |   28.550 |       0.225 |                 nan     |                 nan     |                   nan     |

Cruise contributes the largest absolute change in signed energy relative to residual16-leaky (-0.047 Wh per flight).

Cruise contributes the largest absolute change in signed energy relative to tcn16-leaky (-0.130 Wh per flight).

## random, 35 training flights

35 distinct scored flights. The omitted warmup averages 2.86 seconds. Values average equally over flights within each run, then equally over runs.

| arch             |   mape |   regime_re |   signed_wh |   absolute_wh |   underprediction_wh |
|:-----------------|-------:|------------:|------------:|--------------:|---------------------:|
| adaptive16-leaky |  9.198 |       5.310 |       0.427 |         0.613 |                0.093 |
| residual16-leaky |  9.395 |       5.742 |       0.413 |         0.651 |                0.119 |
| tcn16-leaky      | 10.141 |       6.788 |       0.688 |         0.893 |                0.102 |

Against residual16-leaky, adaptive16-leaky changed mean absolute energy error by -0.038 Wh and mean underprediction magnitude by -0.026 Wh. Absolute error fell in 1 of 3 runs and 62 of 105 matched flight/run observations.

Against tcn16-leaky, adaptive16-leaky changed mean absolute energy error by -0.280 Wh and mean underprediction magnitude by -0.009 Wh. Absolute error fell in 2 of 3 runs and 73 of 105 matched flight/run observations.

|                                 |   bias_w |   signed_wh |   induced_correction_wh |   profile_correction_wh |   parasitic_correction_wh |
|:--------------------------------|---------:|------------:|------------------------:|------------------------:|--------------------------:|
| ('adaptive16-leaky', 'climb')   |    5.330 |      -0.025 |                   0.092 |                  -0.092 |                     0.001 |
| ('adaptive16-leaky', 'cruise')  |   14.528 |       0.371 |                   0.144 |                   0.010 |                    -0.004 |
| ('adaptive16-leaky', 'descent') |    9.709 |       0.081 |                   0.015 |                   0.006 |                    -0.001 |
| ('residual16-leaky', 'climb')   |   10.409 |      -0.009 |                 nan     |                 nan     |                   nan     |
| ('residual16-leaky', 'cruise')  |   14.597 |       0.356 |                 nan     |                 nan     |                   nan     |
| ('residual16-leaky', 'descent') |    8.363 |       0.066 |                 nan     |                 nan     |                   nan     |
| ('tcn16-leaky', 'climb')        |   19.363 |       0.028 |                 nan     |                 nan     |                   nan     |
| ('tcn16-leaky', 'cruise')       |   22.865 |       0.546 |                 nan     |                 nan     |                   nan     |
| ('tcn16-leaky', 'descent')      |   13.861 |       0.114 |                 nan     |                 nan     |                   nan     |

Climb contributes the largest absolute change in signed energy relative to residual16-leaky (-0.016 Wh per flight).

Cruise contributes the largest absolute change in signed energy relative to tcn16-leaky (-0.175 Wh per flight).

## random, 52 training flights

35 distinct scored flights. The omitted warmup averages 2.86 seconds. Values average equally over flights within each run, then equally over runs.

| arch             |   mape |   regime_re |   signed_wh |   absolute_wh |   underprediction_wh |
|:-----------------|-------:|------------:|------------:|--------------:|---------------------:|
| adaptive16-leaky |  8.793 |       4.698 |       0.167 |         0.525 |                0.179 |
| residual16-leaky |  8.775 |       4.777 |       0.148 |         0.548 |                0.200 |
| tcn16-leaky      |  8.807 |       4.911 |      -0.039 |         0.527 |                0.283 |

Against residual16-leaky, adaptive16-leaky changed mean absolute energy error by -0.023 Wh and mean underprediction magnitude by -0.021 Wh. Absolute error fell in 2 of 3 runs and 59 of 105 matched flight/run observations.

Against tcn16-leaky, adaptive16-leaky changed mean absolute energy error by -0.002 Wh and mean underprediction magnitude by -0.104 Wh. Absolute error fell in 2 of 3 runs and 59 of 105 matched flight/run observations.

|                                 |   bias_w |   signed_wh |   induced_correction_wh |   profile_correction_wh |   parasitic_correction_wh |
|:--------------------------------|---------:|------------:|------------------------:|------------------------:|--------------------------:|
| ('adaptive16-leaky', 'climb')   |    1.687 |      -0.046 |                   0.094 |                  -0.114 |                     0.004 |
| ('adaptive16-leaky', 'cruise')  |    4.932 |       0.139 |                   0.135 |                  -0.243 |                     0.035 |
| ('adaptive16-leaky', 'descent') |    9.181 |       0.074 |                   0.011 |                   0.004 |                     0.002 |
| ('residual16-leaky', 'climb')   |    4.009 |      -0.040 |                 nan     |                 nan     |                   nan     |
| ('residual16-leaky', 'cruise')  |    5.379 |       0.156 |                 nan     |                 nan     |                   nan     |
| ('residual16-leaky', 'descent') |    4.120 |       0.032 |                 nan     |                 nan     |                   nan     |
| ('tcn16-leaky', 'climb')        |   -8.497 |      -0.094 |                 nan     |                 nan     |                   nan     |
| ('tcn16-leaky', 'cruise')       |    0.328 |       0.028 |                 nan     |                 nan     |                   nan     |
| ('tcn16-leaky', 'descent')      |    3.194 |       0.027 |                 nan     |                 nan     |                   nan     |

Descent contributes the largest absolute change in signed energy relative to residual16-leaky (+0.042 Wh per flight).

Cruise contributes the largest absolute change in signed energy relative to tcn16-leaky (+0.111 Wh per flight).

## random, 87 training flights

35 distinct scored flights. The omitted warmup averages 2.86 seconds. Values average equally over flights within each run, then equally over runs.

| arch             |   mape |   regime_re |   signed_wh |   absolute_wh |   underprediction_wh |
|:-----------------|-------:|------------:|------------:|--------------:|---------------------:|
| adaptive16-leaky |  8.462 |       4.238 |      -0.099 |         0.513 |                0.306 |
| residual16-leaky |  8.341 |       4.098 |      -0.088 |         0.469 |                0.278 |
| tcn16-leaky      |  8.529 |       4.555 |       0.116 |         0.520 |                0.202 |

Against residual16-leaky, adaptive16-leaky changed mean absolute energy error by +0.044 Wh and mean underprediction magnitude by +0.027 Wh. Absolute error fell in 1 of 3 runs and 42 of 105 matched flight/run observations.

Against tcn16-leaky, adaptive16-leaky changed mean absolute energy error by -0.007 Wh and mean underprediction magnitude by +0.104 Wh. Absolute error fell in 2 of 3 runs and 55 of 105 matched flight/run observations.

|                                 |   bias_w |   signed_wh |   induced_correction_wh |   profile_correction_wh |   parasitic_correction_wh |
|:--------------------------------|---------:|------------:|------------------------:|------------------------:|--------------------------:|
| ('adaptive16-leaky', 'climb')   |  -11.403 |      -0.084 |                   0.002 |                  -0.062 |                     0.010 |
| ('adaptive16-leaky', 'cruise')  |   -0.526 |       0.031 |                  -0.063 |                  -0.233 |                     0.095 |
| ('adaptive16-leaky', 'descent') |   -4.948 |      -0.046 |                  -0.003 |                  -0.107 |                     0.004 |
| ('residual16-leaky', 'climb')   |   -6.672 |      -0.068 |                 nan     |                 nan     |                   nan     |
| ('residual16-leaky', 'cruise')  |   -1.024 |       0.011 |                 nan     |                 nan     |                   nan     |
| ('residual16-leaky', 'descent') |   -2.902 |      -0.031 |                 nan     |                 nan     |                   nan     |
| ('tcn16-leaky', 'climb')        |    4.661 |      -0.029 |                 nan     |                 nan     |                   nan     |
| ('tcn16-leaky', 'cruise')       |    4.921 |       0.155 |                 nan     |                 nan     |                   nan     |
| ('tcn16-leaky', 'descent')      |   -0.431 |      -0.010 |                 nan     |                 nan     |                   nan     |

Cruise contributes the largest absolute change in signed energy relative to residual16-leaky (+0.020 Wh per flight).

Cruise contributes the largest absolute change in signed energy relative to tcn16-leaky (-0.124 Wh per flight).

## random, 114 training flights

35 distinct scored flights. The omitted warmup averages 2.86 seconds. Values average equally over flights within each run, then equally over runs.

| arch             |   mape |   regime_re |   signed_wh |   absolute_wh |   underprediction_wh |
|:-----------------|-------:|------------:|------------:|--------------:|---------------------:|
| adaptive16-leaky |  8.391 |       4.098 |       0.118 |         0.427 |                0.154 |
| residual16-leaky |  8.380 |       4.354 |       0.170 |         0.429 |                0.130 |
| tcn16-leaky      |  8.680 |       4.654 |       0.245 |         0.486 |                0.121 |

Against residual16-leaky, adaptive16-leaky changed mean absolute energy error by -0.002 Wh and mean underprediction magnitude by +0.025 Wh. Absolute error fell in 1 of 1 runs and 19 of 35 matched flight/run observations.

Against tcn16-leaky, adaptive16-leaky changed mean absolute energy error by -0.060 Wh and mean underprediction magnitude by +0.033 Wh. Absolute error fell in 1 of 1 runs and 20 of 35 matched flight/run observations.

|                                 |   bias_w |   signed_wh |   induced_correction_wh |   profile_correction_wh |   parasitic_correction_wh |
|:--------------------------------|---------:|------------:|------------------------:|------------------------:|--------------------------:|
| ('adaptive16-leaky', 'climb')   |  -12.142 |      -0.077 |                  -0.036 |                  -0.023 |                     0.009 |
| ('adaptive16-leaky', 'cruise')  |    4.850 |       0.128 |                  -0.084 |                  -0.123 |                     0.099 |
| ('adaptive16-leaky', 'descent') |    8.348 |       0.067 |                  -0.007 |                   0.011 |                     0.005 |
| ('residual16-leaky', 'climb')   |   -9.136 |      -0.083 |                 nan     |                 nan     |                   nan     |
| ('residual16-leaky', 'cruise')  |    6.803 |       0.171 |                 nan     |                 nan     |                   nan     |
| ('residual16-leaky', 'descent') |   10.055 |       0.082 |                 nan     |                 nan     |                   nan     |
| ('tcn16-leaky', 'climb')        |  -16.134 |      -0.102 |                 nan     |                 nan     |                   nan     |
| ('tcn16-leaky', 'cruise')       |   10.860 |       0.266 |                 nan     |                 nan     |                   nan     |
| ('tcn16-leaky', 'descent')      |    9.738 |       0.081 |                 nan     |                 nan     |                   nan     |

Cruise contributes the largest absolute change in signed energy relative to residual16-leaky (-0.043 Wh per flight).

Cruise contributes the largest absolute change in signed energy relative to tcn16-leaky (-0.138 Wh per flight).

Signed error is predicted minus measured. Underprediction magnitude clips each scored flight’s signed deficit at zero before averaging. Phase Wh decomposes signed error. Only total power is measured, so term corrections describe the fitted model.
