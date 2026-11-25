# Experiments 5 and 6 — power and battery-energy error under wind shift

All 105 planned 30-epoch runs and their evaluations finished on 21 November 2026. The sweep started from PR #11 at `a2f1fd8c9dd7df63fe258832a67229325e8d5863`. The results below average over the same scored flights within each run and then over the three model seeds or training subsets. The full-data point of the learning curve uses one seed.

## Experiment 5: calm flights from windy dates

On the 11 calm flights from four held-out windy dates, the leaky residual hybrid lowered mean absolute energy error in all three seeds. The reduction in mean underprediction was small.

| Training split | Model | Power MAPE (%) | Regime RE (%) | Absolute error (Wh) | Underprediction (Wh) |
|---|---|---:|---:|---:|---:|
| `windy_days` | Plain TCN | 10.172 | 5.124 | 0.511 | 0.243 |
| `windy_days` | Residual | 10.077 | 5.020 | 0.460 | 0.234 |
| `random_sub_windy_days` | Plain TCN | 10.010 | 5.619 | 0.614 | 0.544 |
| `random_sub_windy_days` | Residual | 9.818 | 5.367 | 0.570 | 0.512 |

Holding out the dates raised MAPE by 0.16 percentage points for plain and 0.26 for residual on these calm flights. Holding out wind raised MAPE by 1.06 points for plain and 0.51 for residual on the separate cohort of 20 windy flights. The comparisons use different flight cohorts and support descriptive conclusions. [Day results](day/interpretation.md) · [Wind versus day gaps](day/wind_vs_day.csv) · [Wind context](wind_context/interpretation.md)

## Experiment 6: adaptive physics

On the 20 common unseen windy flights, adaptive had the lowest power MAPE, regime energy error, absolute energy error and underprediction. Mean underprediction decreased by **0.311 Wh per flight** versus the plain TCN and **0.125 Wh** versus residual, approximately 25% and 12% respectively. Absolute energy error improved in all three seeds versus plain and in two of three versus residual.

| Model trained on the wind holdout | Power MAPE (%) | Regime RE (%) | Signed error (Wh) | Absolute error (Wh) | Underprediction (Wh) |
|---|---:|---:|---:|---:|---:|
| Plain leaky TCN | 11.909 | 7.451 | −1.234 | 1.234 | 1.234 |
| Leaky residual | 11.253 | 6.432 | −1.046 | 1.048 | 1.047 |
| Leaky adaptive | **11.162** | **6.229** | **−0.912** | **0.933** | **0.923** |

Cruise accounted for the largest improvement in signed energy error, **+0.249 Wh per flight** versus plain and **+0.148 Wh** versus residual. Relative to residual, climb worsened by about 0.018 Wh and descent improved by about 0.004 Wh. Within the adaptive model, the cruise parasitic correction added 0.199 Wh relative to refitted Liu, while induced and profile corrections each subtracted about 0.14 Wh. Those term corrections use Liu as the reference. The cruise improvement above uses the TCN baselines as the reference. [Wind results](wind/interpretation.md) · [Per-flight phase and term data](wind/per_flight.csv)

The gain depends on training regime. On the wind-exposed control, adaptive underprediction averaged **0.705 Wh**, versus **0.550 Wh** for plain and **0.622 Wh** for residual. On eight off-route flights, adaptive absolute error was **1.085 Wh**, versus **1.615 Wh** for plain and **1.117 Wh** for residual, while all three models overpredicted scored-flight energy. On the learning curve, adaptive beat plain on mean power MAPE at every budget and stayed within 0.20 percentage points of residual. Its underprediction relative to residual varied by budget. The fitted parasitic coefficient reached the lower bound of about `1e-6` in seven reduced-budget subsets, including two of the three five-flight subsets. Interpret fitted term multipliers in light of those weakly constrained terms. [Route results](route/interpretation.md) · [Learning curve](curve/learning_curve.csv) · [Liu fits](curve/liu_fits.csv)

The held-out wind comparison used 20 flights and three seeds from one dataset. The results support a smaller average energy deficit under this wind holdout. The model still underpredicted on average, and the wind control shows the gain did not generalize to both training regimes.

## Result files

The [run matrix](run_matrix.csv) lists the checkpoint for each configuration. The [final handoff](HANDOFF.md) records completion. Each comparison directory contains per-flight and phase rows, paired differences, per-seed averages, aggregate results and fitted Liu coefficients. The [learning-curve figure](curve/learning_curve.png) includes refitted Liu as a physics reference. The smoke results document the engineering checks.

## Data and checkpoints

187 processed flights were present. We fetched the missing KiltHub raw flight archive, checked its SHA-256 against `scripts/download_data.sh`, and generated `data/kilthub/wind.csv` with `python -m utils.wind`. The wind holdout/control comparison scores 20 common unseen flights. The day comparison scores 11 calm flights from four windy dates. These are observational comparisons on different flights.

The shared **Project 38 Model Checkpoints** folder contained 79 final checkpoints at inventory. Their metadata identified 13 small ReLU TCNs, 45 large ReLU TCNs, three large TCNs with a variance head, and 18 LSTMs. No checkpoint matched the leaky small-TCN, residual or adaptive configurations. The local pre-existing training outputs contained LSTM runs. For future shared files, place matching final checkpoints under the canonical paths in [the matrix](run_matrix.csv) before resuming. The runner validates configuration, model and subset seed, train/validation/test IDs, and physics buffers before reuse. A conflicting checkpoint at an expected path stops the run.

For historical context, the committed three-seed wind table reports 11.32% held-out MAPE and 6.60% regime RE for leaky residual. The row with 11.55% and 6.62% is **ReLU** plain. Neither row supplies paired Wh estimates for the new experiment. [Existing results](../../physics_hybrid/gaps.csv)

## Model and scoring

At each scored timestep the s-TCN encodes the last 20 samples and combines them with the static inputs. A leaky hidden head produces three logits. For each of Liu's induced, profile and parasitic terms, the multiplier is `exp(clamp(logit, -ln(2), ln(2)))`. Zero output weights and biases initialize all three to one. The model sums adjusted terms in watts, then normalizes total power once. Each run refits the five Liu constants after selecting the training subset. All architectures use the team's existing input and target scaling (extrema from train plus validation). Consequently a five-flight learning-curve point uses five flights for model fitting and the usual validation flights for scaling.

The phase evaluator selects canonical paths for every required architecture and seed. It validates split membership independently, excludes all train and validation flights from scoring, and uses identical timestamps across architectures. For the learning curve it uses the 35 unique paper-test flights. For the wind/route/day contrasts it uses the held-out/control intersection of unseen flights. The first 19 steps are warmup. Each per-flight row gives the actual elapsed time omitted.

Climb, cruise and descent follow the existing 90%-of-peak altitude rule on the scored trajectory. Each trapezoid endpoint gets half its adjacent interval, including across phase boundaries. `signed_wh` is predicted minus measured on the scored portion of the flight. `absolute_wh` takes the absolute value after integration. `underprediction_wh` is the positive part of measured minus predicted, computed before averaging across flights. `regime_re` uses the existing regime segmentation and integration. MAPE uses steps with positive measured power. All timestamps contribute to Wh. Per-phase and per-flight term corrections include watts, Wh, mean multiplier and fractions at both bounds. Only the total electrical power is observed. Small fitted terms can have a large multiplier with little influence on total energy. Inspect `liu_fits.csv` and the raw-term watts at each training budget.

## Reproduction

From the repository root in the project's Python environment, with the processed and raw dataset available:

```bash
python -m utils.wind   # once, after scripts/download_data.sh on a fresh clone
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Three matched engineering checks in a separate checkpoint root
for net in tcn residual adaptive; do
  python train_torch.py --net_type "$net" --head_act leaky --nb_filters 16 --n_layers 5 \
    --kernel_size 2 --train_size 5 --split wind --total_epochs 1 --save_every 1 \
    --device cpu -o Results/adaptive_smoke
done
p=Results/adaptive_smoke/Results/ckpts/adaptive_20_16_2_1_5_leaky_wind_n5_k0
python scripts/experiments/verify_adaptive.py "$p/epoch_0000.pt"
python evaluate_torch.py -m "$p/final.pt" -e wind --unseen
python scripts/experiments/adaptive_scores.py wind --smoke \
  --ckpts Results/adaptive_smoke/Results/ckpts --out docs/experiments/adaptive_physics/smoke
python scripts/experiments/adaptive_report.py docs/experiments/adaptive_physics/smoke
```

A single command runs or resumes the full sweep. Set `PYTHON` to the interpreter from the project environment if needed. Use `DEVICE=cuda` in a shell with CUDA access. The default is CPU.

```bash
DEVICE=cuda bash scripts/experiments/run_adaptive.sh
```

[run_matrix.csv](run_matrix.csv) enumerates all 105 distinct configurations, including comparison baselines. To list exact missing/reusable runs without training, use:

```bash
for stage in day context shift curve baselines; do
  python scripts/experiments/adaptive_runs.py "$stage"
done
```

| Stage | Required 30-epoch runs | Experiment |
|---|---:|---|
| day | 12 | Plain and residual, both day splits, 3 seeds |
| context | 12 | The same two models on both wind splits, 3 seeds |
| shift | 12 | Adaptive on wind, route and their controls, 3 seeds |
| curve | 19 | Adaptive on six reduced budgets × 3 subsets plus one full-data run |
| baselines | 50 | Plain and residual on route and its control (12) and all seven curve budgets (38) |

The runner writes independent day results before training the wind context. It generates fresh stress-test caches from validated manifest paths, calls the existing wind/day and learning-curve scripts, and writes `per_flight.csv`, `paired.csv`, `per_seed.csv`, `summary.csv`, `liu_fits.csv` and an interpretation for each completed cohort. A run with an interrupted training process restarts that configuration if it lacks a final checkpoint.

## Verified smoke comparison

The normal adaptive CLI scored 12.69% power MAPE and 6.33% regime RE on all 36 wind-holdout flights after one training epoch on five flights. The paired phase check below uses the 20 flights in the protocol's common cohort. Their omitted warmup averages 2.85 seconds.

| One epoch, seed 0 | Power MAPE | Regime RE | Signed Wh | Absolute Wh | Underprediction Wh |
|---|---:|---:|---:|---:|---:|
| Plain leaky | 21.95% | 14.86% | −1.206 | 1.223 | 1.215 |
| Residual leaky | 12.46% | 6.51% | −0.627 | 0.698 | 0.662 |
| Adaptive leaky | 12.59% | 6.38% | −0.513 | 0.633 | 0.573 |

The adaptive smoke run reduced mean underprediction by 0.089 Wh relative to residual. Cruise contributed the largest signed energy change (+0.058 Wh relative to residual). Pointwise MAPE was slightly higher for adaptive than for residual. These measurements verify the evaluation path. The research comparison requires the three seeds and 30 epochs listed above. [Phase breakdown and paired results](smoke/interpretation.md)

Verification covered raw fitted Liu, zero-correction residual and adaptive initialization in watts, extreme-logit bounds, finite gradients and a nonzero head gradient, checkpoint round-trip, prediction/diagnostic agreement, legacy plain checkpoint inference, and the −20 W over 60 s example (−1/3 Wh). The real-flight scorer checks conservation across phases and equality between the three corrections and adaptive minus refitted Liu at every scored timestep.
