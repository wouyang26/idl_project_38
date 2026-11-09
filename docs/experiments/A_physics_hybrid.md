# Experiment A: physics + TCN hybrid

**Question.** Does starting from Liu's physics model let a TCN learn power with less data, or generalize better to
the conditions where the plain TCN struggles (`wind`, `route`)?

**Model.** The TCN predicts only the correction to Liu: `P̂_t = P_Liu(x_t) + g_θ(x_{t−19:t})`, with Liu's constants
refit on the run's own training flights. "Physics didn't help" is a reportable result.

Run everything with `bash scripts/experiments/A_physics_hybrid.sh` (from the repo root, after
`scripts/download_data.sh`). It skips runs that already have `final.pt`.

## Code this needs (interface)

The run script assumes these names; whoever implements them can change the internals freely.

| Piece | Interface | Behaviour |
|---|---|---|
| Liu refit | `python fit_liu.py --split <split>` → `Results/liu/<split>.json` | Fits the 5 constants (`k1, k2, c2, c4, c5`, start from `liu_model.optimum_values()`) on the split's **training** flights only (no validation or test flights), with the same `get_split` as training. |
| Hybrid | `train_torch.py --net_type residual` | Same backbone, head and options as `--net_type tcn`. Output = `P_Liu` (refit constants for `--split`) + head output, in normalized power, so `predict_flight`, `stress_test.py` and `scripts/seed_results.py` work unchanged. The last layer starts at zero, so an untrained model is exactly Liu. Run name prefix `residual_` instead of `tcn_`. Liu constants stored in the checkpoint. |
| Train size | `train_torch.py --train_size N --subset_seed k` | Cuts the split's training flights to a random `N` (`RandomState(k)`), keeping val and test. Run name suffix `_n<N>_k<k>`. |
| Stress test label | `stress_test.py`: `size = nb_filters` for `net_type in ('tcn', 'residual')` | Today it uses `units` for anything that isn't `tcn`, which would label the hybrid `residual32`. |
| α·Liu baseline | `python fit_liu.py --split <split> --scale_only` → `Results/liu/<split>_alpha.json` | Fits one factor α on the training flights, authors' constants kept: `P̂ = α·P_Liu`. |

Liu takes raw (not normalized) airspeed, vertical speed, angle of attack and payload: denormalize with the
checkpoint's `data_min` / `data_max`, or pass the raw columns alongside. `liu_model.power` ignores air density.

## Fixed settings

Same as the s-TCN baseline, so only the physics term differs: s-TCN backbone (`--nb_filters 16 --kernel_size 2
--n_layers 5`), Adam 1e-3, batch 32, lookback 20, 30 epochs, `final.pt`, seeds 0 1 2. No tuning on test flights.

## A1. Liu refit (no training, minutes on a CPU)

Liu with the authors' constants vs Liu refit per split, on every held-out split's unseen flights. Fixes the leak noted
in the midterm (the authors fit Liu on all flights) and replaces the Liu column of Table 3.

## A2. Wind and route (12 hybrid runs + existing s-TCN runs)

| Model | Splits | Runs |
|---|---|---|
| Liu, authors' constants / refit / α·Liu | `wind`, `random_sub_wind`, `route`, `random_subsampled` | no training |
| s-TCN | same | exist (Aaryan's stress test) |
| residual hybrid | same | 4 splits × 3 seeds = 12 |

Report MAPE / RE on the unseen flights and the gap to the control, as in Table 3 (`stress_test.py`).

## A3. Learning curve (2 × 6 × 3 = 36 runs)

`random` split, training flights cut to N ∈ {5, 10, 20, 35, 52, 87}, three random subsets per N (`--subset_seed`
0 1 2, model seed 0), s-TCN and residual hybrid. Test: the random split's 35 test flights. The full 114 is the
existing baseline. Plot MAPE and RE vs N, mean ± std over subsets, with Liu (refit on the same subset) as a flat-ish
reference.

## Success, decided before training

- **Data efficiency:** the hybrid beats the s-TCN by more than the subset spread at N ≤ 20.
- **Generalization:** the hybrid's `wind` or `route` gap is smaller than the s-TCN's by more than the seed spread.
- Neither → physics didn't help this TCN; report it with the curve.

## Later (only if A2 or A3 shows a win)

- **Parameter adaptation:** the TCN outputs three bounded multipliers on Liu's induced, profile and parasitic terms
  (`m = exp(clip(Δ, ±0.7))`) instead of a residual; `--net_type adaptive`. Same splits as A2. Precedent: Zhang et al.
  2022/2025 (latent "intrinsics" from flight history). The residual hybrid's precedent is NeuroBEM (Bauersfeld et
  al., RSS 2021).
- `windy_days` / `random_sub_windy_days` (experiment C) for the hybrid.

## Compute

TCN-sized: ~2.5 min per run on a 5070 Ti, so A2 + A3 (48 runs) take about 2 h; A1 needs no GPU.
