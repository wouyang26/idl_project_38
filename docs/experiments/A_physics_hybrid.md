# Experiment A: physics + TCN hybrid

**Question.** Does starting from Liu's physics model let a TCN learn power with less data, or generalize better to
the conditions where the plain TCN struggles (`wind`, `route`)?

**Model** (`--net_type residual`, `ResidualEnergyModel` in `torch_model.py`). The TCN predicts only the correction to
Liu: `P̂_t = P_Liu(x_t) + g_θ(x_{t−19:t})`. Liu's 5 constants are refit by least squares on the run's own training
flights (never validation or test) and stored in the checkpoint. The last layer starts at zero, so the untrained model
is exactly refit Liu. Everything else (backbone, inputs, normalization, loss, `final.pt`) is the s-TCN's, and the model
outputs normalized power, so `stress_test.py` and the other evaluation scripts work unchanged. "Physics didn't help"
is a reportable result.

Run everything with

```bash
bash scripts/experiments/A_physics_hybrid.sh     # from the repo root, after scripts/download_data.sh
```

It skips runs that already have `final.pt`, so it can be re-run after an interruption.

## Code

| Piece | What it does |
|---|---|
| `physics_torch.py` | Liu in PyTorch, split into induced / profile / parasitic power (matches `liu_model.power` to 5e-16); `fit_liu` (least squares, starts at the authors' constants), `fit_alpha` (`P = α·P_Liu`). |
| `fit_liu.py` | A1: refit Liu and α on every held-out split's training flights, score on the stress test's unseen flights. |
| `train_torch.py --net_type residual` | The hybrid. Run names start with `residual_`. |
| `train_torch.py --train_size N --subset_seed k` | Cuts the training flights to a random `N` (`splits.subsample_train`), keeps val / test. Run name suffix `_n<N>_k<k>`. |
| `scripts/learning_curve.py` | A3: MAPE / RE vs `N` from `per_flight.csv`, with Liu refit on the same subsets. |

## Fixed settings

Same as the s-TCN baseline, so only the physics term differs: `--nb_filters 16 --kernel_size 2 --n_layers 5`, Adam
1e-3, batch 32, lookback 20, 30 epochs, `final.pt`. No tuning on test flights.

## A1. Liu refit (no training) — done

`python fit_liu.py` → `Results/liu/liu_refit.csv`. Power MAPE (%) on the stress test's unseen flights:

| split | n | authors' constants | refit on train | α·Liu | s-TCN (Table 3) |
|---|---|---|---|---|---|
| payload500 | 21 | 11.04 | 8.78 | 10.23 | 7.46 |
| payload250 | 16 | 10.79 | 8.51 | 10.26 | 7.31 |
| speed12 | 9 | 13.49 | 10.94 | 12.87 | 8.87 |
| alt100 | 16 | 10.69 | 9.01 | 10.39 | 8.22 |
| route | 8 | 13.67 | 10.88 | 12.29 | 10.89 |
| speed10-12 | 27 | 13.68 | 10.82 | 13.10 | 9.89 |
| speed8-12 | 57 | 12.85 | 10.59 | 12.49 | 8.82 |
| wind | 20 | 13.94 | 12.61 | 13.50 | 11.55 |
| windy_days | 11 | 13.92 | 11.39 | 13.39 | – |

The authors' column reproduces Table 3. Refitting on the training flights only makes Liu 1.3–2.9 points *better*
everywhere, so the authors' all-flight fit did not favor Liu (the midterm assumed it did); on `route` refit Liu ties
the s-TCN.

## A2. Wind and route (24 runs, 12 of them s-TCN runs that may already exist)

s-TCN (`tcn16`) and hybrid (`residual16`) on `wind`, `random_sub_wind`, `route`, `random_subsampled`, seeds 0 1 2.
`stress_test.py` puts both in `gaps.csv`; compare MAPE / RE on the unseen flights and the gap to the control, with
refit Liu from A1 as the physics-only row.

## A3. Learning curve (36 runs + 2 full-size runs)

`random` split, training flights cut to N ∈ {5, 10, 20, 35, 52, 87} (three subsets each, `--subset_seed` 0 1 2,
model seed 0), s-TCN and hybrid, plus both at the full 114. Test: the random split's 35 test flights.
`scripts/learning_curve.py` → `learning_curve.{csv,png}`, mean ± std over subsets, with Liu refit on the same
subsets. With few flights the refit constants are poorly determined (at N = 5, c4 can hit its bound of ~0).

## Success, decided before training

- **Data efficiency:** the hybrid beats the s-TCN by more than the subset spread at N ≤ 20.
- **Generalization:** the hybrid's `wind` or `route` error or gap is lower than the s-TCN's by more than the seed
  spread.
- Neither: physics didn't help this TCN; report it with the curve.

## Later (only if A2 or A3 shows a win)

- **Parameter adaptation** (`--net_type adaptive`): the TCN outputs three bounded multipliers on Liu's induced,
  profile and parasitic terms (`m = exp(clip(Δ, ±0.7))`) instead of a residual. Precedent: Zhang et al. 2022 / 2025
  (a latent vector from flight history). The residual hybrid's precedent is NeuroBEM (Bauersfeld et al., RSS 2021).
- `windy_days` / `random_sub_windy_days` (experiment C) for the hybrid.

## Compute

~2.5 min per run on a 5070 Ti (under 3 h for everything), ~10 min per run on a laptop CPU. A1 needs no GPU.
