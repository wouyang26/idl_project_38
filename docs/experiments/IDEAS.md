# Experiments to try

A running list of follow-up ideas. Each one says what it would answer and what it would take. Move an item into its
own spec in `docs/experiments/` once someone picks it up.

## Longer lookback window (for wind)

**Question.** Does a longer history help on unseen windy flights? The window is 20 steps (~4 s), the original
paper's setting, never tuned. Gusts play out over more than a few seconds, so wind may need more context.

**How.** `train_torch.py --lookback N` already exists. Train the s-TCN with N ∈ {20, 40, 80} on `wind` and
`random_sub_wind`, 3 seeds, and compare the wind gap with `stress_test.py`. Above 32 steps the TCN can't see the whole
window (5 layers, dilations 1–16), so use `--n_layers 6` for 40 and `--n_layers 7` for 80. A longer window also
drops more steps at the start of each flight from scoring (`FIRST` in `stress_test.py`), so score all N on the same
steps.

**Cost.** 18 runs, under an hour on a GPU.
