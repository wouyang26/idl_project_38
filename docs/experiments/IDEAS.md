# Experiments to try

A running list of follow-up ideas. Each one says what it would answer and what it would take. Move an item into its
own spec in `docs/experiments/` once someone picks it up.

## Longer lookback window (for wind)

**Question.** Does a longer history help on unseen windy flights? The window is 20 steps (~4 s), the original
paper's setting, never tuned. Gusts play out over more than a few seconds, so wind may need more context.

**How.** `train_torch.py --lookback N` already exists. Train the s-TCN with N ∈ {20, 40, 80} on `wind` and
`random_sub_wind`, 3 seeds, and compare the wind gap with `stress_test.py`. Above 32 steps the TCN can't see the whole
window (5 layers, dilations 1–16), so use `--n_layers 6` for 40 and `--n_layers 7` for 80 (5.2k → 6.3k / 7.3k parameters, so it stays small; see
below). A longer window also
drops more steps at the start of each flight from scoring (`FIRST` in `stress_test.py`), so score all N on the same
steps.

**Cost.** 18 runs, under an hour on a GPU.

## Accuracy per parameter (model size budget)

**Question.** How small can the model be and keep its accuracy? It would run continuously on the drone's onboard
computer, so parameters, memory and time per prediction are a real budget, not just an accuracy footnote.

What we already know: the s-TCN has 5,225 parameters and the b-TCN 76,073 (15× more), yet their stress-test gaps
are about the same (`wind` +1.05 vs +0.99 MAPE) and their held-out MAPE is within ~0.1–0.2 points. The physics
hybrid adds only Liu's 5 constants, and Liu alone is 5 parameters. So far the extra size buys little.

**How.**
- Sweep the s-TCN smaller: `--nb_filters` 4 / 8 / 16 and `--n_layers` 3–5, on `random` and `wind`, 3 seeds. Plot
  MAPE / energy error vs parameter count, with Liu (5) and the b-TCN (76k) as end points.
- Repeat for the hybrid: if physics does most of the work, its correcting TCN may shrink further than the plain one
  at the same accuracy.
- Time one prediction on a CPU at batch size 1 (the drone needs one every ~0.2 s), and if possible on the onboard
  computer itself.
- Later: 8-bit quantization of the chosen model and its accuracy cost.

**Cost.** ~36–54 small runs, about an hour on a GPU; timing needs no training.
