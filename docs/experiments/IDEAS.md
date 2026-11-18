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

## Why retrained LSTMs are still bad (relevant to D)

**Finding.** The leaky head (`--head_act leaky`, `adesai/leaky-head`) stops the LSTM collapsing to a constant, but
does not make it good. s-LSTM, 3 seeds, power MAPE (%) held-out / control:

| split | s-LSTM leaky | s-LSTM ReLU | s-TCN |
|---|---|---|---|
| `wind` | 17.9 / 20.0 | 17.7 / 19.4 | 11.6 / 10.8 |
| `route` | 16.1 / 14.7 | 13.5 / 16.2 (1 seed) | 10.9 / 9.8 |

So the dead head was one problem, not the only one: retrained LSTMs are ~6 points worse than the TCN, and their
held-out vs control gaps are meaningless at that error level.

**Question.** Is it the training recipe or the model? Suspects, cheapest first:
- **Optimizer and batch size.** The paper's setup is RMSprop at batch size 1 (stateful), ~90k noisy steps per epoch
  vs the TCN's ~2.8k Adam steps at batch 32. Try Adam, a lower learning rate (1e-4), gradient clipping.
- **Stateful training.** Try non-stateful, batch 32 windows like the TCN (`--stateful` off). Also much faster.
- **Too few epochs** for an LSTM at this learning rate; plot validation MAPE over 60–100 epochs.
- **Checkpoint choice.** `final.pt` is the last epoch; the LSTM's validation MAPE swings 3–6 points between epochs,
  so keep the best-validation epoch instead.
- **Port vs training:** score the ported pretrained s-LSTM on the same random-split flights. If it is close to the
  TCN, the model is fine and our training is the problem.

**How.** s-LSTM on `random` (and then `wind` / `random_sub_wind`), one change at a time, 3 seeds, compare
validation and test MAPE with the s-TCN. Non-stateful runs take minutes on a GPU; stateful ones ~50 min each.

**Fallback.** If none of these closes the gap, report the final results on the TCNs and state that retrained LSTMs
were unreliable in our setup.
