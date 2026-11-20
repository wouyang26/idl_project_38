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
- **Port vs training — done (2026-11-17).** On the random split's test flights (`evaluate_torch.py -e test`):
  pretrained s-LSTM 12.36 / 8.23 (MAPE / RE), b-LSTM 11.47 / 7.59; pretrained s-TCN 9.13 / 5.75, b-TCN 9.06 / 5.37;
  our retrained s-LSTM (ReLU, seed 0, the paper's recipe) 13.45 / 9.23. So the paper's own LSTMs are ~3 points worse
  than its TCNs, and our retrain is only ~1 point behind the paper's LSTM on random flights. Most of the gap to the
  TCN is the LSTM itself, not our port; the extra loss on `wind` / `route` is what is left to explain.

**How.** s-LSTM on `random` (and then `wind` / `random_sub_wind`), one change at a time, 3 seeds, compare
validation and test MAPE with the s-TCN. Non-stateful runs take minutes on a GPU; stateful ones ~50 min each.

**Fallback.** If none of these closes the gap, report the final results on the TCNs and state that retrained LSTMs
were unreliable in our setup.

## Online adaptation during the flight (Neural-Fly style) — Yashasvi

**Question.** Can the model fix its own wind error mid-flight, from the power it has measured so far? The model is
worse on unseen wind because it cannot adapt; Neural-Fly (O'Connell et al., *Science Robotics* 2022) handles unseen
wind by learning a wind-independent representation offline and adapting a few linear coefficients online from the
measured error. Our setup has the same shape: the drone measures its power throughout the flight.

**Scope.** This helps predict the *remaining* energy part-way through a flight ("can I still finish the delivery?"),
not the pre-takeoff estimate, where nothing has been measured yet.

**How.** No training; a scoring script on existing checkpoints.
- Freeze the trained model except its last layer (`power_reg`: 10 weights + 1 bias). Its 10 inputs are the head
  features, so the output is `w·h + b` (plus Liu for the hybrid).
- On each test flight, fit `w, b` by recursive least squares (with a prior pulling them toward the trained values,
  so a few seconds of data can't throw them far) on the first T seconds of measured power, then predict the rest.
- T ∈ {15, 30, 60} s (flights are 84–194 s, median ~130 s). Score MAPE / energy error on the remaining steps only,
  for both adapted and unadapted models, so the comparison uses the same steps.
- Models: s-TCN `wind` and `random_sub_wind` (C checkpoints, 3 seeds, already on the laptop), then the leaky hybrid
  once its checkpoints are shared. Flights: the 20 unseen windy flights, and the random split's test flights as a
  check that adapting doesn't hurt calm flights.

**Success.** The `wind` model's error on the rest of unseen windy flights drops by more than the seed spread, and
the gap to `random_sub_wind` shrinks.

**Caveat.** The first seconds are takeoff and climb, which load the motors differently from cruise, so coefficients
fit there may not transfer. If short T hurts, also try adapting on a sliding window (the last T seconds) instead.

**Cost.** Minutes on a laptop CPU.

**Result (2026-11-19, `scripts/online_adapt.py`, C checkpoints, 3 seeds).** Online mode, error over the rest of
the flight after T s, % (`docs/experiments/online_adapt/summary.csv`):

| model, flights | T | energy error none → online | power MAPE none → online |
|---|---|---|---|
| s-TCN `wind`, 20 unseen windy | 15 s | 5.16 → **2.38** | 11.75 → 11.84 |
| s-TCN `wind`, 20 unseen windy | 30 s | 4.91 → **3.11** | 11.53 → 11.96 |
| b-TCN `wind`, 20 unseen windy | 15 s | 5.07 → **2.36** | 11.70 → 11.70 |
| s-TCN `random_sub_wind` (control) | 15 s | 3.52 → 1.89 | 10.78 → 10.94 |
| s-TCN `windy_days`, 11 calm | 15 s | 3.34 → 1.03 | 10.21 → 9.59 |
| s-TCN `windy_days`, 11 calm | 60 s | 2.13 → 2.96 | 9.31 → 9.31 |

- Refitting 11 numbers from the power measured so far roughly halves the error of the remaining-energy estimate on
  unseen windy flights, for both TCNs, and shrinks the wind vs control energy gap from ~1.6–2.4 to ~0.4–0.5 points
  (T = 15 s).
- It fixes the slow bias, not the step-by-step error: power MAPE barely moves on windy flights.
- It is not wind-specific: it also helps the control and calm flights, most where the bias is largest (wind).
- It can hurt: on calm flights at T = 60 s the energy error rises (2.1 → 3.0). Fixed mode (fit once on the first T
  s, mostly takeoff) is worse than no adaptation, as the caveat above expected.
- lam is picked per model on its validation flights (online, T = 30 s, energy error); 23 of 24 picked 10 or 100.
