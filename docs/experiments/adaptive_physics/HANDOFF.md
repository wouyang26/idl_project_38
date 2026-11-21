# Agent handoff for adaptive physics

Last refreshed **2026-11-21T09:44:03-05:00**. This generated file refreshes every minute while the watcher runs.

**105 of 105 planned final checkpoints are on disk.** The counts exclude smoke runs. The sweep validates checkpoint metadata before reuse.

| Stage | Final checkpoints | Planned |
|---|---:|---:|
| day | 12 | 12 |
| context | 12 | 12 |
| shift | 12 | 12 |
| curve | 19 | 19 |
| baselines | 50 | 50 |

Latest log: `Results/adaptive_sweep.log`, modified 2026-11-21T02:35:46-05:00.
Last run started in this log: `residual_20_16_2_1_5_leaky`. Last logged epoch: 30/30.

Last nonblank log lines:

```text
     tcn16-leaky        87.0       8.53      0.13           3     4.55    0.13         3
     tcn16-leaky       114.0       8.68       NaN           1     4.65     NaN         1
saved docs/experiments/adaptive_physics/curve
```

**Evaluation artifacts**

Production interpretations present: [day](day/interpretation.md), [wind_context](wind_context/interpretation.md), [wind](wind/interpretation.md), [route](route/interpretation.md), [curve](curve/interpretation.md).
Additional outputs present: [day/wind_vs_day.csv](day/wind_vs_day.csv), [curve/learning_curve.csv](curve/learning_curve.csv), [curve/learning_curve.png](curve/learning_curve.png).
Training and evaluation are complete. [Final results and reproduction](README.md).

**Pickup instructions**

- Use `ashutosh/adaptive-physics`, based on PR #11 at `a2f1fd8c9dd7df63fe258832a67229325e8d5863`. Work is in the shared checkout. Check `git status -sb` and preserve existing edits, including the earlier untracked status brief.
- Objective: compare leaky plain and residual on calm flights from windy dates (#5), then determine whether adaptive Liu improves on both baselines on unseen wind, where signed Wh changes by phase, and whether battery-energy underprediction decreases (#6).
- Adaptive implementation and smoke verification passed, including refit-Liu identity, gradients, reload, term correction and phase Wh conservation. [Smoke results](smoke/interpretation.md). Use 30-epoch matched runs for the research conclusion.
- [Runbook](README.md) covers data, the 79 shared checkpoints screened, validation, and reproduction. [Run matrix](run_matrix.csv) gives every checkpoint path. Results/ is local and ignored by Git.
- The first day runner writes `Results/adaptive_day.log`. A host continuation waits for it before starting the full sweep and writing `Results/adaptive_sweep.log`. Check both log modification times and the host processes before restarting. The sandbox process list may hide GPU jobs.
- Run these from the repository root in a host terminal:

```bash
pgrep -af 'adaptive_runs.py|run_adaptive.sh|train_torch.py'
nvidia-smi
tail -n 5 Results/adaptive_day.log
test ! -f Results/adaptive_sweep.log || tail -n 5 Results/adaptive_sweep.log
# Resume only if the original queue has exited and the matrix is incomplete:
PYTHON=/home/ashutosh/miniconda3/envs/idl_project/bin/python DEVICE=cuda bash scripts/experiments/run_adaptive.sh
# Restart the progress writer after a reboot, or omit --watch for a single refresh:
python3 scripts/experiments/adaptive_progress.py --watch
```
