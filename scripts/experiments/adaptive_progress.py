"""Refresh the #5/#6 handoff from local artifacts. With --watch, refresh once a minute."""
import argparse
import csv
from datetime import datetime
import fcntl
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/experiments/adaptive_physics'
RESULTS = ROOT / 'Results'
REPORTS = ['day', 'wind_context', 'wind', 'route', 'curve']
BASE = 'a2f1fd8c9dd7df63fe258832a67229325e8d5863'
BRANCH = 'ashutosh/adaptive-physics'


def local_time(stamp=None):
    dt = datetime.now().astimezone() if stamp is None else datetime.fromtimestamp(stamp).astimezone()
    return dt.isoformat(timespec='seconds')


def refresh():
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    if branch != BRANCH:
        raise SystemExit(f'Progress writer stopped on branch {branch!r}. Expected {BRANCH!r}.')
    with (OUT / 'run_matrix.csv').open(newline='') as stream:
        matrix = list(csv.DictReader(stream))
    present = {row['checkpoint'] for row in matrix if (ROOT / row['checkpoint']).is_file()}
    logs = [p for p in (RESULTS / 'adaptive_day.log', RESULTS / 'adaptive_sweep.log') if p.exists()]
    lines = [
        '# Agent handoff for adaptive physics', '',
        f'Last refreshed **{local_time()}**. This generated file refreshes every minute while the watcher runs.',
        '', f'**{len(present)} of {len(matrix)} planned final checkpoints are on disk.** '
        'The counts exclude smoke runs. The sweep validates checkpoint metadata before reuse.', '',
        '| Stage | Final checkpoints | Planned |', '|---|---:|---:|',
    ]
    for stage in dict.fromkeys(row['stage'] for row in matrix):
        group = [row for row in matrix if row['stage'] == stage]
        lines.append(f"| {stage} | {sum(row['checkpoint'] in present for row in group)} | {len(group)} |")
    if logs:
        log = max(logs, key=lambda p: p.stat().st_mtime)
        history = log.read_text(errors='replace').splitlines()
        resume_markers = [i for i, line in enumerate(history) if line.startswith('--- Resuming at ')]
        if resume_markers:
            history = history[resume_markers[-1] + 1:]
        starts = [i for i, line in enumerate(history) if line.startswith('missing ')]
        run = history[starts[-1] + 1:] if starts else []
        epochs = [m[1] for line in run if (m := re.match(r'epoch\s+(\d+)\s*\|', line))]
        lines += ['', f'Latest log: `Results/{log.name}`, modified {local_time(log.stat().st_mtime)}.']
        if starts:
            run_name = history[starts[-1]].removeprefix('missing ')
            lines.append(f'Last run started in this log: `{run_name}`. '
                         + (f'Last logged epoch: {epochs[-1]}/30.' if epochs else 'Awaiting its first logged epoch.'))
        lines += ['', 'Last nonblank log lines:', '', '```text',
                  *[line for line in history if line.strip()][-3:], '```']
    else:
        lines += ['', 'No training log is present at the expected paths.']
    reports = [d for d in REPORTS if (OUT / d / 'interpretation.md').exists()]
    extra = [p for p in ['day/wind_vs_day.csv', 'curve/learning_curve.csv', 'curve/learning_curve.png']
             if (OUT / p).exists()]
    complete = len(present) == len(matrix) and len(reports) == len(REPORTS) and len(extra) == 3
    lines += ['', '**Evaluation artifacts**', '',
              'Production interpretations present: ' + (', '.join(f'[{d}]({d}/interpretation.md)' for d in reports) or 'none yet') + '.',
              'Additional outputs present: ' + (', '.join(f'[{p}]({p})' for p in extra) or 'none yet') + '.',
              'Training and evaluation are complete. [Final results and reproduction](README.md).' if complete else
              'Next action: follow the active queue, then review the matched production comparisons.',
              '', '**Pickup instructions**', '',
              f'- Use `{BRANCH}`, based on PR #11 at `{BASE}`. Work is in the shared checkout. '
              'Check `git status -sb` and preserve existing edits, including the earlier untracked status brief.',
              '- Objective: compare leaky plain and residual on calm flights from windy dates (#5), then determine '
              'whether adaptive Liu improves on both baselines on unseen wind, where signed Wh changes by phase, '
              'and whether battery-energy underprediction decreases (#6).',
              '- Adaptive implementation and smoke verification passed, including refit-Liu identity, gradients, '
              'reload, term correction and phase Wh conservation. [Smoke results](smoke/interpretation.md). '
              'Use 30-epoch matched runs for the research conclusion.',
              '- [Runbook](README.md) covers data, the 79 shared checkpoints screened, validation, and reproduction. '
              '[Run matrix](run_matrix.csv) gives every checkpoint path. Results/ is local and ignored by Git.',
              '- The first day runner writes `Results/adaptive_day.log`. A host continuation waits for it before '
              'starting the full sweep and writing `Results/adaptive_sweep.log`. Check both log modification times '
              'and the host processes before restarting. The sandbox process list may hide GPU jobs.',
              '- Run these from the repository root in a host terminal:', '',
              '```bash', "pgrep -af 'adaptive_runs.py|run_adaptive.sh|train_torch.py'","nvidia-smi",
              'tail -n 5 Results/adaptive_day.log', 'test ! -f Results/adaptive_sweep.log || tail -n 5 Results/adaptive_sweep.log',
              '# Resume only if the original queue has exited and the matrix is incomplete:',
              'PYTHON=/home/ashutosh/miniconda3/envs/idl_project/bin/python DEVICE=cuda bash scripts/experiments/run_adaptive.sh',
              '# Restart the progress writer after a reboot, or omit --watch for a single refresh:',
              'python3 scripts/experiments/adaptive_progress.py --watch', '```', '']
    path = OUT / 'HANDOFF.md'
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text('\n'.join(lines))
    os.replace(tmp, path)
    return complete


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    if args.watch:
        RESULTS.mkdir(exist_ok=True)
        with (RESULTS / 'adaptive-progress.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit('The progress watcher is already running.')
            while not refresh():
                time.sleep(60)
    else:
        refresh()
