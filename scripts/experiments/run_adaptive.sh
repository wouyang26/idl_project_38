#!/usr/bin/env bash
# Run from a checkout with scripts/download_data.sh completed. Resume with the same command.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${ADAPTIVE_BLAS_THREADS:-1}
PY=${PYTHON:-python}
DEVICE=${DEVICE:-cpu}
out=docs/experiments/adaptive_physics
mkdir -p Results "$out"
exec 9>Results/adaptive-sweep.lock
flock -n 9 || { echo 'Another adaptive sweep is running'; exit 1; }
train_stage() {
    "$PY" scripts/experiments/adaptive_runs.py "$1" --device "$DEVICE" --run --manifest "$cache/$1.paths"
}
score() {
    "$PY" scripts/experiments/adaptive_scores.py "$1" --out "$out/$2" "${@:3}"
    "$PY" scripts/experiments/adaptive_report.py "$out/$2"
}
# Use a new metrics cache on each invocation: stress_test keys its cache by run name.
cache=$(mktemp -d Results/adaptive-eval-XXXXXX)
train_stage day
score windy_days day
train_stage context
score wind wind_context --two-models
cat "$cache/day.paths" "$cache/context.paths" > "$cache/experiment5.paths"
"$PY" stress_test.py --device "$DEVICE" --manifest "$cache/experiment5.paths" --out "$cache/day"
"$PY" scripts/wind_vs_day.py --per_flight "$cache/day/per_flight.csv" --out "$out/day"
train_stage shift
score wind wind
train_stage curve
train_stage baselines
score route route
score random curve
cat "$cache/experiment5.paths" "$cache/shift.paths" "$cache/curve.paths" "$cache/baselines.paths" > "$cache/all.paths"
"$PY" stress_test.py --device "$DEVICE" --manifest "$cache/all.paths" --out "$cache/all"
"$PY" scripts/learning_curve.py --per_flight "$cache/all/per_flight.csv" --out "$out/curve"
