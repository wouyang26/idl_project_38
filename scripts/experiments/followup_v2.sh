#!/usr/bin/env bash
# 2026-11-21 follow-up: (1) Adam s-LSTM on random_subsampled (route control), 3 seeds; (2) seeds 1-2 of the four
# full-size random-split models; (3) stress test into Results/stress_test_v2 with the new arch labels, reusing the
# cached per-flight errors (only the Adam LSTM rows change label) and scoring only the new runs.
set -uo pipefail
cd "$(dirname "$0")/../..
export PATH=$HOME/miniconda3/envs/idl-proj/bin:$PATH
JOBS=Results/followup_v2.jobs
: > "$JOBS"

TCN="--nb_filters 16 --kernel_size 2 --n_layers 5 --total_epochs 30"
LSTM="--net_type lstm --units 16 --n_hidden 3 --optimizer Adam --batch_size 32 --total_epochs 30 --head_act leaky"
sfx() { [ "$1" != 0 ] && echo "_s$1"; }

for seed in 0 1 2; do
    echo "lstm_20_0.0_16_3_Adam_leaky_random_subsampled$(sfx $seed) $LSTM --split random_subsampled --seed $seed" >> "$JOBS"
done
for seed in 1 2; do
    echo "tcn_20_16_2_1_5_s$seed --net_type tcn $TCN --seed $seed" >> "$JOBS"
    echo "tcn_20_16_2_1_5_leaky_s$seed --net_type tcn $TCN --head_act leaky --seed $seed" >> "$JOBS"
    echo "residual_20_16_2_1_5_leaky_s$seed --net_type residual $TCN --head_act leaky --seed $seed" >> "$JOBS"
    echo "lstm_20_0.0_16_3_Adam_leaky_s$seed $LSTM --seed $seed" >> "$JOBS"
done

run() {
    local name=$1; shift
    if [ -f "Results/ckpts/$name/final.pt" ]; then echo "$(date +%T) skip $name (exists)"; return 0; fi
    echo "$(date +%T) start $name"
    if python -u train_torch.py "$@" > "Results/train_${name}.log" 2>&1; then
        echo "$(date +%T) done  $name"
    else
        echo "$(date +%T) FAIL  $name (see Results/train_${name}.log)"
    fi
}
export -f run
echo "$(date '+%F %T') $(wc -l < "$JOBS") jobs, 6 at a time"
xargs -P 6 -L 1 bash -c 'run "$@"' _ < "$JOBS"
echo "$(date +%T) training finished"

mkdir -p Results/stress_test_v2
python - <<'EOF'
import pandas as pd
df = pd.read_csv('Results/stress_test/per_flight.csv')
adam = df['run'].str.contains('_Adam_') & (df['arch'] == 'lstm16-leaky')
df.loc[adam, 'arch'] = 'lstm16-adam-leaky'
df.to_csv('Results/stress_test_v2/per_flight.csv', index=False)
print('relabelled', adam.sum(), 'cached rows of', df.loc[adam, 'run'].nunique(), 'Adam LSTM runs')
EOF
python stress_test.py --out Results/stress_test_v2
echo "$(date '+%F %T') ALL DONE"
