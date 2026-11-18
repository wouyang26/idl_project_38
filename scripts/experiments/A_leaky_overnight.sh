#!/usr/bin/env bash
# Overnight rerun after the leaky-head fix (2026-11-15). Plan: 28 leaky hybrid runs (A2 + A3) and 9 leaky s-LSTM
# runs (wind, random_sub_wind, route), 6 at a time, longest first; then stress_test.py and learning_curve.py.
# Skips runs that already have final.pt, so it can be re-run after an interruption.
#   setsid nohup bash Results/overnight_leaky.sh > Results/overnight_leaky.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH=$HOME/miniconda3/envs/idl-proj/bin:$PATH
JOBS=Results/overnight_leaky.jobs
: > "$JOBS"

TCN="--nb_filters 16 --kernel_size 2 --n_layers 5 --total_epochs 30"
LSTM="--units 16 --n_hidden 3 --optimizer RMSprop --stateful --batch_size 1 --total_epochs 30"
sfx() { [ "$1" != 0 ] && echo "_s$1"; }

# LSTMs first: ~50 min each vs ~10 min for a hybrid, so they should not be left to the end
for split in wind random_sub_wind route; do
    for seed in 0 1 2; do
        echo "lstm_20_0.0_16_3_RMSprop_leaky_${split}$(sfx $seed) --net_type lstm --head_act leaky $LSTM --split $split --seed $seed" >> "$JOBS"
    done
done
# A2: hybrid on wind / route and their controls (random_subsampled already done; listed so it is skipped)
for split in wind random_sub_wind route random_subsampled; do
    for seed in 0 1 2; do
        echo "residual_20_16_2_1_5_leaky_${split}$(sfx $seed) --net_type residual --head_act leaky $TCN --split $split --seed $seed" >> "$JOBS"
    done
done
# A3: learning curve on the random split, plus the full-size point
for n in 5 10 20 35 52 87; do
    for k in 0 1 2; do
        echo "residual_20_16_2_1_5_leaky_n${n}_k${k} --net_type residual --head_act leaky $TCN --train_size $n --subset_seed $k" >> "$JOBS"
    done
done
echo "residual_20_16_2_1_5_leaky --net_type residual --head_act leaky $TCN" >> "$JOBS"

run() {  # run <run name> <train_torch args...>
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

echo "$(date +%T) training finished; stress test"
python stress_test.py --out Results/stress_test
python scripts/learning_curve.py --per_flight Results/stress_test/per_flight.csv --out Results/learning_curve
echo "$(date '+%F %T') ALL DONE"
