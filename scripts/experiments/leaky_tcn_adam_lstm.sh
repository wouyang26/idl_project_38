#!/usr/bin/env bash
# Yashasvi's 11-17 plan: (1) leaky s-TCN baseline, 31 runs; (2) s-LSTM with Adam, batch 32, leaky head.
# 6 at a time, longest first; skips runs that already have final.pt.
# Differences from her message: the 9 wind / route LSTM runs always run (cheap at batch 32; report them only if
# the random-split test beats 12.4 MAPE), and no --stateful (only meaningful at batch size 1).
#   setsid nohup systemd-inhibit --what=sleep:idle:handle-lid-switch --why=training \
#       bash Results/overnight_tcn_lstm.sh > Results/overnight_tcn_lstm.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/../..
export PATH=$HOME/miniconda3/envs/idl-proj/bin:$PATH
JOBS=Results/overnight_tcn_lstm.jobs
: > "$JOBS"

TCN="--nb_filters 16 --kernel_size 2 --n_layers 5 --total_epochs 30"
LSTM="--net_type lstm --units 16 --n_hidden 3 --optimizer Adam --batch_size 32 --total_epochs 30"
sfx() { [ "$1" != 0 ] && echo "_s$1"; }

# (2) s-LSTM, Adam: the test on the random split, plus wind / random_sub_wind / route x 3 seeds
echo "lstm_20_0.0_16_3_Adam_leaky $LSTM --head_act leaky" >> "$JOBS"
for split in wind random_sub_wind route; do
    for seed in 0 1 2; do
        echo "lstm_20_0.0_16_3_Adam_leaky_${split}$(sfx $seed) $LSTM --head_act leaky --split $split --seed $seed" >> "$JOBS"
    done
done
# (1) leaky s-TCN: A2 splits x 3 seeds, A3 learning curve, full size
for split in wind random_sub_wind route random_subsampled; do
    for seed in 0 1 2; do
        echo "tcn_20_16_2_1_5_leaky_${split}$(sfx $seed) --net_type tcn $TCN --head_act leaky --split $split --seed $seed" >> "$JOBS"
    done
done
for n in 5 10 20 35 52 87; do
    for k in 0 1 2; do
        echo "tcn_20_16_2_1_5_leaky_n${n}_k${k} --net_type tcn $TCN --head_act leaky --train_size $n --subset_seed $k" >> "$JOBS"
    done
done
echo "tcn_20_16_2_1_5_leaky --net_type tcn $TCN --head_act leaky" >> "$JOBS"

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

echo "$(date +%T) training finished"
echo "== (2) s-LSTM Adam on the paper's test flights; beat 12.4 MAPE to keep the wind/route runs =="
python evaluate_torch.py -m s-LSTM Results/ckpts/lstm_20_0.0_16_3_Adam_leaky/final.pt -e test
echo "== (1) stress test + learning curve =="
python stress_test.py --out Results/stress_test
python scripts/learning_curve.py --per_flight Results/stress_test/per_flight.csv --out Results/learning_curve
echo "$(date '+%F %T') ALL DONE"
