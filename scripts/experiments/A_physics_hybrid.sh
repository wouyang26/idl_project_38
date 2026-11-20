#!/usr/bin/env bash
# Experiment A: physics + TCN hybrid. Spec and interface: docs/experiments/A_physics_hybrid.md
#
#   bash scripts/experiments/A_physics_hybrid.sh            # from the repo root, after scripts/download_data.sh
#
# Up to 62 training runs (~2.5 min each on a 5070 Ti, ~10 min on a laptop CPU). Runs that already have final.pt are skipped,
# so it can be re-run after an interruption. A2 needs the s-TCN runs of wind / random_sub_wind / route /
# random_subsampled in Results/ckpts (it trains any that are missing).
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p Results

TCN="--nb_filters 16 --kernel_size 2 --n_layers 5 --total_epochs 30"
A2_SPLITS="wind random_sub_wind route random_subsampled"
SEEDS="0 1 2"
SIZES="5 10 20 35 52 87"
SUBSETS="0 1 2"

train() {  # train <run name> <args...>
    local run=$1; shift
    if [ -f "Results/ckpts/$run/final.pt" ]; then echo "skip $run (exists)"; return; fi
    echo "train $run"
    python train_torch.py "$@" > "Results/train_${run}.log" 2>&1
}

# A1: Liu refit (and alpha * Liu) on every held-out split's training flights; no training, ~15 s
python fit_liu.py

# A2: residual hybrid vs s-TCN on wind / route and their controls, 3 seeds
for net in tcn residual; do
    for split in $A2_SPLITS; do
        for seed in $SEEDS; do
            run="${net}_20_16_2_1_5_${split}"; [ "$seed" != 0 ] && run="${run}_s${seed}"
            train "$run" --net_type "$net" $TCN --split "$split" --seed "$seed"
        done
    done
done

# A3: learning curve on the random split, s-TCN vs residual hybrid, 3 training subsets per size
for n in $SIZES; do
    for k in $SUBSETS; do
        train "tcn_20_16_2_1_5_n${n}_k${k}" --net_type tcn $TCN --train_size "$n" --subset_seed "$k"
        train "residual_20_16_2_1_5_n${n}_k${k}" --net_type residual $TCN --train_size "$n" --subset_seed "$k"
    done
done

# the full-size random-split runs are the N = 114 point of the curve
train "tcn_20_16_2_1_5" --net_type tcn $TCN
train "residual_20_16_2_1_5" --net_type residual $TCN

python stress_test.py --ckpts 'Results/ckpts/*_20_16_2_1_5*/final.pt' --out Results/stress_test
python scripts/learning_curve.py --per_flight Results/stress_test/per_flight.csv --out Results/learning_curve
echo "done: A1 Results/liu/liu_refit.csv, A2 Results/stress_test/gaps.csv (tcn16 vs residual16 rows),"
echo "      A3 Results/learning_curve/learning_curve.{csv,png}"
