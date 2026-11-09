#!/usr/bin/env bash
# Experiment A: physics + TCN hybrid. Spec and interface: docs/experiments/A_physics_hybrid.md
#
#   bash scripts/experiments/A_physics_hybrid.sh            # from the repo root, after scripts/download_data.sh
#
# Needs the hybrid code (fit_liu.py, train_torch.py --net_type residual / --train_size / --subset_seed); stops with a
# message if it is missing. Runs that already have final.pt are skipped, so it can be re-run after an interruption.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p Results

help=$(python train_torch.py --help)
for need in "\{[^}]*residual[^}]*\}" "--train_size" "--subset_seed"; do  # residual: a --net_type choice
    grep -qE -- "$need" <<<"$help" || { echo "train_torch.py has no '$need' yet (see docs/experiments/A_physics_hybrid.md)"; exit 1; }
done
[ -f fit_liu.py ] || { echo "fit_liu.py is missing (see docs/experiments/A_physics_hybrid.md)"; exit 1; }

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

# A1: Liu refit (and alpha * Liu) on every split's training flights; no GPU
for split in random payload500 payload250 speed12 alt100 route speed10-12 speed8-12 wind windy_days \
             random_subsampled random_sub_speed10-12 random_sub_speed8-12 random_sub_wind random_sub_windy_days; do
    python fit_liu.py --split "$split"
    python fit_liu.py --split "$split" --scale_only
done

# A2: residual hybrid on wind / route and their controls, 3 seeds (s-TCN runs already exist)
for split in $A2_SPLITS; do
    for seed in $SEEDS; do
        run="residual_20_16_2_1_5_${split}"; [ "$seed" != 0 ] && run="${run}_s${seed}"
        train "$run" --net_type residual $TCN --split "$split" --seed "$seed"
    done
done

# A3: learning curve on the random split, s-TCN vs residual hybrid, 3 training subsets per size
for n in $SIZES; do
    for k in $SUBSETS; do
        train "tcn_20_16_2_1_5_n${n}_k${k}" --net_type tcn $TCN --train_size "$n" --subset_seed "$k"
        train "residual_20_16_2_1_5_n${n}_k${k}" --net_type residual $TCN --train_size "$n" --subset_seed "$k"
    done
done

python stress_test.py --ckpts 'Results/ckpts/residual_*/final.pt' --out Results/stress_test
python stress_test.py --ckpts 'Results/ckpts/*_n*_k*/final.pt' --out Results/stress_test
echo "done: A2 in Results/stress_test/gaps.csv; A3 per-flight errors in Results/stress_test/per_flight.csv"
