#!/usr/bin/env bash
# Experiment C: is the wind gap wind, or a "different day" effect? (s-TCN and b-TCN, 30 epochs, seeds 0 1 2)
#
#   bash scripts/experiments/C_wind_date.sh            # from the repo root; needs scripts/download_data.sh first
#
# Trains windy_days and its date-matched control random_sub_windy_days (12 runs, ~2.5 min each on a 5070 Ti), and
# wind / random_sub_wind only if those checkpoints are missing. Runs that already have final.pt are skipped, so the
# script can be re-run after an interruption. Then scores everything with stress_test.py and splits the wind gap
# into windy-day vs other-day flights with scripts/wind_vs_day.py.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p Results

declare -A MODELS=(
    [s-TCN]="--nb_filters 16 --kernel_size 2 --n_layers 5"
    [b-TCN]="--nb_filters 64 --kernel_size 2 --n_layers 5"
)
SPLITS="windy_days random_sub_windy_days wind random_sub_wind"
SEEDS="0 1 2"

for model in "${!MODELS[@]}"; do
    nf=$(echo "${MODELS[$model]}" | awk '{print $2}')
    for split in $SPLITS; do
        for seed in $SEEDS; do
            run="tcn_20_${nf}_2_1_5_${split}"
            [ "$seed" != 0 ] && run="${run}_s${seed}"
            if [ -f "Results/ckpts/$run/final.pt" ]; then
                echo "skip $run (exists)"
                continue
            fi
            echo "train $run"
            python train_torch.py ${MODELS[$model]} --total_epochs 30 --split "$split" --seed "$seed" \
                > "Results/train_${run}.log" 2>&1
        done
    done
done

python stress_test.py --ckpts 'Results/ckpts/tcn_20_*_2_1_5*/final.pt' --out Results/stress_test
python scripts/wind_vs_day.py --per_flight Results/stress_test/per_flight.csv --out Results/wind_vs_day
echo "done: Results/stress_test/gaps.csv (windy_days row) and Results/wind_vs_day/wind_vs_day.csv"
