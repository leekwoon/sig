#!/bin/bash
set -e

EXPLAINER_NAME=big

for dataset in cifar10; do
    python scripts/roar.py \
        --dataset_name=$dataset \
        --save_dir=results/benchmark_roar/$dataset/$EXPLAINER_NAME \
        --config-name=$EXPLAINER_NAME \
        max_sigma=6
done
