#!/bin/bash
set -e

EXPLAINER_NAME=ig

for dataset in cifar10; do
    python scripts/roar.py \
        --dataset_name=$dataset \
        --save_dir=results/benchmark_roar/$dataset/$EXPLAINER_NAME \
        --config-name=$EXPLAINER_NAME
done
