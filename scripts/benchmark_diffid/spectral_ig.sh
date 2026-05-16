#!/bin/bash
set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <overlap>"
    echo "  overlap: 0.1, 0.2, 0.3, 0.4, 0.5"
    exit 1
fi

OVERLAP=$1
SAVE_NAME=spectral_ig_overlap_${OVERLAP}
EXPLAINER_NAME=spectral_ig

for dataset in imagenet oxfordpet oxfordflower; do
    for model_name in vit; do
    python scripts/diffid.py \
        --config-name=$EXPLAINER_NAME \
        dataset=$dataset \
        model_name=$model_name \
        overlap=$OVERLAP \
        max_eval_samples=500 \
        save_dir=results/benchmark_diffid/$dataset/$SAVE_NAME/$model_name \
        dataset.image_size=224
    done
done

# for dataset in oxfordpet oxfordflower imagenet; do
#     for model_name in vgg16 resnet18 inception; do
#     python scripts/diffid.py \
#         --config-name=$EXPLAINER_NAME \
#         dataset=$dataset \
#         model_name=$model_name \
#         overlap=$OVERLAP \
#         max_eval_samples=500 \
#         save_dir=results/benchmark_diffid/$dataset/$SAVE_NAME/$model_name
#     done
# done
