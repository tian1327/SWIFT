#!/bin/bash

datasets=(
    "semi-aves"
    "fgvc-aircraft"
    "stanford_cars"
    "eurosat"
    "dtd"
)

for dataset in "${datasets[@]}"; do
    echo ""
    echo "fewshot finetuning on $dataset"
    bash scripts/run_dataset_seed_FSFT_LP-init.sh $dataset 1
    # bash scripts/run_dataset_seed_FSFT_LP-init_INet50.sh $dataset 1
done