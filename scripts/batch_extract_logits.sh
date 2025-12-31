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
    echo "Extract logits and confidences on $dataset"
    bash scripts/run_dataset_seed_extract_logits.sh $dataset 1
    # bash scripts/run_dataset_seed_extract_logits.sh $dataset 2
    # bash scripts/run_dataset_seed_extract_logits.sh $dataset 3

done