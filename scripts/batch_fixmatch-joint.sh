#!/bin/bash

datasets=(
    # "semi-aves"
    "fgvc-aircraft"
    "eurosat"
    "dtd"
    "stanford_cars"
)

for dataset in "${datasets[@]}"; do
    echo ""
    echo "FixMatch-Joint on $dataset"
    bash scripts/run_dataset_seed_fixmatch-joint.sh $dataset 1
done