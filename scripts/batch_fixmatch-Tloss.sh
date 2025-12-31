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
    echo "FixMatch-Tloss on $dataset"
    bash scripts/run_dataset_seed_fixmatch-Tloss.sh $dataset 1
done