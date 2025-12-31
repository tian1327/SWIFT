#!/bin/bash

datasets=(
    # "semi-aves"
    # "fgvc-aircraft"
    # "stanford_cars"
    # "eurosat"
    "dtd"
)

for dataset in "${datasets[@]}"; do
    echo ""
    echo "SWIFT on $dataset"

    bash scripts/run_dataset_seed_swift.sh $dataset 1
    # bash scripts/run_dataset_seed_fixmatch_dinov2_LP-init.sh $dataset 1
done