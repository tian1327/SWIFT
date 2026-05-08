#!/bin/bash

datasets=(
    # "eurosat"
    # "dtd"    
    # "semi-aves"
    # "fgvc-aircraft"
    "stanford_cars"
)

for dataset in "${datasets[@]}"; do
    echo ""
    echo "SWIFT on $dataset"
    bash scripts/run_dataset_seed_swift_random_init.sh $dataset 1
    # bash scripts/run_dataset_seed_FSFT_LP-init.sh $dataset 1
    # bash scripts/run_dataset_seed_FSFT_LP-init_INet50.sh $dataset 1
done