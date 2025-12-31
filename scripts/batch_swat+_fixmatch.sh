#!/bin/bash

datasets=(
    "semi-aves"
    # "flowers102"
    "fgvc-aircraft"
    "eurosat"
    "dtd"
    # "oxford_pets"
    "stanford_cars"
    # "food101"
    # "imagenet"
)

for dataset in "${datasets[@]}"; do
    echo ""
    echo "SWAT+_FixMatch on $dataset"
    bash scripts/run_dataset_seed_swat+_fixmatch.sh $dataset 1
done