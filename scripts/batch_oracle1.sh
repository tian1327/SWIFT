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
    echo "Oracle1 on $dataset"
    bash scripts/run_dataset_seed_oracle1.sh $dataset 1
done