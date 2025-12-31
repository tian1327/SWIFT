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
    echo "DebiasPL on $dataset"
    bash scripts/run_dataset_seed_debiasPL_direct-VLM_text-cls.sh $dataset 1
done