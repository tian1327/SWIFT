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
    echo "FixMatch on $dataset"
    bash scripts/run_dataset_seed_fixmatch_text-cls_w_TT.sh $dataset 1
done