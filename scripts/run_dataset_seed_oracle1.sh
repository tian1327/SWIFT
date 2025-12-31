#!/bin/bash

# Define arrays of values for each parameter

# prefix="batch_Oracle1"
prefix="batch_Oracle1_dinov2"

# methods=("mixup" "saliencymix" "CMO" "cutmix-fs" "resizemix" "CMLP" "probing" "finetune" "FLYP" "cutmix" "fixmatch")
methods=("finetune")

# data_sources=("fewshot" "retrieved" "fewshot+retrieved" "fewshot+unlabeled" "fewshot+retrieved+unlabeled")
# data_sources=("fewshot+retrieved+unlabeled")
data_sources=("fewshot+unlabeled")


shot_values=(4 8 16)
# shot_values=(16)

retrieval_splits=("T2T500")

unlabeled_splits=("u_train_in.txt")

batch_size=32

loss="CE"

# epochs=1
epochs=50


model_cfgs=(
    # "resnet50_scratch" \
    # "resnet50_imagenet_pretrained" \
    # "resnet50_inat_pretrained" \
    # "vitb32_openclip_laion400m" \
    # "vitb16_openclip_laion400m" \
    # "dinov2_vits14_reg" \
    "dinov2_vitb14_reg" \
    # "dinov2_vitl14_reg" \
    # "dinov2_vitg14_reg" \
    # "vitb32_clip_inat" \
    # "vitb32_clip_nabirds" \
    # "vitb32_clip_cub" \
    )

# log_mode="file"
log_mode="both"


#------------------------------
# DO NOT MODIFY BELOW THIS LINE !!!
#------------------------------

for model_cfg in "${model_cfgs[@]}"; do

    first_item=$(echo $model_cfg | cut -d'_' -f1)
    second_item=$(echo $model_cfg | cut -d'_' -f2)

    # resnet50_imagenet_pretrained
    if [ "$first_item" = "resnet50" ] && [ "$second_item" = "imagenet" ]; then
        lr_classifier=1e-3
        lr_backbone=1e-3
        wd=1e-4
        cls_inits=("random")
        optim="SGD"
        mu=5
        threshold=0.8
        lambda_u=1.0
        T=1.0
        temp_scheme='none'
        temperature=1.0

    elif [ "$first_item" = "vitb32" ] || [ "$first_item" = "vitb16" ]; then
        lr_classifier=1e-4
        lr_backbone=1e-6
        wd=1e-2
        cls_inits=("REAL-Prompt")
        optim="AdamW"
        T=0.01
        temp_scheme='fewshot+retrieved+unlabeled'
        temperature=0.07

    elif [ "$first_item" = "dinov2" ]; then
        lr_classifier=1e-4
        lr_backbone=1e-6
        wd=1e-2
        cls_inits=("random")
        optim="AdamW"
        T=1.0
        temp_scheme='none'
        temperature=1.0        

    else
        echo "Model not found"
        exit 1
    fi


    # update folder by adding the model_cfg
    folder="${prefix}_${model_cfg}_${epochs}epochs"
    echo "Folder: $folder"

    # Split the model_cfg by underscore and get the second item
    second_item=$(echo $model_cfg | cut -d'_' -f2)

    if [ "$second_item" = "openclip" ] || [ "$second_item" = "clip" ]; then
        script="main.py"
    else
        script="main_ssl.py"
    fi


    # Check if command-line arguments were provided
    if [ "$#" -ge 2 ]; then
        datasets=("$1")  # Use the provided command-line argument for the dataset
        seeds=("$2")
    else
        echo "Usage: $0 <dataset> [seed]"
    fi

    # Check if the results folder exists, if not create it
    if [ ! -d "results/$folder" ]; then
        mkdir -p "results/$folder"
    fi

    output_folder="output/$folder"
    if [ ! -d "$output_folder" ]; then
        mkdir -p "$output_folder"
    fi


    # Dynamically set the filename based on the dataset
    output_file="results/${folder}/${datasets[0]}.csv"

    # Create or clear the output file
    echo "Dataset,Method,Loss,Model,DataSource,Init,Shots,Seed,Retrieve,Stage1Acc,Stage2Acc" > "$output_file"

    # Loop through all combinations and run the script
    for dataset in "${datasets[@]}"; do
        for method in "${methods[@]}"; do
            for data_source in "${data_sources[@]}"; do
                for shots in "${shot_values[@]}"; do
                    for init in "${cls_inits[@]}"; do
                        for seed in "${seeds[@]}"; do
                            for retrieval_split in "${retrieval_splits[@]}"; do
                                for unlabeled_split in "${unlabeled_splits[@]}"; do
                                    echo "Running: $script $dataset $method $loss $model_cfg $data_source $init $shots $seed $retrieval_split $unlabeled_in_split"

                                    # set the cls_path based on linear probing learned weights
                                    #     SSL/output/LinearProbing_vitb32_openclip_laion400m_50epochs/output_semi-aves/semi-aves_probing_fewshot_REAL-Prompt_4shots_seed1/stage1_model_best.pth
                                    #cls_path="output/LinearProbing_${model_cfg}_50epochs/output_${dataset}/${dataset}_probing_fewshot_${init}_${shots}shots_seed${seed}/stage1_model_best.pth"

                                    # DINOv2 probing learned weights
                                                #output/LinearProbing_DINOv2_random-init_dinov2_vitb14_reg_50epochs/output_semi-aves/semi-aves_probing_fewshot_random_4shots_seed1/stage1_model_best.pth
                                    cls_path="output/LinearProbing_DINOv2_random-init_${model_cfg}_50epochs/output_${dataset}/${dataset}_probing_fewshot_${init}_${shots}shots_seed${seed}/stage1_model_best.pth"

                                    # Run the script and capture the output
                                    output=$(python -W ignore "$script" --dataset "$dataset" --method "$method" --data_source "$data_source"  --cls_init "$init" \
                                            --shots "$shots" --seed "$seed" --epochs "$epochs" --bsz "$batch_size" --lr_classifier "$lr_classifier"  \
                                             --lr_backbone "$lr_backbone" --wd "$wd" --optim "$optim" --loss_name "$loss"\
                                             --T "$T" --temp_scheme "$temp_scheme" --temperature "$temperature" \
                                            --log_mode "$log_mode" --retrieval_split "${retrieval_split}.txt" --unlabeled_split "$unlabeled_split" \
                                            --model_cfg "$model_cfg" --folder "$output_folder" \
                                            --cls_path "$cls_path" \
                                            --check_zeroshot \
                                            )

                                    # Print the output to the console
                                    echo "$output"

                                    # Append the results to the CSV file
                                    echo "$output" >> "$output_file"

                                    echo ""
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done