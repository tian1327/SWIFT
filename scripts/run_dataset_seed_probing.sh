#!/bin/bash

# prefix="LinearProbing_random-init"
# prefix="LinearProbing_text-init"
# prefix="LinearProbing_text-init-scaled0.01"
# prefix="LinearProbing_fixedtemp0.01"
# prefix="LinearProbing_text-init_fixed-temp_0.01"
# prefix="LinearProbing_text-init_learn-temp_0.01"

#prefix="LinearProbing_DINOv2_random-init"
# prefix="LinearProbing_DINOv2_random-init-fixed-temp-0.01"
# prefix="LinearProbing_DINOv2_random-init-fixed-temp-0.07"
# prefix="LinearProbing_DINOv2_random-init-learn-temp-0.07"
prefix="ablate_Tloss_LinearProbing_text-init"
# prefix="ablate_temp_LinearProbing"

# methods=("mixup" "saliencymix" "CMO" "cutmix-fs" "resizemix" "CMLP" "probing" "finetune" "FLYP" "cutmix" "fixmatch")
methods=("probing")

# data_sources=("fewshot" "retrieved" "fewshot+retrieved" "fewshot+unlabeled" "fewshot+retrieved+unlabeled")
data_sources=("fewshot")

# shot_values=(4 8 16)
shot_values=(16)
# shot_values=(4)


retrieval_splits=("T2T500")

# unlabeled_splits=("u_train_in_oracle.txt" "u_train_in.txt")
unlabeled_splits=("u_train_in.txt")

batch_size=32

loss="CE"

epochs=50

model_cfgs=(
    # "vitb32_imagenet_pretrained" \
    # "resnet50_scratch" \
    # "resnet50_imagenet_pretrained" \
    # "resnet50_inat_pretrained" \
    "vitb32_openclip_laion400m" \
    # "vitb16_openclip_laion400m" \
    #  "resnet50_clip" \
    # "dinov2_vits14_reg" \
    # "dinov2_vitb14_reg" \
    # "dinov2_vitl14_reg" \
    # "dinov2_vitg14_reg" \
    # "vitb32_clip_inat" \
    # "vitb32_clip_nabirds" \
    # "vitb32_clip_cub" \
    )

log_mode="both"

# temp_scheme_list=('fewshot+retrieved+unlabeled' 'none')
temp_scheme_list=('fewshot+retrieved+unlabeled')
# temp_scheme_list=('none')

temperature_list=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 2.0 3.0)
# temperature_list=(0.01 0.07 0.1 1.0)
# temperature_list=(0.1 1.0)
# temperature_list=(0.07)
# temperature_list=(0.01)
# temperature_list=(1.0)


#------------------------------
# DO NOT MODIFY BELOW THIS LINE !!!
#------------------------------

for model_cfg in "${model_cfgs[@]}"; do

    # update learning rate based on the first item of the model_cfg
    first_item=$(echo $model_cfg | cut -d'_' -f1)
    second_item=$(echo $model_cfg | cut -d'_' -f2)
    echo "Model Config: $model_cfg"
    echo "First Item: $first_item"
    echo "Second Item: $second_item"

    if [[ "$first_item" = "resnet50" && ( "$second_item" = "imagenet" || "$second_item" = "inat" ) ]]; then
        lr_classifier=1e-3
        wd=1e-2
        cls_inits=("random")
        optim="AdamW"

    elif [ "$first_item" = "vitb32" ] && [ "$second_item" = "imagenet" ]; then
        lr_classifier=1e-3
        wd=1e-2
        cls_inits=("random")
        optim="AdamW"

    elif [ "$first_item" = "resnet50" ] && [ "$second_item" = "clip" ]; then
        lr_classifier=1e-3
        wd=1e-2
        cls_inits=("REAL-Prompt")
        optim="AdamW"

    elif [[ "$first_item" = "vitb32" || "$first_item" = "vitb16" ]] && [ "$second_item" = "openclip" ]; then
        lr_classifier=1e-4
        wd=1e-2
        cls_inits=("REAL-Prompt")
        # cls_inits=("random")
        optim="AdamW"

    elif [ "$first_item" = "dinov2" ]; then
        lr_classifier=1e-4
        wd=1e-2
        cls_inits=("random")
        optim="AdamW"

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
                                    for temp_scheme in "${temp_scheme_list[@]}"; do
                                        for temperature in "${temperature_list[@]}"; do

                                            echo "Running: $script $dataset $method $model_cfg $data_source $init $shots $seed $temp_scheme $temperature"

                                            # Run the script and capture the output
                                            output=$(python -W ignore "$script" --dataset "$dataset" --method "$method" --data_source "$data_source"  --cls_init "$init" \
                                                    --shots "$shots" --seed "$seed" --epochs "$epochs" --bsz "$batch_size" --lr_classifier "$lr_classifier"  \
                                                    --wd "$wd" --optim "$optim" --loss_name "$loss"\
                                                    --log_mode "$log_mode" --retrieval_split "${retrieval_split}.txt" --unlabeled_split "$unlabeled_split" \
                                                    --model_cfg "$model_cfg" --folder "$output_folder" \
                                                    --temp_scheme "$temp_scheme" --temperature "$temperature" \
                                                    --skip_stage2 \
                                                    --check_zeroshot \
                                                    # --recal_fea \
                                                    # --scale_text_embedding \
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
    done
done