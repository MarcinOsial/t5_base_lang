#!/bin/bash
#SBATCH --job-name=mid_merging
#SBATCH --gpus=ampere:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Print GPU information for monitoring
nvidia-smi -L

# Display current working directory and add project to PYTHONPATH if needed
echo "Working directory: $(pwd)"
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Activate conda environment (adjust the environment name as needed)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate atlas  # Replace with your actual environment name

# Set model and run
MODEL=ViT-B-16
# MODEL=ViT-L-14
PORT=29566 #15  # Using a specific port to avoid conflicts

# MODEL=ViT-L-14
# PORT=29565 #15  # Using a specific port to avoid conflicts


# Display GPU memory information
echo "GPU Memory Information:"
nvidia-smi

# Display CPU memory information available to the job
echo -e "\nCPU Memory Information:"
echo "Total Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "Used Memory: $(free -h | grep Mem | awk '{print $3}')" 
echo "Free Memory: $(free -h | grep Mem | awk '{print $4}')"
echo "Available Memory: $(free -h | grep Mem | awk '{print $7}')"


export WANDB_DIR=/shared/results/gmosial/wandb
set -e
# RESUME_IDX=$1
# END_IDX=$2 "ImageNet" # "CIFAR10" "SVHN"

pool=(
"Cars" "DTD" "EuroSAT" "GTSRB" "MNIST" "RESISC45" "SUN397" "SVHN"
"CIFAR10" "CIFAR100" "STL10" "Food101" "Caltech101" "Caltech256"
"FGVCAircraft" "Flowers102" "OxfordIIITPet" "CUB200" "PascalVOC" "Country211" "UCF101"
)

# DATASET_NAME=$1

# DATASET_NAME=$1

# min is 1, max is 20
# RESUME_IDXs=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
# # # always 1 more than the resume idx
# END_IDXs=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21)

# min is 1, max is 20
RESUME_IDXs=(20)
# always 1 more than the resume idx
END_IDXs=(21)


# RESUME_IDXs=(20)
# END_IDXs=(21)

# RESUME_IDXs=$1
# END_IDXs=$2

# 1 0.8 0.6 0.4 0.2 

N_THRESHOLDS=(0.6)
SEEDs=(1)
#_random_vectors.py

for SEED in "${SEEDs[@]}"; do
    for SVD_THRESHOLD in "${N_THRESHOLDS[@]}"; do
        for DATASET_NAME in "${pool[@]}"; do
            for i in "${!RESUME_IDXs[@]}"; do
                RESUME_IDX=${RESUME_IDXs[$i]}
                END_IDX=${END_IDXs[$i]}
                echo "Running learn coef finetuning target task for model: $MODEL"
                python src/xtrue_top_global_batch.py \
                    --svd-threshold=$SVD_THRESHOLD \
                    --model=$MODEL \
                    --target-dataset-name=$DATASET_NAME \
                    --blockwise-coef \
                    --isoc \
                    --resume-from-idx=$RESUME_IDX \
                    --end-index=$END_IDX \
                    --seed=$SEED \
                    --num-workers=6
                    # --lp-reg=0.001
                    # --subsample=0.05

                    # --svd-threshold-first=0.1 \
                    # --svd-threshold-second=0.6
            done   
        done
    done
done

# xtrue_stock.py prawdopodobnie 69628
# xtrue_dare.py
# xtrue_down_global.py 69631 scanceled
# xtrue_random_global.py 69630
# xtrue_toprawavg_global.py 69629
# xtrue_top_global_routed.py 69632
# xtrue_ties_merging.py 69628 - 2 i 3 seed




    # --keep-top-values
    # --batch-size=128 \
    # --memory-efficient
#--keep-top-values
# --blockwise-coef
#--keep-top-values --partition=80
 

echo "Job completed at $(date)" 

# pool = [
#     , "SUN397", "MNIST", "CIFAR100",
#      "EuroSAT", "GTSRB", "RESISC45", "SVHN", 
#     "CIFAR10", "ImageNet", "STL10", "Food101", "Caltech101", "Caltech256",
#     "FGVCAircraft", "Flowers102", "OxfordIIITPet", "CUB200", "PascalVOC", "Country211", "UCF101",
# 
