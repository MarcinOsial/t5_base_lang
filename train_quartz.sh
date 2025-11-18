#!/bin/bash
#SBATCH --job-name=t5_base_quartz
#SBATCH --gpus=ampere:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32

# Script to train T5-base on QuaRTz dataset
# Trains for 75,000 steps as per train_tips.txt
# Saves checkpoint in: models/quartz/best.pt

set -e

# Print GPU information for monitoring
nvidia-smi -L

# Display current working directory and add project to PYTHONPATH if needed
echo "Working directory: $(pwd)"

# Change to project directory to ensure correct paths
PROJECT_DIR="/raid/NFS_SHARE/home/marcin.osial/ties-merging"
cd "$PROJECT_DIR"

# Initialize conda and activate environment
source /raid/NFS_SHARE/home/marcin.osial/miniconda3/etc/profile.d/conda.sh
conda activate "$PROJECT_DIR/env"

export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR"

# Train T5-base on QuaRTz dataset
# inference_dataset_mixture=None ensures evaluation only on QuaRTz (same as train_dataset)
python src/training.py \
    -c configs/t5_base.json \
    -k \
    train_dataset=quartz \
    train_dataset_mixture=None \
    inference_dataset_mixture=None \
    project_name=t5_finetuning \
    experiment_name=t5-base-quartz \
    num_batches=75000

echo "Training completed for QuaRTz dataset"

