#!/bin/bash
#SBATCH --job-name=t5_base_paws
#SBATCH --gpus=ampere:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32

# Script to train T5-base on PAWS dataset
# Trains for 75,000 steps as per train_tips.txt
# Saves checkpoint in: models/paws/best.pt

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

# Fix compatibility issues: datasets==2.8.0 requires pyarrow <15.0.0 (for PyExtensionType)
# and pyarrow <15.0.0 requires numpy <2.0 (compiled with numpy 1.x)
# NOTE: Already installed in environment, commented out to avoid reinstalling
# echo "Installing compatible numpy and pyarrow versions..."
# pip install --quiet "numpy<2.0" "pyarrow>=6.0.0,<15.0.0"

# Fix compatibility issues with datasets==2.8.0
# datasets==2.8.0 requires compatible versions of huggingface_hub and fsspec
# Current fsspec 2025.10.0 and even 2024.2.0 have stricter glob pattern validation causing ValueError: Invalid pattern '**' can only be an entire path component
# Installing compatible versions while keeping datasets==2.8.0 as per requirements.txt
# NOTE: Already installed in environment, commented out to avoid reinstalling
# echo "Installing compatible huggingface_hub and fsspec versions for datasets==2.8.0..."
# pip install --quiet --force-reinstall "huggingface_hub>=0.14.0,<0.20.0" "fsspec==2022.11.0"

# pip install wandb

export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR"

# Train T5-base on PAWS dataset
# inference_dataset_mixture=None ensures evaluation only on PAWS (same as train_dataset)
python src/training.py \
    -c configs/t5_base.json \
    -k \
    train_dataset=paws \
    train_dataset_mixture=None \
    inference_dataset_mixture=None \
    project_name=t5_finetuning \
    experiment_name=t5-base-paws \
    num_batches=75000

echo "Training completed for PAWS dataset"

