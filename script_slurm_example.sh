#!/bin/bash
#SBATCH --job-name=t5_base_training
#SBATCH --gpus=ampere:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Script to train T5-base checkpoints on datasets from t5_mixture
# Based on instructions from ties-merging/README.md lines 27-33
# Saves checkpoints in structure: models_lang/fullshot/t5-base/{dataset}/best.pt

set -e

# Print GPU information for monitoring
nvidia-smi -L

# Display current working directory and add project to PYTHONPATH if needed
echo "Working directory: $(pwd)"

# Remove existing virtual environment if it exists (to avoid Python version conflicts)
# Using Python 3.9 because promptsource==0.2.3 requires Python >=3.7,<3.10 (max Python 3.9)
if [ -d "env" ]; then
    echo "Removing existing virtual environment..."
    rm -rf env
fi

# Initialize conda (required before using conda commands in non-interactive shell)
echo "Initializing conda..."
source /raid/NFS_SHARE/home/marcin.osial/miniconda3/etc/profile.d/conda.sh

# Get absolute path to environment directory
ENV_DIR="$(pwd)/env"

# Create conda environment with Python 3.9 (required for promptsource==0.2.3)
# Using -p (path) instead of -n (name) to create environment in project directory
echo "Creating conda environment with Python 3.9..."
conda create -p "$ENV_DIR" python=3.9 -y

# Activate conda environment using full path (required when using -p instead of -n)
echo "Activating conda environment..."
conda activate "$ENV_DIR"

# Verify Python version
echo "Python version: $(python --version)"

# Install dependencies according to README.md setup instructions
# All package versions from requirements.txt are preserved exactly as specified
python -m pip install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html

export PYTHONPATH="$PYTHONPATH:$(pwd)"