#!/bin/bash
#SBATCH --job-name=download_t5_large
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

# Script to download T5-Large models from Google Cloud Storage
# Downloads models from gs://merging_by_matching_models_in_task_subspaces/exp_out/p3/*
# to local directory: T5_Large/

set -e

# Print system information for monitoring
echo "Starting download job at $(date)"
echo "Working directory: $(pwd)"

# Change to project directory
PROJECT_DIR="/raid/NFS_SHARE/home/marcin.osial/ties-merging"
cd "$PROJECT_DIR"

# Create target directory if it doesn't exist
TARGET_DIR="$PROJECT_DIR/T5_Large"
mkdir -p "$TARGET_DIR"
echo "Target directory: $TARGET_DIR"

# Initialize conda
source /raid/NFS_SHARE/home/marcin.osial/miniconda3/etc/profile.d/conda.sh

# Check if gsutil is available in PATH
if command -v gsutil &> /dev/null; then
    echo "gsutil found in PATH, using system installation"
    GSUTIL_CMD="gsutil"
else
    echo "gsutil not found in PATH, creating temporary conda environment..."
    # Create temporary conda environment for gsutil (won't affect main env)
    TEMP_ENV_NAME="temp_gsutil_$$"
    conda create -n "$TEMP_ENV_NAME" -y python=3.9
    conda activate "$TEMP_ENV_NAME"
    
    # Install Google Cloud SDK using conda-forge (includes gsutil)
    echo "Installing Google Cloud SDK in temporary environment..."
    conda install -c conda-forge google-cloud-sdk -y || {
        # Fallback: download and install Google Cloud SDK manually to temp location
        echo "Conda package not available, downloading Google Cloud SDK..."
        TEMP_SDK_DIR="$PROJECT_DIR/temp_gcloud_sdk_$$"
        mkdir -p "$TEMP_SDK_DIR"
        cd "$TEMP_SDK_DIR"
        
        # Download and install Google Cloud SDK
        curl -sSL https://sdk.cloud.google.com | bash -s -- --disable-prompts --install-dir="$TEMP_SDK_DIR" || {
            echo "ERROR: Could not install Google Cloud SDK. Please install gsutil manually or ensure it's in PATH."
            rm -rf "$TEMP_SDK_DIR"
            exit 1
        }
        
        # Add gcloud SDK to PATH
        export PATH="$TEMP_SDK_DIR/google-cloud-sdk/bin:$PATH"
        cd "$PROJECT_DIR"
    }
    
    # Verify gsutil is now available
    if command -v gsutil &> /dev/null; then
        GSUTIL_CMD="gsutil"
        echo "gsutil installed successfully in temporary environment"
    else
        echo "ERROR: gsutil installation failed or not found after installation"
        exit 1
    fi
fi

# Verify gsutil is working
echo "Verifying gsutil installation..."
$GSUTIL_CMD version || {
    echo "ERROR: gsutil verification failed"
    exit 1
}

# Download models from GCS
# Using -m flag for parallel transfers and -r for recursive
echo "Starting download from Google Cloud Storage..."
echo "Source buckets:"
echo "  - gs://merging_by_matching_models_in_task_subspaces/exp_out/p3/paws"
echo "  - gs://merging_by_matching_models_in_task_subspaces/exp_out/p3/qasc"
echo "  - gs://merging_by_matching_models_in_task_subspaces/exp_out/p3/quartz"
echo "  - gs://merging_by_matching_models_in_task_subspaces/exp_out/p3/story_cloze"
echo "  - gs://merging_by_matching_models_in_task_subspaces/exp_out/p3/wiki_qa"
echo "  - gs://merging_by_matching_models_in_task_subspaces/exp_out/p3/winogrande"
echo "  - gs://merging_by_matching_models_in_task_subspaces/exp_out/p3/wsc"
echo "Target: $TARGET_DIR"

cd "$TARGET_DIR"

$GSUTIL_CMD -m cp -r \
  "gs://merging_by_matching_models_in_task_subspaces/exp_out/p3/paws" \
  "gs://merging_by_matching_models_in_task_subspaces/exp_out/p3/qasc" \
  "gs://merging_by_matching_models_in_task_subspaces/exp_out/p3/quartz" \
  "gs://merging_by_matching_models_in_task_subspaces/exp_out/p3/story_cloze" \
  "gs://merging_by_matching_models_in_task_subspaces/exp_out/p3/wiki_qa" \
  "gs://merging_by_matching_models_in_task_subspaces/exp_out/p3/winogrande" \
  "gs://merging_by_matching_models_in_task_subspaces/exp_out/p3/wsc" \
  .

# Clean up temporary conda environment if created
if [ -n "$TEMP_ENV_NAME" ] && [ "$(conda env list | grep -c "$TEMP_ENV_NAME")" -gt 0 ]; then
    echo "Cleaning up temporary conda environment..."
    conda deactivate
    conda env remove -n "$TEMP_ENV_NAME" -y
fi

# Clean up temporary SDK directory if created
if [ -n "$TEMP_SDK_DIR" ] && [ -d "$TEMP_SDK_DIR" ]; then
    echo "Cleaning up temporary SDK directory..."
    rm -rf "$TEMP_SDK_DIR"
fi

echo "Download completed successfully at $(date)"
echo "Models saved to: $TARGET_DIR"

