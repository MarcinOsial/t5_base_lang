#!/bin/bash
#SBATCH --job-name=t5_axis
#SBATCH --gpus=ampere:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

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

# Set model
MODEL=t5-base

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

# ============================================================================
# WORKFLOW CONFIGURATION
# ============================================================================
# Two workflows are supported:
#
# 1. OLD WORKFLOW (backward compatible):
#    - Uses --resume-from-idx and --end-index
#    - Iterates over sequential source datasets (first k tasks)
#    - Example: resume_from_idx=0, end_index=1 → source: [paws], targets: [qasc, ..., wsc]
#
# 2. NEW WORKFLOW (recommended):
#    - Uses --target, --min-sources, --max-sources
#    - Outer loop: iterates over target datasets
#    - Inner loop: generates ALL combinations of source datasets (1 to 6 sources)
#    - Example: --target=qasc --min-sources=1 --max-sources=6
#      → For target=qasc, tests all combinations: [paws], [quartz], ..., [paws,quartz], ..., [all 6]
# ============================================================================

# Choose workflow mode: "old" or "new"
WORKFLOW_MODE="new"  # Change to "old" for backward compatible workflow

if [ "$WORKFLOW_MODE" = "new" ]; then
    # NEW WORKFLOW: Outer loop over targets, inner loop over all source combinations
    echo "Using NEW WORKFLOW: Outer loop over targets, all source combinations"
    
    # Target datasets to iterate over (outer loop)
    # "paws" "qasc" "quartz" "story_cloze" "wiki_qa" "winogrande" "wsc"
    TARGETS=("wsc")

    
    # Source combination range
    MIN_SOURCES=1  # Minimum number of source datasets
    MAX_SOURCES=6  # Maximum number of source datasets
    
    # SVD thresholds to test
    SVD_THRESHOLDS=(1.0) # 0.2 0.4 0.6 0.8 1.0
    SEEDs=(42)
    
    # Experiment status file path (can be customized)
    STATUS_FILE="${PROJECT_DIR}/exp_out/t5_axis/experiment_status_3.csv"
    
    for SEED in "${SEEDs[@]}"; do
        for SVD_THRESHOLD in "${SVD_THRESHOLDS[@]}"; do
            for TARGET in "${TARGETS[@]}"; do
                echo "Running NEW WORKFLOW for T5 model: $MODEL"
                echo "Target dataset: $TARGET"
                echo "Source combinations: $MIN_SOURCES to $MAX_SOURCES sources"
                echo "This will test ALL combinations of source datasets (excluding target)"
                echo "Status file: $STATUS_FILE"
                python -m axis.t5_axis_merging \
                    --target=$TARGET \
                    --min-sources=$MIN_SOURCES \
                    --max-sources=$MAX_SOURCES \
                    --svd-threshold=$SVD_THRESHOLD \
                    --model=$MODEL \
                    --seed=$SEED \
                    --config=axis/configs/t5_axis_training_3.json \
                    --status-file="$STATUS_FILE"
            done
        done
    done
else
    # OLD WORKFLOW: Backward compatible (original implementation)
    echo "Using OLD WORKFLOW: Sequential source datasets"
    
    # T5 source datasets pool
    pool=(
    "paws" "qasc" "quartz" "story_cloze" "wiki_qa" "winogrande" "wsc"
    )
    
    # Configuration arrays
    # resume_from_idx: number of source datasets to start from (0 = start with 1 source dataset)
    # end_index: number of source datasets to end at (exclusive, 1 = only 1 source dataset)
    # Example: resume_from_idx=0, end_index=1 → source: [paws], targets: [qasc, quartz, story_cloze, wiki_qa, winogrande, wsc]
    # Example: resume_from_idx=0, end_index=2 → source: [paws], then [paws, qasc], with all possible targets
    
    RESUME_IDXs=(0 1 2 3 4 5)
    END_IDXs=(1 2 3 4 5 6) # Test with 1 source dataset first, then can increase to 2, 3, etc.
    
    # SVD thresholds to test
    SVD_THRESHOLDS=(0.1 0.2 0.4 0.6 0.8 1.0)
    SEEDs=(42)
    
    # Experiment status file path (can be customized)
    STATUS_FILE="${PROJECT_DIR}/exp_out/t5_axis/experiment_status_2.csv"
    
    for SEED in "${SEEDs[@]}"; do
        for SVD_THRESHOLD in "${SVD_THRESHOLDS[@]}"; do
            for i in "${!RESUME_IDXs[@]}"; do
                RESUME_IDX=${RESUME_IDXs[$i]}
                END_IDX=${END_IDXs[$i]}
                echo "Running OLD WORKFLOW for T5 model: $MODEL"
                echo "Resume from idx: $RESUME_IDX, End index: $END_IDX"
                echo "This will iterate over:"
                echo "  - Source datasets: from ${pool[@]:0:$((RESUME_IDX+1))} to ${pool[@]:0:$END_IDX}"
                echo "  - Target datasets: all remaining datasets (not in source)"
                echo "Status file: $STATUS_FILE"
                python -m axis.t5_axis_merging \
                    --svd-threshold=$SVD_THRESHOLD \
                    --model=$MODEL \
                    --resume-from-idx=$RESUME_IDX \
                    --end-index=$END_IDX \
                    --seed=$SEED \
                    --config=axis/configs/t5_axis_training_2.json \
                    --status-file="$STATUS_FILE"
            done
        done
    done
fi

echo "Job completed at $(date)"

