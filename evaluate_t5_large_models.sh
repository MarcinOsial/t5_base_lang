#!/bin/bash
#SBATCH --job-name=t5_large_eval
#SBATCH --gpus=ampere:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32

# Skrypt do ewaluacji modeli T5-Large:
# 1. Ewaluuje pretrenowany T5-Large (base) na wszystkich datasetach
# 2. Ewaluuje każdy checkpoint dla "test" setu każdego z nich
# Zestaw test taki sam jak w modelu t5-base

set -e

# Print GPU information for monitoring
nvidia-smi -L

# Display current working directory
echo "Working directory: $(pwd)"

# Change to project directory
PROJECT_DIR="/raid/NFS_SHARE/home/marcin.osial/ties-merging"
cd "$PROJECT_DIR"

# Initialize conda and activate environment
source /raid/NFS_SHARE/home/marcin.osial/miniconda3/etc/profile.d/conda.sh
conda activate "$PROJECT_DIR/env"

export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR"

# Konfiguracja
CONFIG_FILE="configs/t5_large.json"
EVAL_SPLIT="test"  # Używamy "test" split jak w t5-base
PROJECT_NAME="t5-large-evaluation"
BASE_MODELS_DIR="/raid/NFS_SHARE/home/marcin.osial/ties-merging/T5_Large"

# Lista wszystkich datasetów (taka sama jak w t5-base)
DATASETS=("paws" "qasc" "quartz" "story_cloze" "wiki_qa" "winogrande" "wsc")

echo "=========================================="
echo "Ewaluacja modeli T5-Large"
echo "Rozpoczęto: $(date)"
echo "=========================================="
echo ""

# Sprawdź dostępność checkpointów przed rozpoczęciem (tylko lm-adapt, pomijamy ia3)
echo "Sprawdzanie dostępności checkpointów (tylko lm-adapt, pomijamy ia3)..."
total_checkpoints_found=0
for dataset in "${DATASETS[@]}"; do
    count=$(find "$BASE_MODELS_DIR/$dataset" -name "checkpoint_*.pt" -type f 2>/dev/null | grep -v gram_matrix | grep -v "/ia3/" | wc -l)
    total_checkpoints_found=$((total_checkpoints_found + count))
    echo "  $dataset: $count checkpoint(ów) (lm-adapt)"
done
echo "  Łącznie: $total_checkpoints_found checkpoint(ów) lm-adapt"
echo ""

# ============================================
# CZĘŚĆ 1: Ewaluacja checkpointów fine-tuned dla każdego datasetu
# ===========================================
echo "=========================================="
echo "CZĘŚĆ 1: Ewaluacja checkpointów fine-tuned dla każdego datasetu"
echo "=========================================="

# ============================================
# CZĘŚĆ 1: Ewaluacja każdego checkpointa fine-tuned dla "test" setu każdego datasetu
# ============================================
total_checkpoints=0
processed_checkpoints=0

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "=== Dataset: $dataset ==="
    
    # Znajdź wszystkie checkpoints dla tego datasetu (tylko lm-adapt, pomijamy ia3 i gram_matrix)
    checkpoints=$(find "$BASE_MODELS_DIR/$dataset" -name "checkpoint_*.pt" -type f | grep -v gram_matrix | grep -v "/ia3/" | sort)
    
    if [ -z "$checkpoints" ]; then
        echo "      ⚠ Brak checkpointów dla $dataset"
        continue
    fi
    
    checkpoint_count=$(echo "$checkpoints" | wc -l)
    echo "      Znaleziono $checkpoint_count checkpoint(ów)"
    
    # Iteruj przez każdy checkpoint
    checkpoint_num=0
    while IFS= read -r checkpoint_path; do
        checkpoint_num=$((checkpoint_num + 1))
        total_checkpoints=$((total_checkpoints + 1))
        
        # Wyciągnij nazwę checkpointu z ścieżki (np. checkpoint_1299.pt)
        checkpoint_name=$(basename "$checkpoint_path" .pt)
        
        # Wszystkie checkpoints są z lm-adapt (ia3 zostały pominięte)
        # Stwórz unikalną nazwę eksperymentu
        experiment_name="${dataset}-lm-adapt-${checkpoint_name}"
        
        echo ""
        echo "  [$checkpoint_num/$checkpoint_count] Checkpoint: $checkpoint_name (lm-adapt)"
        echo "      Ścieżka: $checkpoint_path"
        echo "      Ewaluacja na dataset: $dataset (split: $EVAL_SPLIT)"
        
        # Usuń cache'owane wyniki dla tego eksperymentu, aby wymusić ponowną ewaluację
        # Każdy checkpoint powinien być ewaluowany od nowa, nie używając cache'owanych wyników
        experiment_dir="exp_out/$PROJECT_NAME/google-t5-large-lm-adapt/$experiment_name"
        if [ -d "$experiment_dir/predictions" ]; then
            echo "      Usuwanie cache'owanych wyników z: $experiment_dir/predictions"
            rm -rf "$experiment_dir/predictions"
        fi
        
        # Uruchom ewaluację z checkpointem
        python src/inference.py \
            -c "$CONFIG_FILE" \
            -i "$dataset" \
            --kwargs \
            split="$EVAL_SPLIT" \
            project_name="$PROJECT_NAME" \
            experiment_name="$experiment_name" \
            checkpoint_to_directly_load_model="$checkpoint_path"
        
        if [ $? -eq 0 ]; then
            processed_checkpoints=$((processed_checkpoints + 1))
            echo "      ✓ Ewaluacja zakończona pomyślnie"
        else
            echo "      ✗ Błąd podczas ewaluacji"
        fi
        
        echo "      ----------------------------------------"
    done <<< "$checkpoints"
done

# CZĘŚĆ 2: Pomijamy ewaluację pretrenowanego modelu (już wykonana wcześniej)
# echo ""
# echo "=========================================="
# echo "CZĘŚĆ 2: Ewaluacja pretrenowanego T5-Large (base) na wszystkich datasetach"
# echo "=========================================="
# echo "Pominięto - ewaluacja pretrenowanego modelu już wykonana wcześniej"

echo ""
echo "=========================================="
echo "Podsumowanie:"
echo "  - Przetworzono checkpointów fine-tuned: $processed_checkpoints/$total_checkpoints"
echo "  - Wyniki zapisane w: exp_out/$PROJECT_NAME/"
echo "Zakończono: $(date)"
echo "=========================================="
echo "Ewaluacja zakończona."

