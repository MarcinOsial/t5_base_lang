#!/bin/bash
#SBATCH --job-name=t5_base_pretrained_eval
#SBATCH --gpus=ampere:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32

# Skrypt do ewaluacji pretrenowanego T5-base na wszystkich datasetach
# Używa pretrenowanego modelu (bez fine-tuningu) dla porównania z fine-tuned modelami
# Wyniki zapisuje w oddzielnym folderze: exp_out/t5-base-pretrained/

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

# Katalog bazowy z modelami fine-tuned (używamy tylko do wyciągnięcia listy datasetów)
BASE_DIR="/raid/NFS_SHARE/home/marcin.osial/ties-merging/exp_out/t5_finetuning/t5-base"

# Konfiguracja
CONFIG_FILE="configs/t5_base.json"
# Używamy "test" zamiast "validation", bo "validation" używa tylko pierwszych 32 próbek
# "test" zawiera wszystkie pozostałe próbki z pełnego validation setu (wszystkie próbki poza pierwszymi 32)
EVAL_SPLIT="test"
PROJECT_NAME="t5-base-pretrained"  # Oddzielny folder dla wyników pretrenowanego modelu

# Sprawdzenie czy katalog istnieje
if [ ! -d "$BASE_DIR" ]; then
    echo "Błąd: Katalog $BASE_DIR nie istnieje!"
    exit 1
fi

# Znajdź wszystkie pliki best_model.pt, aby wyciągnąć listę datasetów
echo "Szukam datasetów w katalogu: $BASE_DIR"
echo "Będę ewaluować pretrenowany T5-base na tych samych datasetach"
echo "=========================================="

# Licznik przetworzonych datasetów
count=0

#         --multiple_prompts \

# Przeszukaj wszystkie podkatalogi w poszukiwaniu best_model.pt
# Używamy tego tylko do wyciągnięcia listy datasetów
find "$BASE_DIR" -name "best_model.pt" -type f | while read -r checkpoint_path; do
    count=$((count + 1))
    
    # Wyciągnij ścieżkę katalogu zawierającego model
    model_dir=$(dirname "$checkpoint_path")
    
    # Wyciągnij nazwę datasetu z nazwy katalogu (np. t5-base-story_cloze -> story_cloze)
    # Usuń ścieżkę bazową i prefix "t5-base-"
    dataset_name=$(basename "$model_dir" | sed 's/^t5-base-//')
    
    echo ""
    echo "[$count] Dataset: $dataset_name"
    echo "      Uruchamiam ewaluację pretrenowanego T5-base..."
    
    # Uruchom ewaluację BEZ checkpoint_to_directly_load_model (użyje pretrenowanego modelu)
    # Używamy --multiple_prompts aby raportować medianę z wielu template'ów (jak w artykule)
    python src/inference.py \
        -c "$CONFIG_FILE" \
        -i "$dataset_name" \
        --kwargs \
        split="$EVAL_SPLIT" \
        project_name="$PROJECT_NAME" \
        experiment_name="$dataset_name"
    
    # Sprawdź kod wyjścia poprzedniej komendy
    if [ $? -eq 0 ]; then
        echo "      ✓ Ewaluacja zakończona pomyślnie dla $dataset_name (pretrenowany model)"
    else
        echo "      ✗ Błąd podczas ewaluacji dla $dataset_name"
    fi
    
    echo "      ----------------------------------------"
done

echo ""
echo "=========================================="
echo "Ewaluacja pretrenowanego modelu na wszystkich datasetach zakończona."
echo "Wyniki zapisane w: exp_out/$PROJECT_NAME/"

