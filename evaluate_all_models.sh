#!/bin/bash
#SBATCH --job-name=t5_base_eval_all
#SBATCH --gpus=ampere:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32

# Skrypt do ewaluacji wszystkich modeli best_model.pt w katalogu t5-base
# Dla każdego znalezionego modelu uruchamia ewaluację na odpowiednim datasacie

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

# Katalog bazowy z modelami
BASE_DIR="/raid/NFS_SHARE/home/marcin.osial/ties-merging/exp_out/t5_finetuning/t5-base"

# Konfiguracja
CONFIG_FILE="configs/t5_base.json"
# Używamy "test" zamiast "validation", bo "validation" używa tylko pierwszych 32 próbek
# "test" zawiera wszystkie pozostałe próbki z pełnego validation setu (wszystkie próbki poza pierwszymi 32)
EVAL_SPLIT="test"
PROJECT_NAME="t5-base"

# Sprawdzenie czy katalog istnieje
if [ ! -d "$BASE_DIR" ]; then
    echo "Błąd: Katalog $BASE_DIR nie istnieje!"
    exit 1
fi

# Znajdź wszystkie pliki best_model.pt
echo "Szukam modeli best_model.pt w katalogu: $BASE_DIR"
echo "=========================================="

# Licznik przetworzonych modeli
count=0

# Przeszukaj wszystkie podkatalogi w poszukiwaniu best_model.pt
find "$BASE_DIR" -name "best_model.pt" -type f | while read -r checkpoint_path; do
    count=$((count + 1))
    
    # Wyciągnij ścieżkę katalogu zawierającego model
    model_dir=$(dirname "$checkpoint_path")
    
    # Wyciągnij nazwę datasetu z nazwy katalogu (np. t5-base-story_cloze -> story_cloze)
    # Usuń ścieżkę bazową i prefix "t5-base-"
    dataset_name=$(basename "$model_dir" | sed 's/^t5-base-//')
    
    echo ""
    echo "[$count] Znaleziono model: $checkpoint_path"
    echo "      Dataset: $dataset_name"
    echo "      Uruchamiam ewaluację..."
    
    # Usuń cache'owane wyniki dla tego eksperymentu, aby wymusić ponowną ewaluację
    # Każdy checkpoint powinien być ewaluowany od nowa, nie używając cache'owanych wyników
    experiment_dir="exp_out/$PROJECT_NAME/t5-base/$dataset_name"
    if [ -d "$experiment_dir/predictions" ]; then
        echo "      Usuwanie cache'owanych wyników z: $experiment_dir/predictions"
        rm -rf "$experiment_dir/predictions"
    fi
    
    # Uruchom ewaluację
    python src/inference.py \
        -c "$CONFIG_FILE" \
        -i "$dataset_name" \
        --kwargs \
        checkpoint_to_directly_load_model="$checkpoint_path" \
        split="$EVAL_SPLIT" \
        project_name="$PROJECT_NAME" \
        experiment_name="$dataset_name"
    
    # Sprawdź kod wyjścia poprzedniej komendy
    if [ $? -eq 0 ]; then
        echo "      ✓ Ewaluacja zakończona pomyślnie dla $dataset_name"
    else
        echo "      ✗ Błąd podczas ewaluacji dla $dataset_name"
    fi
    
    echo "      ----------------------------------------"
done

echo ""
echo "=========================================="
echo "Ewaluacja wszystkich modeli zakończona."

