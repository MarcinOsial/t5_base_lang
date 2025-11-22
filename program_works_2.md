# Dokumentacja: Trening T5-base na PAWS z ewaluacją tylko na PAWS

## Przegląd

Ten dokument opisuje konfigurację i reprodukcję treningu modelu T5-base na zbiorze danych PAWS z ewaluacją wyłącznie na tym samym zbiorze (PAWS), bez ewaluacji na innych zbiorach z mieszanki `t5_mixture`.

## Kluczowe zmiany w kodzie

### 1. Automatyczne ustawianie inference_dataset na train_dataset

**Plik:** `src/training.py`

**Zmiana:** Funkcja `evaluate_checkpoint` została rozszerzona o logikę, która automatycznie ustawia `inference_dataset` na `train_dataset`, gdy:
- `inference_dataset_mixture` jest `None`
- `inference_dataset` jest `None`

**Kod:**
```python
elif evaluation_config.inference_dataset is None:
    # If inference_dataset_mixture is None and inference_dataset is None,
    # use train_dataset for evaluation (default behavior: evaluate on training dataset)
    train_dataset = None
    if training_config is not None and hasattr(training_config, 'train_dataset') and training_config.train_dataset is not None:
        train_dataset = training_config.train_dataset
    elif hasattr(evaluation_config, 'train_dataset') and evaluation_config.train_dataset is not None:
        train_dataset = evaluation_config.train_dataset
    
    if train_dataset is not None:
        evaluation_config.inference_dataset = train_dataset
        logger.info(f"inference_dataset was None, using train_dataset: {train_dataset}")
```

**Efekt:** Gdy `inference_dataset_mixture=None`, model automatycznie ewaluuje na zbiorze używanym do treningu (np. `paws`), zamiast na wszystkich zbiorach z mieszanki `t5_mixture`.

### 2. Naprawa problemu z story_cloze

**Problem:** Biblioteka HuggingFace `datasets` oczekuje plików o bardzo specyficznych nazwach dla zbioru `story_cloze`:
- `cloze_test_val__spring2016 - cloze_test_ALL_val.csv` (dla validation)
- `cloze_test_test__spring2016 - cloze_test_ALL_test.csv` (dla test)

**Rozwiązanie:** Utworzono linki symboliczne z oczekiwanych nazw do rzeczywistych plików.

**Lokalizacja:** `/raid/NFS_SHARE/home/marcin.osial/ties-merging/datasets/story_cloze/`

**Komendy do wykonania:**
```bash
cd /raid/NFS_SHARE/home/marcin.osial/ties-merging/datasets/story_cloze
ln -sf cloze_testval_spring2016.csv "cloze_test_val__spring2016 - cloze_test_ALL_val.csv"
ln -sf cloze_testtest_spring2016.csv "cloze_test_test__spring2016 - cloze_test_ALL_test.csv"
```

**Weryfikacja:**
```bash
ls -la /raid/NFS_SHARE/home/marcin.osial/ties-merging/datasets/story_cloze/
```

Powinny być widoczne:
- Oryginalne pliki: `cloze_testval_spring2016.csv`, `cloze_testtest_spring2016.csv`
- Linki symboliczne: `cloze_test_val__spring2016 - cloze_test_ALL_val.csv`, `cloze_test_test__spring2016 - cloze_test_ALL_test.csv`

## Konfiguracja treningu

### Plik skryptu: `train_paws.sh`

**Kluczowe parametry:**
- `train_dataset=paws` - zbiór danych do treningu
- `train_dataset_mixture=None` - brak mieszanki zbiorów do treningu
- `inference_dataset_mixture=None` - **WAŻNE:** wyłącza ewaluację na wszystkich zbiorach z mieszanki, powoduje ewaluację tylko na PAWS
- `num_batches=75000` - liczba iteracji treningowych

**Pełna komenda:**
```bash
python src/training.py \
    -c configs/t5_base.json \
    -k \
    train_dataset=paws \
    train_dataset_mixture=None \
    inference_dataset_mixture=None \
    project_name=t5_finetuning \
    experiment_name=t5-base-paws \
    num_batches=75000
```

### Plik konfiguracyjny: `configs/t5_base.json`

**Domyślne wartości (nadpisywane przez `-k` w skrypcie):**
```json
{
    "pretrained_model": "t5-base",
    "train_batch_size": 700,
    "eval_batch_size": 700,
    "lr": 8e-5,
    "checkpoint_frequency": 100,
    "inference_dataset_mixture": "t5_mixture",  // NADPISYWANE przez inference_dataset_mixture=None
    "train_dataset_mixture": "t5_mixture",      // NADPISYWANE przez train_dataset_mixture=None
    "num_batches": 75000
}
```

**Uwaga:** Parametry przekazane przez `-k` w skrypcie mają priorytet nad wartościami w pliku JSON.

## Reprodukcja środowiska

### 1. Środowisko Python

**Wersja Python:** 3.9

**Ścieżka do środowiska:**
```bash
/raid/NFS_SHARE/home/marcin.osial/ties-merging/env
```

**Aktywacja:**
```bash
source /raid/NFS_SHARE/home/marcin.osial/miniconda3/etc/profile.d/conda.sh
conda activate /raid/NFS_SHARE/home/marcin.osial/ties-merging/env
```

### 2. Zależności

**Główne biblioteki:**
- `torch` (PyTorch)
- `datasets==2.8.0` (HuggingFace)
- `transformers` (HuggingFace)
- `wandb` (Weights & Biases)

**Ważne kompatybilności:**
- `datasets==2.8.0` wymaga `pyarrow <15.0.0`
- `pyarrow <15.0.0` wymaga `numpy <2.0`
- `datasets==2.8.0` wymaga `huggingface_hub>=0.14.0,<0.20.0` i `fsspec==2022.11.0`

**Instalacja zależności:**
```bash
pip install -r requirements.txt
pip install wandb
```

### 3. Struktura katalogów

**Katalog projektu:**
```
/raid/NFS_SHARE/home/marcin.osial/ties-merging/
├── src/                    # Kod źródłowy
├── configs/                # Pliki konfiguracyjne
├── datasets/               # Lokalne zbiory danych
│   └── story_cloze/        # Pliki story_cloze z linkami symbolicznymi
├── .cache/                 # Cache (templates, tokenization, huggingface)
├── exp_out/                # Wyniki eksperymentów
└── train_paws.sh          # Skrypt treningowy
```

**Zmienne środowiskowe:**
```bash
export PYTHONPATH="$PYTHONPATH:/raid/NFS_SHARE/home/marcin.osial/ties-merging"
```

### 4. Zasoby sprzętowe

**GPU:** NVIDIA A100-SXM4-80GB (2x)

**SLURM konfiguracja:**
```bash
#SBATCH --gpus=ampere:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
```

## Uruchomienie treningu

### Metoda 1: Bezpośrednie uruchomienie skryptu

```bash
cd /raid/NFS_SHARE/home/marcin.osial/ties-merging
bash train_paws.sh
```

### Metoda 2: Przez SLURM

```bash
sbatch train_paws.sh
```

### Metoda 3: Bezpośrednia komenda Python

```bash
cd /raid/NFS_SHARE/home/marcin.osial/ties-merging
source /raid/NFS_SHARE/home/marcin.osial/miniconda3/etc/profile.d/conda.sh
conda activate /raid/NFS_SHARE/home/marcin.osial/ties-merging/env
export PYTHONPATH="$PYTHONPATH:/raid/NFS_SHARE/home/marcin.osial/ties-merging"

python src/training.py \
    -c configs/t5_base.json \
    -k \
    train_dataset=paws \
    train_dataset_mixture=None \
    inference_dataset_mixture=None \
    project_name=t5_finetuning \
    experiment_name=t5-base-paws \
    num_batches=75000
```

## Weryfikacja działania

### 1. Sprawdzenie logów

**Oczekiwane komunikaty w logach:**
```
INFO:root:Evaluating checkpoint
INFO:root:inference_dataset was None, using train_dataset: paws
INFO:root:	Evaluating model on paws dataset
```

**NIE powinno być:**
- Ewaluacji na innych zbiorach (qasc, quartz, story_cloze, wiki_qa, winogrande, wsc)
- Błędów `FileNotFoundError` dla story_cloze

### 2. Sprawdzenie wyników

**Lokalizacja wyników:**
```
exp_out/t5_finetuning/t5-base/t5-base-paws/predictions/batch_<N>/validation/paws_template_0/
```

**Przykładowy wynik po 99 batchach:**
```json
{
    "batch_idx": 99,
    "loss": 0.694,
    "validation": {
        "accuracy": 0.796,
        "average": 0.796
    },
    "score_to_select_checkpoint": 0.796
}
```

### 3. Sprawdzenie cache

**Cache tokenizacji:**
```
.cache/tokenized/paws_validation_template0_t5-base_vocab32100_maxlen512.pkl
```

**Cache templates:**
```
.cache/templates/paws_validation_template0_eval.pkl
```

## Rozwiązywanie problemów

### Problem 1: Ewaluacja na wszystkich zbiorach zamiast tylko PAWS

**Przyczyna:** `inference_dataset_mixture` nie jest ustawione na `None` w skrypcie.

**Rozwiązanie:** Upewnij się, że w `train_paws.sh` jest:
```bash
inference_dataset_mixture=None
```

### Problem 2: Błąd FileNotFoundError dla story_cloze

**Przyczyna:** Brak linków symbolicznych dla plików story_cloze.

**Rozwiązanie:** Wykonaj komendy z sekcji "Naprawa problemu z story_cloze".

### Problem 3: Błąd "train_dataset is not available"

**Przyczyna:** `training_config` nie jest przekazywane do `evaluate_checkpoint`.

**Rozwiązanie:** Upewnij się, że w `src/training.py` wszystkie wywołania `evaluate_checkpoint` mają parametr `training_config=training_config`.

### Problem 4: Błędy kompatybilności bibliotek

**Przyczyna:** Niezgodne wersje bibliotek.

**Rozwiązanie:** Zainstaluj zgodne wersje:
```bash
pip install "numpy<2.0" "pyarrow>=6.0.0,<15.0.0" "huggingface_hub>=0.14.0,<0.20.0" "fsspec==2022.11.0"
```

## Podsumowanie kluczowych punktów

1. **Ewaluacja tylko na PAWS:** Ustaw `inference_dataset_mixture=None` w skrypcie treningowym
2. **Naprawa story_cloze:** Utwórz linki symboliczne dla plików story_cloze
3. **Automatyczne ustawianie:** Kod automatycznie używa `train_dataset` jako `inference_dataset` gdy oba są `None`
4. **Reprodukcja:** Użyj dokładnie tych samych wersji bibliotek i konfiguracji
5. **Weryfikacja:** Sprawdź logi, czy ewaluacja odbywa się tylko na PAWS

## Oczekiwane wyniki

**Po 99 batchach:**
- Accuracy: ~0.796
- Loss: ~0.694

**Po 199 batchach:**
- Accuracy: ~0.915
- Loss: ~0.174

**Czas treningu:** ~1.32 iteracji/sekundę (na A100)

## Kontakt i wsparcie

W przypadku problemów sprawdź:
1. Logi SLURM: `slurm-<job_id>.out`
2. Logi treningu w `exp_out/`
3. Cache w `.cache/`

