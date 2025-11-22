# Postęp Pracy - TIES-Merging Setup

## Data rozpoczęcia: 2024-11-17

## Status: ✅ Środowisko skonfigurowane, pretrained model pobrany, problemy z zależnościami rozwiązane, skrypty treningowe gotowe, optymalizacje cache i multiprocessing zaimplementowane

---

## Wykonane kroki

### 1. ✅ Utworzenie środowiska wirtualnego
- Środowisko utworzone: `env/` (conda environment)
- Python 3.9 (wymagany dla promptsource==0.2.3)
- Lokalizacja: `/raid/NFS_SHARE/home/marcin.osial/ties-merging/env/`
- Zainstalowane zależności zgodnie z `requirements.txt`

### 2. ✅ Aktualizacja ścieżek w kodzie

#### `src/training.py` (linia 5)
- **Zmieniono**: `HF_HOME` na `/raid/NFS_SHARE/home/marcin.osial/ties-merging/.cache/huggingface/`
- **Cel**: Cache modeli HuggingFace w lokalizacji projektu

#### `src/utils/merge_utils.py` (linie 19-35)
- **Zmieniono**: `BASIC_INFORMATION` dla T5-base:
  - `dump_dir`: `/raid/NFS_SHARE/home/marcin.osial/ties-merging/dump_dir`
  - `load_dir`: `/raid/NFS_SHARE/home/marcin.osial/ties-merging/models`
- **Status**: T5-base skonfigurowany, T5-large i T0_3B pozostają z pustymi ścieżkami (nieużywane)

#### `src/model/checkpoint_filepaths.py` (linia 16)
- **Zmieniono**: `basedir` na `/raid/NFS_SHARE/home/marcin.osial/ties-merging/models`
- **Cel**: Bazowy katalog dla checkpointów modeli

### 3. ✅ Pobranie pretrenowanego modelu T5-base
- **Model**: `t5-base` z HuggingFace Hub
- **Cache HuggingFace**: `/raid/NFS_SHARE/home/marcin.osial/ties-merging/.cache/huggingface/`
- **Checkpoint zapisany**: `/raid/NFS_SHARE/home/marcin.osial/ties-merging/models/pretrained.pt`
- **Rozmiar**: 0.83 GB
- **Parametry**: 222,903,552
- **Skrypt**: `download_pretrained_t5.py` (utworzony do pobrania)

### 4. ✅ Utworzenie struktury katalogów
```
/raid/NFS_SHARE/home/marcin.osial/ties-merging/models/
├── pretrained.pt          # ✅ Zapisany pretrained T5-base
├── paws/                  # Gotowy na best.pt
├── qasc/                  # Gotowy na best.pt
├── quartz/                # Gotowy na best.pt
├── story_cloze/           # Gotowy na best.pt
├── wiki_qa/               # Gotowy na best.pt
├── winogrande/            # Gotowy na best.pt
└── wsc/                   # Gotowy na best.pt
```

---

## Następne kroki

### Do wykonania:
1. ✅ Skrypty treningowe przygotowane - gotowe do uruchomienia
2. ⏳ Trenowanie modeli T5-base na poszczególnych datasetach:
   - `paws`, `qasc`, `quartz`, `story_cloze`, `wiki_qa`, `winogrande`, `wsc`
   - Każdy do 75,000 kroków zgodnie z `train_tips.txt`
   - Skrypty SLURM: `train_paws.sh`, `train_qasc.sh`, `train_quartz.sh`, `train_story_cloze.sh`, `train_wiki_qa.sh`, `train_winogrande.sh`, `train_wsc.sh`
   - Uruchomienie: `sbatch train_{dataset}.sh`

3. ⏳ Po trenowaniu: Merging modeli zgodnie z README.md

---

## Uwagi techniczne

- **Automatyczne pobieranie modeli**: Kod w `src/model/load_model.py` (linie 49-51) używa `AutoModelForSeq2SeqLM.from_pretrained()`, które automatycznie pobiera modele z HuggingFace Hub przy pierwszym użyciu
- **Cache HuggingFace**: Modele będą cache'owane w `HF_HOME` przy każdym użyciu
- **Struktura checkpointów**: Po trenowaniu każdy dataset powinien mieć `best.pt` w odpowiednim podkatalogu `models/{dataset}/`

---

## Pliki pomocnicze

- `download_pretrained_t5.py` - Skrypt do pobrania pretrained modelu
- `script_slurm_example.sh` - Skrypt SLURM z automatyczną aktywacją środowiska i instalacją zależności
- `train_paws.sh`, `train_qasc.sh`, `train_quartz.sh`, `train_story_cloze.sh`, `train_wiki_qa.sh`, `train_winogrande.sh`, `train_wsc.sh` - Skrypty treningowe SLURM dla każdego datasetu

---

## Rozwiązane problemy techniczne

### ✅ Problem z PyArrow i NumPy (2024-11-17)
- **Problem**: `AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'` oraz `ImportError: numpy.core.multiarray failed to import`
- **Przyczyna**: 
  - `datasets==2.8.0` wymaga `pyarrow <15.0.0` (dla `PyExtensionType`)
  - `pyarrow <15.0.0` został skompilowany z NumPy 1.x
  - W środowisku zainstalowano NumPy 2.0.2, co powodowało konflikt
- **Rozwiązanie**: 
  - Obniżenie NumPy do `<2.0` (zainstalowano NumPy 1.x)
  - Zainstalowanie `pyarrow>=6.0.0,<15.0.0` kompatybilnego z `datasets==2.8.0`
  - Wersje zainstalowane w środowisku: `numpy<2.0`, `pyarrow>=6.0.0,<15.0.0`

### ✅ Skrypty treningowe SLURM
- Utworzono 7 skryptów treningowych dla każdego datasetu
- Każdy skrypt automatycznie aktywuje środowisko conda
- Skrypty gotowe do użycia: `train_paws.sh`, `train_qasc.sh`, `train_quartz.sh`, `train_story_cloze.sh`, `train_wiki_qa.sh`, `train_winogrande.sh`, `train_wsc.sh`

### ✅ Problem z datasetem PAWS - rozwiązany (2024-11-17)
- **Problem**: `NonMatchingSplitsSizesError` - oczekiwano 49,401 przykładów train, otrzymano 725,450
- **Przyczyna**: 
  - Konfiguracja `labeled_final` w HuggingFace Hub (`paws`) łączyła 3 różne pliki parquet:
    - `labeled_final/train-00000-of-00001.parquet` (49,401 przykładów)
    - `labeled_swap/train-00000-of-00001.parquet` (dodatkowe przykłady)
    - `unlabeled_final/train-00000-of-00001.parquet` (dodatkowe przykłady)
  - To powodowało łącznie 725,450 przykładów zamiast oczekiwanych 49,401
- **Rozwiązanie**:
  - Zmieniono `dataset_stash` w `PAWSReader` z `("paws", "labeled_final")` na `("google-research-datasets/paws", "labeled_final")`
  - Zmodyfikowano `PAWSReader._read_origin_dataset()` aby używał bezpośrednich URL do plików parquet tylko z `labeled_final`
  - Użyto `load_dataset('parquet', data_files={...})` z bezpośrednimi URL do właściwych plików
- **Wynik**: ✅ Wszystkie rozmiary się zgadzają:
  - Train: 49,401 przykładów (oczekiwane: 49,401)
  - Validation: 8,000 przykładów (oczekiwane: 8,000)
  - Test: 8,000 przykładów (oczekiwane: 8,000)
- **Zmiany w kodzie**: `src/data/dataset_readers.py` (linie 796-821)

### ✅ Optymalizacje cache i multiprocessing (2024-11-17)
- **Problem**: Przetwarzanie templates trwało bardzo długo (~41h dla PAWS z 49,401 przykładami)
- **Rozwiązanie**: Zaimplementowano trzy optymalizacje:

#### 1. Cache na dysk dla przetworzonych templates
- **Lokalizacja**: `.cache/templates/` w katalogu głównym projektu
- **Format**: Pliki pickle z hash weryfikacyjnym dla cache invalidation
- **Efekt**: Przy ponownym uruchomieniu templates są wczytywane z dysku w ~2-3 minuty zamiast przetwarzania od nowa
- **Implementacja**: 
  - `_get_cache_path()` - generuje ścieżkę cache
  - `_compute_dataset_hash()` - hash dla weryfikacji cache
  - `_load_from_cache()` / `_save_to_cache()` - operacje cache
  - Automatyczna weryfikacja hash przed użyciem cache

#### 2. Multiprocessing dla aplikacji templates
- **Workers**: Domyślnie `min(CPU_count - 1, 8)` - ograniczone do 8 workers (zgodnie z `--cpus-per-task=8` w SLURM)
- **Efekt**: Przetwarzanie templates ~8x szybsze (dla 8 CPU: z ~41h do ~5-6h)
- **Implementacja**:
  - `_process_chunk_worker()` - globalna funkcja worker (pickleable)
  - `_applyTemplate_toData()` - używa `multiprocessing.Pool` dla datasetów >1000 przykładów
  - Automatyczny fallback do sequential processing dla małych datasetów
  - Ograniczenie do max 8 workers zapobiega nadmiernemu wykorzystaniu CPU na klastrach
- **Zmiany w kodzie**: `src/data/dataset_readers.py` (linie 20-70, 117-124, 261-331)

#### 3. Multiprocessing w DataLoader z CUDA (poprawione)
- **Zmiana**: 
  - Użycie 'spawn' start method zamiast 'fork' dla multiprocessing z CUDA
  - `num_workers=16` (zgodnie z SLURM --cpus-per-task=32)
  - Przeniesienie `.to(device)` z `collate_fn` do głównego procesu
  - **Optymalizacje DataLoader**:
    - `pin_memory=True` gdy CUDA dostępne (szybszy transfer CPU→GPU)
    - `persistent_workers=True` (workers pozostają żywe między epokami)
    - `prefetch_factor=2` (workers przygotowują batchy z wyprzedzeniem)
- **Efekt**: 
  - Multiprocessing działa nawet z CUDA (dzięki 'spawn' method)
  - Równoległe ładowanie batchy z 16 workers
  - Przenoszenie tensory na GPU w głównym procesie (szybkie, nie blokuje workers)
  - Szybszy transfer CPU→GPU dzięki pin_memory
  - Mniejszy overhead dzięki persistent_workers
- **Implementacja**: 
  - `src/data/Batcher.py` - ustawienie 'spawn' start method i optymalizacje (linie 70-125)
  - `src/data/PytorchDataset.py` - usunięcie `.to(device)` z collate_fn (linie 119-126)
  - `src/training.py` - przeniesienie `.to(device)` do training loop (linie 312-324)
- **Uwaga**: 'spawn' start method tworzy nowe procesy Python (nie fork), co pozwala na CUDA

#### 4. Zwiększenie batch size dla lepszego wykorzystania GPU
- **Zmiana**: 
  - `train_batch_size`: 256 → 512 (lepsze wykorzystanie A100 80GB)
  - `eval_batch_size`: 256 → 512
  - `gradient_accumulation_factor`: 4 → 2 (zachowanie efektywnego batch size ~1024)
- **Efekt**: 
  - Lepsze wykorzystanie GPU (mniej małych operacji)
  - Mniej iteracji dla tej samej liczby przykładów
  - Stabilniejszy gradient (lepsza zbieżność)
- **Implementacja**: `configs/t5_base.json` (linie 3-5)

#### Konfiguracja
- **Cache directory**: Automatycznie tworzony w `.cache/templates/`
- **Num workers dla templates**: Domyślnie `min(CPU_count - 1, 16)` - max 16 workers (zgodnie z SLURM --cpus-per-task=32)
- **Num workers dla DataLoader**: `num_workers=16` (zgodnie z SLURM --cpus-per-task=32)
  - Używa 'spawn' start method gdy CUDA dostępne (umożliwia multiprocessing z CUDA)
  - Używa domyślnego 'fork' gdy CPU only
  - `pin_memory=True` gdy CUDA dostępne (szybszy transfer)
  - `persistent_workers=True` (mniejszy overhead)
  - `prefetch_factor=2` (lepsze wykorzystanie workers)
- **Batch size**: 
  - `train_batch_size=512` (zwiększone z 256 dla lepszego wykorzystania A100)
  - `gradient_accumulation_factor=2` (efektywny batch size ~1024)
- **Możliwość konfiguracji**: Przez `dataset_kwargs` w `get_datasetReader()`:
  ```python
  reader = get_datasetReader("paws", {
      "cache_dir": "/custom/path",
      "num_workers": 8  # Override dla template processing
  })
  ```

#### Szacowane przyspieszenie
- **Pierwsze uruchomienie**: ~16x szybsze (z multiprocessing, max 16 workers)
- **Ponowne uruchomienia**: ~1000x szybsze (z cache na dysk: ~2-3 min vs ~5-6h)
- **Trening**: Oczekiwane 2-3x przyspieszenie dzięki większemu batch size i optymalizacjom DataLoader

### ✅ Naprawa błędów multiprocessing i CUDA (2024-11-17)
- **Problem 1**: `num_workers=255` dla template processing (za dużo na klastrze)
  - **Przyczyna**: `cpu_count() - 1` na węźle klastrowym z wieloma CPU dawało 255 workers
  - **Rozwiązanie**: Ograniczenie do `min(cpu_count() - 1, 16)` - max 16 workers (zgodnie z `--cpus-per-task=32`)
  
- **Problem 2**: `RuntimeError: Cannot re-initialize CUDA in forked subprocess`
  - **Przyczyna**: 
    - DataLoader z `num_workers=4` używał domyślnego 'fork' start method na Linux
    - `collate_fn` próbował przenieść tensory na GPU (`.to(device)`) w worker procesie
    - CUDA nie może być reinicjalizowana w forked procesach
  - **Rozwiązanie**: 
    - Użycie 'spawn' start method zamiast 'fork' dla multiprocessing z CUDA
    - Przeniesienie `.to(device)` z `collate_fn` do głównego procesu (w training loop)
    - Ustawienie `num_workers=16` (zgodnie z SLURM --cpus-per-task=32)
    - Multiprocessing działa teraz nawet z CUDA dzięki 'spawn' method
  - **Zmiany w kodzie**: 
    - `src/data/Batcher.py` (linie 70-93) - ustawienie 'spawn' start method
    - `src/data/PytorchDataset.py` (linie 119-126) - usunięcie `.to(device)` z collate_fn
    - `src/training.py` (linie 312-324) - przeniesienie `.to(device)` do training loop

### ✅ Optymalizacja batch size i DataLoader (2024-11-17)
- **Problem**: Trening nadal wolny (~3.7 it/s) mimo zwiększenia CPU i workers
  - **Analiza**: Zwiększenie CPU/workers nie poprawiło prędkości → bottleneck nie w CPU
  - **Zidentyfikowane problemy**:
    1. Batch size za mały dla A100 (256) - GPU nie w pełni wykorzystane
    2. Brak `pin_memory` - wolniejszy transfer CPU→GPU
    3. Brak `persistent_workers` - overhead przy każdym uruchomieniu
    4. Brak `prefetch_factor` - workers nie przygotowują batchy z wyprzedzeniem
- **Rozwiązanie**:
  1. **Zwiększenie batch size**: `256 → 512` (lepsze wykorzystanie A100 80GB)
  2. **Gradient accumulation**: `4 → 2` (zachowanie efektywnego batch size ~1024)
  3. **pin_memory=True**: Gdy CUDA dostępne - szybszy transfer CPU→GPU (~10-30% przyspieszenia)
  4. **persistent_workers=True**: Gdy num_workers > 0 - workers pozostają żywe między epokami
  5. **prefetch_factor=2**: Workers przygotowują 2 batchy z wyprzedzeniem
- **Zmiany w kodzie**:
  - `configs/t5_base.json` (linie 3-5) - batch size i gradient accumulation
  - `src/data/Batcher.py` (linie 110-112, 122-124) - pin_memory, persistent_workers, prefetch_factor
- **Oczekiwane przyspieszenie**: 2-3x (z ~3.7 it/s do ~7-11 it/s)

---

## 2024-11-17: Cache tokenizacji - eliminacja głównego bottlenecku

### Problem
- **Tokenizacja była głównym bottleneckem** - dzieje się w czasie rzeczywistym w każdym DataLoader workerze
- Każdy przykład był tokenizowany wielokrotnie (w każdym workerze, w każdej epoce)
- Mimo cache templates, tokenizacja spowalniała trening

### Rozwiązanie: Pre-tokenizacja z cache na dysk
- **Pre-tokenizacja całego datasetu** przed utworzeniem DataLoader
- **Cache na dysk** (`.cache/tokenized/`) - tokenizacja raz, użycie wielokrotne
- **Automatyczna detekcja** - jeśli dataset ma `input_ids`, używa cache (bez tokenizacji)
- **Działa dla wszystkich datasetów** - train, validation, test, eval

### Implementacja
1. **Nowy moduł**: `src/data/tokenization_cache.py`
   - `pre_tokenize_dataset()` - tokenizuje cały dataset z cache
   - Hash-based cache validation - automatyczna invalidation przy zmianie danych
   - Cache path: `.cache/tokenized/{dataset}_{split}_template{idx}_{tokenizer}_maxlen{len}.pkl`

2. **Modyfikacja `PytorchDataset`**:
   - Auto-detekcja pre-tokenizowanych danych (sprawdza `input_ids`)
   - Jeśli pre-tokenized → zwraca bezpośrednio (bez tokenizacji)
   - Jeśli nie → legacy mode (tokenizuje on-the-fly)

3. **Modyfikacja `Batcher`**:
   - Nowe parametry: `tokenizer`, `max_seq_len`, `use_tokenization_cache`
   - Pre-tokenizacja przed utworzeniem PytorchDataset w:
     - `get_trainBatches()` - trening
     - `get_splitOfBatches()` - validation/test
     - `get_evalBatches()` - evaluation

4. **Integracja**:
   - `src/training.py` - przekazuje tokenizer i max_seq_len do Batcher
   - `src/eval/evaluate.py` - przekazuje tokenizer do Batcher (max_seq_len=512 default)

### Zmiany w kodzie
- `src/data/tokenization_cache.py` - nowy moduł (270 linii)
- `src/data/PytorchDataset.py` - auto-detekcja pre-tokenizacji
- `src/data/Batcher.py` - pre-tokenizacja przed DataLoader
- `src/training.py` - przekazanie tokenizer i max_seq_len
- `src/eval/evaluate.py` - przekazanie tokenizer

### Oczekiwane przyspieszenie
- **Pierwszy run**: Tokenizacja raz (zapis do cache) - może być wolniejsze
- **Kolejne runy**: **5-10x przyspieszenie** - brak tokenizacji w workers
- **Eliminacja bottlenecku**: Tokenizacja nie blokuje już DataLoader workers

### Uwagi
- Cache jest automatycznie invalidowany przy zmianie danych (hash-based)
- Cache działa dla wszystkich datasetów (train, eval, validation, test)
- Tokenizacja jest sekwencyjna (tokenizer nie jest pickleable) - ale dzieje się raz, cache na dysk

---

## 2024-11-17: Naprawa multiprocessing - numpy arrays zamiast tensors

### Problem
- **Błąd mmap przy multiprocessing**: `RuntimeError: unable to mmap 16 bytes` - PyTorch tensors próbowały być share'owane przez mmap między procesami
- Multiprocessing z 'spawn' nie działał z PyTorch tensors w cache

### Rozwiązanie: Numpy arrays w cache
- **Zapis jako numpy arrays**: Pre-tokenizowane dane są zapisywane jako numpy arrays (nie PyTorch tensors)
- **Konwersja w __getitem__**: Numpy arrays są konwertowane z powrotem na PyTorch tensors w `PytorchDataset.__getitem__`
- **Eliminacja mmap**: Numpy arrays są łatwiejsze do pickle'owania, nie wymagają mmap/shared memory

### Implementacja
1. **Modyfikacja `tokenization_cache.py`**:
   - `_tokenize_single_example()` - konwertuje tensors na numpy arrays przed zwróceniem (`.cpu().numpy()`)
   - Wszystkie tensors (input_ids, input_mask, target_ids, target_mask, all_choices) → numpy arrays

2. **Modyfikacja `PytorchDataset.py`**:
   - `__getitem__()` - konwertuje numpy arrays z powrotem na PyTorch tensors (`torch.from_numpy()`)
   - Auto-detekcja sprawdza czy to numpy array lub tensor
   - Obsługa list numpy arrays (all_choices_ids, all_choices_masks)

3. **Zwiększenie workers**:
   - `num_workers` zwiększony z powrotem do 16 (numpy arrays działają z multiprocessing)

4. **Dodanie torch.compile**:
   - Kompilacja modelu z `torch.compile(mode="reduce-overhead")` dla 20-30% przyspieszenia
   - Kompilacja po DDP wrapping (DDP + compile działają razem)

### Zmiany w kodzie
- `src/data/tokenization_cache.py` - konwersja tensors → numpy arrays
- `src/data/PytorchDataset.py` - konwersja numpy arrays → tensors + auto-detekcja
- `src/data/Batcher.py` - zwiększenie num_workers do 16
- `src/training.py` - dodanie torch.compile po DDP wrapping

### Oczekiwane efekty
- **Eliminacja błędów mmap**: Multiprocessing działa bez problemów z pamięcią
- **Więcej workers**: 16 workers zamiast 8 = lepsze wykorzystanie CPU
- **torch.compile**: 20-30% przyspieszenie treningu (po pierwszym warmup)

### Ważne
- **Stary cache trzeba usunąć**: Cache z tensors nie zadziała - trzeba usunąć `.cache/tokenized/*.pkl` przed pierwszym uruchomieniem
- **Pierwszy run**: Tokenizacja + kompilacja modelu może być wolniejsza
- **Kolejne runy**: Pełna prędkość z cache + compile

---

## 2024-11-17: Zwiększenie batch_size do 700 i naprawa błędu ewaluacji

### Zmiany w konfiguracji
- **batch_size**: 512 → 700 (optymalne dla A100)
- **eval_batch_size**: 512 → 700
- **gradient_accumulation_factor**: 2 → 1 (batch 700 jest już duży)
- **lr**: 1e-4 → 8e-5 (sqrt scaling: 1e-4 × √(700/1024) ≈ 8e-5)
- **num_batches**: 75000 (bez zmian - liczba iteracji pozostaje stała)

### Naprawa błędu ewaluacji
- **Problem**: `prediction_dir` był None w configach tworzonych przez `MultiEvaluationConfig`
- **Przyczyna**: `get_allConfigs()` nie przekazywał `prediction_dir` do nowych configów
- **Rozwiązanie**:
  1. W `training.py`: Upewniamy się że `prediction_dir` jest w `base_config_dict` przed tworzeniem `MultiEvaluationConfig`
  2. W `MultiEvaluationConfig.py`: Dodajemy `prediction_dir` do `updated_fields` jeśli istnieje w base config

### Obserwacje z logów (slurm-71039.out)
- ✅ torch.compile działa (kompilacja modelu)
- ✅ Cache tokenizacji działa (wczytanie z cache)
- ✅ 16 workers działa (multiprocessing bez błędów)
- ⚠️ Wydajność: ~3.5 it/s (stabilny trening) - wolniej niż oczekiwane ~7-10 it/s
- ⚠️ Możliwe przyczyny wolności:
  - Batch size 700 może być za duży dla T5-base
  - torch.compile może nie działać optymalnie (graph breaks)
  - Możliwy bottleneck w forward/backward pass

### Zmiany w kodzie
- `configs/t5_base.json` - batch_size, lr, gradient_accumulation_factor
- `src/training.py` - naprawa prediction_dir w evaluate_checkpoint
- `src/eval/MultiEvaluationConfig.py` - zachowanie prediction_dir w get_allConfigs()

### Naprawa graph breaks w torch.compile
- **Problem**: Graph breaks z `Tensor.item()` powodowały mniejszą efektywność kompilacji
- **Rozwiązanie**: Ustawiono `torch._dynamo.config.capture_scalar_outputs = True` przed kompilacją
- **Efekt**: Operacje jak `loss.detach().cpu().item()` nie będą powodować graph breaks
- **Zmiany**: `src/training.py` - dodano konfigurację przed `torch.compile()`

### Batch size 700 - optymalizacja i wyniki
- **Status**: Batch size 700 działa dobrze na A100
- **Czas treningu**: ~9 godzin dla 75,000 kroków (2.32-2.33 it/s)
- **Konfiguracja**: `train_batch_size=700`, `eval_batch_size=700`, `gradient_accumulation_factor=1`, `lr=8e-5`
- **Wnioski**: Obecna konfiguracja jest optymalna dla dostępnych zasobów - czas treningu jest akceptowalny

### Naprawa błędu prediction_dir=None w ewaluacji (ostateczna)
- **Problem**: `TypeError: join() argument must be str, bytes, or os.PathLike object, not 'NoneType'` w `getAndMake_specificPredictionDir` podczas ewaluacji co 200 kroków
- **Przyczyna**: 
  1. `evaluation_config.prediction_dir` mogło być None przed wywołaniem `evaluate_checkpoint`
  2. `MultiEvaluationConfig.get_allConfigs()` nie zawsze zachowywało `prediction_dir` w `updated_fields` jeśli `base_dict["prediction_dir"]` było None
- **Rozwiązanie**:
  1. W `src/training.py` (linie 148-188):
     - Ustawienie `evaluation_config.prediction_dir` jeśli None (przed utworzeniem base_config_dict)
     - Wielopoziomowa walidacja: `evaluation_config.prediction_dir`, `base_config_dict["prediction_dir"]`, `fields_toUpdate["prediction_dir"]`
     - Finalna walidacja przed i po utworzeniu `MultiEvaluationConfig`
  2. W `src/eval/MultiEvaluationConfig.py` (linie 52-70):
     - Priorytetyzacja: najpierw `self.prediction_dir` (ustawione przez `fields_toUpdate` w `__init__`), potem `base_dict["prediction_dir"]`
     - Jeśli nadal None, podniesienie `ValueError` z szczegółowym komunikatem (pomaga w debugowaniu)
- **Zmiany**: 
  - `src/training.py` - wielopoziomowa walidacja i ustawianie `prediction_dir` w trzech miejscach
  - `src/eval/MultiEvaluationConfig.py` - priorytetyzacja i walidacja `prediction_dir` w `get_allConfigs()`
- **Efekt**: Błąd `prediction_dir=None` nie powinien już występować - jeśli wystąpi, zostanie złapany z jasnym komunikatem błędu

### Rekomendacje
- Monitorować logi czy błąd `prediction_dir=None` zniknął
- Sprawdzić wykorzystanie GPU (nvidia-smi podczas treningu)
- Monitorować czy graph breaks zniknęły w logach

---

*Ostatnia aktualizacja: 2024-11-17*

