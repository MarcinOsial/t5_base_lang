# Analiza logów slurm-71039.out - Szczegółowe wnioski

## 📊 Podsumowanie wykonawcze

**Status**: Trening działa, ale wolniej niż oczekiwano. Ewaluacja ma błąd (naprawiony).

**Wydajność**: ~3.5 it/s (stabilny trening) vs oczekiwane ~7-10 it/s

**Czas treningu**: ~14 godzin na 150k iteracji (przy obecnej prędkości)

---

## ✅ Co działa dobrze

### 1. Inicjalizacja i cache
- ✅ **torch.compile działa** (linia 13-14): Model skompilowany pomyślnie
- ✅ **Cache tokenizacji działa** (linia 30-32): Wczytano 49,401 przykładów z cache
- ✅ **Cache templates działa** (linia 25-27): Wczytano templates z cache
- ✅ **16 workers działa** (linia 33): Multiprocessing bez błędów mmap
- ✅ **Numpy arrays w cache**: Eliminacja problemów z mmap

### 2. Stabilność treningu
- ✅ Trening działa stabilnie przez 199 iteracji
- ✅ Brak błędów pamięciowych
- ✅ Brak błędów CUDA
- ✅ Konsystentna prędkość (~3.5 it/s)

---

## ⚠️ Problemy i obserwacje

### 1. Wydajność treningu - wolniejsza niż oczekiwana

**Obserwacje**:
- **Pierwsze iteracje (3-15)**: Bardzo wolne (50s → 1.1s/it)
  - To jest normalne - warmup + kompilacja torch.compile
- **Stabilizacja (16-30)**: Przyspieszenie (1.3 → 3.5 it/s)
- **Stabilny trening (31-199)**: ~3.46-3.50 it/s
  - To jest **~2x wolniej** niż oczekiwane (~7-10 it/s)

**Możliwe przyczyny**:

1. **Batch size 700 może być za duży**:
   - A100 ma 80GB, ale batch 700 może powodować:
     - Większe zużycie pamięci → wolniejsze operacje
     - Dłuższy forward/backward pass
     - Mniej równoległości w GPU

2. **torch.compile może nie działać optymalnie**:
   - Linie 130-148: Warnings o graph breaks (`Tensor.item()`)
   - Linie 141-145, 252-256: Warnings o "Online softmax disabled"
   - Linia 148: "skipping cudagraphs due to cpu device"
   - **Wnioski**: Kompilacja może nie być w pełni efektywna

3. **Konwersja numpy→tensor w każdym __getitem__**:
   - Każdy worker konwertuje numpy arrays na tensors
   - To może być bottleneck jeśli dataset jest duży
   - Ale powinno być szybkie (torch.from_numpy jest szybkie)

4. **Możliwy bottleneck w forward/backward pass**:
   - GPU może nie być w pełni wykorzystane
   - Batch 700 może być nieoptymalny dla T5-base
   - Możliwe że model jest I/O bound, nie compute bound

### 2. Błąd ewaluacji (naprawiony)

**Problem**:
- Linia 361: Ewaluacja zaczyna się (batch 199)
- Linia 365-382: `TypeError: prediction_dir is None`
- Błąd występuje w `evaluate_fromConfig` → `getAndMake_specificPredictionDir`

**Przyczyna**:
- `MultiEvaluationConfig.get_allConfigs()` tworzy nowe configi
- Nowe configi nie dziedziczą `prediction_dir` z base config
- `updated_fields` zawiera tylko pola z `fields_toIterateOver` (split, inference_dataset)
- `prediction_dir` nie jest w `updated_fields` → None w nowych configach

**Naprawa**:
1. W `training.py`: Upewniamy się że `prediction_dir` jest w `base_config_dict` przed tworzeniem `MultiEvaluationConfig`
2. W `MultiEvaluationConfig.py`: Dodajemy `prediction_dir` do `updated_fields` jeśli istnieje w base config

---

## 📈 Analiza wydajności

### Timeline treningu

| Iteracja | Czas/it | Prędkość | Uwagi |
|----------|---------|----------|-------|
| 3 | 50.01s | 0.02 it/s | Warmup + kompilacja |
| 10 | 4.05s | 0.25 it/s | Przyspieszenie |
| 15 | 0.90s | 1.11 it/s | Dalsze przyspieszenie |
| 30 | 0.29s | 3.47 it/s | Stabilizacja |
| 100 | 0.29s | 3.46 it/s | Stabilny |
| 199 | 0.29s | 3.50 it/s | Stabilny |

### Porównanie z oczekiwaniami

| Metryka | Oczekiwane | Rzeczywiste | Status |
|---------|------------|-------------|--------|
| Prędkość treningu | 7-10 it/s | 3.5 it/s | ⚠️ 2x wolniej |
| Cache tokenizacji | Działa | ✅ Działa | ✅ OK |
| Multiprocessing | Działa | ✅ Działa | ✅ OK |
| torch.compile | 20-30% speedup | ❓ Nie widać | ⚠️ Może nie działać |

---

## 🔍 Szczegółowe obserwacje

### 1. torch.compile warnings

**Linie 130-148**: Graph breaks z `Tensor.item()`
- `loss.detach().cpu().item()` powoduje graph break
- To może zmniejszyć efektywność kompilacji
- **Rozwiązanie**: Ustawić `torch._dynamo.config.capture_scalar_outputs = True`

**Linie 141-145, 252-256**: "Online softmax disabled"
- Inductor decyduje się podzielić reduction
- Może wpływać na wydajność
- **To jest warning, nie błąd** - można zignorować

**Linia 148**: "skipping cudagraphs due to cpu device"
- Cudagraphs są pomijane bo niektóre operacje są na CPU
- To może zmniejszyć efektywność kompilacji
- **Możliwe przyczyny**: `.cpu().item()` w kodzie

### 2. Batch size 700

**Obecna konfiguracja**:
- `train_batch_size`: 700
- `gradient_accumulation_factor`: 1
- `Efektywny batch`: 700

**Możliwe problemy**:
- Batch 700 może być za duży dla T5-base na A100
- Optymalny batch dla T5-base to zazwyczaj 256-512
- Większy batch = dłuższy forward/backward pass
- Może powodować mniejsze wykorzystanie GPU (mniej równoległości)

**Rekomendacja**: Spróbować batch_size=512 lub 600

### 3. Konwersja numpy→tensor

**Obecna implementacja**:
- W `PytorchDataset.__getitem__()` konwertujemy numpy arrays na tensors
- Dzieje się to w każdym workerze dla każdego przykładu
- `torch.from_numpy()` jest szybkie, ale może być bottleneck przy dużym dataset

**Możliwe optymalizacje**:
- Batch conversion (konwertować cały batch naraz)
- Ale to wymagałoby zmian w DataLoader

---

## 🎯 Rekomendacje

### Natychmiastowe (naprawione)
1. ✅ **Naprawić błąd z prediction_dir** - DONE
   - Dodano fallback w `training.py`
   - Dodano zachowanie `prediction_dir` w `MultiEvaluationConfig.get_allConfigs()`

### Krótkoterminowe (do testowania)
1. **Zmniejszyć batch_size do 512-600**:
   - Batch 700 może być za duży
   - Testować batch_size=512 lub 600
   - Sprawdzić czy prędkość się poprawi

2. **Sprawdzić wykorzystanie GPU**:
   - Uruchomić `nvidia-smi` podczas treningu
   - Sprawdzić czy GPU jest w pełni wykorzystane (powinno być >90%)
   - Jeśli <50%, to bottleneck jest gdzie indziej

3. **Wyłączyć torch.compile tymczasowo**:
   - Sprawdzić czy rzeczywiście przyspiesza
   - Jeśli nie, wyłączyć i porównać prędkość

4. **Naprawić graph breaks w torch.compile**:
   - Ustawić `torch._dynamo.config.capture_scalar_outputs = True`
   - To może poprawić efektywność kompilacji

### Długoterminowe (opcjonalne)
1. **Batch conversion numpy→tensor**:
   - Konwertować cały batch naraz zamiast przykład po przykładzie
   - Wymagałoby zmian w DataLoader/collate_fn

2. **Profiling treningu**:
   - Użyć `torch.profiler` do znalezienia bottlenecków
   - Sprawdzić gdzie spędza się najwięcej czasu

---

## 📝 Wnioski końcowe

### Co działa
- ✅ Wszystkie optymalizacje działają (cache, multiprocessing, compile)
- ✅ Trening jest stabilny i działa bez błędów
- ✅ Błąd z ewaluacją został naprawiony

### Co wymaga uwagi
- ⚠️ Wydajność jest ~2x wolniejsza niż oczekiwana
- ⚠️ Batch size 700 może być nieoptymalny
- ⚠️ torch.compile może nie działać w pełni efektywnie

### Następne kroki
1. Naprawić błąd ewaluacji ✅ (DONE)
2. Przetestować z batch_size=512-600
3. Sprawdzić wykorzystanie GPU
4. Rozważyć wyłączenie torch.compile jeśli nie pomaga

---

*Analiza wykonana: 2024-11-17*

