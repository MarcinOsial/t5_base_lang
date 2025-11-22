# Wyjaśnienie: Batch Size, Learning Rate i num_batches

## Związek między Batch Size a Learning Rate

### Zasada skalowania Learning Rate

Gdy zwiększamy batch size, musimy odpowiednio dostosować learning rate, aby zachować stabilność treningu i jakość modelu.

**Dlaczego?**
- Większy batch size → bardziej stabilny gradient (mniejsza wariancja)
- Stabilniejszy gradient → możemy użyć większego learning rate
- Ale zbyt duży LR z dużym batch → niestabilność treningu

### Metody skalowania

1. **Linear Scaling** (proste):
   ```
   lr_new = lr_old × (batch_size_new / batch_size_old)
   ```
   - Działa dobrze dla małych zmian batch size
   - Przykład: 512 → 700, lr: 1e-4 → 1.37e-4

2. **Square Root Scaling** (bezpieczniejsze):
   ```
   lr_new = lr_old × √(batch_size_new / batch_size_old)
   ```
   - Bardziej konserwatywne, bezpieczniejsze
   - Przykład: 512 → 700, lr: 1e-4 → 1.17e-4

3. **Conservative Scaling** (najbezpieczniejsze):
   - Mniejsza zmiana niż linear
   - Dobre dla dużych zmian batch size

## Nasza zmiana

### Przed:
- `train_batch_size`: 512
- `gradient_accumulation_factor`: 2
- **Efektywny batch**: 512 × 2 = **1024**
- `lr`: 1e-4

### Po:
- `train_batch_size`: 700
- `gradient_accumulation_factor`: 1
- **Efektywny batch**: 700 × 1 = **700**
- `lr`: 8e-5 (sqrt scaling: 1e-4 × √(700/1024) ≈ 8.27e-5)

### Dlaczego zmniejszyliśmy gradient_accumulation_factor?
- Batch size 700 jest już duży dla A100
- Gradient accumulation = 1 jest prostsze i wymaga mniej pamięci
- Efektywny batch 700 jest nadal rozsądny

## Czy num_batches powinien się zmienić?

### Odpowiedź: **NIE** (zostawiamy 75000)

**Dlaczego?**

1. **num_batches = liczba iteracji treningowych**
   - To jest liczba kroków optymalizacji, nie zależy od batch size
   - Ważniejsza jest liczba iteracji niż całkowita liczba przykładów

2. **Zachowanie liczby przykładów vs iteracji**
   - Jeśli chcemy zachować **tę samą liczbę przykładów**:
     - Stare: 1024 × 75000 = 76,800,000 przykładów
     - Nowe: 700 × 75000 = 52,500,000 przykładów (mniej!)
   - Ale zazwyczaj ważniejsza jest **liczba iteracji** (kroków optymalizacji)
   - Model uczy się przez iteracje, nie przez całkowitą liczbę przykładów

3. **Praktyka**
   - W większości przypadków num_batches jest stały niezależnie od batch size
   - Batch size wpływa na stabilność i prędkość, nie na liczbę potrzebnych iteracji
   - Jeśli chcesz zachować tę samą liczbę przykładów, możesz zwiększyć num_batches:
     - Nowe num_batches: 76,800,000 / 700 ≈ 109,714
     - Ale to nie jest konieczne!

### Rekomendacja

**Zostawiamy num_batches = 75000** (bez zmian)
- Model będzie trenowany przez 75,000 iteracji
- To jest standardowa praktyka
- Batch size wpływa na stabilność i prędkość, nie na liczbę potrzebnych iteracji

## Podsumowanie zmian

| Parametr | Stara wartość | Nowa wartość | Uzasadnienie |
|----------|---------------|--------------|--------------|
| `train_batch_size` | 512 | **700** | Optymalne dla A100 |
| `eval_batch_size` | 512 | **700** | Spójność z train |
| `gradient_accumulation_factor` | 2 | **1** | Batch 700 jest już duży |
| `lr` | 1e-4 | **8e-5** | Sqrt scaling dla bezpieczeństwa |
| `num_batches` | 75000 | **75000** | Liczba iteracji bez zmian |

## Oczekiwane efekty

1. **Szybszy trening**: Większy batch size = mniej iteracji na epokę
2. **Lepsze wykorzystanie GPU**: Batch 700 lepiej wykorzystuje A100
3. **Stabilniejszy trening**: Większy batch = mniejsza wariancja gradientu
4. **Mniej pamięci**: gradient_accumulation=1 zamiast 2

