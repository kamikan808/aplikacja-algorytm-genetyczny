# Algorytm Genetyczny — dokumentacja projektu

## Struktura projektu

```
genetic_algorithm/
├── requirements.txt          zależności (numpy, matplotlib, jupyter)
├── ewoluucyjne.ipynb         notatnik do testowania — tu uruchamiasz algorytm,
│                             zmieniasz parametry i sprawdzasz wyniki
└── ga/                       pakiet algorytmu genetycznego
    ├── __init__.py           eksportuje wszystko, wystarczy: from ga import ...
    ├── functions.py          gotowe funkcje celu: himmelblau (2 zmienne),
    │                         sphere, rastrigin, rosenbrock (dowolna liczba zmiennych)
    ├── encoding.py           kodowanie wartości rzeczywistych na chromosom binarny
    │                         i dekodowanie z powrotem; obliczanie liczby bitów
    │                         na zmienną na podstawie zakresu i dokładności
    ├── selection.py          3 metody selekcji rodziców: best, roulette, tournament
    ├── crossover.py          4 metody krzyżowania chromosomów: single_point,
    │                         two_point, uniform, granular
    ├── mutation.py           3 metody mutacji: boundary, single_point, two_point
    ├── operators.py          operator inwersji (odwrócenie fragmentu chromosomu)
    │                         i strategia elitarna (przenoszenie najlepszych osobników)
    ├── results.py            zapis wyników do pliku CSV — każde uruchomienie
    │                         algorytmu to jeden wiersz z pełną konfiguracją i wynikiem
    └── algorithm.py          główny silnik GA: GAConfig (konfiguracja), GAResult (wyniki),
                              run_ga() (uruchamia algorytm i zwraca wyniki)
```

## Instalacja

```bash
pip install -r requirements.txt
```

---

## Przykładowe uruchomienie

```python
from ga import GAConfig, run_ga, himmelblau, save_result

cfg = GAConfig(
    func=himmelblau,          # gotowa funkcja lub własna def/lambda
    n_vars=2,                 # liczba zmiennych (himmelblau wymaga 2)
    lower=-5, upper=5,        # zakres poszukiwań
    maximize=False,           # False = minimalizacja, True = maksymalizacja
    pop_size=80,
    n_epochs=300,
    selection_method='tournament',
    crossover_method='single_point',
    mutation_method='single_point',
    use_elitism=True,
)

result = run_ga(cfg, seed=42)
save_result(result, "results.csv", func_name="himmelblau")

print(result.best_value)       # najlepsza wartość funkcji
print(result.best_individual)  # wektor zmiennych
print(result.history_best)     # zbieżność: najlepsza wartość w każdej epoce
```

Własną funkcję podaje się bezpośrednio: `func=lambda x: x[0]**2 + x[1]**2` lub `func=moja_funkcja`.
Funkcje `sphere`, `rastrigin`, `rosenbrock` działają z dowolnym `n_vars`. `himmelblau` wymaga `n_vars=2`.

---

## Pełna konfiguracja (GAConfig)

| Parametr | Domyślnie | Opis |
|---|---|---|
| `func` | — | funkcja celu |
| `n_vars` | — | liczba zmiennych |
| `lower`, `upper` | — | zakres poszukiwań |
| `maximize` | `False` | `True` = maksymalizacja |
| `pop_size` | `50` | wielkość populacji |
| `n_epochs` | `100` | liczba generacji |
| `precision` | `6` | dokładność kodowania (miejsca po przecinku) |
| `selection_method` | `tournament` | `best` / `roulette` / `tournament` |
| `crossover_method` | `single_point` | `single_point` / `two_point` / `uniform` / `granular` |
| `mutation_method` | `single_point` | `boundary` / `single_point` / `two_point` |
| `crossover_prob` | `0.8` | prawdopodobieństwo krzyżowania |
| `mutation_prob` | `0.01` | prawdopodobieństwo mutacji |
| `use_inversion` | `False` | operator inwersji |
| `inversion_prob` | `0.05` | prawdopodobieństwo inwersji |
| `use_elitism` | `True` | strategia elitarna |
| `n_elite` | `1` | liczba elitarnych osobników |
| `tournament_size` | `3` | dla `selection_method='tournament'` |
| `swap_prob` | `0.5` | dla `crossover_method='uniform'` |
| `grain_size` | `4` | dla `crossover_method='granular'` |

---

## Co zostało do zrobienia

- **Punkt 7** — moduł wielu eksperymentów: automatyczne uruchamianie dla wielu konfiguracji z zadaną liczbą powtórzeń
- **Punkt 8** — tabela rankingowa najlepszych konfiguracji (osobny plik podsumowujący CSV)
- **Punkt 9** — wykresy: zbieżność, wykres 3D i heatmapa (dla funkcji 2 zmiennych)
