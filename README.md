# Algorytm Genetyczny — Optymalizacja Funkcji (Aplikacja Webowa)

Projekt realizujący klasyczny algorytm genetyczny z interfejsem graficznym stworzonym w frameworku Streamlit. Pozwala na optymalizację (minimalizację/maksymalizację) funkcji wielu zmiennych z możliwością pełnej konfiguracji parametrów ewolucyjnych i wygenerowania tabelarycznego rankingu metod.

## Struktura projektu

```text
genetic_algorithm/
├── app.py                    główna aplikacja Streamlit (interfejs graficzny, wykresy, logika sesji)
├── requirements.txt          zależności projektu (numpy, matplotlib, pandas, streamlit)
└── ga/                       pakiet logiki algorytmu genetycznego
    ├── __init__.py           eksportuje główne moduły
    ├── functions.py          gotowe funkcje celu: himmelblau (2 zmienne), sphere, rastrigin, rosenbrock (n-zmiennych)
    ├── encoding.py           kodowanie wartości rzeczywistych na chromosom binarny i dekodowanie
    ├── selection.py          3 metody selekcji rodziców: best, roulette, tournament
    ├── crossover.py          4 metody krzyżowania chromosomów: single_point, two_point, uniform, granular
    ├── mutation.py           3 metody mutacji: boundary, single_point, two_point
    ├── operators.py          operator inwersji i strategia elitarna
    ├── algorithm.py          główna pętla ewolucyjna (GAConfig, GAResult, run_ga)
    └── results.py            zapis do pliku dyskowego (wykorzystywane przy działaniu poza chmurą)
```

## Jak uruchomić lokalnie

Zalecane jest uruchomienie projektu w wirtualnym środowisku Pythona. Aby uruchomić bez wirtualnego środowiska należy pominąć krok 1 i 2.

W terminalu wykonaj następujące polecenia:
```bash
# 1. Utworzenie wirtualnego środowiska
python3 -m venv venv

# 2. Aktywacja środowiska
source venv/bin/activate

# 3. Instalacja niezbędnych bibliotek
pip install -r requirements.txt

# 4. Uruchomienie aplikacji w przeglądarce
streamlit run app.py
```
Aplikacja otworzy się automatycznie pod adresem `http://localhost:8501`.

## Wersja w chmurze (Hosting)

Aplikacja jest w pełni przystosowana do działania w chmurze (np. na darmowym **Streamlit Community Cloud**). Mechanizm zapisu wyników działa w oparciu o pamięć sesji (RAM), co pozwala na płynne pobieranie statystyk, historii pojedynczych uruchomień oraz pełnych rankingów benchmarkowych jako pliki `.csv` bezpośrednio z interfejsu użytkownika.

---

## Pełna konfiguracja algorytmu (GAConfig)

Konfiguracja jest obsługiwana w tle przez interfejs graficzny, ale silnik bazuje na obiekcie `GAConfig`. 

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
| `tournament_size` | `3` | dla `tournament` |
| `grain_size` | `1` | dla `granular` |
| `swap_prob` | `0.5` | dla `uniform` |
