"""
selection.py
------------
Metody selekcji osobników do reprodukcji.

Każda funkcja przyjmuje populację, wektor przystosowania i parametry,
a zwraca tablicę wybranych osobników (rodziców).

Dostępne metody:
    - best       (selekcja najlepszych / truncation)
    - roulette   (selekcja ruletkowa)
    - tournament (selekcja turniejowa)
"""

import numpy as np


def _fitness_for_selection(fitness: np.ndarray, maximize: bool) -> np.ndarray:
    """
    Przelicz wartości funkcji celu na wartości przystosowania ≥ 0,
    tak by większa wartość zawsze oznaczała lepszego osobnika.

    Dla minimalizacji: fitness_sel = max(f) - f  (najniższe f → największe fitness_sel)
    Dla maksymalizacji: fitness_sel = f - min(f) + epsilon
    """
    eps = 1e-10
    if maximize:
        shifted = fitness - fitness.min() + eps
    else:
        shifted = fitness.max() - fitness + eps
    return shifted


# ---------------------------------------------------------------------------
# Selekcja najlepszych (truncation)
# ---------------------------------------------------------------------------

def selection_best(population: np.ndarray, fitness: np.ndarray,
                   n_select: int, maximize: bool = False, **kwargs) -> np.ndarray:
    """
    Wybierz `n_select` najlepszych osobników (z powtórzeniami jeśli n_select > pop_size).

    Args:
        population: array (pop_size, chrom_len)
        fitness:    array (pop_size,) — wartości funkcji celu
        n_select:   liczba rodziców do wybrania
        maximize:   True = maksymalizacja, False = minimalizacja

    Returns:
        Array (n_select, chrom_len) wybranych rodziców.
    """
    if maximize:
        sorted_idx = np.argsort(fitness)[::-1]
    else:
        sorted_idx = np.argsort(fitness)

    # Jeśli n_select > liczba najlepszych, cyklicznie powtarzaj
    selected_idx = [sorted_idx[i % len(sorted_idx)] for i in range(n_select)]
    return population[selected_idx]


# ---------------------------------------------------------------------------
# Selekcja ruletkowa (fitness proportionate)
# ---------------------------------------------------------------------------

def selection_roulette(population: np.ndarray, fitness: np.ndarray,
                       n_select: int, maximize: bool = False, **kwargs) -> np.ndarray:
    """
    Selekcja ruletkowa proporcjonalna do przystosowania.

    Prawdopodobieństwo wybrania osobnika i: p_i = fitness_sel_i / sum(fitness_sel)

    Args:
        population: array (pop_size, chrom_len)
        fitness:    array (pop_size,) — wartości funkcji celu
        n_select:   liczba rodziców do wybrania
        maximize:   True = maksymalizacja, False = minimalizacja

    Returns:
        Array (n_select, chrom_len) wybranych rodziców.
    """
    fit_sel = _fitness_for_selection(fitness, maximize)
    probabilities = fit_sel / fit_sel.sum()
    idx = np.random.choice(len(population), size=n_select, replace=True, p=probabilities)
    return population[idx]


# ---------------------------------------------------------------------------
# Selekcja turniejowa
# ---------------------------------------------------------------------------

def selection_tournament(population: np.ndarray, fitness: np.ndarray,
                         n_select: int, maximize: bool = False,
                         tournament_size: int = 3, **kwargs) -> np.ndarray:
    """
    Selekcja turniejowa: losuj `tournament_size` osobników i wybierz najlepszego.
    Powtarzaj n_select razy.

    Args:
        population:      array (pop_size, chrom_len)
        fitness:         array (pop_size,) — wartości funkcji celu
        n_select:        liczba rodziców do wybrania
        maximize:        True = maksymalizacja, False = minimalizacja
        tournament_size: liczba uczestników każdego turnieju (domyślnie 3)

    Returns:
        Array (n_select, chrom_len) wybranych rodziców.
    """
    pop_size = len(population)
    selected = []

    for _ in range(n_select):
        contenders_idx = np.random.randint(0, pop_size, size=tournament_size)
        contenders_fit = fitness[contenders_idx]
        if maximize:
            winner_local = np.argmax(contenders_fit)
        else:
            winner_local = np.argmin(contenders_fit)
        winner_idx = contenders_idx[winner_local]
        selected.append(population[winner_idx])

    return np.array(selected)


# ---------------------------------------------------------------------------
# Rejestr metod selekcji
# ---------------------------------------------------------------------------

SELECTION_METHODS = {
    "best":       selection_best,
    "roulette":   selection_roulette,
    "tournament": selection_tournament,
}


def select(method: str, population: np.ndarray, fitness: np.ndarray,
           n_select: int, maximize: bool = False, **kwargs) -> np.ndarray:
    """
    Punkt wejścia: wybierz rodziców podaną metodą.

    Args:
        method:     nazwa metody ('best', 'roulette', 'tournament')
        population: array (pop_size, chrom_len)
        fitness:    array (pop_size,) — wartości funkcji celu
        n_select:   liczba rodziców do wybrania
        maximize:   True = maksymalizacja, False = minimalizacja
        **kwargs:   dodatkowe parametry specyficzne dla metody
                    (np. tournament_size dla 'tournament')

    Returns:
        Array (n_select, chrom_len) wybranych rodziców.
    """
    if method not in SELECTION_METHODS:
        raise ValueError(f"Nieznana metoda selekcji: '{method}'. "
                         f"Dostępne: {list(SELECTION_METHODS.keys())}")
    return SELECTION_METHODS[method](population, fitness, n_select, maximize, **kwargs)
