"""
crossover.py
------------
Metody krzyżowania chromosomów binarnych.

Każda funkcja przyjmuje parę rodziców (1D numpy array bitów) i parametry,
a zwraca parę potomków tego samego kształtu.

Dostępne metody:
    - single_point  (krzyżowanie jednopunktowe)
    - two_point     (krzyżowanie dwupunktowe)
    - uniform       (krzyżowanie jednorodne)
    - granular      (krzyżowanie ziarniste)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Krzyżowanie jednopunktowe
# ---------------------------------------------------------------------------

def crossover_single_point(parent1: np.ndarray, parent2: np.ndarray,
                            crossover_prob: float = 0.8, **kwargs):
    """
    Krzyżowanie jednopunktowe.

    Losuje jeden punkt podziału cp ∈ [1, chrom_len-1].
    Potomek A = parent1[:cp] + parent2[cp:]
    Potomek B = parent2[:cp] + parent1[cp:]

    Args:
        parent1, parent2: chromosomy rodziców (1D array)
        crossover_prob:   prawdopodobieństwo krzyżowania

    Returns:
        (child1, child2) — para potomków.
    """
    child1, child2 = parent1.copy(), parent2.copy()
    if np.random.rand() < crossover_prob:
        cp = np.random.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:cp], parent2[cp:]])
        child2 = np.concatenate([parent2[:cp], parent1[cp:]])
    return child1, child2


# ---------------------------------------------------------------------------
# Krzyżowanie dwupunktowe
# ---------------------------------------------------------------------------

def crossover_two_point(parent1: np.ndarray, parent2: np.ndarray,
                         crossover_prob: float = 0.8, **kwargs):
    """
    Krzyżowanie dwupunktowe.

    Losuje dwa punkty podziału cp1 < cp2.
    Potomek A = parent1[:cp1] + parent2[cp1:cp2] + parent1[cp2:]
    Potomek B = parent2[:cp1] + parent1[cp1:cp2] + parent2[cp2:]

    Args:
        parent1, parent2: chromosomy rodziców (1D array)
        crossover_prob:   prawdopodobieństwo krzyżowania

    Returns:
        (child1, child2) — para potomków.
    """
    child1, child2 = parent1.copy(), parent2.copy()
    if np.random.rand() < crossover_prob:
        chrom_len = len(parent1)
        cp1, cp2 = sorted(np.random.choice(range(1, chrom_len), size=2, replace=False))
        child1 = np.concatenate([parent1[:cp1], parent2[cp1:cp2], parent1[cp2:]])
        child2 = np.concatenate([parent2[:cp1], parent1[cp1:cp2], parent2[cp2:]])
    return child1, child2


# ---------------------------------------------------------------------------
# Krzyżowanie jednorodne (uniform)
# ---------------------------------------------------------------------------

def crossover_uniform(parent1: np.ndarray, parent2: np.ndarray,
                       crossover_prob: float = 0.8, swap_prob: float = 0.5, **kwargs):
    """
    Krzyżowanie jednorodne.

    Dla każdego bitu niezależnie losuje, czy pochodzi od rodzica 1 czy 2.
    Prawdopodobieństwo zamiany bitu: swap_prob (domyślnie 0.5).

    Args:
        parent1, parent2: chromosomy rodziców (1D array)
        crossover_prob:   prawdopodobieństwo krzyżowania (ogólne)
        swap_prob:        prawdopodobieństwo zamiany pojedynczego bitu

    Returns:
        (child1, child2) — para potomków.
    """
    child1, child2 = parent1.copy(), parent2.copy()
    if np.random.rand() < crossover_prob:
        mask = np.random.rand(len(parent1)) < swap_prob
        child1 = np.where(mask, parent2, parent1)
        child2 = np.where(mask, parent1, parent2)
    return child1, child2


# ---------------------------------------------------------------------------
# Krzyżowanie ziarniste (granular / segment-uniform)
# ---------------------------------------------------------------------------

def crossover_granular(parent1: np.ndarray, parent2: np.ndarray,
                        crossover_prob: float = 0.8, grain_size: int = 4, **kwargs):
    """
    Krzyżowanie ziarniste (granularne).

    Chromosom dzielony jest na segmenty ('ziarna') o długości `grain_size`.
    Dla każdego segmentu losowo wybieramy, od którego rodzica pochodzi cały segment.
    Zapewnia większą spójność bloków genów niż uniform.

    Args:
        parent1, parent2: chromosomy rodziców (1D array)
        crossover_prob:   prawdopodobieństwo krzyżowania
        grain_size:       długość segmentu w bitach (domyślnie 4)

    Returns:
        (child1, child2) — para potomków.
    """
    child1, child2 = parent1.copy(), parent2.copy()
    if np.random.rand() < crossover_prob:
        chrom_len = len(parent1)
        child1 = parent1.copy()
        child2 = parent2.copy()
        i = 0
        while i < chrom_len:
            end = min(i + grain_size, chrom_len)
            if np.random.rand() < 0.5:
                child1[i:end] = parent2[i:end]
                child2[i:end] = parent1[i:end]
            i += grain_size
    return child1, child2


# ---------------------------------------------------------------------------
# Rejestr metod krzyżowania
# ---------------------------------------------------------------------------

CROSSOVER_METHODS = {
    "single_point": crossover_single_point,
    "two_point":    crossover_two_point,
    "uniform":      crossover_uniform,
    "granular":     crossover_granular,
}


def crossover(method: str, parent1: np.ndarray, parent2: np.ndarray,
              crossover_prob: float = 0.8, **kwargs):
    """
    Punkt wejścia: skrzyżuj dwoje rodziców podaną metodą.

    Args:
        method:         nazwa metody ('single_point', 'two_point', 'uniform', 'granular')
        parent1/parent2: chromosomy rodziców (1D numpy array bitów)
        crossover_prob: prawdopodobieństwo krzyżowania
        **kwargs:       dodatkowe parametry specyficzne dla metody
                        (np. swap_prob dla 'uniform', grain_size dla 'granular')

    Returns:
        (child1, child2) — para potomków jako numpy arrays.
    """
    if method not in CROSSOVER_METHODS:
        raise ValueError(f"Nieznana metoda krzyżowania: '{method}'. "
                         f"Dostępne: {list(CROSSOVER_METHODS.keys())}")
    return CROSSOVER_METHODS[method](parent1, parent2, crossover_prob, **kwargs)
