"""
operators.py
------------
Operator inwersji i strategia elitarna.

Operator inwersji:
    Odwraca losowo wybrany segment chromosomu.

Strategia elitarna:
    Przenosi najlepszych osobników z poprzedniej populacji
    bezpośrednio do nowej (bez zmian).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Operator inwersji
# ---------------------------------------------------------------------------

def inversion(chromosome: np.ndarray, inversion_prob: float = 0.05,
              **kwargs) -> np.ndarray:
    """
    Operator inwersji.

    Z prawdopodobieństwem `inversion_prob` losuje dwa punkty podziału
    i odwraca (reverse) fragment chromosomu między nimi.

    Przykład:
        Przed: [1, 0, 1, 1, 0, 0, 1]
        Punkty: cp1=2, cp2=5
        Po:     [1, 0, 0, 1, 1, 0, 1]   ← segment [1,1,0] → [0,1,1]

    Args:
        chromosome:     1D array bitów
        inversion_prob: prawdopodobieństwo wykonania inwersji

    Returns:
        Chromosom po (ewentualnej) inwersji.
    """
    result = chromosome.copy()
    if len(result) < 2:
        return result
    if np.random.rand() < inversion_prob:
        cp1, cp2 = sorted(np.random.choice(range(len(result) + 1), size=2, replace=False))
        if cp1 < cp2:
            result[cp1:cp2] = result[cp1:cp2][::-1]
    return result


# ---------------------------------------------------------------------------
# Strategia elitarna
# ---------------------------------------------------------------------------

def apply_elitism(old_population: np.ndarray, old_fitness: np.ndarray,
                  new_population: np.ndarray, n_elite: int,
                  maximize: bool = False) -> np.ndarray:
    """
    Strategia elitarna.

    Zastępuje `n_elite` najgorszych osobników nowej populacji
    przez `n_elite` najlepszych osobników poprzedniej populacji.

    Dzięki temu najlepsze dotychczas znalezione rozwiązania
    nie zostaną utracone w wyniku krzyżowania i mutacji.

    Args:
        old_population: array (pop_size, chrom_len) — poprzednia populacja
        old_fitness:    array (pop_size,) — wartości funkcji celu poprzedniej populacji
        new_population: array (pop_size, chrom_len) — nowo wygenerowana populacja
        n_elite:        liczba elitarnych osobników do zachowania
        maximize:       True = maksymalizacja, False = minimalizacja

    Returns:
        Zmodyfikowana nowa populacja z wstawionymi elitarnymi osobnikami.
    """
    if n_elite <= 0:
        return new_population

    result = new_population.copy()
    pop_size = len(old_population)
    n_elite = min(n_elite, pop_size)

    # Wyznacz indeksy najlepszych w starej populacji
    if maximize:
        elite_idx = np.argsort(old_fitness)[-n_elite:][::-1]
    else:
        elite_idx = np.argsort(old_fitness)[:n_elite]

    # Wstaw elitarne chromosomy na początku nowej populacji
    # (nadpisują n_elite pierwszych osobników nowej populacji)
    result[:n_elite] = old_population[elite_idx]
    return result
