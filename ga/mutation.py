# mutation.py - moduł mutacji w algorytmie genetycznym

import numpy as np


# Mutacja brzegowa (boundary mutation)
def mutation_boundary(chromosome: np.ndarray, mutation_prob: float = 0.01,
                       **kwargs) -> np.ndarray:
    """
    Mutacja brzegowa.

    Dla każdego osobnika z prawdopodobieństwem `mutation_prob` zastępuje
    losowy gen wartością z brzegu (0 lub 1 — negacja bitu brzegowego
    w sensie zakresu). W implementacji binarnej traktujemy ją jako:
    wybierz losowy bit i wymuś wartość 0 albo 1 (skrajną — czyli zamień na 0 lub 1
    niezależnie od obecnej wartości, losując którą z dwóch wartości brzegowych).
    """
    result = chromosome.copy()
    if np.random.rand() < mutation_prob:
        idx = np.random.randint(0, len(result))
        # wartość brzegowa: 0 lub 1 (losowo)
        result[idx] = np.random.randint(0, 2)
    return result

# Mutacja jednopunktowa (single point / bit-flip)

def mutation_single_point(chromosome: np.ndarray, mutation_prob: float = 0.01,
                           **kwargs) -> np.ndarray:
    """
    Mutacja jednopunktowa (odwrócenie jednego bitu).

    Z prawdopodobieństwem `mutation_prob` losuje jeden bit i neguje go.
    """
    result = chromosome.copy()
    if np.random.rand() < mutation_prob:
        idx = np.random.randint(0, len(result))
        result[idx] ^= 1  # bit flip
    return result

# Mutacja dwupunktowa (two-point / two-bit flip)

def mutation_two_point(chromosome: np.ndarray, mutation_prob: float = 0.01,
                        **kwargs) -> np.ndarray:
    """
    Mutacja dwupunktowa (odwrócenie dwóch bitów).

    Z prawdopodobieństwem `mutation_prob` losuje dwa różne bity i neguje oba.
    Jeśli chromosom ma tylko jeden bit, zachowuje się jak mutacja jednopunktowa.
    """
    result = chromosome.copy()
    if np.random.rand() < mutation_prob:
        chrom_len = len(result)
        if chrom_len < 2:
            result[0] ^= 1
        else:
            idx1, idx2 = np.random.choice(chrom_len, size=2, replace=False)
            result[idx1] ^= 1
            result[idx2] ^= 1
    return result


# Rejestr metod mutacji

MUTATION_METHODS = {
    "boundary":    mutation_boundary,
    "single_point": mutation_single_point,
    "two_point":   mutation_two_point,
}


def mutate(method: str, chromosome: np.ndarray,
           mutation_prob: float = 0.01, **kwargs) -> np.ndarray:
    # Wywołaj wybraną metodę mutacji na danym chromosomie
    if method not in MUTATION_METHODS:
        raise ValueError(f"Nieznana metoda mutacji: '{method}'. "
                         f"Dostępne: {list(MUTATION_METHODS.keys())}")
    return MUTATION_METHODS[method](chromosome, mutation_prob, **kwargs)
