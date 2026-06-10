"""
encoding.py
-----------
Kodowanie binarne i dekodowanie chromosomów.

Chromosom reprezentuje wektor n zmiennych rzeczywistych.
Każda zmienna jest kodowana na `bits_per_var` bitach z zakresu [lower, upper].
"""

import numpy as np


def calculate_bits_per_var(lower: float, upper: float, precision: int) -> int:
    """
    Oblicz minimalną liczbę bitów potrzebną do zakodowania zmiennej
    z danego zakresu z zadaną dokładnością (liczbą miejsc po przecinku).

    Args:
        lower: dolna granica zakresu
        upper: górna granica zakresu
        precision: liczba miejsc po przecinku (np. 3 → 0.001)

    Returns:
        Liczba bitów na zmienną.
    """
    n_points = (upper - lower) * (10 ** precision)
    bits = int(np.ceil(np.log2(n_points + 1)))
    return max(bits, 1)


def encode_individual(values: np.ndarray, lower: float, upper: float, bits_per_var: int) -> np.ndarray:
    """
    Zakoduj wektor wartości rzeczywistych do chromosomu binarnego.

    Args:
        values:       wektor n wartości rzeczywistych
        lower:        dolna granica zakresu
        upper:        górna granica zakresu
        bits_per_var: liczba bitów na zmienną

    Returns:
        Płaski numpy array int8 z bitami chromosomu (n_vars * bits_per_var bitów).
    """
    max_int = 2 ** bits_per_var - 1
    chromosome = []
    for v in values:
        v_clipped = np.clip(v, lower, upper)
        int_val = int(round((v_clipped - lower) / (upper - lower) * max_int))
        bits = [(int_val >> (bits_per_var - 1 - i)) & 1 for i in range(bits_per_var)]
        chromosome.extend(bits)
    return np.array(chromosome, dtype=np.int8)


def decode_individual(chromosome: np.ndarray, lower: float, upper: float,
                      bits_per_var: int, n_vars: int) -> np.ndarray:
    """
    Dekoduj chromosom binarny do wektora wartości rzeczywistych.

    Args:
        chromosome:   płaski numpy array bitów
        lower:        dolna granica zakresu
        upper:        górna granica zakresu
        bits_per_var: liczba bitów na zmienną
        n_vars:       liczba zmiennych

    Returns:
        Numpy array n wartości rzeczywistych.
    """
    max_int = 2 ** bits_per_var - 1
    values = []
    for i in range(n_vars):
        segment = chromosome[i * bits_per_var:(i + 1) * bits_per_var]
        int_val = int("".join(str(b) for b in segment), 2)
        real_val = lower + int_val / max_int * (upper - lower)
        values.append(real_val)
    return np.array(values)


def random_individual(n_vars: int, bits_per_var: int) -> np.ndarray:
    """Wygeneruj losowy chromosom binarny."""
    return np.random.randint(0, 2, size=n_vars * bits_per_var, dtype=np.int8)


def init_population(pop_size: int, n_vars: int, bits_per_var: int) -> np.ndarray:
    """
    Zainicjuj losową populację.

    Returns:
        Array kształtu (pop_size, n_vars * bits_per_var).
    """
    return np.random.randint(0, 2, size=(pop_size, n_vars * bits_per_var), dtype=np.int8)
