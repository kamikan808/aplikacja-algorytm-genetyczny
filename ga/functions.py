"""
functions.py
------------
Gotowe funkcje celu dostępne w projekcie.

Każda funkcja przyjmuje numpy array x i zwraca skalar float.
Można też podać własną funkcję — wystarczy przekazać ją bezpośrednio
do GAConfig jako parametr `func`.

Dostępne funkcje:
    himmelblau  — 2 zmienne, 4 minima globalne = 0
    sphere      — n zmiennych, minimum = 0 w (0,...,0)
    rastrigin   — n zmiennych, minimum = 0 w (0,...,0)
    rosenbrock  — n zmiennych, minimum = 0 w (1,...,1)
"""

import numpy as np


def himmelblau(x: np.ndarray) -> float:
    """
    f(x0, x1) = (x0^2 + x1 - 11)^2 + (x0 + x1^2 - 7)^2
    Wymaga dokładnie 2 zmiennych.
    Zakres: [-5, 5]
    Minima globalne (wartość 0.0):
        (3.0, 2.0), (-2.805, 3.131), (-3.779, -3.283), (3.584, -1.848)
    """
    x0, x1 = x[0], x[1]
    return float((x0 ** 2 + x1 - 11) ** 2 + (x0 + x1 ** 2 - 7) ** 2)


def sphere(x: np.ndarray) -> float:
    """
    f(x) = sum(xi^2)
    Zakres: [-5.12, 5.12]
    Minimum globalne: 0.0 w (0,...,0)
    """
    return float(np.sum(x ** 2))


def rastrigin(x: np.ndarray) -> float:
    """
    f(x) = 10*n + sum(xi^2 - 10*cos(2*pi*xi))
    Zakres: [-5.12, 5.12]
    Minimum globalne: 0.0 w (0,...,0)
    """
    n = len(x)
    return float(10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


def rosenbrock(x: np.ndarray) -> float:
    """
    f(x) = sum(100*(x_{i+1} - xi^2)^2 + (1 - xi)^2)
    Zakres: [-5, 10]
    Minimum globalne: 0.0 w (1,...,1)
    """
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))
