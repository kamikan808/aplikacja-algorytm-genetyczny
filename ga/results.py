"""
results.py
----------
Zapis wyników algorytmu genetycznego do pliku CSV.

Każde uruchomienie algorytmu to jeden wiersz w pliku.
"""

import csv
import os
from datetime import datetime
from typing import List

from .algorithm import GAResult

CSV_COLUMNS = [
    "timestamp",
    "func_name",
    "n_vars",
    "maximize",
    "lower",
    "upper",
    "pop_size",
    "n_epochs",
    "precision",
    "selection_method",
    "crossover_method",
    "mutation_method",
    "crossover_prob",
    "mutation_prob",
    "use_inversion",
    "inversion_prob",
    "use_elitism",
    "n_elite",
    "best_value",
    "best_individual",
    "elapsed_time",
]


def save_result(result: GAResult, filepath: str = "results.csv",
                func_name: str = "unknown") -> None:
    """
    Dopisz jeden wynik do pliku CSV (tworzy plik jeśli nie istnieje).

    Args:
        result:    obiekt GAResult z wynikiem uruchomienia
        filepath:  ścieżka do pliku CSV
        func_name: nazwa funkcji celu (string, bo callable nie ma nazwy)
    """
    cfg = result.config
    file_exists = os.path.isfile(filepath)

    row = {
        "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "func_name":        func_name,
        "n_vars":           cfg.n_vars,
        "maximize":         cfg.maximize,
        "lower":            cfg.lower,
        "upper":            cfg.upper,
        "pop_size":         cfg.pop_size,
        "n_epochs":         cfg.n_epochs,
        "precision":        cfg.precision,
        "selection_method": cfg.selection_method,
        "crossover_method": cfg.crossover_method,
        "mutation_method":  cfg.mutation_method,
        "crossover_prob":   cfg.crossover_prob,
        "mutation_prob":    cfg.mutation_prob,
        "use_inversion":    cfg.use_inversion,
        "inversion_prob":   cfg.inversion_prob,
        "use_elitism":      cfg.use_elitism,
        "n_elite":          cfg.n_elite,
        "best_value":       round(result.best_value, 10),
        "best_individual":  str([round(v, 6) for v in result.best_individual.tolist()]),
        "elapsed_time":     round(result.elapsed_time, 4),
    }

    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_results(results: List[tuple], filepath: str = "results.csv") -> None:
    """
    Zapisz listę wyników naraz.

    Args:
        results:  lista krotek (GAResult, func_name)
        filepath: ścieżka do pliku CSV
    """
    for result, func_name in results:
        save_result(result, filepath, func_name)
