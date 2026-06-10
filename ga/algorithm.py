# algorithm.py - główny moduł algorytmu genetycznego

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .encoding import init_population, decode_individual, calculate_bits_per_var
from .selection import select
from .crossover import crossover
from .mutation import mutate
from .operators import inversion, apply_elitism


@dataclass
class GAConfig:
    # Parametry konfiguracyjne algorytmu genetycznego.
    # Minimalne wymagane pola:
    #     func   — funkcja celu: przyjmuje np.ndarray, zwraca float
    #     n_vars — liczba zmiennych (wymiar przestrzeni poszukiwań)
    #     lower  — dolna granica zakresu dla każdej zmiennej
    #     upper  — górna granica zakresu dla każdej zmiennej

    func: Callable          # funkcja celu: f(x: np.ndarray) -> float
    n_vars: int             # liczba zmiennych
    lower: float            # dolna granica zakresu
    upper: float            # górna granica zakresu

    # --- Typ optymalizacji ---
    maximize: bool = False  # False = minimalizacja, True = maksymalizacja

    # --- Parametry populacji ---
    pop_size: int = 50
    n_epochs: int = 100
    precision: int = 6      # dokładność kodowania: liczba miejsc po przecinku

    # --- Metody ---
    selection_method: str = "tournament"    # 'best' | 'roulette' | 'tournament'
    crossover_method: str = "single_point"  # 'single_point' | 'two_point' | 'uniform' | 'granular'
    mutation_method: str = "single_point"   # 'boundary' | 'single_point' | 'two_point'

    # --- Prawdopodobieństwa ---
    crossover_prob: float = 0.8
    mutation_prob: float = 0.01

    # --- Operator inwersji ---
    use_inversion: bool = False
    inversion_prob: float = 0.05

    # --- Strategia elitarna ---
    use_elitism: bool = True
    n_elite: int = 1

    # --- Parametry specyficzne dla metod ---
    tournament_size: int = 3    # dla selection_method='tournament'
    swap_prob: float = 0.5      # dla crossover_method='uniform'
    grain_size: int = 4         # dla crossover_method='granular'


@dataclass
class GAResult:
    """Wyniki jednego uruchomienia algorytmu."""
    best_value: float           # najlepsza wartość funkcji celu
    best_individual: np.ndarray # wektor zmiennych najlepszego osobnika
    history_best: List[float]   # najlepsza wartość w każdej epoce
    history_avg: List[float]    # średnia wartość populacji w każdej epoce
    elapsed_time: float         # czas obliczeń w sekundach
    config: GAConfig = field(repr=False)


def run_ga(config: GAConfig, seed: Optional[int] = None) -> GAResult:

    if seed is not None:
        np.random.seed(seed)

    t_start = time.perf_counter()

    bits_per_var = calculate_bits_per_var(config.lower, config.upper, config.precision)

    population = init_population(config.pop_size, config.n_vars, bits_per_var)
    fitness = _evaluate(population, config, bits_per_var)

    history_best: List[float] = []
    history_avg: List[float] = []

    # Zapamiętaj globalnie najlepszego osobnika
    global_best_chrom, global_best_value = _get_best(population, fitness, config.maximize)

    for _ in range(config.n_epochs):

        # Selekcja
        parents = select(
            method=config.selection_method,
            population=population,
            fitness=fitness,
            n_select=config.pop_size,
            maximize=config.maximize,
            tournament_size=config.tournament_size,
        )

        # Krzyżowanie
        new_pop = []
        for i in range(0, config.pop_size - 1, 2):
            p1 = parents[i]
            p2 = parents[(i + 1) % config.pop_size]
            c1, c2 = crossover(
                method=config.crossover_method,
                parent1=p1, parent2=p2,
                crossover_prob=config.crossover_prob,
                swap_prob=config.swap_prob,
                grain_size=config.grain_size,
            )
            new_pop.extend([c1, c2])
        if len(new_pop) < config.pop_size:
            new_pop.append(parents[-1].copy())
        new_pop = np.array(new_pop[:config.pop_size], dtype=np.int8)

        # Mutacja
        for i in range(config.pop_size):
            new_pop[i] = mutate(
                method=config.mutation_method,
                chromosome=new_pop[i],
                mutation_prob=config.mutation_prob,
            )

        # Inwersja
        if config.use_inversion:
            for i in range(config.pop_size):
                new_pop[i] = inversion(new_pop[i], config.inversion_prob)

        # Elitaryzm
        if config.use_elitism:
            new_pop = apply_elitism(
                old_population=population,
                old_fitness=fitness,
                new_population=new_pop,
                n_elite=config.n_elite,
                maximize=config.maximize,
            )

        population = new_pop
        fitness = _evaluate(population, config, bits_per_var)

        epoch_best_chrom, epoch_best_value = _get_best(population, fitness, config.maximize)

        is_better = (epoch_best_value > global_best_value) if config.maximize \
                    else (epoch_best_value < global_best_value)
        if is_better:
            global_best_value = epoch_best_value
            global_best_chrom = epoch_best_chrom.copy()

        history_best.append(epoch_best_value)
        history_avg.append(float(np.mean(fitness)))

    t_end = time.perf_counter()

    best_individual = decode_individual(
        chromosome=global_best_chrom,
        lower=config.lower,
        upper=config.upper,
        bits_per_var=bits_per_var,
        n_vars=config.n_vars,
    )

    return GAResult(
        best_value=global_best_value,
        best_individual=best_individual,
        history_best=history_best,
        history_avg=history_avg,
        elapsed_time=t_end - t_start,
        config=config,
    )


def _evaluate(population: np.ndarray, config: GAConfig, bits_per_var: int) -> np.ndarray:
    fitness = np.empty(len(population))
    for i, chrom in enumerate(population):
        x = decode_individual(chrom, config.lower, config.upper, bits_per_var, config.n_vars)
        fitness[i] = config.func(x)
    return fitness


def _get_best(population: np.ndarray, fitness: np.ndarray, maximize: bool):
    idx = int(np.argmax(fitness)) if maximize else int(np.argmin(fitness))
    return population[idx].copy(), float(fitness[idx])
