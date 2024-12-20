"""
Course: Stochastic Simulation
Names: Petr Chalupský, Henry Zwart, Tika van Bennekum
Student IDs: 15719227, 15393879, 13392425
Assignement: Solving Traveling Salesman Problem using Simulated Annealing

File description:
    Simulations are performed with different chain lenghts.
    The error between these simulation results are calculated.
"""

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from tsp_simulated_annealing.cooling_schedules import Cooling
from tsp_simulated_annealing.data import Problem
from tsp_simulated_annealing.tsp import solve_tsp, tune_temperature


def run_simulation(
    par_idx,
    problem,
    initial_temp,
    final_temp,
    chain_lengths,
    n_iters,
    optimal_cost,
    base_seed,
):
    """
    Runs simulation, varying the chain length.
    """
    rng = np.random.default_rng(base_seed + par_idx)
    s0 = problem.random_solution(rng)
    error_matrix = np.empty((len(Cooling), len(chain_lengths)), dtype=np.float64)

    for algo_idx, cooling_alg in enumerate(Cooling):
        for i, c in enumerate(chain_lengths):
            result = solve_tsp(
                s0,
                problem,
                cooling_alg,
                init_temp=initial_temp,
                final_temp=final_temp,
                rng=rng,
                cool_time=n_iters,
                iters_per_temp=c,
            )
            error_matrix[algo_idx, i] = abs(
                problem.distance(result.states[-1]) - optimal_cost
            )
    return par_idx, error_matrix


def main():
    """
    Calls function to provide simulations with different chain lengths.
    Uses the data gathered to calculate the error between them.
    """
    base_seed = 125
    rng = np.random.default_rng(base_seed)

    # Load problem
    problem = Problem.MEDIUM.load()
    optimal_cost = problem.optimal_distance()

    # Varying chain length, balancing total work
    n_iters = 1000
    small_chains = np.arange(1, 50, dtype=np.int64)
    large_chains = np.arange(50, 1501, 50, dtype=np.int64)
    chain_lengths = np.concatenate((small_chains, large_chains))
    # chain_lengths = np.arange(0, 1001, 25, dtype=np.int64)
    # chain_lengths[0] += 1
    repeats = 30

    # Configure cooling algorithms
    temperatures = tune_temperature(
        problem.random_solution(rng),
        problem=problem,
        init_accept=0.8,
        rng=rng,
    )

    final_error = np.empty(
        (repeats, len(Cooling), len(chain_lengths)), dtype=np.float64
    )

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                run_simulation,
                par_idx,
                problem,
                temperatures.initial,
                temperatures.final,
                chain_lengths,
                n_iters,
                optimal_cost,
                base_seed,
            )
            for par_idx in range(repeats)
        ]

        for future in tqdm(as_completed(futures), total=repeats):
            r, error_matrix = future.result()
            final_error[r] = error_matrix

    np.save(Path("data/chain_length_error.npy"), final_error)

    metadata = {
        "n_iters": n_iters,
        "chain_lengths": chain_lengths.tolist(),
        "repeats": repeats,
        "temperature": {
            "initial": temperatures.initial,
            "final": temperatures.final,
            "initial_ci": temperatures.initial_ci,
            "final_ci": temperatures.final_ci,
        },
    }

    with Path("data/chain_length_error.meta").open("w") as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    main()
