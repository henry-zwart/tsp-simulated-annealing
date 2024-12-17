from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from tsp_simulated_annealing.cooling_schedules import Cooling
from tsp_simulated_annealing.data import Problem
from tsp_simulated_annealing.tsp import solve_tsp, tune_temperature


def run_single_repeat(
    par_idx,
    problem,
    initial_temp,
    final_temp,
    chain_length,
    n_iters,
    optimal_cost,
    base_seed,
):
    rng = np.random.default_rng(base_seed + par_idx)
    s0 = problem.random_solution(rng)
    error = {n: [] for n in n_iters}
    for n in n_iters:
        for algorithm in Cooling:
            results = solve_tsp(
                s0,
                problem,
                algorithm,
                init_temp=initial_temp,
                final_temp=final_temp,
                rng=rng,
                cool_time=n,
                iters_per_temp=chain_length,
            )
            error[n].append(abs(problem.distance_many(results.states) - optimal_cost))

    return error


def main():
    base_seed = 125
    rng = np.random.default_rng(base_seed)

    # Load small problem
    problem = Problem.MEDIUM.load()

    # Sample an initial state
    optimal_dist = problem.optimal_distance()

    # Solve for each cooling schedule, printing the final solution and cost
    chain_length = 1000  # How long we stay at one temperature
    repeats = 30
    temperatures = tune_temperature(
        problem.random_solution(rng),
        problem,
        init_accept=0.8,
        rng=rng,
    )
    n_iters = np.array([500, 1000, 2000])

    all_errors = {n: [] for n in n_iters}
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                run_single_repeat,
                par_idx,
                problem,
                temperatures.initial,
                temperatures.final,
                chain_length,
                n_iters,
                optimal_dist,
                base_seed,
            )
            for par_idx in range(repeats)
        ]

        for future in tqdm(as_completed(futures), total=repeats):
            errors = future.result()
            for n, e in errors.items():
                all_errors[n].append(e)

    for n, e in all_errors.items():
        all_errors[n] = np.array(e)

    for n, e in all_errors.items():
        for c_i, c in enumerate(Cooling):
            np.save(Path(f"data/a280_{c}_{n}_errors.npy"), e[:, c_i].copy())


if __name__ == "__main__":
    main()
