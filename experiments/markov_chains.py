import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import numpy as np
from tqdm import tqdm

from tsp_simulated_annealing.cooling_schedules import (
    Cooling,
    exponential_cooling,
    fit_exponential,
    fit_inverse_log,
    fit_linear,
    inverse_exponential,
    inverse_linear,
    inverse_log_cooling,
    linear_cooling,
)
from tsp_simulated_annealing.data import Problem
from tsp_simulated_annealing.tsp import (
    sample_at_temperature,
    solve_tsp,
    tune_temperature,
)


def run_single_repeat(
    par_idx,
    problem,
    initial_temp,
    final_temp,
    cooling_methods,
    select_times,
    chain_length,
    cool_time,
    n_samples,
    optimal_cost,
    base_seed,
):
    rng = np.random.default_rng(base_seed + par_idx)
    initial_states = np.empty(
        (len(Cooling), cool_time + 1, len(problem.optimal_tour)),
        dtype=np.int64,
    )
    costs = np.empty(
        (len(Cooling), len(select_times[Cooling.Linear]), n_samples),
        dtype=np.int64,
    )
    error = np.empty_like(costs)
    temperatures = np.empty((len(Cooling), len(select_times[Cooling.Linear])))

    s0 = problem.random_solution(rng)
    for algo_idx, algo in enumerate(Cooling):
        # Run TSP solver for each cooling algorithm
        result = solve_tsp(
            s0,
            problem,
            algo,
            init_temp=initial_temp,
            final_temp=final_temp,
            rng=rng,
            cool_time=cool_time,
            iters_per_temp=chain_length,
        )
        initial_states[algo_idx] = result.states

        for t_idx, t in enumerate(select_times[algo]):
            temperatures[algo_idx, t_idx] = cooling_methods[algo_idx](t)
            samples = sample_at_temperature(
                result.states[t],
                problem,
                n_samples=n_samples,
                temperature=temperatures[algo_idx, t_idx],
                rng=rng,
            )
            sample_costs = problem.distance_many(samples)
            costs[algo_idx, t_idx] = sample_costs
            error[algo_idx, t_idx] = abs(sample_costs - optimal_cost)

    return par_idx, initial_states, costs, error, temperatures


def main(chain_length):
    base_seed = 125
    rng = np.random.default_rng(base_seed)

    # Load problem
    problem = Problem.MEDIUM.load()
    optimal_cost = problem.optimal_distance()

    # Varying chain length, balancing total work
    cool_time = 1000
    n_iterations = cool_time + 1
    repeats = 30
    n_samples = 30000

    # Configure cooling algorithms
    temp_tune_result = tune_temperature(
        problem.random_solution(rng),
        problem=problem,
        init_accept=0.8,
        rng=rng,
    )
    initial_temp = temp_tune_result.initial
    final_temp = temp_tune_result.final

    eta = fit_linear(initial_temp, final_temp, n_iterations)
    alpha = fit_exponential(initial_temp, final_temp, n_iterations)
    a, b = fit_inverse_log(initial_temp, final_temp, n_iterations)

    cooling_methods = [
        partial(linear_cooling, eta=eta, T_0=initial_temp),
        partial(exponential_cooling, alpha=alpha, T_0=initial_temp),
        partial(inverse_log_cooling, a=a, b=b),
    ]

    # Determine times, t1, t2, where exponential and linear temperatures
    #   equal inverse-log temperature.
    select_times = {Cooling.InverseLog: [1, 2]}
    inv_log_temps = [cooling_methods[2](t) for t in select_times[Cooling.InverseLog]]

    select_times[Cooling.Linear] = [
        inverse_linear(eta, initial_temp, temp) for temp in inv_log_temps
    ]
    select_times[Cooling.Exponential] = [
        inverse_exponential(alpha, initial_temp, temp) for temp in inv_log_temps
    ]

    all_initial_states = np.empty(
        (repeats, len(Cooling), n_iterations, len(problem.optimal_tour)),
        dtype=np.int64,
    )
    all_costs = np.empty(
        (repeats, len(Cooling), len(select_times[Cooling.Linear]), n_samples),
        dtype=np.int64,
    )
    all_errors = np.empty_like(all_costs)

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                run_single_repeat,
                par_idx,
                problem,
                initial_temp,
                final_temp,
                cooling_methods,
                select_times,
                chain_length,
                cool_time,
                n_samples,
                optimal_cost,
                base_seed,  # Ensure unique RNG seed per repeat
            )
            for par_idx in range(repeats)
        ]

        for future in tqdm(
            as_completed(futures),
            total=repeats,
            desc=f"Processing repeats (L_k={chain_length})",
        ):
            r, initial_states, costs, errors, temps = future.result()
            all_initial_states[r] = initial_states
            all_costs[r] = costs
            all_errors[r] = errors
            all_sample_temps = temps

    np.save(Path(f"data/chain_costs_{chain_length}.npy"), all_costs)
    np.save(Path(f"data/chain_error_{chain_length}.npy"), all_errors)

    metadata = {
        "cooling": {
            "initial": float(initial_temp),
            "final": float(final_temp),
            "initial_ci": float(temp_tune_result.initial_ci),
            "final_ci": float(temp_tune_result.final_ci),
            "linear": {"eta": float(eta), "temperatures": all_sample_temps[0].tolist()},
            "exponential": {
                "alpha": float(alpha),
                "temperatures": all_sample_temps[1].tolist(),
            },
            "inverse_log": {
                "a": float(a),
                "b": float(b),
                "temperatures": all_sample_temps[2].tolist(),
            },
        },
        "measure_times": select_times,
        "measure_chain_length": n_samples,
        "estimation_chain_length": chain_length,
        "temperature_steps": cool_time,
        "repeats": repeats,
    }

    with Path(f"data/markov_chains_{chain_length}.meta").open("w") as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    chain_length = int(sys.argv[1])
    main(chain_length)
