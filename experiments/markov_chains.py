import json
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


def main():
    base_seed = 125
    rng = np.random.default_rng(base_seed)

    # Load problem
    problem = Problem.MEDIUM.load()
    optimal_cost = problem.optimal_distance()

    # Varying chain length, balancing total work
    cool_time = 2000
    n_iterations = cool_time + 1
    chain_length = 1000
    repeats = 15
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

    select_times = {
        Cooling.Linear: [0, 750, 1250, 2000],
        Cooling.Exponential: [0, 750, 1250, 2000],
        Cooling.InverseLog: [0, 1, 5, 100],
    }

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
            as_completed(futures), total=repeats, desc="Processing repeats"
        ):
            r, initial_states, costs, errors, temps = future.result()
            all_initial_states[r] = initial_states
            all_costs[r] = costs
            all_errors[r] = errors
            all_sample_temps = temps

    np.save(Path("data/chain_costs.npy"), all_costs)
    np.save(Path("data/chain_error.npy"), all_errors)

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

    with Path("data/markov_chains.meta").open("w") as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    main()
