import json
from functools import partial
from pathlib import Path

import numpy as np
from tqdm import trange

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


def main():
    rng = np.random.default_rng(125)

    # Load problem
    problem = Problem.MEDIUM.load()
    optimal_cost = problem.optimal_distance()

    # Varying chain length, balancing total work
    cool_time = 1500
    chain_length = 1000
    repeats = 5

    initial_states = np.empty(
        (repeats, len(Cooling), cool_time + 1, len(problem.optimal_tour)),
        dtype=np.int64,
    )

    # Configure cooling algorithms
    temp_tune_result = tune_temperature(
        problem.random_solution(rng),
        problem=problem,
        init_accept=0.95,
        rng=rng,
    )
    initial_temp = temp_tune_result.initial
    final_temp = temp_tune_result.final

    for r in trange(repeats):
        s0 = problem.random_solution(rng)
        initial_states[r, :] = s0
        for algo_idx, algo in enumerate(Cooling):
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
            initial_states[r, algo_idx, 1:] = result.states

    # Create temperature functions
    eta = fit_linear(initial_temp, final_temp, cool_time)
    alpha = fit_exponential(initial_temp, final_temp, cool_time)
    a, b = fit_inverse_log(initial_temp, final_temp, cool_time)

    cooling_methods = [
        partial(linear_cooling, eta=eta, T_0=initial_temp),
        partial(exponential_cooling, alpha=alpha, T_0=initial_temp),
        partial(inverse_log_cooling, a=a, b=b),
    ]

    # Sample from state distribution for a few select times,
    #   record cost and error at each
    select_times = [0, 15, 150, 1500]
    n_samples = 10000
    costs = np.empty(
        (repeats, len(Cooling), len(select_times), n_samples), dtype=np.int64
    )
    error = np.empty_like(costs)
    for r in trange(repeats):
        for i in range(len(Cooling)):
            for t_idx, t in enumerate(select_times):
                samples = sample_at_temperature(
                    initial_states[r, i, t],
                    problem,
                    n_samples=n_samples,
                    temperature=cooling_methods[i](t),
                    rng=rng,
                )
                sample_costs = problem.distance_many(samples)
                sample_errors = abs(sample_costs - optimal_cost)
                costs[r, i, t_idx] = sample_costs
                error[r, i, t_idx] = sample_errors

    np.save(Path("data/chain_costs.npy"), costs)
    np.save(Path("data/chain_error.npy"), error)

    metadata = {
        "cooling": {
            "initial": float(initial_temp),
            "final": float(final_temp),
            "initial_ci": float(temp_tune_result.initial_ci),
            "final_ci": float(temp_tune_result.final_ci),
            "linear": {"eta": float(eta)},
            "exponential": {"alpha": float(alpha)},
            "inverse_log": {"a": float(a), "b": float(b)},
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
