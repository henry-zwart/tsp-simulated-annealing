import json
from pathlib import Path

import numpy as np
from tqdm import trange

from tsp_simulated_annealing.cooling_schedules import Cooling
from tsp_simulated_annealing.data import Problem
from tsp_simulated_annealing.tsp import solve_tsp, tune_temperature


def main():
    rng = np.random.default_rng(125)

    # Load problem
    problem = Problem.MEDIUM.load()
    optimal_cost = problem.optimal_distance()

    # Varying chain length, balancing total work
    n_iters = 15
    chain_lengths = np.arange(0, 301, 25, dtype=np.int64)
    chain_lengths[0] += 1
    repeats = 15

    # Configure cooling algorithms
    temperatures = tune_temperature(
        problem.random_solution(rng),
        problem=problem,
        init_accept=0.95,
        rng=rng,
    )

    final_error = np.empty(
        (repeats, len(Cooling), len(chain_lengths)), dtype=np.float64
    )

    for r in trange(repeats):
        s0 = problem.random_solution(rng)
        for algo_idx, cooling_alg in enumerate(Cooling):
            for i, c in enumerate(chain_lengths):
                result = solve_tsp(
                    s0,
                    problem,
                    cooling_alg,
                    init_temp=temperatures.initial,
                    final_temp=temperatures.final,
                    rng=rng,
                    cool_time=n_iters,
                    iters_per_temp=c,
                )
                final_error[r, algo_idx, i] = abs(
                    problem.distance(result.states[-1]) - optimal_cost
                )

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
    # print(final_error.mean(axis=0))
    # np.save(Path("data/inv_log_error.npy"), final_error)

    # fig, ax = plt.subplots()
    # mean_error = final_error.mean(axis=0)
    # ci = 1.97 * final_error.std(ddof=1, axis=0) / np.sqrt(repeats)
    # for i, algo in enumerate(Cooling):
    #    ax.plot(chain_lengths, mean_error[i], label=algo)
    #    ax.fill_between(
    #        chain_lengths, mean_error[i] - ci[i], mean_error[i] + ci[i], alpha=0.3
    #    )
    # fig.legend()
    # fig.tight_layout()
    # fig.savefig("inv_log_err.pdf", dpi=300)


if __name__ == "__main__":
    main()
