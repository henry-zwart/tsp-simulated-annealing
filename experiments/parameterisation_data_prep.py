"""
Course: Stochastic Simulation
Names: Petr Chalupský, Henry Zwart, Tika van Bennekum
Student IDs: 15719227, 15393879, 13392425
Assignement: Solving Traveling Salesman Problem using Simulated Annealing

File description:
    Prepares the data for different parameterisation experiments.
"""

import numpy as np

from tsp_simulated_annealing.cooling_schedules import (
    fit_exponential,
    fit_inverse_log,
    fit_linear,
)
from tsp_simulated_annealing.data import Problem
from tsp_simulated_annealing.tsp import tune_temperature


def main():
    """
    Prepares the data for different parameterisation experiments.
    """
    rng = np.random.default_rng(25)
    init_accept = 0.8
    final_accept = 0.01
    n_samples_list = [501, 1001, 2001]

    p = Problem.MEDIUM
    print(p)
    problem = p.load()
    s0 = problem.random_solution(rng)

    # Estimate initial temperature
    temp_tune_results = tune_temperature(
        s0,
        problem,
        rng=rng,
        init_accept=init_accept,
        warmup_repeats=100,
    )
    print(" ------- Temperature -------")
    print(f"Initial: {init_accept*100:.2f}%")
    print(f"Final: {final_accept*100:.2f}%")
    print()

    print(f"T0: {temp_tune_results.initial:.2f} +- {temp_tune_results.initial_ci:.4f}")
    np.save(f"data/T_0_{p}", temp_tune_results.initial)
    print(" ------- Cooling -------")
    for n_samples in n_samples_list:
        eta = fit_linear(temp_tune_results.initial, temp_tune_results.final, n_samples)
        print(f"Linear: eta={eta:.4f}")

        alpha = fit_exponential(
            temp_tune_results.initial, temp_tune_results.final, n_samples
        )
        print(f"Exponential: alpha={alpha:.4f}")

        a, b = fit_inverse_log(
            temp_tune_results.initial, temp_tune_results.final, n_samples
        )
        print(f"Inverse log: a={a:.4f}, b={b:.8f}")

        print()
        np.save(f"data/{p}_{n_samples}_eta.npy", eta)
        np.save(f"data/{p}_{n_samples}_alpha.npy", alpha)
        np.save(f"data/{p}_{n_samples}_a.npy", a)
        np.save(f"data/{p}_{n_samples}_b.npy", b)


if __name__ == "__main__":
    main()
