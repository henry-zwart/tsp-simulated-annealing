"""
Course: Stochastic Simulation
Names: Petr Chalupský, Henry Zwart, Tika van Bennekum
Student IDs: 15719227, 15393879, 13392425
Assignement: Solving Traveling Salesman Problem using Simulated Annealing

File description:
    An experiment is performed for each cooling schedule.
"""

from cProfile import Profile
from pstats import SortKey, Stats

import numpy as np

from tsp_simulated_annealing.cooling_schedules import Cooling
from tsp_simulated_annealing.data import Problem
from tsp_simulated_annealing.tsp import solve_tsp


def main():
    """
    Performs an experiment for each cooling schedule.
    """
    rng = np.random.default_rng(125)

    # Load small problem
    problem = Problem.LARGE.load()

    # Sample an initial state
    initial_solution = problem.random_solution(rng)

    n_samples = 300
    cool_time = 1500
    for algorithm in Cooling:
        print("hi")
        _ = solve_tsp(
            initial_solution,
            problem,
            algorithm,
            cool_time,
            rng,
            n_samples,
        )


if __name__ == "__main__":
    with Profile() as profile:
        main()
        Stats(profile).strip_dirs().sort_stats(SortKey.TIME).print_stats(25)
