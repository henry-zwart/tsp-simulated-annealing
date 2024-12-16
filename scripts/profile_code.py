from cProfile import Profile
from pstats import SortKey, Stats

import numpy as np

from tsp_simulated_annealing.cooling_schedules import Cooling
from tsp_simulated_annealing.data import Problem
from tsp_simulated_annealing.tsp import solve_tsp


def main():
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
            final_accept=0.001,
        )


if __name__ == "__main__":
    with Profile() as profile:
        main()
        Stats(profile).strip_dirs().sort_stats(SortKey.TIME).print_stats(25)
