import numpy as np

from tsp_simulated_annealing.cooling_schedules import Cooling
from tsp_simulated_annealing.data import Problem
from tsp_simulated_annealing.tsp import solve_tsp


def main():
    rng = np.random.default_rng(125)

    # Load small problem
    problem = Problem.SMALL.load()

    # Sample an initial state
    initial_solution = problem.random_solution(rng)

    # Solve for each cooling schedule, printing the final solution and cost
    n_samples = 10000
    cool_time = 10000
    for algorithm in Cooling:
        print(f"Solving with {algorithm}, chain-length = {n_samples}...")
        states = solve_tsp(initial_solution, problem, n_samples, algorithm, cool_time)
        final_state = states[-1]
        cost = problem.distance(final_state)
        print(f"Solution: {final_state}")
        print(f"Cost: {cost}")
        print(f"Error: {cost - problem.optimal_distance()}")
        print()


if __name__ == "__main__":
    main()
