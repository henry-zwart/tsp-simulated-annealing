import numpy as np

from tsp_simulated_annealing.cooling_schedules import Cooling
from tsp_simulated_annealing.data import Problem
from tsp_simulated_annealing.tsp import solve_tsp


def main():
    rng = np.random.default_rng(125)

    # Load small problem
    problem = Problem.MEDIUM.load()

    # Sample an initial state
    initial_solution = problem.random_solution(rng)
    optimal_dist = problem.optimal_distance()

    # Solve for each cooling schedule, printing the final solution and cost
    n_samples = 1  # How long we stay at one temperature
    for cool_time in [500, 1000, 2000]:
        for algorithm in Cooling:
            print(f"Solving with {algorithm}, chain-length = {n_samples}...")
            states = solve_tsp(
                initial_solution,
                problem,
                algorithm,
                cool_time,
                n_samples,
                final_accept=0.001,
            )
            errors = []
            for i in range(len(states)):
                cost = problem.distance(states[i])
                error = cost - optimal_dist
                errors.append(error)
            np.save(f"data/a280_{algorithm}_{cool_time}_errors.npy", errors)
            print()


if __name__ == "__main__":
    main()
