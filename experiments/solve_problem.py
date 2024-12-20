import matplotlib.pyplot as plt
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

    fig, ax = plt.subplots()

    # Solve for each cooling schedule, printing the final solution and cost
    n_samples = 500
    cool_time = 2000
    for algorithm in Cooling:
        print(f"Solving with {algorithm}, chain-length = {n_samples}...")
        for _ in range(5):
            results = solve_tsp(
                initial_solution,
                problem,
                algorithm,
                cool_time,
                rng,
                n_samples,
                init_accept=0.8,
            )
            states = results.states
            final_state = states[-1]
            cost = problem.distance(final_state)
            # print(f"Solution: {final_state}")
            print(f"Cost: {cost}")
            print(f"Error: {cost - problem.optimal_distance()}")
            print()

        ax.plot(np.arange(cool_time), problem.distance_many(states))

    fig.savefig("cost.png", dpi=700)


if __name__ == "__main__":
    main()
