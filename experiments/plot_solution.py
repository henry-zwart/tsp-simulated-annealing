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

    plot_states = [0, 250, 1000, -1]

    # Solve for each cooling schedule, printing the final solution and cost
    n_samples = 300
    cool_time = 1500
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
        final_state = states[-1]
        cost = problem.distance(final_state)
        print(f"Solution: {final_state}")
        print(f"Cost: {cost}")
        print(f"Error: {cost - problem.optimal_distance()}")
        print()

        fig, axes = plt.subplots(
            1,
            4,
            sharex=True,
            sharey=True,
            subplot_kw=dict(box_aspect=1),
            figsize=(12, 4),
        )
        for state_idx, ax in zip(plot_states, axes, strict=False):
            solution_coords = problem.locations[states[state_idx]]
            ax.scatter(
                problem.locations[:, 0],
                problem.locations[:, 1],
                c="red",
                s=5,
                label="Cities",
            )
            ax.plot(
                solution_coords[:, 0],
                solution_coords[:, 1],
                "-",
                label="Solution",
                linewidth=0.5,
            )

        fig.savefig(f"solutions_{algorithm}.png", dpi=700)


if __name__ == "__main__":
    main()
