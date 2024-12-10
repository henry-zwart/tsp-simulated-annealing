import numpy as np

from tsp_simulated_annealing.data import Problem


def main():
    """
    For each problem, load the problem and associated solution.
    Calculate cost for the solution.
    """
    rng = np.random.default_rng(125)
    problem = Problem.MEDIUM.load()
    print(problem.cities)
    print(problem.locations)
    print(problem.optimal_tour)
    initial_solution = problem.random_solution(rng)
    print(f"Initial distance: {problem.distance(initial_solution)}")
    print(f"Optimal distance: {problem.optimal_distance()}")


if __name__ == "__main__":
    main()
