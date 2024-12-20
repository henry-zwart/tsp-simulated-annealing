"""
Course: Stochastic Simulation
Names: Petr Chalupský, Henry Zwart, Tika van Bennekum
Student IDs: 15719227, 15393879, 13392425
Assignement: Solving Traveling Salesman Problem using Simulated Annealing

File description:
    For each problem, loads the problem and associated solution.
    Calculates cost for the solution.
"""

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
