"""
Course: Stochastic Simulation
Names: Petr Chalupský, Henry Zwart, Tika van Bennekum
Student IDs: 15719227, 15393879, 13392425
Assignement: Solving Traveling Salesman Problem using Simulated Annealing

File description:
    This file handles all data.
    Classes are written to handle the data easily.
"""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np

from tsp_simulated_annealing.main import distance_route

TSP_PROBLEMS = Path("tsp_problems")


@dataclass
class ProblemData:
    cities: np.ndarray
    locations: np.ndarray
    optimal_tour: np.ndarray

    def random_solution(self, rng: np.random.Generator) -> np.ndarray:
        """Generate a random route through the cities."""
        cities = self.cities.copy()
        rng.shuffle(cities)
        return np.concat((cities, cities[:1]))

    def distance(self, route: np.ndarray) -> int:
        """Calculate the total distance (cost) of a solution."""
        return distance_route(route, self.locations)

    def distance_many(self, routes: np.ndarray) -> np.ndarray:
        """Calculate total distance (cost) for each solution in an array."""
        distances = [distance_route(r, self.locations) for r in routes]
        return np.array(distances)

    def optimal_distance(self) -> int:
        """Calculate the optimal route cost."""
        return self.distance(self.optimal_tour)


class Problem(StrEnum):
    SMALL = "eil51"
    MEDIUM = "a280"
    LARGE = "pcb442"

    def load(self) -> ProblemData:
        """
        Load TSP problem, returning the city IDs and locations as numpy arrays.
        """
        with self.problem_path().open("r") as f:
            data = [d.strip().split(" ") for d in f.readlines()]

        # Some lines of a280 have extra spaces
        data = [[x for x in row if x != ""] for row in data]

        # Separate out cities and renormalise IDs
        cities = np.array([city for city, *_ in data], dtype=np.int64)
        normalised_cities = cities - 1

        # Separate out locations
        locations = np.array([location for _, *location in data], dtype=np.float64)

        # Load optimal route, and return as unified dataclass
        solution = self.load_solution()
        return ProblemData(normalised_cities, locations, solution)

    def load_solution(self) -> np.ndarray:
        """
        Load the optimal tour for the TSP problem.
        """
        with self.solution_path().open("r") as f:
            tour = f.readlines()

        # Add the first city to the end, to match our expected format
        tour.append(tour[0])
        normalised_tour = np.array(tour, dtype=np.int64) - 1
        return normalised_tour

    def problem_path(self) -> Path:
        """
        Load path.
        """
        return TSP_PROBLEMS / f"{self}.tsp.txt"

    def solution_path(self) -> Path:
        """
        Load optimal route.
        """
        return TSP_PROBLEMS / f"{self}.opt.tour.txt"
