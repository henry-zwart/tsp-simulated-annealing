"""
Course: Stochastic Simulation
Names: Petr Chalupský, Henry Zwart, Tika van Bennekum
Student IDs: 15719227, 15393879, 13392425
Assignement: Solving Traveling Salesman Problem using Simulated Annealing

File description:
    In this file the main code functions needed for the experiments is written.
    Simmulated annealing usign two-opt is implemented.
    An experiment will call this file to run a simulation of the system.
"""

import math
from pathlib import Path

import numpy as np
from numba import njit

rng = np.random.default_rng(seed=42)

small_path = Path("tsp_problems/eil51.tsp.txt")
medium_path = Path("tsp_problems/a280.tsp.txt")
large_path = Path("tsp_problems/pcb442.tsp.txt")
test_path = Path("tsp_problems/test.txt")

small_path_sol = Path("tsp_problems/eil51.opt.tour.txt")
medium_path_sol = Path("tsp_problems/a280.opt.tour.txt")
large_path_sol = Path("tsp_problems/pcb442.opt.tour.txt")

data_small = small_path.read_text().split("\n")[6:][:-2]
data_medium = medium_path.read_text().split("\n")[6:][:-2]
data_large = large_path.read_text().split("\n")[6:][:-2]
data_test = test_path.read_text().split("\n")[6:][:-2]

optimal_route_small = np.array(
    [int(x) for x in small_path_sol.read_text().split("\n")[6:][:-3]]
)
optimal_route_medium = np.array(
    [int(x) for x in medium_path_sol.read_text().split("\n")[5:][:-2]]
)
optimal_route_large = np.array(
    [int(x) for x in large_path_sol.read_text().split("\n")[6:][:-3]]
)


class Node:
    """Class for a node, has attributes: id, x-coordinate, y-coordinate."""

    def __init__(self, id, x, y):
        self.id = int(id)
        self.x = float(x)
        self.y = float(y)


def data_to_nodes(data):
    """Takes a data file, and transforms it into
    a list of Nodes and a list of coordinates."""
    nodes = []
    coordinates = []
    for elem in data:
        info = elem.split(" ")
        info = " ".join(info).split()
        id, x, y = info
        # id - 1 so the id corresponds to index
        nodes.append(Node(str(int(id) - 1), x, y))
        coordinates.append((float(x), float(y)))
    return (nodes, coordinates)


def show_data(nodes):
    """Shows all existing nodes and their attributes."""
    for node in nodes:
        print(f"id: {node.id}, x: {node.x}, y: {node.y}")


@njit()
def distance_two_nodes(id1, id2, coordinates):
    """Calculates the distance between two nodes."""
    x1, y1 = coordinates[id1]
    x2, y2 = coordinates[id2]
    return round(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))


def make_initial_solution(highest_id, rng):
    """Will produce a randomly generated initial solution for STP.
    The same seed will always produce the same intitial solution."""
    sol = np.linspace(0, highest_id, highest_id)
    rng.shuffle(sol)
    sol = np.append(sol, sol[0])
    return np.array([int(x) for x in sol])


def sample_non_adjacent(low, high, rng: np.random.Generator):
    """Samples two non-adjecent edges."""
    i = rng.integers(low, high)
    while True:
        j = rng.integers(low, high)
        if abs(i - j) > 1:
            break
    if i < j:
        return i, j
    return j, i


@njit()
def reverse_route(solution, idx_1, idx_2, locations, new_solution):
    """Reverses segment of route resulting form the removal of two edges."""
    new_solution[idx_1 : idx_2 + 1] = new_solution[idx_1 : idx_2 + 1][::-1]
    c1, c2 = solution[idx_1 - 1], solution[idx_1]
    c3, c4 = solution[idx_2], solution[idx_2 + 1]

    old_edge_cost = distance_two_nodes(c1, c2, locations) + distance_two_nodes(
        c3, c4, locations
    )
    new_edge_cost = distance_two_nodes(c1, c3, locations) + distance_two_nodes(
        c2, c4, locations
    )
    return new_solution, (new_edge_cost - old_edge_cost)


def two_opt(solution, locations, rng: np.random.Generator):
    """Two-opt operation: removes two edges from list and reverses the list inbetween
    creating two new edges."""
    idx_1, idx_2 = sample_non_adjacent(1, len(solution) - 1, rng)
    new_solution = solution.copy()

    new_solution, cost_update = reverse_route(
        solution, idx_1, idx_2, locations, new_solution
    )

    return new_solution, cost_update


def distance_route(solution, coordinates) -> int:
    """For a certain route and a list of the point coordinates,
    the route distance is calculated."""
    total_dis = 0
    for i in range(len(solution) - 1):
        total_dis += distance_two_nodes(solution[i], solution[i + 1], coordinates)
    return int(total_dis)


def main_algorithm(data, markov_chain_length, cooling_schedule, T_0, rng, acceptance):
    nodes, coordinates = data_to_nodes(data)
    """
    Simmulated annealing algorithm.
    """
    highest_id = len(nodes) - 1
    T = T_0
    cur_sol = make_initial_solution(highest_id, rng)
    print("initial sol: ", cur_sol)
    cur_dis = distance_route(cur_sol, coordinates)
    for t in range(1, markov_chain_length):
        new_sol, dist_delta = two_opt(cur_sol, coordinates, rng)
        new_dis = cur_dis + dist_delta

        if new_dis < cur_dis:
            cur_sol = new_sol
            cur_dis = new_dis
        else:
            chance = acceptance(new_dis, cur_dis, T)
            if rng.uniform(0, 1) <= chance:
                cur_sol = new_sol
                cur_dis = new_dis
        # Cool after each loop
        T = cooling_schedule(t)

    return cur_sol, cur_dis
