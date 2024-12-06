"""
Course: Stochastic Simulation
Names: Petr Chalupský, Henry Zwart, Tika van Bennekum
Student IDs: 15719227, 15393879, 13392425
Assignement: Solving Traveling Salesman Problem using Simulated Annealing

File description:
    In this file all the code needed for the experiments is written.
    An experiment will call this file to run a simulation of the system.
"""

import math
import random
from pathlib import Path

import numpy as np

small_path = Path("../../tsp_problems/eil51.tsp.txt")
medium_path = Path("../../tsp_problems/a280.tsp.txt")
large_path = Path("../../tsp_problems/pcb442.tsp.txt")
test_path = Path("../../tsp_problems/test.txt")

data_small = small_path.read_text().split("\n")[6:][:-2]
data_medium = medium_path.read_text().split("\n")[6:][:-2]
data_large = large_path.read_text().split("\n")[6:][:-2]
data_test = test_path.read_text().split("\n")[6:][:-2]


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
    coordinates = [None]  # so the id corresponds to index
    for elem in data:
        info = elem.split(" ")
        info = " ".join(info).split()
        id, x, y = info
        nodes.append(Node(id, x, y))
        coordinates.append((float(x), float(y)))
    return (nodes, coordinates)


def show_data(nodes):
    """Shows all existing nodes and their attributes."""
    for node in nodes:
        print(f"id: {node.id}, x: {node.x}, y: {node.y}")


def distance_two_nodes(id1, id2, coordinates):
    """Calculates the distance between two nodes."""
    x1, y1 = coordinates[id1]
    x2, y2 = coordinates[id2]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def make_initial_solution(highest_id, seed):
    """Will produce a randomly generated initial solution for STP.
    The same seed will always produce the same intitial solution."""
    random.seed(seed)
    sol = np.linspace(1, highest_id, highest_id)
    random.shuffle(sol)
    return np.array([int(x) for x in sol])


def two_opt(solution, seed):
    """Two opt func. add description"""
    new_solution = solution.copy()
    print(f"Initial_solution: {solution}")
    random.seed(seed)

    index_edge_1 = np.random.randint(0, len(solution) - 1)
    while True:
        index_edge_2 = np.random.randint(0, len(solution) - 1)
        # checks edges are nonadjecent
        if index_edge_1 - 1 > index_edge_2 or index_edge_2 > index_edge_1 + 1:
            break

    # print(f"index_edge_1: {index_edge_1}, index_edge_2: {index_edge_2}")

    if index_edge_1 < index_edge_2:
        new_solution[index_edge_1 + 1 : index_edge_2 + 1] = solution[
            index_edge_1 + 1 : index_edge_2 + 1
        ][::-1]
    else:
        new_solution[index_edge_2 + 1 : index_edge_1 + 1] = solution[
            index_edge_2 + 1 : index_edge_1 + 1
        ][::-1]

    return np.array([int(x) for x in new_solution])


def distance_route(solution, coordinates):
    """For a certain route and a list of the point coordinates,
    the route distance is calculated."""
    total_dis = 0
    for i in range(len(solution) - 1):
        total_dis += distance_two_nodes(solution[i], solution[i + 1], coordinates)
    # adds dis between first and last node (making a cicle)
    total_dis += distance_two_nodes(solution[0], solution[-1], coordinates)
    return total_dis


# def main_algorithm(data):
#     nodes, coordinates = data_to_nodes(data)

#     highest_id = len(nodes)
#     seed = 123

#     cur_sol = make_initial_solution(highest_id, seed)
#     cur_dis = distance_route(cur_sol, coordinates)
#     while something:
#         new_sol = two_opt(cur_sol, seed)
#         new_dis = distance_route(new_sol, coordinates)

#         if new_dis < cur_dis:
#             cur_sol = new_sol
#             cur_dis = new_dis
#         else:
#             chance = cooling_schedule1
#             if chance:
#                 cur_sol = new_sol
#                 cur_dis = new_dis


# main_algorithm(data_test)
