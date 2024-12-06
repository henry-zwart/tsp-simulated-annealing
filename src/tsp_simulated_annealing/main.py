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

data_small = small_path.read_text().split("\n")[6:][:-2]
data_medium = medium_path.read_text().split("\n")[6:][:-2]
data_large = large_path.read_text().split("\n")[6:][:-2]


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
    coordinates = [None]
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
    print(x1, y1, x2, y2)
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def make_initial_solution(highest_id, seed):
    """Will produce a randomly generated initial solution for STP.
    The same seed will always produce the same intitial solution."""
    random.seed(seed)
    sol = np.linspace(1, highest_id, highest_id)
    random.shuffle(sol)
    return sol


nodes, coordinates = data_to_nodes(data_small)
