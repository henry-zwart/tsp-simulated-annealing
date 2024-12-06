import math
from pathlib import Path

small_path = Path("../../tsp_problems/eil51.tsp.txt")
medium_path = Path("../../tsp_problems/a280.tsp.txt")
large_path = Path("../../tsp_problems/pcb442.tsp.txt")

data_small = small_path.read_text().split("\n")[6:][:-2]
data_medium = medium_path.read_text().split("\n")[6:][:-2]
data_large = large_path.read_text().split("\n")[6:][:-2]


class Node:
    def __init__(self, id, x, y):
        self.id = int(id)
        self.x = float(x)
        self.y = float(y)


def data_to_nodes(data):
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
    for node in nodes:
        print(f"id: {node.id}, x: {node.x}, y: {node.y}")


def distance_two_nodes(id1, id2, coordinates):
    x1, y1 = coordinates[id1]
    x2, y2 = coordinates[id2]
    print(x1, y1, x2, y2)
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


nodes, coordinates = data_to_nodes(data_small)
