from pathlib import Path

small_path = Path("../../tsp_problems/eil51.tsp.txt")
medium_path = Path("../../tsp_problems/a280.tsp.txt")
large_path = Path("../../tsp_problems/pcb442.tsp.txt")

data_small = small_path.read_text().split("\n")[6:][:-2]
data_medium = medium_path.read_text().split("\n")[6:][:-2]
data_large = large_path.read_text().split("\n")[6:][:-2]


class Node:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y


def data_to_nodes(data):
    nodes = []
    for elem in data:
        info = elem.split(" ")
        info = " ".join(info).split()
        id, x, y = info
        nodes.append(Node(id, x, y))
    return nodes


def show_data(nodes):
    for node in nodes:
        print(f"id: {node.id}, x: {node.x}, y: {node.y}")


nodes = data_to_nodes(data_small)
show_data(nodes)
