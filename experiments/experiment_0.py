from functools import partial
from pathlib import Path

from tsp_simulated_annealing.cooling_schedules import (
    inverse_log_cooling,
)
from tsp_simulated_annealing.main import main_algorithm

cooling_schedule = partial(inverse_log_cooling, a=5, b=1)


test_path_2 = Path("../tsp_problems/eil51.tsp.txt")
print(test_path_2)

data_small = test_path_2.read_text().split("\n")[6:][:-2]

solution, dist = main_algorithm(data_small, 100, cooling_schedule, 20)

print(f"solution: {solution}, distance {dist} ")
