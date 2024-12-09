from functools import partial
from pathlib import Path

import numpy as np

from tsp_simulated_annealing.cooling_schedules import (
    inverse_log_cooling,
)
from tsp_simulated_annealing.main import main_algorithm

if __name__ == "__main__":
    cooling_schedule = partial(inverse_log_cooling, a=5, b=1)

    rng = np.random.default_rng(42)
    small_path = Path("../tsp_problems/eil51.tsp.txt")
    data_small = small_path.read_text().split("\n")[6:][:-2]

    solution, dist = main_algorithm(data_small, 100, cooling_schedule, 20, rng)
    print(f"solution: {solution}, distance {dist} ")
