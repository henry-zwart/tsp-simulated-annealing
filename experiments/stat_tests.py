import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_ind_from_stats

from tsp_simulated_annealing.cooling_schedules import Cooling


def main():
    RESULTS_PATH = Path("results")
    DATA_PATH = Path("data")
    with (DATA_PATH / "cooling.meta").open("r") as f:
        metadata = json.load(f)

    test_results = {}
    for cooling in Cooling:
        sample_size = metadata["repeats"]
        mean_1 = metadata[cooling]["500"]["final_error_mean"]
        mean_2 = metadata[cooling]["2000"]["final_error_mean"]
        std_1 = metadata[cooling]["500"]["final_error_ci"] * np.sqrt(sample_size)
        std_2 = metadata[cooling]["2000"]["final_error_ci"] * np.sqrt(sample_size)

        t_stat, p_value = ttest_ind_from_stats(
            mean_1, std_1, mean_2, std_2, sample_size, sample_size, equal_var=False
        )

        test_results[cooling] = {"t_stat": t_stat, "p_value": p_value}

    with (RESULTS_PATH / "tests.json").open("w") as f:
        json.dump(test_results, f)


if __name__ == "__main__":
    main()
