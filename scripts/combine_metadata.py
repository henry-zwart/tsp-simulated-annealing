"""
Course: Stochastic Simulation
Names: Petr Chalupský, Henry Zwart, Tika van Bennekum
Student IDs: 15719227, 15393879, 13392425
Assignement: TSP Simulated annealing

File description:
    Script to combine metadata.
"""

import json
from pathlib import Path


def main(data_dir: Path, results_dir: Path):
    """Function to lead Jason files."""
    big_meta = {}

    with (data_dir / "chain_length_error.meta").open("r") as f:
        big_meta["chain_length_error"] = json.load(f)

    with (data_dir / "markov_chains.meta").open("r") as f:
        big_meta["markov_chains"] = json.load(f)

    with (results_dir / "experiment_metadata.json").open("w") as f:
        json.dump(big_meta, f)


if __name__ == "__main__":
    DATA_DIR = Path("data")
    RESULTS_DIR = Path("results")
    main(DATA_DIR, RESULTS_DIR)
