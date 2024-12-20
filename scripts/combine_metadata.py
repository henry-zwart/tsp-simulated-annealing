"""
Course: Stochastic Simulation
Names: Petr Chalupský, Henry Zwart, Tika van Bennekum
Student IDs: 15719227, 15393879, 13392425
Assignement: TSP Simulated annealing

File description:
    Script to combine metadata.
"""

import json
import sys
from pathlib import Path


def main(data_dir: Path, results_dir: Path, chain_lengths: list[int]):
    """Function to lead Jason files."""
    big_meta = {}

    with (data_dir / "chain_length_error.meta").open("r") as f:
        big_meta["chain_length_error"] = json.load(f)

    big_meta["markov_chains"] = {}
    for chain_length in chain_lengths:
        with (data_dir / f"markov_chains_{chain_length}.meta").open("r") as f:
            big_meta["markov_chains"][chain_length] = json.load(f)

    with (results_dir / "experiment_metadata.json").open("w") as f:
        json.dump(big_meta, f)


if __name__ == "__main__":
    DATA_DIR = Path("data")
    RESULTS_DIR = Path("results")
    chain_lengths = [int(x) for x in sys.argv[1:]]
    main(DATA_DIR, RESULTS_DIR, chain_lengths)
