import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tsp_simulated_annealing.cooling_schedules import Cooling


def main():
    DATA_DIR = Path("data")
    RESULTS_DIR = Path("results")
    FIGURES_DIR = Path("results/figures")

    # Estimation error vs. chain length
    with (DATA_DIR / "chain_length_error.meta").open("r") as f:
        metadata = json.load(f)
    error = np.load(DATA_DIR / "chain_length_error.npy")
    mean_error = error.mean(axis=0)
    pct = np.percentile(error, [5, 95], axis=0)
    fig, ax = plt.subplots(layout="constrained")
    for i, cooling in enumerate(Cooling):
        ax.plot(metadata["chain_lengths"], mean_error[i], label=cooling)
        ax.fill_between(metadata["chain_lengths"], pct[0, i], pct[1, i], alpha=0.3)
    fig.savefig(FIGURES_DIR / "chain_length_error.pdf", dpi=300)

    # Trace plots
    with (DATA_DIR / "markov_chains.meta").open("r") as f:
        metadata = json.load(f)
    chain_costs = np.load(DATA_DIR / "chain_costs.npy")
    samples = np.arange(1, metadata["measure_chain_length"] + 1)
    for i, cooling in enumerate(Cooling):
        fig, axes = plt.subplots(
            1,
            len(metadata["measure_times"]),
            layout="constrained",
            figsize=(6.8, 2),
            sharex=True,
        )
        for t_i, _t in enumerate(metadata["measure_times"]):
            chain = chain_costs[:, i, t_i]
            axes[t_i].plot(samples, chain.T, linewidth=0.5)
            axes[t_i].set_xlabel("Sample")
            axes[t_i].set_title("Temperature = ??")

        axes[0].set_ylabel("Cost")
        fig.suptitle(f"{cooling} cooling trace")
        fig.savefig(FIGURES_DIR / f"{cooling}_trace.pdf", dpi=300)

    # Distribution plots
    chain_split_len = metadata["measure_chain_length"] // 3
    for i, cooling in enumerate(Cooling):
        fig, axes = plt.subplots(
            2,
            len(metadata["measure_times"]),
            layout="constrained",
            figsize=(6.8, 5),
            sharex="col",
        )
        for t_i, _t in enumerate(metadata["measure_times"]):
            chain = chain_costs[:, i, t_i]
            axes[0, t_i].hist(chain.T[:chain_split_len], density=True)
            axes[1, t_i].hist(chain.T[-chain_split_len:], density=True)
            axes[1, t_i].set_xlabel("Cost")
            axes[0, t_i].set_title("Temperature = ??")

        fig.supylabel("PDF")
        fig.savefig(FIGURES_DIR / f"{cooling}_dist.pdf", dpi=300)

    metadata = {"chain_length_error": {"percentiles": {"lower": 5, "upper": 95}}}

    with (RESULTS_DIR / "plot_metadata.json").open("w") as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    main()
