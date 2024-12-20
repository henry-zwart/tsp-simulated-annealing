"""
Course: Stochastic Simulation
Names: Petr Chalupský, Henry Zwart, Tika van Bennekum
Student IDs: 15719227, 15393879, 13392425
Assignement: Solving Traveling Salesman Problem using Simulated Annealing

File description:
    Produces plots that didn't fall into a certain category.
    Plots are created from the experiment results.
"""

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from tsp_simulated_annealing.cooling_schedules import Cooling


def main(chain_lengths: list[int]):
    """
    Produces plots for experiment results.
    """
    FONT_SIZE_SMALL = 9
    FONT_SIZE_DEFAULT = 10

    plt.rc("font", family="Georgia")
    plt.rc("font", weight="normal")  # controls default font
    plt.rc("mathtext", fontset="stix")
    plt.rc("font", size=FONT_SIZE_DEFAULT)  # controls default text sizes
    plt.rc("axes", titlesize=FONT_SIZE_DEFAULT)  # fontsize of the axes title
    plt.rc("axes", labelsize=FONT_SIZE_DEFAULT)  # fontsize of the x and y labels
    plt.rc("figure", labelsize=FONT_SIZE_DEFAULT)
    plt.rc("xtick", labelsize=FONT_SIZE_SMALL)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=FONT_SIZE_SMALL)  # fontsize of the tick labels

    plt.rc("figure", dpi=700)  # fix output resolution

    sns.set_context(
        "paper",
        rc={
            "axes.linewidth": 0.5,
            "axes.labelsize": FONT_SIZE_DEFAULT,
            "axes.titlesize": FONT_SIZE_DEFAULT,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "ytick.minor.width": 0.4,
        },
    )

    DATA_DIR = Path("data")
    RESULTS_DIR = Path("results")
    FIGURES_DIR = Path("results/figures")

    palette = list(mcolors.TABLEAU_COLORS)

    # Estimation error vs. chain length
    with (DATA_DIR / "chain_length_error.meta").open("r") as f:
        metadata = json.load(f)
    error = np.load(DATA_DIR / "chain_length_error.npy")
    mean_error = error.mean(axis=0)
    a = error.std(ddof=1, axis=0) / np.sqrt(error.shape[0])
    fig, ax = plt.subplots(layout="constrained", figsize=(3, 2.5))
    for i, cooling in enumerate(Cooling):
        ax.plot(
            metadata["chain_lengths"],
            mean_error[i],
            label=cooling.replace("_", " ").title(),
        )
        ax.fill_between(
            metadata["chain_lengths"],
            mean_error[i] - a[i],
            mean_error[i] + a[i],
            alpha=0.3,
        )
    ax.set_xlabel(r"$L_k$")
    ax.set_ylabel("Log error at $t_{1000}$")
    ax.set_yscale("log")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.legend(frameon=False)
    fig.savefig(FIGURES_DIR / "chain_length_error.pdf", dpi=700, bbox_inches="tight")

    fig, axes = plt.subplots(
        3,
        2,
        layout="constrained",
        figsize=(3.25, 5),
        sharex=True,
        sharey="row",
    )
    with (DATA_DIR / "markov_chains_200.meta").open("r") as f:
        metadata = json.load(f)

    chain_error_200 = np.load(DATA_DIR / "chain_error_200.npy")[..., 1:3, :]
    chain_error_1500 = np.load(DATA_DIR / "chain_error_1500.npy")[..., 1:3, :]
    for c_i, cooling in enumerate(Cooling):
        measure_times = metadata["measure_times"][cooling][1:3]
        temperatures = metadata["cooling"][cooling]["temperatures"][1:3]
        samples = np.arange(1, metadata["measure_chain_length"] + 1)

        row = 2 - c_i
        for t_i, t in enumerate(temperatures):
            axis = axes[row, t_i]
            for chain_error in (chain_error_200, chain_error_1500):
                chain = chain_error[:, c_i, t_i]
                mean = chain.mean(axis=0)
                std = chain.std(axis=0, ddof=1)
                lower = np.clip(mean - 2 * std, a_min=0, a_max=None)
                upper = mean + 2 * std
                axis.plot(samples, mean, linewidth=1)
                axis.fill_between(samples, lower, upper, alpha=0.3)

            axis.xaxis.set_label_position("top")
            axis.set_xlabel(r"$T($" + str(measure_times[t_i]) + r"$) = $" + f"{t:.2f}")

        axes[row, -1].text(
            1.03,
            0.5,
            cooling.replace("_", " ").title(),
            transform=axes[row, -1].transAxes,
            va="center",
            rotation=90,
        )
        if cooling == Cooling.InverseLog:
            axes[row, 0].set_yscale("log")
            axes[row, 0].set_yticks([2000, 10000])
            axes[row, 0].get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        else:
            axes[row, 0].set_ylim(0, None)

    for i in range(2):
        ticks = np.arange(10000, 30001, 10000)
        axes[-1, i].set_xticks(
            ticks,
            labels=map(str, ticks),
            rotation=45,
            ha="right",
        )

    for ax in axes.flatten():
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    fig.supylabel("Error")
    fig.supxlabel("Samples")

    # Add legend
    handles = [
        plt.Line2D([0], [0], color="tab:blue", label="$L_k = 200$"),
        plt.Line2D([0], [0], color="tab:orange", label="$L_k = 1500$"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.57, 1.05),
        frameon=False,
    )

    # Adjust spacing between subplots
    fig.get_layout_engine().set(w_pad=4 / 72, h_pad=8 / 72, hspace=0, wspace=0)

    fig.savefig(FIGURES_DIR / "traces.pdf", dpi=300, bbox_inches="tight")

    # === Plot inverse-log trace plot
    for chain_length in (200, 1500):
        # Only get inverse-log, for subset of times
        chain_error = np.load(DATA_DIR / f"chain_error_{chain_length}.npy")[
            ..., [0, 1, 3, 4], :
        ]

        for c_i, c in enumerate(Cooling):
            measure_times = np.array(metadata["measure_times"][c])[[0, 1, 3, 4]]
            temperatures = np.array(metadata["cooling"][c]["temperatures"])[
                [0, 1, 3, 4]
            ]
            samples = np.arange(1, metadata["measure_chain_length"] + 1)

            fig, axes = plt.subplots(
                1, 4, layout="constrained", figsize=(6.5, 2.25), sharex=True
            )

            for t_i, t in enumerate(temperatures):
                axis = axes[t_i]
                chain = chain_error[:, c_i, t_i]
                mean = chain.mean(axis=0)
                std = chain.std(axis=0, ddof=1)
                lower = np.clip(mean - 2 * std, a_min=0, a_max=None)
                upper = mean + 2 * std
                axis.plot(samples, mean, linewidth=1)
                axis.fill_between(samples, lower, upper, alpha=0.3)
                axis.xaxis.set_label_position("top")
                axis.set_xlabel(
                    r"$T($" + str(measure_times[t_i]) + r"$) = $" + f"{t:.2f}"
                )
                axis.set_ylim(0, None)

            axes[0].set_ylabel("Error")

            for ax in axes:
                ticks = np.arange(10000, 30001, 10000)
                ax.set_xticks(
                    ticks,
                    labels=map(str, ticks),
                    rotation=45,
                    ha="right",
                )

            fig.supxlabel("Samples")
            fig.suptitle(f"{c.replace("_", " ").title()} cooling trace")
            fig.savefig(
                FIGURES_DIR / f"trace_{c}_{chain_length}.pdf",
                dpi=300,
                bbox_inches="tight",
            )

    for chain_length in chain_lengths:
        with (DATA_DIR / f"markov_chains_{chain_length}.meta").open("r") as f:
            metadata = json.load(f)
        chain_costs = np.load(DATA_DIR / f"chain_costs_{chain_length}.npy")
        chain_error = np.load(DATA_DIR / f"chain_error_{chain_length}.npy")
        samples = np.arange(1, metadata["measure_chain_length"] + 1)
        fig, axes = plt.subplots(
            3,
            len(metadata["measure_times"][Cooling.Linear]),
            layout="constrained",
            figsize=(4, 5),
            sharex=True,
            sharey="row",
        )
        for i, cooling in enumerate(Cooling):
            measure_times = metadata["measure_times"][cooling]
            temperatures = metadata["cooling"][cooling]["temperatures"]

            color = palette[i]
            row = 2 - i
            for t_i, t in enumerate(temperatures):
                axis = axes[row, t_i]
                chain = chain_costs[:, i, t_i]
                chain = chain_error[:, i, t_i]
                mean = chain.mean(axis=0)
                std = chain.std(axis=0, ddof=1)
                lower = np.clip(mean - 2 * std, a_min=0, a_max=None)
                upper = mean + 2 * std
                axis.plot(samples, mean, color=color, linewidth=1)
                axis.fill_between(samples, lower, upper, color=color, alpha=0.3)
                axis.xaxis.set_label_position("top")
                axis.set_xlabel(
                    r"$T($" + str(measure_times[t_i]) + r"$) = $" + f"{t:.2f}"
                )

            axes[row, -1].text(
                1.02,
                0.5,
                cooling.replace("_", " ").title(),
                transform=axes[row, -1].transAxes,
                va="center",
                rotation=0,
            )
            axes[row, 0].set_ylim(0, None)

        for i in range(len(metadata["measure_times"][Cooling.Linear])):
            ticks = np.arange(10000, 30001, 10000)
            axes[-1, i].set_xticks(
                ticks,
                labels=map(str, ticks),
                rotation=45,
                ha="right",
            )

        for ax in axes.flatten():
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
        fig.supylabel("Error")
        fig.supxlabel("Samples")
        fig.suptitle(f"Chain length: {chain_length}")
        fig.savefig(FIGURES_DIR / f"traces_{chain_length}.pdf", dpi=300)

    metadata = {"chain_length_error": {"percentiles": {"lower": 5, "upper": 95}}}

    with (RESULTS_DIR / "plot_metadata.json").open("w") as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    chain_lengths = [int(x) for x in sys.argv[1:]]
    main(chain_lengths)
