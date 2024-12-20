# graph
# 3 subplots, each subplot one cooling schedule a280
# each subplot 3 times 500, 1000, 2000

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from tsp_simulated_annealing.cooling_schedules import (
    exponential_cooling,
    inverse_log_cooling,
    linear_cooling,
)

FONT_SIZE_TINY = 7
FONT_SIZE_SMALL = 9
FONT_SIZE_DEFAULT = 10
FONT_SIZE_LARGE = 12

plt.rc("font", family="Georgia")
plt.rc("font", weight="normal")  # controls default font
plt.rc("mathtext", fontset="stix")
plt.rc("font", size=FONT_SIZE_DEFAULT)  # controls default text sizes
plt.rc("axes", titlesize=FONT_SIZE_DEFAULT)  # fontsize of the axes title
plt.rc("axes", labelsize=FONT_SIZE_DEFAULT)  # fontsize of the x and y labels
plt.rc("figure", labelsize=FONT_SIZE_DEFAULT)
plt.rc("xtick", labelsize=FONT_SIZE_SMALL)  # fontsize of the tick labels
plt.rc("ytick", labelsize=FONT_SIZE_SMALL)  # fontsize of the tick labels

# plt.rc("axes", titlepad=10)  # add space between title and plot
plt.rc("figure", dpi=700)  # fix output resolution

sns.set_context(
    "paper",
    rc={
        "axes.linewidth": 0.5,
        "axes.labelsize": FONT_SIZE_LARGE,
        "axes.titlesize": FONT_SIZE_DEFAULT,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "ytick.minor.width": 0.4,
    },
)

etas = []
alphas = []
a = []
b = []

for n_samples in [500, 1000, 2000]:
    etas.append(np.load(f"data/a280_{n_samples}_eta.npy"))
    alphas.append(np.load(f"data/a280_{n_samples}_alpha.npy"))
    a.append(np.load(f"data/a280_{n_samples}_a.npy"))
    b.append(np.load(f"data/a280_{n_samples}_b.npy"))

lin_error = []
exp_error = []
log_error = []

for n_samples in [500, 1000, 2000]:
    lin_error.append(np.load(f"data/a280_linear_{n_samples}_errors.npy"))
    exp_error.append(np.load(f"data/a280_exponential_{n_samples}_errors.npy"))
    log_error.append(np.load(f"data/a280_inverse_log_{n_samples}_errors.npy"))

T_0 = np.load("data/T_0_a280.npy")


fig, axes = plt.subplots(
    ncols=3,
    nrows=2,
    figsize=(6.5, 3),
    layout="constrained",
    gridspec_kw={"height_ratios": [1, 2]},
    sharex=True,
    sharey="row",
)

t = []
t.append(np.arange(0, 501))
t.append(np.arange(0, 1001))
t.append(np.arange(0, 2001))

axes[0, 0].set_title("Linear", pad=35)
axes[0, 1].set_title("Exponential", pad=35)
axes[0, 2].set_title("Inverse Log", pad=35)

for i in range(3):
    axes[0, 0].plot(t[i], linear_cooling(t[i], etas[i], T_0), label=f"{etas[i]:.2f}")
    axes[0, 1].plot(
        t[i],
        exponential_cooling(t[i], alphas[i], T_0),
        label=f"{alphas[i]:.3f}",
    )
    axes[0, 2].plot(
        t[i],
        inverse_log_cooling(t[i], a[i], b[i]),
        label=f"{a[i]:.2f},{b[i]:.2f}",
    )

    lin_mean = lin_error[i].mean(axis=0)
    lin_std = lin_error[i].std(axis=0, ddof=1)
    axes[1, 0].plot(t[i], lin_mean, linewidth=0.5)
    axes[1, 0].fill_between(t[i], lin_mean - lin_std, lin_mean + lin_std, alpha=0.5)
    exp_mean = exp_error[i].mean(axis=0)
    exp_std = exp_error[i].std(axis=0, ddof=1)
    axes[1, 1].plot(t[i], exp_mean, linewidth=0.5)
    axes[1, 1].fill_between(t[i], exp_mean - exp_std, exp_mean + exp_std, alpha=0.5)

    log_mean = log_error[i].mean(axis=0)
    log_std = log_error[i].std(axis=0, ddof=1)
    axes[1, 2].plot(t[i], log_mean, linewidth=0.5)
    axes[1, 2].fill_between(t[i], log_mean - log_std, log_mean + log_std, alpha=0.5)

for ax in axes.flatten():
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

legend_titles = [r"$\eta$", r"$\alpha$", r"$(a,b)$"]
for ax, legend_title in zip(axes[0], legend_titles, strict=False):
    ax.legend(
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        fontsize=FONT_SIZE_TINY,
        title=legend_title,
        handlelength=1,
        labelspacing=0.4,
        frameon=False,
    )

axes[0, 0].set_ylabel(r"T(t)")
axes[1, 0].set_ylabel("Error")
axes[1, 1].set_xlabel("Iteration")
axes[1, 0].set_yscale("log")
axes[1, 0].set_ylim(100, None)

fig.savefig("results/figures/plot_temperature.pdf", dpi=700)

with Path("data/cooling.meta").open("w") as f:
    json.dump({"n_samples": [500, 1000, 2000]}, f)
