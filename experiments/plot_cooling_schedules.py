# graph
# 3 subplots, each subplot one cooling schedule a280
# each subplot 3 times 500, 1000, 2000

import matplotlib.pyplot as plt
import numpy as np

from tsp_simulated_annealing.cooling_schedules import (
    exponential_cooling,
    inverse_log_cooling,
    linear_cooling,
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


plt.rc("xtick", labelsize=14)
plt.rc("ytick", labelsize=14)
plt.rc("axes", labelsize=18)

fig, axes = plt.subplots(
    ncols=3,
    nrows=2,
    figsize=(12, 5),
    gridspec_kw={"height_ratios": [1, 2]},
    sharex=True,
    sharey="row",
)

t = []
t.append(np.arange(0, 500))
t.append(np.arange(0, 1000))
t.append(np.arange(0, 2000))

axes[0, 0].set_title("Lin")
axes[0, 1].set_title("Exp")
axes[0, 2].set_title("Log")

for i in range(3):
    axes[0, 0].plot(
        t[i], linear_cooling(t[i], etas[i], T_0), label=f"eta = {etas[i]:.2f}"
    )
    axes[0, 1].plot(
        t[i],
        exponential_cooling(t[i], alphas[i], T_0),
        label=f"alpha = {alphas[i]:.3f}",
    )
    axes[0, 2].plot(
        t[i],
        inverse_log_cooling(t[i], a[i], b[i]),
        label=f"a = {a[i]:.2f}, b = {b[i]:.2f}",
    )

    axes[1, 0].plot(t[i], lin_error[i])
    axes[1, 1].plot(t[i], exp_error[i])
    axes[1, 2].plot(
        t[i],
        log_error[i],
    )

axes[0, 0].legend(prop={"size": 12})
axes[0, 1].legend(prop={"size": 12})
axes[0, 2].legend(prop={"size": 12})

axes[0, 0].set_ylabel("Temperature")
axes[1, 0].set_ylabel("Error")
axes[1, 1].set_xlabel("Time")

# fig.supxlabel(r"Time", fontsize=20)
# fig.supylabel("T", fontsize=20)
fig.tight_layout()
plt.show()
fig.savefig("plot_temperature.pdf", dpi=300)
