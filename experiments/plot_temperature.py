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

T_0 = np.load("data/T_0_a280.npy")


plt.rc("xtick", labelsize=14)
plt.rc("ytick", labelsize=14)
plt.rc("axes", labelsize=18)

fig, axes = plt.subplots(ncols=3, figsize=(12, 6), sharey=True)

t = []
t.append(np.arange(0, 500))
t.append(np.arange(0, 1000))
t.append(np.arange(0, 2000))

axes[0].set_title("Lin")
axes[1].set_title("Exp")
axes[2].set_title("Log")

for i in range(3):
    axes[0].plot(t[i], linear_cooling(t[i], etas[i], T_0), label=f"eta = {etas[i]:.2f}")
    axes[1].plot(
        t[i],
        exponential_cooling(t[i], alphas[i], T_0),
        label=f"alpha = {alphas[i]:.3f}",
    )
    axes[2].plot(
        t[i],
        inverse_log_cooling(t[i], a[i], b[i]),
        label=f"a = {a[i]:.2f}, b = {b[i]:.2f}",
    )

axes[0].legend()
axes[1].legend()
axes[2].legend()


fig.supxlabel(r"Time", fontsize=20)
fig.supylabel("T", fontsize=20)
fig.tight_layout()
plt.savefig("figures/plot_temperature.png", dpi=300)
