"""
Course: Stochastic Simulation
Names: Petr Chalupský, Henry Zwart, Tika van Bennekum
Student IDs: 15719227, 15393879, 13392425
Assignement: Solving Traveling Salesman Problem using Simulated Annealing

File description:
    This file contains all the cooling schedules researched in our paper.
"""

import math
from enum import StrEnum
from functools import partial

import numpy as np
from scipy import optimize


class Cooling(StrEnum):
    Linear = "linear"
    Exponential = "exponential"
    InverseLog = "inverse_log"


def get_scheduler(
    init_temp: float,
    final_temp: float,
    time_n: float,
    algorithm: Cooling,
):
    """Function to handle the different cooling schedules."""
    match algorithm:
        case Cooling.Linear:
            eta = fit_linear(init_temp, final_temp, time_n + 1)
            fn = partial(linear_cooling, eta=eta, T_0=init_temp)
        case Cooling.Exponential:
            alpha = fit_exponential(init_temp, final_temp, time_n + 1)
            fn = partial(exponential_cooling, alpha=alpha, T_0=init_temp)
        case Cooling.InverseLog:
            a, b = fit_inverse_log(init_temp, final_temp, time_n + 1)
            fn = partial(inverse_log_cooling, a=a, b=b)

    return fn


def fit_linear(init_temp, final_temp, n_samples) -> float:
    """Determine eta which fits initial conditions."""
    return (init_temp - final_temp) / n_samples


def fit_exponential(init_temp, final_temp, n_samples) -> float:
    """Determine alpha which fits initial conditions."""
    return np.exp((np.log(final_temp) - np.log(init_temp)) / (n_samples))


def fit_inverse_log(init_temp, final_temp, n_samples) -> tuple[float, float]:
    """Determine a and b which fit initial conditions.

    This requires the use of a root-finding method to solve the equation:
        b^(T0/Tn) - b - n = 0
    From which we can calculate a:
        a = T0 log(b)
    """
    # define k = T0/Tn
    k = init_temp / final_temp

    def f(b):
        # return b**k - b - (n_samples - 1)
        return k * math.log(b) - math.log(b + n_samples)

    # Attempt to find root
    result = optimize.root_scalar(f, bracket=[0.001, 2], method="brentq")
    if not result.converged:
        raise ValueError("Solving for 'b' failed to converge.")
    b = result.root

    a = init_temp * np.log(b)

    return (a, b)


def inverse_log_cooling(t, a, b):
    "Inverse log cooling schedule."
    T = a / np.log(t + b)
    return T


def exponential_cooling(t, alpha, T_0):
    "Exponential cooling schedule."
    T = T_0 * alpha**t
    return T


def linear_cooling(t, eta, T_0):
    """
    Linear cooling schedule.
    eta, needs to be calculate as eta = T_0/t_f,
    where t_f is the total simulation time (number of steps)
    """
    T = T_0 - eta * t
    return T


def inverse_linear(eta, T_0, T_n):
    """
    Calculate the time at which linear cooling reaches a given temperature.
    """
    return int(np.round((T_0 - T_n) / eta))


def inverse_exponential(alpha, T_0, T_n):
    """
    Calculate the time at which exponential cooling reaches a given temperature.
    """
    return int(np.round((np.log(T_n / T_0)) / np.log(alpha)))
