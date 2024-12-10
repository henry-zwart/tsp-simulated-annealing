"""
Course: Stochastic Simulation
Names: Petr Chalupský, Henry Zwart, Tika van Bennekum
Student IDs: 15719227, 15393879, 13392425
Assignement: Solving Traveling Salesman Problem using Simulated Annealing

File description:
    This file contains all the cooling schedules researched in our paper.
"""

from enum import StrEnum
from functools import partial

import numpy as np


class Cooling(StrEnum):
    Linear = "linear"
    Exponential = "exponential"
    InverseLog = "inverse_log"


def get_scheduler(temp_0: float, temp_n: float, time_n: float, algorithm: Cooling):
    match algorithm:
        case Cooling.Linear:
            eta = (temp_0 - temp_n) / time_n
            return partial(linear_cooling, eta=eta, T_0=temp_0)
        case Cooling.Exponential:
            alpha = np.pow(temp_n / temp_0, time_n)
            return partial(exponential_cooling, alpha=alpha, T_0=temp_0)
        case Cooling.InverseLog:
            # b = optimize.root_scalar(
            #    inv_log_rootfinding_fn,

            # )
            raise ValueError("Not implemented for InverseLog")


def inv_log_rootfinding_fn(temp_0: float, temp_n: float, time_n: float, b: float):
    return (np.log10(b) / np.log10(b + time_n)) - (temp_n / temp_0)


def inv_log_rootfinding_jac(_temp_0: float, _temp_n: float, time_n: float, b: float):
    numerator = (np.log(b + time_n) / b) - (
        np.log(b) / (b + time_n) * np.log(b + time_n)
    )
    return numerator / (np.log(b + time_n) ** 2)


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
