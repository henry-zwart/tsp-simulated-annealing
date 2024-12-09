"""
Course: Stochastic Simulation
Names: Petr Chalupský, Henry Zwart, Tika van Bennekum
Student IDs: 15719227, 15393879, 13392425
Assignement: Solving Traveling Salesman Problem using Simulated Annealing

File description:
    This file contains all the cooling schedules researched in our paper.
"""

import numpy as np


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
