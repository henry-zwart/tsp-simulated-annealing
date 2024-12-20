"""
Course: Stochastic Simulation
Names: Petr Chalupský, Henry Zwart, Tika van Bennekum
Student IDs: 15719227, 15393879, 13392425
Assignement: Solving Traveling Salesman Problem using Simulated Annealing

File description:
    This file contains the acceptance function, that determines wheter a worse
    route will be accepted.
"""

from scipy.special import expit


def guarantee(h_next, h_current, T):
    """When new solution is better, it is the quarented to be accepted."""
    return 1


def acceptance(h_next, h_current, T):
    """Acceptance calculates alpha.
    Alpha determines wheter we accept or reject the adapted route if
    the total distance is greater then the previous route."""
    alpha = expit((-h_next + h_current) / T)

    return alpha
