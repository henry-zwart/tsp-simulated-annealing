import numpy as np


def inverse_log_cooling(t, a, b):
    T = a / np.log(t + b)
    return T


def exponential_cooling(t, alpha, T_0):
    T = T_0 * alpha**t
    return T


def linear_cooling(t, eta, T_0):
    """
    eta, needs to be calculate as eta = T_0/t_f,
    where t_f is the total simulation time (number of steps)
    """
    T = T_0 - eta * t
    return T
