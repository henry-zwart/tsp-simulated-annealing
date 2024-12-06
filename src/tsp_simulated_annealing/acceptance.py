import numpy as np


def acceptance(h_next, h_current, T):
    alpha = np.exp((-h_next + h_current) / T)

    return alpha
