import numpy as np


def acceptance(h_next, h_current, T):
    alpha = np.min(np.exp((-h_next + h_current) / T), 1)

    return alpha
