import numpy as np


def cost_function(extra_energy, potential_increase=None, pred=None):
    cost = np.sum(-extra_energy * (potential_increase ** 2) / 1000)
    return cost
