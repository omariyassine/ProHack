import numpy as np
import logging

logger = logging.getLogger(__file__)


def cost_function(extra_energy, potential_increase=None, pred=None):
    cost = np.sum(-extra_energy * (potential_increase ** 2) / 1000)
    logger.debug("Cost function: %s", cost)
    return cost
