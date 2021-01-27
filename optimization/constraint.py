import pandas as pd
import numpy as np

from config.own import MINIMAL_ENERGY_UNDER_THRESHOLD, THRESHOLD, TOTAL_ENERGY


def under_threshold(extra_energy, existency_index=None):
    here = pd.DataFrame(
        {
            "extra_energy": extra_energy,
            "existency_index": existency_index,
        }
    )
    sum_ = here[here.existency_index <= THRESHOLD]["extra_energy"].sum()
    return sum_ - MINIMAL_ENERGY_UNDER_THRESHOLD * TOTAL_ENERGY


def total_energy(extra_energy):
    sum_ = np.sum(extra_energy)
    return TOTAL_ENERGY - sum_
