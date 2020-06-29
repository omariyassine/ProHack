""" 
Configure the variables used for the cli of the current country

"""

from core.logging import LOGGING

# Logging

LOGGING["loggers"][""]["level"] = "INFO"

TOTAL_ENERGY = 50000
MAX_ENERGY_PER_GALAXY = 100
THRESHOLD = 0.7
MINIMAL_ENERGY_UNDER_THRESHOLD = 0.1
