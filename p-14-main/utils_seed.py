# utils_seed.py

import random
import numpy as np

def set_global_seed(seed=42):
    """Set random seed for Python's random module and NumPy.

    This helps to make experiments reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
