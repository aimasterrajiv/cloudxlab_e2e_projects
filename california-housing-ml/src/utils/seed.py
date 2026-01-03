"""
Utilities for reproducibility across the project.
"""

import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    This function ensures that results are repeatable across:
    - Python's built-in random module
    - NumPy operations
    - Hash-based operations (where applicable)

    Parameters
    ----------
    seed : int, default=42
        The seed value to use for all random number generators.
    """

    # Python built-in random
    random.seed(seed)

    # NumPy random
    np.random.seed(seed)

    # Hash-based operations (important for dict/set ordering consistency)
    os.environ["PYTHONHASHSEED"] = str(seed)
