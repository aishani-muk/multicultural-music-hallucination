"""Centralized seed management for reproducibility."""

from __future__ import annotations

import os
import random


def seed_all(seed: int = 848) -> int:
    """Seed python, numpy, torch (if installed), and set PYTHONHASHSEED.

    Returns the seed for logging.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed
