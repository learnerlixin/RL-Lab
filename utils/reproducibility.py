"""Random seeding helpers for repeatable experiments."""
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> int:
    """Seed Python, NumPy and PyTorch RNGs and return the normalized seed.

    Deterministic algorithms can be slower and may raise when an operation has
    no deterministic implementation; warn_only keeps experiments runnable.
    """
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)
    return seed
