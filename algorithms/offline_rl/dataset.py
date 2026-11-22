"""Validated transition dataset helpers for offline RL experiments."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class OfflineTransitionDataset(Dataset):
    """In-memory dataset with timeout metadata kept separate from terminals.

    terminals represent true MDP termination. Time-limit truncations belong in
    timeouts so value targets can continue bootstrapping at a timeout boundary.
    """
    def __init__(self, observations, actions, rewards, next_observations,
                 terminals, timeouts=None):
        self.observations = np.asarray(observations, dtype=np.float32)
        self.actions = np.asarray(actions)
        self.rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
        self.next_observations = np.asarray(next_observations, dtype=np.float32)
        self.terminals = np.asarray(terminals, dtype=np.float32).reshape(-1)
        self.timeouts = (np.zeros_like(self.terminals) if timeouts is None else
                         np.asarray(timeouts, dtype=np.float32).reshape(-1))
        size = len(self.rewards)
        arrays = (self.observations, self.actions, self.next_observations,
                  self.terminals, self.timeouts)
        if size == 0 or any(len(array) != size for array in arrays):
            raise ValueError("all transition arrays must have the same nonzero length")
        if self.observations.ndim < 2 or self.next_observations.shape != self.observations.shape:
            raise ValueError("observations and next_observations need matching [N,...] shapes")
        if not np.isfinite(self.observations).all() or not np.isfinite(self.next_observations).all():
            raise ValueError("observations must contain only finite values")
        if not np.isfinite(self.actions).all() or not np.isfinite(self.rewards).all():
            raise ValueError("actions and rewards must contain only finite values")
        if not np.isin(self.terminals, [0.0, 1.0]).all() or not np.isin(self.timeouts, [0.0, 1.0]).all():
            raise ValueError("terminals and timeouts must be binary")

    @classmethod
    def from_npz(cls, path):
        """Load a safe, non-pickle NPZ file with documented field names."""
        with np.load(Path(path), allow_pickle=False) as data:
            required = {"observations", "actions", "rewards", "next_observations", "terminals"}
            missing = required.difference(data.files)
            if missing:
                raise ValueError(f"dataset is missing fields: {sorted(missing)}")
            return cls(*(data[name] for name in
                         ("observations", "actions", "rewards", "next_observations", "terminals")),
                       timeouts=data["timeouts"] if "timeouts" in data.files else None)

    def save_npz(self, path):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, observations=self.observations,
                            actions=self.actions, rewards=self.rewards,
                            next_observations=self.next_observations,
                            terminals=self.terminals, timeouts=self.timeouts)
        return destination

    def __len__(self):
        return len(self.rewards)

    def __getitem__(self, index):
        return {
            "observations": torch.as_tensor(self.observations[index]),
            "actions": torch.as_tensor(self.actions[index]),
            "rewards": torch.as_tensor(self.rewards[index]),
            "next_observations": torch.as_tensor(self.next_observations[index]),
            "terminals": torch.as_tensor(self.terminals[index]),
            "timeouts": torch.as_tensor(self.timeouts[index]),
        }


__all__ = ["OfflineTransitionDataset"]
