"""Model/optimizer checkpoints with optional RNG-state restoration."""
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


def _rng_state():
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, model, optimizer=None, step=0, metadata=None,
                    include_rng=True):
    """Atomically save model, optimizer, progress metadata and optional RNG state."""
    if step < 0:
        raise ValueError("step must be nonnegative")
    payload = {"model": model.state_dict(), "step": int(step),
               "metadata": dict(metadata or {})}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if include_rng:
        payload["rng_state"] = _rng_state()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
        os.close(descriptor)
        temp_path = Path(temp_name)
        torch.save(payload, temp_path)
        os.replace(temp_path, destination)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
    return destination


def load_checkpoint(path, model, optimizer=None, map_location="cpu",
                    restore_rng=False, strict=True):
    """Restore a checkpoint and return its step and user metadata.

    Only load checkpoint files from trusted sources.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if "model" not in payload or "step" not in payload:
        raise ValueError("invalid checkpoint: missing model or step")
    model.load_state_dict(payload["model"], strict=strict)
    if optimizer is not None:
        if "optimizer" not in payload:
            raise ValueError("checkpoint does not contain optimizer state")
        optimizer.load_state_dict(payload["optimizer"])
    if restore_rng:
        if "rng_state" not in payload:
            raise ValueError("checkpoint does not contain RNG state")
        _restore_rng(payload["rng_state"])
    return {"step": int(payload["step"]), "metadata": dict(payload.get("metadata", {}))}


__all__ = ["save_checkpoint", "load_checkpoint"]
