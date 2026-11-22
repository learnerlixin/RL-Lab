"""Lightweight multi-seed experiment runners and aggregate statistics."""
from collections.abc import Mapping
import math

import numpy as np

from .reproducibility import seed_everything


def summarize_seed_results(results):
    """Aggregate scalar metrics keyed by integer seed.

    Returns per-metric mean, sample standard deviation, median, and a normal
    95% confidence interval for the mean. With one seed, std is zero and the
    interval collapses to the observed mean; this is descriptive only.
    """
    if not isinstance(results, Mapping) or not results:
        raise ValueError("results must be a non-empty mapping of seed to metrics")
    seeds = sorted(results)
    metric_names = None
    normalized = {}
    for seed in seeds:
        metrics = results[seed]
        if not isinstance(metrics, Mapping) or not metrics:
            raise ValueError(f"seed {seed} must contain a non-empty metrics mapping")
        names = set(metrics)
        if any(not isinstance(name, str) for name in names):
            raise TypeError("metric names must be strings")
        if metric_names is None:
            metric_names = names
        elif names != metric_names:
            raise ValueError("all seeds must report the same metric names")
        normalized[seed] = {}
        for name, value in metrics.items():
            scalar = float(value)
            if not math.isfinite(scalar):
                raise ValueError(f"metric {name!r} for seed {seed} must be finite")
            normalized[seed][str(name)] = scalar

    summary = {}
    for name in sorted(metric_names):
        values = np.asarray([normalized[seed][name] for seed in seeds], dtype=np.float64)
        count = len(values)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if count > 1 else 0.0
        half_width = 1.96 * std / math.sqrt(count) if count > 1 else 0.0
        summary[name] = {
            "mean": mean,
            "std": std,
            "median": float(np.median(values)),
            "ci95_low": mean - half_width,
            "ci95_high": mean + half_width,
            "n_seeds": count,
        }
    return {"runs": normalized, "summary": summary}


def run_seed_sweep(seeds, run_fn, deterministic=False):
    """Run callback run_fn(seed) once per independent seed.

    Each run begins with Python, NumPy and PyTorch RNGs reset to that seed.
    Environment RNGs should still be seeded explicitly inside the callback.
    """
    seed_list = list(seeds)
    if not seed_list or any(not isinstance(seed, int) for seed in seed_list):
        raise ValueError("seeds must be a non-empty sequence of integers")
    if len(set(seed_list)) != len(seed_list):
        raise ValueError("seeds must be unique")
    results = {}
    for seed in seed_list:
        seed_everything(seed, deterministic=deterministic)
        metrics = run_fn(seed)
        if not isinstance(metrics, Mapping):
            raise TypeError("run_fn must return a mapping of scalar metrics")
        results[seed] = dict(metrics)
    return summarize_seed_results(results)


__all__ = ["run_seed_sweep", "summarize_seed_results"]
