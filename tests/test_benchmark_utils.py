import numpy as np
import pytest

from utils.benchmark import run_seed_sweep, summarize_seed_results


def test_summarize_seed_results_reports_mean_std_and_interval():
    result = summarize_seed_results({
        1: {"mean_return": 10.0, "steps": 100},
        2: {"mean_return": 14.0, "steps": 120},
        3: {"mean_return": 12.0, "steps": 110},
    })
    summary = result["summary"]["mean_return"]
    assert summary["mean"] == pytest.approx(12.0)
    assert summary["std"] == pytest.approx(2.0)
    assert summary["median"] == pytest.approx(12.0)
    assert summary["ci95_low"] < 12.0 < summary["ci95_high"]
    assert summary["n_seeds"] == 3


def test_summarize_seed_results_rejects_mismatched_or_nonfinite_metrics():
    with pytest.raises(ValueError):
        summarize_seed_results({1: {"reward": 1.0}, 2: {"loss": 0.5}})
    with pytest.raises(ValueError):
        summarize_seed_results({1: {"reward": float("nan")}})


def test_run_seed_sweep_resets_rng_for_each_seed_and_returns_summary():
    observed = []
    def run(seed):
        observed.append((seed, np.random.random()))
        return {"score": float(seed)}

    result = run_seed_sweep([5, 8], run)
    assert list(result["runs"]) == [5, 8]
    assert result["summary"]["score"]["mean"] == pytest.approx(6.5)
    assert len(observed) == 2


def test_run_seed_sweep_rejects_duplicate_seeds():
    with pytest.raises(ValueError):
        run_seed_sweep([1, 1], lambda seed: {"score": 1.0})
