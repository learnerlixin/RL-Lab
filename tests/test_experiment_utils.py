import csv
import random

import numpy as np
import pytest
import torch
from torch import nn

from utils import CSVLogger, evaluate_policy, load_checkpoint, save_checkpoint, seed_everything


def test_seed_everything_repeats_python_numpy_and_torch():
    seed_everything(17)
    first = (random.random(), np.random.rand(), torch.rand(1))
    seed_everything(17)
    second = (random.random(), np.random.rand(), torch.rand(1))
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_csv_logger_writes_header_and_rows(tmp_path):
    path = tmp_path / "nested" / "metrics.csv"
    with CSVLogger(path, ["step", "reward"]) as logger:
        logger.log({"step": 1, "reward": 2.5})
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [{"step": "1", "reward": "2.5"}]


def test_csv_logger_rejects_schema_mismatch(tmp_path):
    with CSVLogger(tmp_path / "metrics.csv", ["step"]) as logger:
        with pytest.raises(ValueError):
            logger.log({"reward": 1})


def test_checkpoint_restores_model_optimizer_and_metadata(tmp_path):
    model = nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    inputs = torch.ones(2, 3)
    model(inputs).sum().backward()
    optimizer.step()
    path = tmp_path / "checkpoints" / "latest.pt"
    expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
    save_checkpoint(path, model, optimizer, step=23, metadata={"run": "unit"})
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10)
    result = load_checkpoint(path, model, optimizer)
    assert result == {"step": 23, "metadata": {"run": "unit"}}
    for key, value in model.state_dict().items():
        assert torch.equal(value, expected[key])


class TinyEnv:
    def __init__(self):
        self.closed = False
        self.episode = 0

    def reset(self, seed=None):
        self.steps = 0
        self.episode += 1
        return np.array([0.0]), {"seed": seed}

    def step(self, action):
        self.steps += 1
        return np.array([self.steps]), 1.0, self.steps == 2, False, {}


def test_evaluate_policy_supports_gymnasium_and_preserves_env():
    env = TinyEnv()
    result = evaluate_policy(env, lambda obs: 0, episodes=3, seed=10)
    assert result["returns"] == [2.0, 2.0, 2.0]
    assert result["lengths"] == [2, 2, 2]
    assert not env.closed


def test_evaluate_policy_uses_deterministic_agent_argument():
    class Agent:
        def act(self, observation, deterministic=False):
            assert deterministic is True
            return 0
    result = evaluate_policy(TinyEnv(), Agent(), episodes=1)
    assert result["mean_return"] == 2.0
