import numpy as np
import pytest
import torch

from algorithms.policy_optimization.ddpg import DDPGAgent
from algorithms.policy_optimization.sac import ContinuousReplayBuffer


def _replay():
    replay = ContinuousReplayBuffer(capacity=32)
    for i in range(16):
        replay.add(np.full(3, i, dtype=np.float32),
                   np.array([0.25, -0.25], dtype=np.float32),
                   float(i) / 10.0,
                   np.full(3, i + 1, dtype=np.float32),
                   terminated=(i == 15))
    return replay


def test_ddpg_action_bounds_and_update():
    agent = DDPGAgent(3, 2, action_low=[-2.0, 0.0], action_high=[1.0, 4.0])
    action = agent.act(np.zeros(3, dtype=np.float32), noise_std=0.2)
    assert action.shape == (2,)
    assert -2.0 <= action[0] <= 1.0
    assert 0.0 <= action[1] <= 4.0
    metrics = agent.update(_replay(), batch_size=8)
    assert set(metrics) == {"critic_loss", "actor_loss"}
    assert all(np.isfinite(value) for value in metrics.values())
    assert all(parameter.grad is None for parameter in agent.critic.parameters())


def test_ddpg_returns_none_until_batch_available():
    agent = DDPGAgent(3, 1)
    assert agent.update(_replay(), batch_size=64) is None


def test_ddpg_rejects_invalid_discount_parameters():
    with pytest.raises(ValueError):
        DDPGAgent(3, 1, gamma=1.5)
    with pytest.raises(ValueError):
        DDPGAgent(3, 1, tau=0.0)


def test_ddpg_rejects_nonpositive_learning_rate():
    with pytest.raises(ValueError):
        DDPGAgent(3, 1, learning_rate=0.0)
