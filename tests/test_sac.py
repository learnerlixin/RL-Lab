import numpy as np
from algorithms.policy_optimization.sac import GaussianActor, SACAgent, ContinuousReplayBuffer
import torch


def test_gaussian_actor_respects_action_bounds():
    actor = GaussianActor(3, 2, action_low=-2.0, action_high=2.0)
    actions, logp, deterministic = actor.sample(torch.randn(10, 3))
    assert actions.shape == (10, 2)
    assert logp.shape == (10, 1)
    assert torch.all(actions <= 2.0) and torch.all(actions >= -2.0)
    assert deterministic.shape == (10, 2)


def test_sac_update_runs():
    replay = ContinuousReplayBuffer(100)
    for i in range(20):
        replay.add(np.zeros(3), np.zeros(2), 1.0, np.ones(3), False)
    agent = SACAgent(3, 2, action_low=-1.0, action_high=1.0)
    metrics = agent.update(replay, batch_size=8)
    assert all(np.isfinite(v) for v in metrics.values())


def test_sac_rejects_invalid_hyperparameters_and_action_bounds():
    import pytest
    with pytest.raises(ValueError):
        SACAgent(3, 2, alpha=0.0)
    with pytest.raises(ValueError):
        SACAgent(3, 2, tau=0.0)
    with pytest.raises(ValueError):
        GaussianActor(3, 2, action_low=[-1.0, -1.0, -1.0], action_high=[1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        ContinuousReplayBuffer(capacity=0)


def test_continuous_replay_buffer_rejects_nonfinite_and_invalid_terminal():
    import pytest
    replay = ContinuousReplayBuffer(4)
    with pytest.raises(ValueError):
        replay.add([0.0], [float("nan")], 0.0, [1.0], False)
    with pytest.raises(ValueError):
        replay.add([0.0], [0.0], 0.0, [1.0], 0.5)
