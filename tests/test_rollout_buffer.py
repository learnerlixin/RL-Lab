import pytest
import torch

from algorithms.policy_based.rollout_buffer import RolloutBuffer


def test_rollout_buffer_bootstraps_time_limit_but_not_terminal():
    buffer = RolloutBuffer(4)
    buffer.add([0.0, 1.0], 0, 1.0, 0.0, -0.5, 10.0,
               terminated=False, truncated=True)
    buffer.add([1.0, 2.0], 1, 2.0, 0.0, -0.4, 100.0,
               terminated=True, truncated=False)
    advantages, returns = buffer.compute_returns_and_advantages(
        gamma=0.9, gae_lambda=1.0)
    assert torch.allclose(advantages, torch.tensor([10.0, 2.0]))
    assert torch.allclose(returns, advantages)


def test_rollout_buffer_gae_does_not_leak_across_episode_boundaries():
    buffer = RolloutBuffer(4)
    buffer.add([0.0], 0, 1.0, 0.0, -0.5, 5.0, False, True)
    buffer.add([1.0], 1, 10.0, 0.0, -0.4, 0.0, True, False)
    advantages, _ = buffer.compute_returns_and_advantages(
        gamma=1.0, gae_lambda=1.0)
    assert torch.equal(advantages, torch.tensor([6.0, 10.0]))


def test_rollout_buffer_minibatches_and_clear():
    buffer = RolloutBuffer(5)
    for index in range(5):
        buffer.add([index, 0.0], index % 2, 1.0, 0.0, -0.3, 0.0,
                   terminated=index == 4)
    buffer.compute_returns_and_advantages()
    batches = list(buffer.minibatches(2, shuffle=False))
    assert [len(batch["observations"]) for batch in batches] == [2, 2, 1]
    assert batches[0]["actions"].dtype == torch.int64
    buffer.clear()
    assert len(buffer) == 0


def test_rollout_buffer_validates_capacity_and_transition_shapes():
    with pytest.raises(ValueError):
        RolloutBuffer(0)
    buffer = RolloutBuffer(1)
    with pytest.raises(ValueError):
        buffer.compute_returns_and_advantages()
    buffer.add([0.0], 0, 1.0, 0.0, -0.1, 0.0, False)
    with pytest.raises(OverflowError):
        buffer.add([1.0], 1, 1.0, 0.0, -0.1, 0.0, False)
