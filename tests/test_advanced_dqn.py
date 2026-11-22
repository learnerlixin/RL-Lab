import numpy as np
import torch
from algorithms.value_based.advanced_dqn import (
    DuelingQNetwork, DoubleDQNAgent, PrioritizedReplayBuffer, double_dqn_targets
)


def test_dueling_network_output_shape():
    net = DuelingQNetwork(4, 3)
    assert net(torch.zeros(5, 4)).shape == (5, 3)


def test_double_dqn_selects_online_action_but_evaluates_target():
    rewards = torch.tensor([1.0])
    terminals = torch.tensor([0.0])
    online = torch.tensor([[0.0, 5.0]])
    target = torch.tensor([[10.0, 2.0]])
    target_value = double_dqn_targets(rewards, terminals, online, target, 0.9)
    assert torch.allclose(target_value, torch.tensor([2.8]))


def test_prioritized_replay_round_trip():
    replay = PrioritizedReplayBuffer(capacity=8)
    for i in range(4):
        replay.add(np.array([i]), i % 2, 1.0, np.array([i + 1]), False)
    sample = replay.sample(3)
    assert sample[0].shape == (3, 1)
    replay.update_priorities(sample[-2], np.array([1.0, 2.0, 3.0]))
    assert len(replay) == 4


def test_double_dqn_update_runs():
    replay = PrioritizedReplayBuffer(capacity=20)
    for i in range(8):
        replay.add(np.array([i, 0.0]), i % 2, 1.0, np.array([i + 1, 0.0]), False)
    agent = DoubleDQNAgent(2, 2, target_update_interval=2)
    assert np.isfinite(agent.update(replay, batch_size=4))
