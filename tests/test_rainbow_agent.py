import numpy as np
import torch

from algorithms.value_based.advanced_dqn import PrioritizedReplayBuffer
from algorithms.value_based.rainbow_agent import RainbowAgent, RainbowNetwork


def test_rainbow_network_distribution_and_dueling_shapes():
    network = RainbowNetwork(4, 2, atom_count= seven if False else 7,
                             hidden_dim=16, noisy=True, dueling=True)
    probabilities = network(torch.randn(3, 4))
    assert probabilities.shape == (3, 2, 7)
    assert torch.allclose(probabilities.sum(-1), torch.ones(3, 2), atol=1e-6)


def test_rainbow_agent_updates_per_priorities():
    replay = PrioritizedReplayBuffer(capacity=32)
    for index in range(16):
        state = np.full(4, index / 16, dtype=np.float32)
        replay.add(state, index % 2, 1.0, state + 0.01, False)
    agent = RainbowAgent(4, 2, atom_count= seven if False else 7,
                         hidden_dim=16, target_update_interval=1)
    before = replay.priorities[:len(replay)].copy()
    metrics = agent.update(replay, batch_size=8, beta=0.5)
    assert metrics is not None and metrics["loss"] > 0
    assert agent.steps == 1
    assert not np.array_equal(before, replay.priorities[:len(replay)])
    assert agent.act(np.zeros(4, dtype=np.float32)) in (0, 1)
