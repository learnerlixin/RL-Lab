import numpy as np
from algorithms.value_based.dqn import DQNAgent, ReplayBuffer


def test_dqn_update_runs_on_toy_batch():
    replay = ReplayBuffer(100)
    for i in range(20):
        replay.add(np.array([i, 0.0]), i % 2, 1.0,
                   np.array([i + 1, 0.0]), False)
    agent = DQNAgent(2, 2, target_update_interval=2)
    loss = agent.update(replay, batch_size=8)
    assert loss is not None and np.isfinite(loss)
    assert agent.steps == 1


def test_dqn_action_is_valid():
    agent = DQNAgent(3, 4)
    action = agent.act(np.zeros(3), epsilon=0.0)
    assert 0 <= action < 4
