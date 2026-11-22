import numpy as np
from algorithms.policy_optimization.sac import ContinuousReplayBuffer
from algorithms.policy_optimization.td3 import TD3Agent


def test_td3_action_respects_bounds():
    agent = TD3Agent(3, 2, action_low=-2.0, action_high=2.0)
    action = agent.act(np.zeros(3), noise_std=0.5)
    assert np.all(action >= -2.0) and np.all(action <= 2.0)


def test_td3_update_runs():
    replay = ContinuousReplayBuffer(100)
    for i in range(16):
        replay.add(np.zeros(3), np.zeros(2), 1.0, np.ones(3), False)
    agent = TD3Agent(3, 2, policy_delay=1)
    metrics = agent.update(replay, batch_size=8)
    assert np.isfinite(metrics["critic_loss"])
    assert np.isfinite(metrics["actor_loss"])
