import numpy as np
from algorithms.tabular.td_control import epsilon_greedy


def test_epsilon_zero_selects_greedy_action():
    rng = np.random.default_rng(0)
    values = np.array([1.0, 3.0, 2.0])
    assert epsilon_greedy(values, 0.0, rng) == 1


def test_epsilon_one_returns_valid_random_action():
    rng = np.random.default_rng(1)
    for _ in range(100):
        action = epsilon_greedy(np.zeros(4), 1.0, rng)
        assert 0 <= action < 4
