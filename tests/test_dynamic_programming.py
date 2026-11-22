import numpy as np
from algorithms.tabular.dynamic_programming import value_iteration, policy_evaluation


def tiny_mdp():
    return [[[(1.0, 0, 1.0, True)]]]


def test_value_iteration_terminal_reward():
    values, policy = value_iteration(tiny_mdp(), 1, 1, gamma=0.9)
    assert np.allclose(values, [1.0])
    assert policy.tolist() == [0]


def test_policy_evaluation_terminal_reward():
    values = policy_evaluation(tiny_mdp(), np.array([[1.0]]), 1, 1, gamma=0.9)
    assert np.allclose(values, [1.0])
