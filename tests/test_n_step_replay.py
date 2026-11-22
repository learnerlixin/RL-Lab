import numpy as np
import torch

from algorithms.value_based.advanced_dqn import NStepAccumulator, PrioritizedReplayBuffer
from algorithms.value_based.rainbow import categorical_projection


def test_n_step_accumulator_emits_full_horizon_and_discount():
    replay = PrioritizedReplayBuffer(capacity=16)
    accumulator = NStepAccumulator(replay, n=3, gamma=0.5)
    for i in range(3):
        accumulator.add(np.array([i]), 0, 1.0, np.array([i + 1]), False)
    assert len(replay) == 1
    sample = replay.sample(1, include_discounts=True)
    assert np.isclose(sample[2][0], 1.75)
    assert np.isclose(sample[5][0], 0.5 ** 3)
    assert not sample[4][0]


def test_n_step_accumulator_flushes_terminal_prefixes():
    replay = PrioritizedReplayBuffer(capacity=16)
    accumulator = NStepAccumulator(replay, n=3, gamma=0.5)
    accumulator.add([0], 0, 1, [1], False)
    accumulator.add([1], 0, 2, [2], True)
    assert len(replay) == 2
    transitions = sorted(replay.data, key=lambda item: int(item[0][0]))
    assert np.isclose(transitions[0][2], 2.0)
    assert np.isclose(transitions[1][2], 2.0)
    assert transitions[0][4] and transitions[1][4]
    assert np.isclose(transitions[0][5], 0.25)
    assert np.isclose(transitions[1][5], 0.5)


def test_n_step_truncation_flushes_without_terminal_mask():
    replay = PrioritizedReplayBuffer(capacity=16)
    accumulator = NStepAccumulator(replay, n=3, gamma=0.9)
    accumulator.add([0], 0, 1, [1], False)
    accumulator.add([1], 0, 1, [2], False, truncated=True)
    assert len(replay) == 2
    assert all(not transition[4] for transition in replay.data)


def test_c51_projection_accepts_per_sample_discounts():
    support = torch.linspace(-2, 2, 5)
    probabilities = torch.full((2, 5), 0.2)
    projected = categorical_projection(
        probabilities, torch.tensor([0.0, 0.0]), torch.zeros(2),
        torch.tensor([0.5, 0.25]), support, -2.0, 2.0
    )
    assert torch.allclose(projected.sum(-1), torch.ones(2), atol=1e-6)
