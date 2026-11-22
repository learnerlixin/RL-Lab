import torch
from algorithms.value_based.rainbow import (
    NoisyLinear, CategoricalQNetwork, categorical_projection, n_step_return
)


def test_noisy_linear_shape_and_resample():
    layer = NoisyLinear(4, 3)
    output = layer(torch.randn(5, 4))
    before = layer.weight_epsilon.clone()
    layer.reset_noise()
    assert output.shape == (5, 3)
    assert not torch.equal(before, layer.weight_epsilon)


def test_categorical_q_network_probabilities():
    net = CategoricalQNetwork(4, 2, atom_count=11)
    probs = net(torch.randn(3, 4))
    assert probs.shape == (3, 2, 11)
    assert torch.allclose(probs.sum(-1), torch.ones(3, 2), atol=1e-6)


def test_c51_projection_preserves_probability_mass():
    support = torch.linspace(-2, 2, 5)
    probs = torch.full((2, 5), 0.2)
    projected = categorical_projection(probs, torch.tensor([0.0, 1.0]),
                                        torch.tensor([0.0, 1.0]), 0.9,
                                        support, -2.0, 2.0)
    assert torch.allclose(projected.sum(-1), torch.ones(2), atol=1e-6)


def test_n_step_return():
    assert abs(n_step_return([1, 2], gamma=0.5, bootstrap_value=4) - 3.0) < 1e-6
