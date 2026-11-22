import numpy as np
import torch

from algorithms.policy_based.a2c import A2CAgent, a2c_loss, discounted_returns


def test_discounted_returns_respects_terminal_mask_and_bootstrap():
    rewards = torch.tensor([1.0, 2.0, 3.0])
    dones = torch.tensor([0.0, 0.0, 0.0])
    returns = discounted_returns(rewards, dones, bootstrap_value=4.0, gamma=0.5)
    assert torch.allclose(returns, torch.tensor([3.25, 4.5, 5.0]))
    terminal = discounted_returns(rewards, torch.tensor([0.0, 1.0, 0.0]),
                                  bootstrap_value=4.0, gamma=0.5)
    assert torch.allclose(terminal, torch.tensor([2.0, 2.0, 5.0]))


def test_a2c_loss_is_finite():
    logits = torch.randn(5, 3, requires_grad=True)
    values = torch.randn(5, requires_grad=True)
    actions = torch.tensor([0, 1, 2, 0, 1])
    returns = torch.randn(5)
    loss, metrics = a2c_loss(logits, values, actions, returns)
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["entropy"] > 0


def test_a2c_agent_update_changes_parameters():
    agent = A2CAgent(4, 2, hidden_dim=16)
    states = np.random.randn(8, 4).astype(np.float32)
    actions = np.random.randint(0, 2, size=8)
    returns = np.random.randn(8).astype(np.float32)
    before = agent.network.policy_head.weight.detach().clone()
    metrics = agent.update(states, actions, returns)
    assert np.isfinite(metrics["loss"])
    assert not torch.equal(before, agent.network.policy_head.weight)
    assert agent.act(np.zeros(4, dtype=np.float32))[0] in (0, 1)
