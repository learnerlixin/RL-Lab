import torch
from algorithms.policy_based.continuous_ppo import ContinuousActorCritic, continuous_ppo_update


def test_continuous_actor_respects_action_bounds_and_shapes():
    model = ContinuousActorCritic(3, 2, action_low=[-2.0, 0.0], action_high=[1.0, 4.0])
    states = torch.randn(8, 3)
    actions, logps, values = model.act(states)
    assert actions.shape == (8, 2)
    assert logps.shape == values.shape == (8,)
    assert torch.all(actions[:, 0] >= -2.0) and torch.all(actions[:, 0] <= 1.0)
    assert torch.all(actions[:, 1] >= 0.0) and torch.all(actions[:, 1] <= 4.0)
    evaluated, entropy, evaluated_values = model.evaluate_actions(states, actions)
    assert evaluated.shape == entropy.shape == evaluated_values.shape == (8,)
    assert torch.isfinite(evaluated).all()


def test_continuous_ppo_update_runs():
    model = ContinuousActorCritic(3, 1, action_low=-2.0, action_high=2.0, hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    states = torch.randn(24, 3)
    with torch.no_grad():
        actions, old_logps, values = model.act(states)
    advantages = torch.randn(24)
    returns = values + advantages
    metrics = continuous_ppo_update(model, optimizer, states, actions, old_logps, returns,
                                    advantages, epochs=2, batch_size=8)
    assert set(metrics) == {"policy_loss", "value_loss", "entropy"}
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())

def test_continuous_ppo_rejects_negative_loss_coefficients():
    import pytest
    model = ContinuousActorCritic(3, 1, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    states = torch.randn(2, 3)
    with torch.no_grad():
        actions, logps, values = model.act(states)
    with pytest.raises(ValueError):
        continuous_ppo_update(model, optimizer, states, actions, logps,
                              values, torch.ones(2), entropy_coef=-0.1)
