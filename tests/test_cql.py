import torch
from algorithms.offline_rl.cql import discrete_cql_penalty


def test_cql_penalty_is_finite():
    q_all = torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 2.0]])
    q_data = torch.tensor([1.0, 0.0])
    assert torch.isfinite(discrete_cql_penalty(q_all, q_data))


def test_cql_penalty_grows_when_unseen_action_q_increases():
    q_data = torch.tensor([1.0])
    low = discrete_cql_penalty(torch.tensor([[1.0, 1.0]]), q_data)
    high = discrete_cql_penalty(torch.tensor([[1.0, 5.0]]), q_data)
    assert high > low
