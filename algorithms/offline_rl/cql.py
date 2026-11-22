"""Conservative Q-Learning (CQL) discrete-action regularizer."""
import torch


def discrete_cql_penalty(all_action_q_values, dataset_action_q_values, temperature=1.0):
    """Penalize high Q for unseen actions relative to actions in the dataset.

    all_action_q_values: [batch, action_count]
    dataset_action_q_values: [batch]
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    conservative_gap = temperature * torch.logsumexp(
        all_action_q_values / temperature, dim=-1) - dataset_action_q_values
    return conservative_gap.mean()
