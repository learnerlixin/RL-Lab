import torch
from torch import nn

from algorithms.llm_rl.reward_model import RewardModelHead
from algorithms.llm_rl.reward_model_trainer import reward_model_train_step


class TinyEncoder(nn.Module):
    def __init__(self, vocab_size=20, hidden_size=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids, attention_mask):
        return (self.embedding(input_ids),)


def test_reward_model_train_step_updates_encoder_and_head():
    torch.manual_seed(3)
    encoder = TinyEncoder()
    head = RewardModelHead(8, dropout=0.0)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=1e-2)
    chosen = torch.tensor([[1, 2, 0], [3, 4, 0]])
    rejected = torch.tensor([[5, 6, 0], [7, 8, 0]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 0]])
    before = [p.detach().clone() for p in encoder.parameters()]
    metrics = reward_model_train_step(encoder, head, optimizer, chosen, mask, rejected, mask)
    assert set(metrics) == {"loss", "preference_accuracy", "reward_margin", "grad_norm"}
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, encoder.parameters()))


def test_reward_model_train_step_rejects_invalid_clip():
    encoder, head = TinyEncoder(), RewardModelHead(8)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()))
    ids = torch.ones(2, 3, dtype=torch.long)
    try:
        reward_model_train_step(encoder, head, optimizer, ids, ids, ids, ids, max_grad_norm=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
