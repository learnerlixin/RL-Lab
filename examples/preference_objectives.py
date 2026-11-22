import torch
from algorithms.llm_rl.preference_objectives import bradley_terry_loss, dpo_loss

chosen = torch.tensor([1.2, 0.8])
rejected = torch.tensor([0.1, 0.4])
print("Reward model loss:", bradley_terry_loss(chosen, rejected).item())
loss, reward_w, reward_l = dpo_loss(
    torch.tensor([-2.0, -1.0]), torch.tensor([-3.0, -1.8]),
    torch.tensor([-2.4, -1.2]), torch.tensor([-2.9, -2.0]))
print("DPO loss:", loss.item())
print("implicit rewards:", reward_w.tolist(), reward_l.tolist())
