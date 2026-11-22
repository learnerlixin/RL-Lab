"""Train the discrete PPO baseline on Gymnasium CartPole-v1."""
import numpy as np
import torch
import gymnasium as gym
from algorithms.policy_based.actor_critic import ActorCritic
from algorithms.policy_based.ppo import ppo_update
from algorithms.policy_based.rollout_buffer import RolloutBuffer


def main(episodes=500, seed=7):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = gym.make("CartPole-v1")
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    returns_history = []
    buffer = RolloutBuffer(capacity=5000)
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed if episode == 0 else None)
        total = 0.0
        while True:
            state = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                action, logp, value = model.act(state.unsqueeze(0))
            next_obs, reward, terminated, truncated, _ = env.step(int(action.item()))
            with torch.no_grad():
                _, next_value = model(torch.as_tensor(next_obs, dtype=torch.float32).unsqueeze(0))
            buffer.add(state.numpy(), int(action.item()), reward, float(value.item()),
                       float(logp.item()), float(next_value.item()),
                       terminated=bool(terminated), truncated=bool(truncated))
            total += reward
            obs = next_obs
            if terminated or truncated:
                break
        buffer.compute_returns_and_advantages(gamma=0.99, gae_lambda=0.95)
        rollout = buffer.get()
        ppo_update(model, optimizer, rollout["observations"], rollout["actions"].long(),
                   rollout["log_probs"], rollout["returns"], rollout["advantages"],
                   epochs=4, batch_size=64)
        buffer.clear()
        returns_history.append(total)
        if (episode + 1) % 20 == 0:
            print(f"episode={episode + 1:4d} mean_return_20={np.mean(returns_history[-20:]):.1f}")
    env.close()


if __name__ == "__main__":
    main()
