"""Train the discrete PPO baseline on Gymnasium CartPole-v1."""
import numpy as np
import torch
import gymnasium as gym
from algorithms.policy_based.actor_critic import ActorCritic, generalized_advantage_estimate
from algorithms.policy_based.ppo import ppo_update


def main(episodes=500, seed=7):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = gym.make("CartPole-v1")
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    returns_history = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed if episode == 0 else None)
        states, actions, logps, rewards, values, next_values, dones = [], [], [], [], [], [], []
        total = 0.0
        while True:
            state = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                action, logp, value = model.act(state.unsqueeze(0))
            next_obs, reward, terminated, truncated, _ = env.step(int(action.item()))
            with torch.no_grad():
                _, next_value = model(torch.as_tensor(next_obs, dtype=torch.float32).unsqueeze(0))
            states.append(state)
            actions.append(action.squeeze(0))
            logps.append(logp.squeeze(0))
            rewards.append(float(reward))
            values.append(value.squeeze(0))
            next_values.append(torch.zeros_like(next_value.squeeze(0)) if terminated else next_value.squeeze(0))
            dones.append(float(terminated))
            total += reward
            obs = next_obs
            if terminated or truncated:
                break
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        values_t = torch.stack(values)
        next_values_t = torch.stack(next_values)
        dones_t = torch.tensor(dones, dtype=torch.float32)
        advantages, value_targets = generalized_advantage_estimate(
            rewards_t, values_t, next_values_t, dones_t)
        ppo_update(model, optimizer, torch.stack(states), torch.stack(actions).long(),
                   torch.stack(logps).detach(), value_targets.detach(), advantages.detach(),
                   epochs=4, batch_size=64)
        returns_history.append(total)
        if (episode + 1) % 20 == 0:
            print(f"episode={episode + 1:4d} mean_return_20={np.mean(returns_history[-20:]):.1f}")
    env.close()


if __name__ == "__main__":
    main()
