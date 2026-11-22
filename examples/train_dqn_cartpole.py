"""Train the teaching DQN agent on Gymnasium CartPole-v1."""
import random
import gymnasium as gym
from algorithms.value_based.dqn import DQNAgent, ReplayBuffer


def main(episodes=300, seed=7):
    env = gym.make("CartPole-v1")
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    replay = ReplayBuffer(50_000)
    random.seed(seed)
    scores = []
    for episode in range(episodes):
        state, _ = env.reset(seed=seed if episode == 0 else None)
        epsilon = max(0.05, 1.0 - episode / (episodes * 0.75))
        total = 0.0
        while True:
            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            replay.add(state, action, reward, next_state, terminated)
            agent.update(replay, batch_size=64)
            state = next_state
            total += reward
            if terminated or truncated:
                break
        scores.append(total)
        if (episode + 1) % 20 == 0:
            print(f"episode={episode + 1:4d} mean_return_20={sum(scores[-20:]) / 20:.1f}")
    env.close()


if __name__ == "__main__":
    main()
