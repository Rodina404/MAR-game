import numpy as np
import random
from catch_env import CatchGameEnv

# =========================
# Training parameters
# =========================
EPISODES = 5000
ALPHA = 0.01
GAMMA = 0.95

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

# =========================
# Policy Gradient Training
# =========================
def train_policy_gradient():
    env = CatchGameEnv(render_mode=None)
    theta = {}

    def get_theta(state):
        if state not in theta:
            theta[state] = np.zeros(env.action_space_size)
        return theta[state]

    rewards_per_episode = []
    baseline = 0  # Initialize baseline

    print("🎯 Starting Policy Gradient Training...")

    for episode in range(EPISODES):
        state, _ = env.reset()
        done = False
        episode_data = []

        while not done:
            probs = softmax(get_theta(state))
            action = np.random.choice(env.action_space_size, p=probs)
            next_state, reward, _, done, _ = env.step(action)
            episode_data.append((state, action, reward))
            state = next_state

        # Compute total reward for baseline
        total_reward = sum(r for _, _, r in episode_data)
        rewards_per_episode.append(total_reward)

        # Update baseline (running average)
        baseline = 0.9 * baseline + 0.1 * total_reward

        # Update policy weights with baseline
        G = 0
        for state, action, reward in reversed(episode_data):
            G = reward + GAMMA * G
            probs = softmax(get_theta(state))
            get_theta(state)[action] += ALPHA * (G - baseline) * (1 - probs[action])

        if episode % 100 == 0:
            print(f"Episode {episode} | Reward: {total_reward:.2f} | Baseline: {baseline:.2f}")

    # Save results
    np.save("policy_gradient_theta.npy", theta)
    np.save("policy_gradient_rewards.npy", rewards_per_episode)
    print("✅ Training completed successfully!")

    # Demonstrate
    demonstrate_policy_gradient(theta)


# =========================
# Demonstration (GUI)
# =========================
def demonstrate_policy_gradient(theta, num_episodes=5):
    env = CatchGameEnv(render_mode='human')

    def get_theta_local(state):
        return theta.get(state, np.zeros(env.action_space_size))

    print("\n🎮 Demonstrating Learned Policy...")

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        print(f"\n--- Demo Episode {episode + 1}/{num_episodes} ---")
        while not done:
            probs = softmax(get_theta_local(state))
            action = np.argmax(probs)
            state, reward, _, done, _ = env.step(action)
            total_reward += reward
            env.render()

            import pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return

        print(f"Total Reward: {total_reward:.2f}")

    env.close()
    print("\n✅ Demonstration finished!")


if __name__ == "__main__":
    train_policy_gradient()
