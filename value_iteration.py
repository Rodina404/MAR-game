import numpy as np
import random
from catch_env import CatchGameEnv

# =========================
# Parameters
# =========================
EPISODES = 20000        # more episodes for convergence
ALPHA = 0.2             # learning rate
GAMMA = 0.95            # discount factor
EPSILON_START = 1.0     # initial exploration
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

# =========================
# Reward shaping function
# =========================
def shaped_reward(obs, next_obs, raw_reward):
    # small positive reward if agent moves closer to the item
    agent_x, agent_y = obs[0], 0  # agent_y fixed
    next_agent_x = next_obs[0]
    item_x, item_y = next_obs[1], next_obs[2]
    
    old_dist = abs(agent_x - item_x)
    new_dist = abs(next_agent_x - item_x)
    
    # closer = +0.1, farther = -0.1
    distance_reward = 0.1 if new_dist < old_dist else -0.1
    return raw_reward + distance_reward

# =========================
# Discretize observation to Q-table state
# =========================
def obs_to_state(obs):
    return (
        obs[0] // 10,  # agent_x index (finer grid)
        obs[1] // 10,  # item_x index
        obs[2] // 10,  # item_y index
        obs[3]          # item type
    )

# =========================
# Training (Model-Free Value Iteration)
# =========================
def train_value_iteration():
    env = CatchGameEnv(render_mode=None)  # no GUI
    Q = {}  # Q-table as dictionary

    def get_Q(state):
        if state not in Q:
            Q[state] = np.zeros(env.action_space_size)
        return Q[state]

    epsilon = EPSILON_START
    rewards_per_episode = []

    print("🎯 Starting Model-Free Value Iteration (stochastic env)...")

    for episode in range(EPISODES):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            state = obs_to_state(obs)

            # epsilon-greedy
            if random.random() < epsilon:
                action = random.randint(0, env.action_space_size - 1)
            else:
                action = np.argmax(get_Q(state))

            next_obs, reward, _, done, _ = env.step(action)
            reward = shaped_reward(obs, next_obs, reward)  # reward shaping

            next_state = obs_to_state(next_obs)

            # Q-learning update
            best_next_q = np.max(get_Q(next_state))
            get_Q(state)[action] += ALPHA * (reward + GAMMA * best_next_q - get_Q(state)[action])

            obs = next_obs
            total_reward += reward

        # decay exploration
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        rewards_per_episode.append(total_reward)

        if episode % 500 == 0:
            print(f"Episode {episode} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")

    np.save("value_iteration_Q.npy", Q)
    np.save("value_iteration_rewards.npy", rewards_per_episode)
    print("✅ Training finished and saved!")
    env.close()

    demonstrate_value_iteration(Q)

# =========================
# Demonstration
# =========================
def demonstrate_value_iteration(Q, num_episodes=5):
    env = CatchGameEnv(render_mode='human')
    print("\n🎮 Demonstrating Learned Policy...")

    def get_Q_state(state):
        return Q.get(state, np.zeros(env.action_space_size))

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        print(f"\n--- Demo Episode {episode + 1}/{num_episodes} ---")

        while not done:
            state = obs_to_state(obs)
            action = np.argmax(get_Q_state(state))
            obs, reward, _, done, _ = env.step(action)
            total_reward += reward
            env.render()

            # handle close button
            import pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return

        print(f"Total Reward: {total_reward:.2f}")

    env.close()
    print("\n✅ Demonstration finished!")

# =========================
# Run
# =========================
if __name__ == "__main__":
    train_value_iteration()
