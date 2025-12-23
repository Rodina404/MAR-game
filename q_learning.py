import numpy as np
import random
from catch_env import CatchGameEnv

# =========================
# Training parameters
# =========================
EPISODES = 5000        # 3adad el games elly el agent hayla3abha

ALPHA = 0.1            # Learning rate (sor3et ta3alom el Q-values)
GAMMA = 0.95           # Discount factor (ahmeyet el mosta2bal)

EPSILON_START = 1.0    # fi el awel explore 100%
EPSILON_MIN = 0.01     # a2al exploration momken
EPSILON_DECAY = 0.995  # kol episode epsilon by2al shwaya

# =========================
# Q-Learning Training
# =========================
def train_q_learning():

    # hena m3mlnash GUI 3ashan el training yeb2a saree3
    env = CatchGameEnv(render_mode=None)

    Q = {}  # Q-table ka dictionary 3ashan el state tuple

    # function btgeeb Q-values lel state
    def get_Q(state):
        if state not in Q:
            Q[state] = np.zeros(env.action_space_size)
        return Q[state]

    epsilon = EPSILON_START
    rewards_per_episode = []  # n5زن reward kol episode

    print("🎯 Starting Q-Learning Training...")

    # loop 3ala kol episode (kol episode = game)
    for episode in range(EPISODES):
        state, _ = env.reset()   # reset el game
        done = False
        total_reward = 0

        # loop gowa el episode (steps)
        while not done:

            # epsilon-greedy:
            # ya explore ya exploit
            if random.random() < epsilon:
                action = random.randint(0, env.action_space_size - 1)
            else:
                action = np.argmax(get_Q(state))

            # na5od step fel environment
            next_state, reward, _, done, _ = env.step(action)

            # Q-learning formula
            best_next_q = np.max(get_Q(next_state))
            get_Q(state)[action] += ALPHA * (
                reward + GAMMA * best_next_q - get_Q(state)[action]
            )

            state = next_state
            total_reward += reward

        # n2alel el exploration shwaya shwaya
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        rewards_per_episode.append(total_reward)

        # print progress kol 100 episode
        if episode % 100 == 0:
            print(
                f"Episode {episode} | "
                f"Reward: {total_reward:.2f} | "
                f"Epsilon: {epsilon:.3f}"
            )

    env.close()

    # save Q-table w rewards
    np.save("q_learning_Q.npy", Q)
    np.save("q_learning_rewards.npy", rewards_per_episode)

    print("✅ Training completed successfully!")

    # n3red el agent ba3d el training
    demonstrate_q_learning(Q)


# =========================
# Demonstration (GUI)
# =========================
def demonstrate_q_learning(Q, num_episodes=5):

    # hena b2a n3ml GUI
    env = CatchGameEnv(render_mode='human')

    def get_Q(state):
        return Q.get(state, np.zeros(env.action_space_size))

    print("\n🎮 Demonstrating Learned Policy...")

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        print(f"\n--- Demo Episode {episode + 1}/{num_episodes} ---")

        while not done:
            # a5tar a7san action (no exploration)
            action = np.argmax(get_Q(state))
            state, reward, _, done, _ = env.step(action)
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
# Run program
# =========================
if __name__ == "__main__":
    train_q_learning()
