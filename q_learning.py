import numpy as np
import random
from catch_env import CatchGameEnv

EPISODES = 3000
ALPHA = 0.1 #How fast Q-values update????"
GAMMA = 0.95
EPSILON = 0.1 #Expolration prob


def train_q_learning():
    env = CatchGameEnv(render_mode=None) 
    Q = {} #Qtable

    def get_Q(state):#ensure elstates w kda 
        if state not in Q:
            Q[state] = np.zeros(env.action_space_size)
        return Q[state]

    rewards_per_episode = []

    for ep in range(EPISODES): #kol loop = 1 game 
        state, _ = env.reset()
        truncated = False #done 
        total_reward = 0 #performance

        while not truncated: #bkrr explore wla exploit 
            if random.random() < EPSILON:
                action = random.randint(0, env.action_space_size - 1)
            else:
                action = np.argmax(get_Q(state))

            next_state, reward, _, truncated, _ = env.step(action)

            best_next = np.max(get_Q(next_state))
            get_Q(state)[action] += ALPHA * (
                reward + GAMMA * best_next - get_Q(state)[action]
            )
            #TD error
            #update Q(s,a) -> the reward + the discounted best future Q-value

            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)

        if ep % 500 == 0:
            print(f"Episode {ep}, Reward: {total_reward:.2f}")

    np.save("q_learning_Q.npy", Q)
    np.save("q_learning_rewards.npy", rewards_per_episode)
    print("Q-Learning training completed.")


if __name__ == "__main__":
    train_q_learning()
