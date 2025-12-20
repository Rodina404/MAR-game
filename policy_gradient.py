import numpy as np
from catch_env import CatchGameEnv

EPISODES = 2000
ALPHA = 0.01
GAMMA = 0.99


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()
#b3ml stochastic policy 

def train_policy_gradient():
    env = CatchGameEnv(render_mode=None)
    theta = {}
    #kol state leha weights l kol action 
    def get_theta(state):
        if state not in theta:
            theta[state] = np.zeros(env.action_space_size)
        return theta[state]
    #zy Q tables 

    rewards_per_episode = []

    for ep in range(EPISODES):
        state, _ = env.reset()
        episode = [] #Stores: (state, action, reward) for entire episode
        truncated = False

        while not truncated:
            probs = softmax(get_theta(state)) 
            action = np.random.choice(env.action_space_size, p=probs) #bkhtar action by probability 
            next_state, reward, _, truncated, _ = env.step(action)

            episode.append((state, action, reward)) #save 3shan y learn 
            state = next_state

        G = 0
        for state, action, reward in reversed(episode):
            G = reward + GAMMA * G
            probs = softmax(get_theta(state)) #>p --> good , <p --> bad
            get_theta(state)[action] += ALPHA * (1 - probs[action]) * G

        rewards_per_episode.append(sum(r for _, _, r in episode))

        if ep % 300 == 0:
            print(f"Episode {ep}, Reward: {rewards_per_episode[-1]:.2f}")

    np.save("policy_gradient_theta.npy", theta)
    np.save("policy_gradient_rewards.npy", rewards_per_episode)
    print("✅ Policy Gradient training completed.")


if __name__ == "__main__":
    train_policy_gradient()
