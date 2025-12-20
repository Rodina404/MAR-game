import numpy as np
from catch_env import CatchGameEnv

GAMMA = 0.9
THETA = 1e-4
NUM_SIMULATIONS = 5


def get_all_states(env):
    states = []
    for ax in range(env.observation_space_shape[0]):
        for ix in range(env.observation_space_shape[1]):
            for iy in range(env.observation_space_shape[2]):
                for t in [0, 1]:
                    states.append((ax, ix, iy, t))
    return states


def value_iteration(env):
    states = get_all_states(env)
    num_actions = env.action_space_size

    V = {s: 0.0 for s in states}
    policy = {s: 0 for s in states}

    while True:
        delta = 0
        for state in states:
            v = V[state]
            action_values = []

            for action in range(num_actions):
                total_reward = 0
                for _ in range(NUM_SIMULATIONS):
                    env.reset()
                    env.agent_x = state[0] * 20
                    env.item_x = state[1] * 20
                    env.item_y = state[2] * 20
                    env.item_type = state[3]

                    obs, reward, _, _, _ = env.step(action)
                    total_reward += reward + GAMMA * V.get(obs, 0)

                action_values.append(total_reward / NUM_SIMULATIONS)

            V[state] = max(action_values)
            policy[state] = np.argmax(action_values)
            delta = max(delta, abs(v - V[state]))

        if delta < THETA:
            break

    return policy, V


if __name__ == "__main__":
    env = CatchGameEnv(render_mode=None)
    policy, V = value_iteration(env)
    np.save("value_iteration_policy.npy", policy)
    print("✅ Value Iteration finished and saved.")
