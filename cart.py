# pip install gymnasium[classic-control] numpy
import gymnasium as gym
import numpy as np
from collections import defaultdict

env = gym.make("CartPole-v1")
n_episodes = 2000
alpha = 0.1
gamma = 1.0                 # episodic: undiscounted steps-to-failure
epsilon_start, epsilon_end = 1.0, 0.02
epsilon_decay = 0.995

# Discretize the continuous state into bins (coarse but simple)
bins = {
    0: np.linspace(-4.8,  4.8, 9),    # cart position
    1: np.linspace(-4.0,  4.0, 9),    # cart velocity (clipped range)
    2: np.linspace(-0.418,0.418,9),   # pole angle (~24°)
    3: np.linspace(-4.0,  4.0, 9),    # pole angular velocity
}

def discretize(obs):
    idxs = []
    for i, x in enumerate(obs):
        b = bins[i]
        idxs.append(np.digitize(x, b))
    return tuple(idxs)

n_actions = env.action_space.n
Q = defaultdict(lambda: np.zeros(n_actions))

def epsilon_greedy(s, eps):
    if np.random.rand() < eps:
        return env.action_space.sample()
    return int(np.argmax(Q[s]))

epsilon = epsilon_start
returns = []

for ep in range(n_episodes):
    obs, _ = env.reset()
    s = discretize(obs)
    done = False
    steps = 0

    while not done:
        a = epsilon_greedy(s, epsilon)
        obs2, reward, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        r = 1.0   # +1 per time step until failure (as in Example 3.4)
        s2 = discretize(obs2)

        # Q-learning update
        Q[s][a] += alpha * (r + (0 if done else gamma * np.max(Q[s2])) - Q[s][a])

        s = s2
        steps += 1

    returns.append(steps)  # steps-to-failure = episodic return
    epsilon = max(epsilon_end, epsilon * epsilon_decay)

    if (ep+1) % 100 == 0:
        print(f"Episode {ep+1}: avg steps (last 100) = {np.mean(returns[-100:]):.1f}")

env.close()
