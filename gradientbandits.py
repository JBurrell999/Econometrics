import numpy as np

k = 10                     # number of arms
H = np.zeros(k)            # preferences
avg_reward = 0
alpha = 0.1
steps = 1000

true_means = np.random.randn(k)  # unknown real reward means
rewards = []

for t in range(1, steps+1):
    probs = np.exp(H) / np.sum(np.exp(H))       # softmax
    action = np.random.choice(k, p=probs)       # sample from policy
    reward = np.random.randn() + true_means[action]  # noisy reward
    avg_reward += (reward - avg_reward) / t
    for a in range(k):
        H[a] += alpha * (reward - avg_reward) * ((a == action) - probs[a])
    rewards.append(reward)