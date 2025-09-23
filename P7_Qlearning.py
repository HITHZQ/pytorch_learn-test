import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from itertools import count
import matplotlib.pyplot as plt
import matplotlib
from torch import optim

MEMORY_CAPACITY = 2000
BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9
TARGET_REPLACE_ITERS = 100
GAMMA = 0.9

env = gym.make('CartPole-v1')
env = env.unwrapped

N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]


class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 10)
        self.fc2 = nn.Linear(10, N_ACTIONS)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_values = self.fc2(x)
        return action_values


class Experience(object):
    def __init__(self):
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))

    def add(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def sample(self):
        sample_index = np.random.choice(min(self.memory_counter, MEMORY_CAPACITY), BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))  # Fixed syntax error
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])
        return b_s, b_a, b_r, b_s_


def choose_action(x, Q):
    x = torch.unsqueeze(torch.FloatTensor(x), 0)
    if np.random.rand() < EPSILON:
        action_values = Q.forward(x)
        action = torch.max(action_values, dim=1)[1].data.numpy()[0]
    else:
        action = np.random.randint(0, N_ACTIONS)
    return action


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Q = QNetwork().to(device)
Q_hat = QNetwork().to(device)
Q_hat.load_state_dict(Q.state_dict())
Q_hat.eval()
experience = Experience()
optimizer = optim.RMSprop(Q.parameters(), lr=LR)  # Should optimize Q, not Q_hat
loss_function = nn.MSELoss()

episode_rewards = []

for i_episode in range(4000):
    state, _ = env.reset()  # Gymnasium returns (state, info)
    ep_r = 0
    while True:
        # env.render()  # Comment out for faster training
        action = choose_action(state, Q)
        next_state, reward, terminated, truncated, info = env.step(action)  # Gymnasium returns 5 values

        # Use the actual thresholds from the environment
        x, _, theta, _ = next_state
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        experience.add(state, action, r, next_state)  # Use shaped reward
        ep_r += reward  # Track actual environment reward

        if experience.memory_counter > MEMORY_CAPACITY:  # Fixed attribute name
            b_s, b_a, b_r, b_s_ = experience.sample()
            b_s, b_a, b_r, b_s_ = b_s.to(device), b_a.to(device), b_r.to(device), b_s_.to(device)

            q_eval = Q(b_s).gather(1, b_a)
            q_next = Q_hat(b_s_).detach()
            q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # Added GAMMA and fixed dimension

            loss = loss_function(q_eval, q_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if terminated or truncated:
            episode_rewards.append(ep_r)
            if i_episode % 100 == 0:
                print(f"Episode: {i_episode}, Reward: {ep_r}, Avg Reward: {np.mean(episode_rewards[-100:])}")
            break

        state = next_state

    if i_episode % TARGET_REPLACE_ITERS == 0:
        Q_hat.load_state_dict(Q.state_dict())

env.close()

# Plot the training progress
plt.plot(episode_rewards)
plt.title('Training Progress')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()