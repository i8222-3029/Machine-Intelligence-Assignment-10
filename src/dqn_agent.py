import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from src.continuous_warehouse import ContinuousWarehouseEnv

class DQNNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(tuple(args))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, env, buffer_size=10000, batch_size=64, gamma=0.99, lr=1e-3, tau=0.005, use_replay=True, use_target=True):
        self.env = env
        self.state_dim = env.state_size
        self.action_dim = env.action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.use_replay = use_replay
        self.use_target = use_target
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNNet(self.state_dim, self.action_dim).to(self.device)
        self.target_net = DQNNet(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.tau = tau
        self.steps = 0

    def select_action(self, state, epsilon=0.1):
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)
        state = torch.FloatTensor(self.env.normalize_state(state)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.policy_net(state)
        return q.argmax().item()

    def update(self):
        if self.use_replay:
            if len(self.replay_buffer) < self.batch_size:
                return
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        else:
            # Use only the most recent transition
            states, actions, rewards, next_states, dones = zip(self.recent_transition)
        states = torch.FloatTensor([self.env.normalize_state(s) for s in states]).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor([self.env.normalize_state(s) for s in next_states]).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.use_target:
                next_q = self.target_net(next_states).max(1)[0]
            else:
                next_q = self.policy_net(next_states).max(1)[0]
            target = rewards + self.gamma * next_q * (1 - dones)
        loss = nn.functional.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.use_target:
            for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train(self, episodes=2000, epsilon=0.1, target_update=100):
        rewards = []
        for ep in range(episodes):
            state = self.env.reset()
            done = False
            ep_reward = 0
            while not done:
                action = self.select_action(state, epsilon)
                next_state, reward, done, _ = self.env.step(action)
                if self.use_replay:
                    self.replay_buffer.push(state, action, reward, next_state, float(done))
                else:
                    self.recent_transition = (state, action, reward, next_state, float(done))
                state = next_state
                ep_reward += reward
                self.update()
                self.steps += 1
                if self.use_target and self.steps % target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
            rewards.append(ep_reward)
        return rewards

if __name__ == "__main__":
    env = ContinuousWarehouseEnv()
    agent = DQNAgent(env)
    rewards = agent.train(episodes=2000)
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter1d
    plt.plot(uniform_filter1d(rewards, size=100), label="DQN")
    plt.xlabel("Episode")
    plt.ylabel("Reward (rolling avg)")
    plt.legend()
    plt.show()
